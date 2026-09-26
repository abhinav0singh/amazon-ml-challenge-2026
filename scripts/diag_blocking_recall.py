"""
diag_blocking_recall.py: FULL-HAYSTACK blocking recall on a subset of S1.

Why this exists: sampled runs shrink the S2/S3 haystack, so every sampled recall
ceiling was optimistic -- B2 at 40k projected recall 0.983 / cover 0.955, the
first full-scale run measured 0.885 / 0.766. Here only the S1 side is subsampled;
the S2/S3 side is the FULL country group, so every block has its true full-scale
size and the top-k / per-entity-cap competition is the real one. Cost is the
TF-IDF fit over the group plus work proportional to the S1 subset.

From ONE blocking pass it reports recall before the per-entity cap and after
several cap sizes and ranking scores (the cap is applied post hoc), plus what
each key scheme contributes. Optional extra passes: doubled per-block K, and
including the oversized blocks that MAX_BLOCK skips.

    .venv\\Scripts\\python scripts\\diag_blocking_recall.py --data <DATA> --country india --n 10000

Numbers from this script are full-haystack measurements on an S1 subset. They
are blocking diagnostics, NOT CV, and go in STATUS.md section 4.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import blocking as B  # noqa: E402
from data_io import load_truth  # noqa: E402

COLS = ["entity_id", "country_n", "name_core", "full_n", "addr_n", "postal1"]
CAPS = [20, 40, 60, 80, 100, 150, 10**9]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_country(work, tag, country):
    path = os.path.join(work, f"norm_{tag}.parquet")
    return pd.read_parquet(path, columns=COLS,
                           filters=[("country_n", "==", country)]).reset_index(drop=True)


def block_pairs(s1, mats, k1, k23, max_block, kmult=1, oversized_only=False):
    """The _block_by_keys loop, instrumented: returns unique (ia, ib) pairs plus,
    per scheme, the pairs that scheme found. `oversized_only` matches ONLY the
    blocks MAX_BLOCK normally skips, to measure what skipping them costs."""
    found, per_scheme, n_blocks, n_over = [], {}, 0, 0
    for name in k1:
        ga = B._group_positions(*k1[name])
        gb = B._group_positions(*k23[name])
        got = []
        for key, pa in ga.items():
            pb = gb.get(key)
            if pb is None:
                continue
            over = pb.size > max_block
            n_over += over
            if over != oversized_only:
                continue
            n_blocks += 1
            for _, (A, Bm, k) in mats.items():
                r, c = B._topk_sparse(A[pa], Bm[pb], k * kmult)
                if r.size:
                    got.append(np.stack([pa[r], pb[c]], axis=1))
        if got:
            g = np.unique(np.concatenate(got), axis=0)
            per_scheme[name] = g
            found.append(g)
    pairs = np.unique(np.concatenate(found), axis=0) if found else np.empty((0, 2), np.int64)
    return pairs, per_scheme, n_blocks, n_over


def pair_keys(ia, ib, n23):
    return ia.astype(np.int64) * n23 + ib.astype(np.int64)


def evaluate(cand, truth_sub, s1_ids, label):
    """Recall / cover before the cap and after each (score, cap) combination."""
    n_true = {s: len(truth_sub.get(s, ())) for s in s1_ids}
    tot = sum(n_true.values())
    y = np.fromiter((c in truth_sub.get(s, ()) for s, c in zip(cand["s1_id"], cand["cand_id"])),
                    dtype=bool, count=len(cand))
    scores = {
        "max(current)": cand[["cos_name", "cos_full", "cos_addr"]].max(axis=1),
        "sum": cand["cos_name"] + cand["cos_full"] + cand["cos_addr"],
        "cos_full": cand["cos_full"],
        "full+addr": cand["cos_full"] + cand["cos_addr"],
    }
    out = {"label": label, "true_pairs": tot, "entities": len(s1_ids),
           "cands_per_s1_precap": len(cand) / len(s1_ids), "rows": {}}
    per = cand.groupby("s1_id").size()
    out["precap_cands_p50_p90_max"] = [float(per.quantile(.5)), float(per.quantile(.9)), int(per.max())]
    for sname, sc in scores.items():
        rank = sc.groupby(cand["s1_id"]).rank(ascending=False, method="first").to_numpy()
        for cap in CAPS:
            keep = y & (rank <= cap)
            hit_per = pd.Series(keep).groupby(cand["s1_id"].to_numpy()).sum()
            hit = int(keep.sum())
            full = sum(1 for s in s1_ids if hit_per.get(s, 0) == n_true[s])
            # Oracle macro-F0.5: a PERFECT matcher on these candidates predicts exactly
            # truth & candidates -- precision 1, recall = hits / |truth|. This is the
            # ceiling the candidate set puts on the metric itself (cover is a proxy).
            f = 0.0
            for s in s1_ids:
                nt, h = n_true[s], hit_per.get(s, 0)
                f += 1.0 if nt == 0 else (0.0 if h == 0 else 1.25 * (h / nt) / (0.25 + h / nt))
            key = f"{sname} cap={'none' if cap >= 10**9 else cap}"
            out["rows"][key] = {"recall": hit / tot, "cover": full / len(s1_ids),
                                "oracle_f05": f / len(s1_ids)}
    return out


def keys_with(df, dfn, dfa, n_name, n_addr):
    """blocking.blocking_keys with the rare-token counts as parameters, to measure
    what indexing more tokens buys. Mirrors blocking_keys exactly otherwise."""
    core_ns = df["name_core"].str.replace(" ", "", regex=False)
    return {"pfx5": B._single(core_ns.str[:5].to_numpy()),
            "postal": B._single(B._postal_key(df, core_ns)),
            "nametok": B._rare_tokens(df["name_core"].to_numpy(), dfn, n_name, "n:"),
            "addrtok": B._rare_tokens(df["addr_n"].to_numpy(), dfa, n_addr, "a:")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--country", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kmult", type=int, default=2, help="extra pass with per-block K multiplied (0 = skip)")
    ap.add_argument("--oversized", action="store_true", help="extra pass over the skipped oversized blocks")
    ap.add_argument("--ntok", nargs="*", default=[], type=lambda s: tuple(int(x) for x in s.split(",")),
                    help="extra key variants as name,addr rare-token counts, e.g. --ntok 3,3 4,3")
    ap.add_argument("--out", default=None)
    # The pipeline names caches f"train_s1{args.sample}", so the full-scale cache
    # is 'train_s10' / 'train_s230' (sample = 0), not 'train_s1'.
    ap.add_argument("--sample-tag", default="0", help="norm cache suffix: 0 = full scale; 2000 = a --sample 2000 run")
    args = ap.parse_args()

    t0 = time.time()
    s1 = load_country(args.work, f"train_s1{args.sample_tag}", args.country)
    s23 = load_country(args.work, f"train_s23{args.sample_tag}", args.country)
    rng = np.random.default_rng(args.seed)
    s1 = s1.iloc[np.sort(rng.choice(len(s1), size=min(args.n, len(s1)), replace=False))].reset_index(drop=True)
    truth = load_truth(args.data)
    s1_ids = s1["entity_id"].tolist()
    truth_sub = {s: truth.get(s, set()) for s in s1_ids}
    del truth
    log(f"{args.country}: {len(s1)} S1 subset vs FULL {len(s23)} S2/S3 (load {time.time()-t0:.0f}s)")

    t = time.time()
    mats = B._fit_views(s1, s23)
    dfn = B._doc_freq(s1["name_core"].to_numpy(), s23["name_core"].to_numpy())
    dfa = B._doc_freq(s1["addr_n"].to_numpy(), s23["addr_n"].to_numpy())
    k1, k23 = B.blocking_keys(s1, dfn, dfa), B.blocking_keys(s23, dfn, dfa)
    log(f"fit views + keys: {time.time()-t:.0f}s")

    report = {"country": args.country, "n_s1": len(s1), "n_s23": len(s23),
              "config": {"K": [B.K_NAME, B.K_FULL, B.K_ADDR], "MAX_BLOCK": B.MAX_BLOCK,
                         "MAX_CANDS_in_code": B.MAX_CANDS, "MAX_DF": B.MAX_DF}}
    t = time.time()
    pairs, per_scheme, nb, nover = block_pairs(s1, mats, k1, k23, B.MAX_BLOCK)
    report["base_blocking_secs"] = time.time() - t
    report["blocks_matched"], report["oversized_keys_seen"] = nb, nover
    cand = B._finish(s1, s23, mats, pairs[:, 0], pairs[:, 1])
    report["base"] = evaluate(cand, truth_sub, s1_ids, "base K, MAX_BLOCK skip")
    log(f"base pass {report['base_blocking_secs']:.0f}s, {len(cand)} pairs")

    # what each scheme finds, and what ONLY it finds (true pairs)
    n23 = len(s23)
    ids1, ids23 = s1["entity_id"].to_numpy(), s23["entity_id"].to_numpy()
    true_keys = set()
    pos1 = {s: i for i, s in enumerate(ids1)}
    pos23 = pd.Series(np.arange(n23), index=ids23)
    for s, ms in truth_sub.items():
        for m in ms:
            if m in pos23.index:
                true_keys.add(pos1[s] * n23 + int(pos23[m]))
    tot = sum(len(v) for v in truth_sub.values())
    scheme_true = {n: set(pair_keys(g[:, 0], g[:, 1], n23).tolist()) & true_keys for n, g in per_scheme.items()}
    report["schemes"] = {}
    for n, st in scheme_true.items():
        others = set().union(*[v for m, v in scheme_true.items() if m != n]) if len(scheme_true) > 1 else set()
        report["schemes"][n] = {"recall_alone": len(st) / tot, "unique_true": len(st - others)}

    for n_name, n_addr in args.ntok:
        t = time.time()
        kv1, kv23 = keys_with(s1, dfn, dfa, n_name, n_addr), keys_with(s23, dfn, dfa, n_name, n_addr)
        pv, _, _, _ = block_pairs(s1, mats, kv1, kv23, B.MAX_BLOCK)
        cv_ = B._finish(s1, s23, mats, pv[:, 0], pv[:, 1])
        k = f"ntok_name{n_name}_addr{n_addr}"
        report[k] = evaluate(cv_, truth_sub, s1_ids, f"n_tok name {n_name} / addr {n_addr}")
        report[k]["secs"] = time.time() - t
        log(f"{k} pass {time.time()-t:.0f}s, {len(cv_)} pairs")

    if args.kmult and args.kmult > 1:
        t = time.time()
        p2, _, _, _ = block_pairs(s1, mats, k1, k23, B.MAX_BLOCK, kmult=args.kmult)
        c2 = B._finish(s1, s23, mats, p2[:, 0], p2[:, 1])
        report[f"kx{args.kmult}"] = evaluate(c2, truth_sub, s1_ids, f"K x{args.kmult}")
        report[f"kx{args.kmult}"]["secs"] = time.time() - t
        log(f"K x{args.kmult} pass {time.time()-t:.0f}s, {len(c2)} pairs")

    if args.oversized:
        t = time.time()
        po, _, nbo, _ = block_pairs(s1, mats, k1, k23, B.MAX_BLOCK, oversized_only=True)
        allp = np.unique(np.concatenate([pairs, po]), axis=0)
        co = B._finish(s1, s23, mats, allp[:, 0], allp[:, 1])
        report["with_oversized"] = evaluate(co, truth_sub, s1_ids, "base + oversized blocks")
        report["with_oversized"]["secs"] = time.time() - t
        report["with_oversized"]["oversized_blocks_matched"] = nbo
        log(f"oversized pass {time.time()-t:.0f}s")

    out = args.out or os.path.join(args.work, f"diag_recall_{args.country}_{args.n}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    log(f"wrote {out} (total {time.time()-t0:.0f}s)")
    for sec in ["base", f"kx{args.kmult}", "with_oversized"] + [
            f"ntok_name{a}_addr{b}" for a, b in args.ntok]:
        if sec in report:
            r = report[sec]["rows"]
            print(f"--- {report[sec]['label']}: precap cands/S1 {report[sec]['cands_per_s1_precap']:.1f}")
            for k in ("max(current) cap=40", "max(current) cap=none", "full+addr cap=40",
                      "full+addr cap=60", "full+addr cap=80", "full+addr cap=none"):
                print(f"   {k:24s} recall {r[k]['recall']:.4f}  cover {r[k]['cover']:.4f}  "
                      f"oracle F0.5 {r[k]['oracle_f05']:.4f}")
    print("schemes:", json.dumps(report["schemes"]))


if __name__ == "__main__":
    main()
