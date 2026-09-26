"""
diag_blocking_combo.py: WHY true pairs are missed, and which blocking changes
recover them -- full haystack, one TF-IDF fit, many configurations.

    .venv\\Scripts\\python scripts\\diag_blocking_combo.py --data <DATA> --country india --n 10000

Builds on diag_blocking_recall.py (same loading and evaluation). For the current
blocking it analyses the true pairs that never become candidates (similarity
levels, shared tokens, examples), then scores each configuration in CONFIGS by
pre-cap / capped recall, cover and oracle macro-F0.5. Diagnostic only, not CV.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import blocking as B  # noqa: E402
from diag_blocking_recall import block_pairs, evaluate, keys_with, load_country, log  # noqa: E402
from data_io import load_truth  # noqa: E402


def rare_prefix_tokens(series, doc_freq, n_tok, plen, prefix):
    """Like blocking._rare_tokens, but keyed on each token's first `plen` chars:
    survives typos, inflections and transliteration variants past the prefix."""
    rows, keys = [], []
    for i, s in enumerate(series):
        t = s.split()
        if not t:
            continue
        for tok in sorted(set(t), key=lambda x: (doc_freq.get(x, 0), x))[:n_tok]:
            if len(tok) >= plen:
                rows.append(i)
                keys.append(prefix + tok[:plen])
    return np.asarray(rows, dtype=np.int64), np.asarray(keys, dtype=object)


def num_bigrams(series, prefix="nb:"):
    """Keys on consecutive pairs of address tokens that contain a digit
    ("3 83", "8 2", "2 601"): house/plot numbers survive when the name is missing
    and the address is reduced to numbers + city, which is exactly how the
    missed India S2/S3 records look."""
    rows, keys = [], []
    for i, s in enumerate(series):
        t = [x for x in s.split() if any(ch.isdigit() for ch in x)]
        for a, b in set(zip(t, t[1:])):
            rows.append(i)
            keys.append(f"{prefix}{a}_{b}")
    return np.asarray(rows, dtype=np.int64), np.asarray(keys, dtype=object)


def make_keys(df, dfn, dfa, cfg, side="s23"):
    n_addr = cfg.get("n_addr_s1", cfg.get("n_addr", 2)) if side == "s1" else cfg.get("n_addr", 2)
    k = keys_with(df, dfn, dfa, cfg.get("n_name", 3), n_addr)
    if cfg.get("num_bigrams"):
        k["numbig"] = num_bigrams(df["addr_n"].to_numpy())
    if cfg.get("name_pfx"):
        n, p = cfg["name_pfx"]
        k["npfx"] = rare_prefix_tokens(df["name_core"].to_numpy(), dfn, n, p, "np:")
    if cfg.get("addr_pfx"):
        n, p = cfg["addr_pfx"]
        k["apfx"] = rare_prefix_tokens(df["addr_n"].to_numpy(), dfa, n, p, "ap:")
    return k


CONFIGS = {
    "v1 (current)":            {},
    "addr4":                   {"n_addr": 4},
    "addr4 + K2":              {"n_addr": 4, "kmult": 2},
    "addr4 + oversized":       {"n_addr": 4, "oversized": True},
    "addr4 + name-pfx(3,4)":   {"n_addr": 4, "name_pfx": (3, 4)},
    "addr4 + addr-pfx(3,5)":   {"n_addr": 4, "addr_pfx": (3, 5)},
    "addr4 + both pfx + K2":   {"n_addr": 4, "name_pfx": (3, 4), "addr_pfx": (3, 5), "kmult": 2},
    # asymmetric: long S1 addresses indexed under many tokens, short S2/S3 under few
    "asym S1 addr8 / S23 addr2": {"n_addr_s1": 8, "n_addr": 2},
    "asym S1 addr8 / S23 addr3": {"n_addr_s1": 8, "n_addr": 3},
    "num-bigrams":             {"num_bigrams": True},
    "asym 8/3 + num-bigrams":  {"n_addr_s1": 8, "n_addr": 3, "num_bigrams": True},
    "asym 12/4 + num-bigrams": {"n_addr_s1": 12, "n_addr": 4, "num_bigrams": True},
}


def miss_analysis(s1, s23, mats, cand, truth_sub, n_show=25):
    """Where do the missed true pairs sit? Similarity, shared tokens, examples."""
    ids1, ids23 = s1["entity_id"].to_numpy(), s23["entity_id"].to_numpy()
    p1 = {s: i for i, s in enumerate(ids1)}
    p23 = pd.Series(np.arange(len(ids23)), index=ids23)
    have = set(zip(cand["s1_id"], cand["cand_id"]))
    ia, ib = [], []
    for s, ms in truth_sub.items():
        for m in ms:
            if (s, m) not in have and m in p23.index:
                ia.append(p1[s]); ib.append(int(p23[m]))
    if not ia:
        return {"missed": 0}
    ia, ib = np.asarray(ia), np.asarray(ib)
    miss = B._finish(s1, s23, mats, ia, ib)
    nc1, nc2 = s1["name_core"].to_numpy()[ia], s23["name_core"].to_numpy()[ib]
    ad1, ad2 = s1["addr_n"].to_numpy()[ia], s23["addr_n"].to_numpy()[ib]
    share_n = np.array([bool(set(a.split()) & set(b.split())) for a, b in zip(nc1, nc2)])
    share_a = np.array([bool(set(a.split()) & set(b.split())) for a, b in zip(ad1, ad2)])
    out = {"missed": int(len(ia)),
           "share_any_name_token": float(share_n.mean()), "share_any_addr_token": float(share_a.mean()),
           "share_neither": float((~share_n & ~share_a).mean()),
           "cos_full_quantiles_10_50_90": [float(x) for x in np.quantile(miss["cos_full"], [.1, .5, .9])],
           "cos_name_quantiles_10_50_90": [float(x) for x in np.quantile(miss["cos_name"], [.1, .5, .9])],
           "cos_addr_quantiles_10_50_90": [float(x) for x in np.quantile(miss["cos_addr"], [.1, .5, .9])]}
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ia), size=min(n_show, len(ia)), replace=False)
    out["examples"] = [{"s1": [nc1[j], ad1[j]], "s23": [nc2[j], ad2[j]],
                        "cos": [round(float(miss[c].iloc[j]), 3) for c in ("cos_name", "cos_full", "cos_addr")]}
                       for j in pick]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--country", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-tag", default="0")
    ap.add_argument("--only", nargs="*", default=None, help="subset of CONFIGS names")
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
    log(f"{args.country}: {len(s1)} S1 vs FULL {len(s23)} S2/S3")
    mats = B._fit_views(s1, s23)
    dfn = B._doc_freq(s1["name_core"].to_numpy(), s23["name_core"].to_numpy())
    dfa = B._doc_freq(s1["addr_n"].to_numpy(), s23["addr_n"].to_numpy())
    log(f"fit done ({time.time()-t0:.0f}s)")

    report = {"country": args.country, "n_s1": len(s1), "configs": {}}
    out = os.path.join(args.work, f"diag_combo_{args.country}_{args.n}.json")
    for name, cfg in CONFIGS.items():
        if args.only and name not in args.only:
            continue
        t = time.time()
        k1, k23 = make_keys(s1, dfn, dfa, cfg, "s1"), make_keys(s23, dfn, dfa, cfg, "s23")
        p, _, _, _ = block_pairs(s1, mats, k1, k23, B.MAX_BLOCK, kmult=cfg.get("kmult", 1))
        if cfg.get("oversized"):
            po, _, _, _ = block_pairs(s1, mats, k1, k23, B.MAX_BLOCK, kmult=cfg.get("kmult", 1),
                                      oversized_only=True)
            p = np.unique(np.concatenate([p, po]), axis=0)
        cand = B._finish(s1, s23, mats, p[:, 0], p[:, 1])
        ev = evaluate(cand, truth_sub, s1_ids, name)
        ev["secs"] = round(time.time() - t)
        if name == "v1 (current)":
            report["miss_analysis_v1"] = miss_analysis(s1, s23, mats, cand, truth_sub)
            m = report["miss_analysis_v1"]
            log(f"v1 misses {m['missed']}: share a name token {m.get('share_any_name_token', 0):.3f}, "
                f"an addr token {m.get('share_any_addr_token', 0):.3f}, neither {m.get('share_neither', 0):.3f}; "
                f"cos_full q10/50/90 {m.get('cos_full_quantiles_10_50_90')}")
        report["configs"][name] = ev
        r = ev["rows"]
        log(f"{name:24s} precap {ev['cands_per_s1_precap']:6.1f}/S1 {ev['secs']:5d}s | "
            + " | ".join(f"cap{c} R {r[f'full+addr cap={c}']['recall']:.4f} "
                         f"C {r[f'full+addr cap={c}']['cover']:.4f} "
                         f"O {r[f'full+addr cap={c}']['oracle_f05']:.4f}" for c in ("80", "100", "none")))
        with open(out, "w") as f:
            json.dump(report, f, indent=1, default=str)
    log(f"wrote {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
