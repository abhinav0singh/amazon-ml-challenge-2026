"""
decide_eval.py: compare decision-layer rules on a SAMPLED run (issue #3).

    python scripts/decide_eval.py --data C:/amlc/dataset --work work_sample --sample 20000

Everything this prints is a SAMPLE number. A sampled run shrinks the haystack, so
its level is optimistic and it must never be quoted as CV or logged as a result in
STATUS.md section 5 (AGENTS.md section 4). What a sample CAN support is a
comparison between decision rules scored on identical pairs, which is what this
script is for.

It does not modify anything under src/. It imports the pipeline's own functions,
so the pair frame it builds is the one run_pipeline.py builds, and it asserts that
against the pipeline's own oof_pairs.parquet before reporting anything.

What it measures
  1. The contested share: how often two S1 entities want the same record above the
     threshold. If that is tiny there is nothing for a smarter conflict rule to fix.
  2. Every decision rule, through decide.cross_fitted_rule -- parameters chosen on
     four folds, scored on the fifth, over the FULL S1 id list including entities
     blocking found nothing for. tune() is never used to produce a reported number.
  3. Seed-to-seed noise, by refitting the same folds with different LightGBM seeds.
     AGENTS.md section 4 accepts a change only if it gains more than 2x that noise
     AND improves at least 4 of 5 folds, so the noise has to be measured, not
     assumed.
  4. Leave-one-country-out, the only proxy available for unseen France. Each rule
     is applied to LOCO probabilities at the parameters cross-fitting chose, which
     is the same convention run_pipeline.py uses for its own LOCO line.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from blocking import generate_candidates  # noqa: E402
from cv_folds import make_s1_folds  # noqa: E402
from data_io import load_split, load_truth, read_tsv  # noqa: E402
from decide import (apply_expected_f05, apply_relative_rule, apply_rule,  # noqa: E402
                    check_one_to_one, contested_share, cross_fitted_rule)
from metric import macro_f05, precision_recall  # noqa: E402
from normalize import add_normalized_columns  # noqa: E402
from pair_features import FEATURES, build_pair_features  # noqa: E402
from run_pipeline import SEED, fit_model, label_pairs, subsample  # noqa: E402

T_GRID = np.round(np.arange(0.20, 0.96, 0.025), 3)          # the grid tune() uses
FLOOR_GRID = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
REL_GRID = [(t, a) for t in (0.05, 0.15, 0.30, 0.45, 0.60)
            for a in (0.0, 0.2, 0.4, 0.6, 0.8)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _countries_for(data: str, s1_ids) -> dict:
    """{s1_id: normalised country} for EVERY sampled entity, read from the source
    file rather than from the pair frame. An entity blocking found no candidates
    for has no row in the pair frame but is still a real prediction of "empty" and
    must still be scored, so its country has to come from somewhere else."""
    from normalize import basic_clean
    s1 = read_tsv(os.path.join(data, "train", "train_source1.tsv"))
    want = set(s1_ids)
    s1 = s1[s1["entity_id"].isin(want)]
    return {e: basic_clean(c) for e, c in zip(s1["entity_id"], s1["country"])}


def build_frame(data: str, work: str, n: int, sample_seed: int):
    """Rebuild the sampled train pair frame exactly as run_pipeline.py does.

    Cached to <work>/pairs_features.parquet: blocking and the feature loop are the
    expensive part, and every model variant below reuses the same frame.
    Returns (pairs, s1_ids, truth, country) where `country` covers every sampled
    entity, not only the ones that got candidates.
    """
    cache = os.path.join(work, f"pairs_features_sample{n}.parquet")
    meta = os.path.join(work, f"sample_meta{n}.json")
    if os.path.exists(cache) and os.path.exists(meta):
        log(f"reusing cached pair frame {cache}")
        with open(meta) as f:
            m = json.load(f)
        if "country" not in m:      # cache written before countries were stored
            log("cached meta has no country map; reading it from train_source1.tsv")
            m["country"] = _countries_for(data, m["s1_ids"])
            with open(meta, "w") as f:
                json.dump(m, f)
        return (pd.read_parquet(cache), m["s1_ids"],
                {s: set(v) for s, v in m["truth"].items()}, m["country"])

    log("loading + normalising train")
    s1, s23 = load_split(data, "train")
    truth = load_truth(data)
    s1, s23 = subsample(s1, s23, n, sample_seed, truth)
    log(f"sampled train: {len(s1)} S1, {len(s23)} S2/S3")
    s1, s23 = add_normalized_columns(s1), add_normalized_columns(s23)
    s1_ids = s1["entity_id"].tolist()
    truth = {s: truth.get(s, set()) for s in s1_ids}

    log("blocking")
    cand = generate_candidates(s1, s23, by_country=True)
    log(f"pair features for {len(cand)} candidate pairs")
    pairs = build_pair_features(cand, s1, s23)
    pairs["y"] = label_pairs(pairs, truth)
    pairs["c"] = pairs["s1_id"].map(s1.set_index("entity_id")["country_n"])

    country = dict(zip(s1["entity_id"], s1["country_n"]))
    os.makedirs(work, exist_ok=True)
    pairs.to_parquet(cache)
    with open(meta, "w") as f:
        json.dump({"s1_ids": s1_ids, "truth": {s: sorted(v) for s, v in truth.items()},
                   "country": country}, f)
    return pairs, s1_ids, truth, country


def oof_predict(pairs: pd.DataFrame, s1_fold: dict, seed: int) -> np.ndarray:
    """Out-of-fold probabilities on the locked folds, with a given model seed."""
    import lightgbm as lgb  # noqa: F401  (imported for the side effect of a clear error)
    from run_pipeline import LGB_PARAMS

    fold = pairs["s1_id"].map(s1_fold).to_numpy()
    p = np.full(len(pairs), np.nan)
    saved = LGB_PARAMS["random_state"]
    LGB_PARAMS["random_state"] = seed
    try:
        for f in sorted(set(s1_fold.values())):
            tr, va = fold != f, fold == f
            m = fit_model(pairs.loc[tr, FEATURES], pairs.loc[tr, "y"],
                          pairs.loc[va, FEATURES], pairs.loc[va, "y"])
            p[va] = m.predict_proba(pairs.loc[va, FEATURES])[:, 1]
            log(f"  seed {seed} fold {f}: best_iter={m.best_iteration_}")
    finally:
        LGB_PARAMS["random_state"] = saved
    return p


def loco_predict(pairs: pd.DataFrame) -> np.ndarray:
    """Leave-one-country-out probabilities: each row scored by a model that never
    saw its country. The unseen-France proxy."""
    c = pairs["c"].to_numpy()
    p = np.full(len(pairs), np.nan)
    for country in sorted(pd.unique(c)):
        tr, va = c != country, c == country
        m = fit_model(pairs.loc[tr, FEATURES], pairs.loc[tr, "y"],
                      pairs.loc[va, FEATURES], pairs.loc[va, "y"])
        p[va] = m.predict_proba(pairs.loc[va, FEATURES])[:, 1]
        log(f"  LOCO without '{country}': best_iter={m.best_iteration_}")
    return p


# --- the rules under test, each as rule(pairs, params) -> {s1_id: set} ---------
RULES = {
    "baseline  threshold + one-to-one": (lambda df, g: apply_rule(df, g, True), list(T_GRID)),
    "V1  expected-F0.5 set selection": (lambda df, g: apply_expected_f05(df, g, True), FLOOR_GRID),
    "V2  relative alpha + floor t": (lambda df, g: apply_relative_rule(df, g[0], g[1], True), REL_GRID),
}


def evaluate(pairs, truth, s1_fold, s1_ids, tag):
    """Cross-fitted score for every rule. Returns {name: (mean, folds, params)}."""
    out = {}
    for name, (rule, grid) in RULES.items():
        mean, folds, params = cross_fitted_rule(pairs, truth, s1_fold, rule, grid)
        out[name] = (mean, folds, params)
        log(f"[{tag}] {name:34s} {mean:.4f}  folds=" +
            " ".join(f"{s:.4f}" for s in folds) + f"  params={params}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="C:/amlc/dataset")
    ap.add_argument("--work", default="work_sample")
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--sample-seed", type=int, default=SEED)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="model seeds; the spread across them is the seed noise")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    print("=" * 78)
    print(f"SAMPLE RUN -- {a.sample} S1 entities, sample seed {a.sample_seed}.")
    print("Every number below is a SAMPLE number: optimistic in level, usable only")
    print("for comparing rules scored on identical pairs. Not CV.")
    print("=" * 78)

    pairs, s1_ids, truth, country = build_frame(a.data, a.work, a.sample, a.sample_seed)
    s1_fold = make_s1_folds(s1_ids, path=os.path.join(a.work, f"folds_sample{a.sample}.csv"),
                            seed=SEED)

    report = {"sample": a.sample, "sample_seed": a.sample_seed,
              "n_s1": len(s1_ids), "n_pairs": int(len(pairs)),
              "singleton_share_sample": float(np.mean([not truth[s] for s in s1_ids])),
              "multi_s1_share_sample": check_one_to_one(truth)}
    log(f"sample: {len(s1_ids)} S1, {len(pairs)} pairs, "
        f"singletons {report['singleton_share_sample']:.4f}, "
        f"records matched to >1 S1 {report['multi_s1_share_sample']:.6f}")

    # Compare this frame against the pipeline's own, and record the agreement.
    #
    # It will NOT match exactly, and that is a finding rather than a bug here:
    # blocking._rare_tokens picks a record's "3 rarest tokens" with
    #     sorted(set(t), key=lambda x: doc_freq.get(x, 0))[:n_tok]
    # and sorted() is stable, so ties on document frequency are broken by the
    # iteration order of a set of strings -- which depends on Python's per-process
    # hash randomisation. Two runs of the same command on the same data therefore
    # produce slightly different candidate sets. Measured here at ~0.16% of pairs.
    # The comparison below is still sound because every rule is scored on THIS
    # frame; what is lost is byte-reproducibility of the pipeline as a whole.
    ref = os.path.join(a.work, "oof_pairs.parquet")
    if os.path.exists(ref):
        r = pd.read_parquet(ref)
        mine = set(zip(pairs["s1_id"], pairs["cand_id"]))
        theirs = set(zip(r["s1_id"], r["cand_id"]))
        jac = len(mine & theirs) / len(mine | theirs)
        report["vs_pipeline_frame"] = {"only_here": len(mine - theirs),
                                       "only_pipeline": len(theirs - mine),
                                       "jaccard": jac}
        log(f"pair frame vs {ref}: jaccard {jac:.6f} "
            f"({len(mine - theirs)} only here, {len(theirs - mine)} only there)")
        if jac < 0.99:
            raise SystemExit("pair frames disagree by more than 1% -- investigate "
                             "before trusting any comparison below")
    else:
        log(f"no {ref} to check against -- run run_pipeline.py --sample first for that check")

    # ---- per-seed OOF predictions -------------------------------------------
    results, contested = {}, {}
    for seed in a.seeds:
        col = os.path.join(a.work, f"oof_p_seed{seed}_sample{a.sample}.npy")
        if os.path.exists(col):
            p = np.load(col)
            log(f"reusing OOF probabilities for seed {seed}")
        else:
            log(f"fitting 5 folds with model seed {seed}")
            p = oof_predict(pairs, s1_fold, seed)
            np.save(col, p)
        frame = pairs[["s1_id", "cand_id"]].assign(p=p)
        if seed == a.seeds[0]:
            contested = {f"t={t}": contested_share(frame, t)
                         for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)}
            for k, v in contested.items():
                log(f"contested {k}: {v['contested_records']}/{v['records_above_t']} "
                    f"records = {v['contested_share']:.5f}")
        results[seed] = evaluate(frame, truth, s1_fold, s1_ids, f"seed {seed}")

    # ---- seed noise and the acceptance test ---------------------------------
    base_name = list(RULES)[0]
    seed_means = {n: [results[s][n][0] for s in a.seeds] for n in RULES}
    noise = float(np.std(seed_means[base_name], ddof=1)) if len(a.seeds) > 1 else float("nan")
    log(f"seed-to-seed noise on the baseline (sd over seeds {a.seeds}): {noise:.5f}")

    ref_seed = a.seeds[0]
    base_mean, base_folds, _ = results[ref_seed][base_name]
    verdict = {}
    for name in RULES:
        if name == base_name:
            continue
        mean, folds, params = results[ref_seed][name]
        wins = sum(1 for v, b in zip(folds, base_folds) if v > b)
        gain = mean - base_mean
        ok = (not np.isnan(noise)) and gain > 2 * noise and wins >= 4
        verdict[name] = {"mean_sample": mean, "gain_vs_baseline": gain,
                         "folds_improved": f"{wins}/5", "seed_noise": noise,
                         "needs_gain_over": 2 * noise,
                         "decision": "accept" if ok else "inconclusive",
                         "params_per_fold": params}
        log(f"VERDICT {name:34s} gain={gain:+.5f} (needs > {2 * noise:.5f}) "
            f"folds improved {wins}/5 -> {verdict[name]['decision'].upper()}")

    # ---- LOCO ---------------------------------------------------------------
    loco_path = os.path.join(a.work, f"loco_p_sample{a.sample}.npy")
    if os.path.exists(loco_path):
        pl = np.load(loco_path)
        log("reusing LOCO probabilities")
    else:
        log("fitting leave-one-country-out models")
        pl = loco_predict(pairs)
        np.save(loco_path, pl)
    lframe = pairs[["s1_id", "cand_id", "c"]].assign(p=pl)
    # Built from the FULL sampled entity list, so an entity blocking found no
    # candidates for is still scored as the "empty" prediction it really is.
    cmap = pd.Series(country)
    loco = {}
    for name, (rule, _) in RULES.items():
        params = results[ref_seed][name][2]
        chosen = max(set(map(str, params)), key=list(map(str, params)).count)
        pick = params[list(map(str, params)).index(chosen)]   # modal cross-fitted choice
        loco[name] = {"params_used": pick}
        for ctry in sorted(cmap.unique()):
            ids_c = cmap[cmap == ctry].index.tolist()
            sub = lframe[lframe["c"] == ctry]
            pred = rule(sub, pick)
            sc = macro_f05(pred, truth, ids_c)
            P, R = precision_recall(pred, truth, ids_c)
            have = set(sub["s1_id"].unique())
            n_nocand = sum(1 for i in ids_c if i not in have)
            loco[name][ctry] = {"macro_f05_sample": sc, "pair_P": P, "pair_R": R,
                                "entities_scored": len(ids_c),
                                "of_which_no_candidates": n_nocand}
            log(f"[LOCO sample] {name:34s} held out '{ctry}': {sc:.4f} "
                f"(P={P:.3f} R={R:.3f}) over {len(ids_c)} entities "
                f"({n_nocand} with no candidates) at params={pick}")

    report.update(contested_share_by_t=contested, seed_noise_baseline=noise,
                  cross_fitted_sample={n: {"mean": results[ref_seed][n][0],
                                           "folds": results[ref_seed][n][1],
                                           "params": [str(x) for x in results[ref_seed][n][2]]}
                                       for n in RULES},
                  per_seed_means={n: dict(zip(map(str, a.seeds), seed_means[n])) for n in RULES},
                  verdicts=verdict, loco_sample=loco)
    out = os.path.join(a.work, f"decide_eval_sample{a.sample}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"report written to {out}")
    print("\nREMINDER: sample numbers. Do not put them in STATUS.md section 5 as CV.")


if __name__ == "__main__":
    main()
