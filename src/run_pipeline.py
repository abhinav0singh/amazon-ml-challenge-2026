"""
run_pipeline.py: end-to-end baseline, data -> blocking -> matching -> output.

    python src/run_pipeline.py --data data/dataset --out output [--loco] [--no-one-to-one]
    python src/run_pipeline.py --data data/dataset --out output_smoke --sample 20000

Steps
  1. Normalise train and test records (normalize.py).
  2. Blocking on train; report the recall ceiling (blocking.py).
  3. Pair features (pair_features.py) and labels from train_ground_truth.tsv.
  4. 5-fold GroupKFold by S1 entity (folds saved to work/folds.csv) -> LightGBM
     out-of-fold (OOF) pair probabilities.
  5. Decision rule tuned on OOF for macro F0.5; honest cross-fitted score reported.
  6. Optional --loco: train on one country, score on another (unseen-France proxy).
  7. Test: blocking -> features -> average of the 5 fold models -> decision rule
     -> output/matching_results.tsv + output/candidate_pairs.tsv.
Everything printed is also saved to work/report.json for the tracker and docs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import blocking_report, generate_candidates  # noqa: E402
from cv_folds import make_s1_folds  # noqa: E402
from data_io import load_split, load_truth, write_id_lists  # noqa: E402
from decide import apply_rule, check_one_to_one, cross_fitted_score, tune  # noqa: E402
from metric import macro_f05, precision_recall  # noqa: E402
from normalize import add_normalized_columns  # noqa: E402
from pair_features import FEATURES, build_pair_features  # noqa: E402

SEED = 42
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=2000, random_state=SEED, verbose=-1, n_jobs=-1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_model(X_tr, y_tr, X_va, y_va):
    """One LightGBM matcher with early stopping on the validation fold."""
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def label_pairs(pairs: pd.DataFrame, truth: dict) -> np.ndarray:
    """1 if the candidate is a true match of the S1 entity, else 0."""
    return np.fromiter((c in truth.get(s, ()) for s, c in zip(pairs["s1_id"], pairs["cand_id"])),
                       dtype=np.int8, count=len(pairs))


def subsample(s1, s23, n, seed, truth=None):
    """SMOKE TEST ONLY. Cut the split down to `n` Source-1 entities.

    Keeps the S2/S3-records-per-S1-entity density of the full split, so the
    pipeline sees realistically sized candidate pools per entity:
      * train (truth given): keep every true match of the kept entities, then
        top up with random other S2/S3 records to reach the original density.
        The top-up records are real distractors -- most belong to S1 entities
        we dropped -- so they are exactly the kind of near-miss the matcher
        must reject.
      * test (no truth): keep a random S1 sample and thin S2/S3 by the same rate.

    The resulting scores are NOT comparable to a full run and must never be
    quoted as CV. The haystack is smaller, so blocking recall and precision are
    both optimistic. This exists to prove the pipeline runs end to end.
    """
    rng = np.random.default_rng(seed)
    ids = s1["entity_id"].to_numpy()
    if n >= len(ids):
        return s1, s23
    density = len(s23) / len(s1)  # S2/S3 records per S1 entity in the FULL split
    keep = rng.choice(ids, size=n, replace=False)
    s1s = s1[s1["entity_id"].isin(set(keep))]

    if truth is None:
        pool = s23["entity_id"].to_numpy()
        take = min(len(pool), int(round(n * density)))
        wanted = set(rng.choice(pool, size=take, replace=False))
    else:
        wanted = set()
        for s in keep:
            wanted |= truth.get(s, set())
        short = int(round(n * density)) - len(wanted)
        if short > 0:
            others = s23.loc[~s23["entity_id"].isin(wanted), "entity_id"].to_numpy()
            take = min(len(others), short)
            wanted |= set(rng.choice(others, size=take, replace=False))
    return s1s, s23[s23["entity_id"].isin(wanted)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset", help="folder containing train/ and test/")
    ap.add_argument("--out", default="output")
    ap.add_argument("--work", default="work")
    ap.add_argument("--loco", action="store_true", help="leave-one-country-out check")
    ap.add_argument("--no-one-to-one", action="store_true")
    ap.add_argument("--no-country-block", action="store_true")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="SMOKE TEST: run on N Source-1 entities instead of the full split. "
                         "Scores from a sampled run are NOT CV and must not be reported as such.")
    ap.add_argument("--sample-seed", type=int, default=SEED)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    report = {}

    # A sampled run must never touch the locked artefacts of a real run: it writes
    # its own folds file and its own report, so work/folds.csv stays authoritative.
    smoke = args.sample > 0
    folds_path = os.path.join(args.work, f"folds_sample{args.sample}.csv" if smoke else "folds.csv")
    report_path = os.path.join(args.work, f"report_sample{args.sample}.json" if smoke else "report.json")
    if smoke:
        log(f"*** SMOKE TEST: {args.sample} S1 entities, seed {args.sample_seed}. "
            f"Numbers are NOT comparable to a full run -- do not quote them as CV. ***")
        report["sample"] = {"n_s1": args.sample, "seed": args.sample_seed,
                            "warning": "subsampled smoke test; scores are optimistic and not CV"}

    # ---------- train ----------
    log("loading + normalising train")
    s1, s23 = load_split(args.data, "train")
    truth = load_truth(args.data)
    if smoke:
        s1, s23 = subsample(s1, s23, args.sample, args.sample_seed, truth)
        log(f"sampled train: {len(s1)} S1, {len(s23)} S2/S3")
    s1, s23 = add_normalized_columns(s1), add_normalized_columns(s23)
    s1_ids = s1["entity_id"].tolist()
    truth = {s: truth.get(s, set()) for s in s1_ids}
    report["train_sizes"] = {"s1": len(s1), "s23": len(s23)}
    report["train_singleton_share"] = float(np.mean([len(v) == 0 for v in truth.values()]))
    report["multi_s1_share"] = check_one_to_one(truth)
    one_to_one = (not args.no_one_to_one) and report["multi_s1_share"] < 0.01
    report["one_to_one_used"] = one_to_one
    log(f"singleton share {report['train_singleton_share']:.3f} | "
        f"ids matched to >1 S1: {report['multi_s1_share']:.4f} -> one_to_one={one_to_one}")

    log("blocking train")
    by_country = not args.no_country_block
    cand = generate_candidates(s1, s23, by_country=by_country)
    report["blocking_train"] = blocking_report(cand, truth, s1_ids)
    log(f"blocking: {report['blocking_train']}")

    log("pair features train")
    pairs = build_pair_features(cand, s1, s23)
    pairs["y"] = label_pairs(pairs, truth)

    # ---------- CV ----------
    s1_fold = make_s1_folds(s1_ids, path=folds_path, seed=SEED)
    pairs["fold"] = pairs["s1_id"].map(s1_fold)
    pairs["p"] = np.nan
    models = []
    for f in sorted(set(s1_fold.values())):
        tr, va = pairs["fold"] != f, pairs["fold"] == f
        m = fit_model(pairs.loc[tr, FEATURES], pairs.loc[tr, "y"], pairs.loc[va, FEATURES], pairs.loc[va, "y"])
        pairs.loc[va, "p"] = m.predict_proba(pairs.loc[va, FEATURES])[:, 1]
        models.append(m)
        log(f"fold {f}: best_iter={m.best_iteration_}")
    pairs[["s1_id", "cand_id", "fold", "y", "p"]].to_parquet(os.path.join(args.work, "oof_pairs.parquet"))

    best_t, best_s, curve = tune(pairs, truth, s1_ids, one_to_one)
    honest, fold_ts = cross_fitted_score(pairs, truth, s1_fold, one_to_one)
    pred = apply_rule(pairs, best_t, one_to_one)
    P, R = precision_recall(pred, truth, s1_ids)
    report.update(oof_best_t=float(best_t), oof_macro_f05_at_best_t=best_s,
                  cv_macro_f05_cross_fitted=honest, fold_thresholds=[float(t) for t in fold_ts],
                  oof_pair_precision=P, oof_pair_recall=R,
                  empty_baseline_macro_f05=macro_f05({}, truth, s1_ids))
    log(f"CV macro-F0.5 (cross-fitted) = {honest:.4f} | at t={best_t}: {best_s:.4f} "
        f"| P={P:.3f} R={R:.3f} | empty baseline = {report['empty_baseline_macro_f05']:.4f}")

    # ---------- leave-one-country-out (proxy for unseen France) ----------
    if args.loco and s1["country_n"].nunique() > 1:
        cmap = s1.set_index("entity_id")["country_n"]
        pairs["c"] = pairs["s1_id"].map(cmap)
        report["loco"] = {}
        for c in sorted(pairs["c"].unique()):
            tr, va = pairs["c"] != c, pairs["c"] == c
            m = fit_model(pairs.loc[tr, FEATURES], pairs.loc[tr, "y"], pairs.loc[va, FEATURES], pairs.loc[va, "y"])
            pv = pairs.loc[va, ["s1_id", "cand_id"]].assign(p=m.predict_proba(pairs.loc[va, FEATURES])[:, 1])
            ids_c = cmap[cmap == c].index.tolist()
            sc = macro_f05(apply_rule(pv, best_t, one_to_one), truth, ids_c)
            report["loco"][c] = sc
            log(f"LOCO: train without '{c}', score on '{c}' at t={best_t}: {sc:.4f}")

    # ---------- test ----------
    log("test: normalise + block + features")
    t1, t23 = load_split(args.data, "test")
    if smoke:
        # No truth for test, so the kept S2/S3 records are a blind random sample:
        # most true matches of the kept test entities are simply not in the file.
        # The test half of a smoke run therefore proves only that the code path
        # runs and writes well-formed output. Its prediction rate is meaningless.
        t1, t23 = subsample(t1, t23, args.sample, args.sample_seed)
        log(f"sampled test: {len(t1)} S1, {len(t23)} S2/S3 "
            f"(random sample -- test prediction rates below are NOT interpretable)")
        report["sample"]["test_caveat"] = (
            "test S2/S3 sampled without truth, so most real matches are absent; "
            "pred_nonempty_share is not comparable to oof_pred_nonempty_share")
    t1, t23 = add_normalized_columns(t1), add_normalized_columns(t23)
    tcand = generate_candidates(t1, t23, by_country=by_country)
    tpairs = build_pair_features(tcand, t1, t23)
    tpairs["p"] = np.mean([m.predict_proba(tpairs[FEATURES])[:, 1] for m in models], axis=0)
    tpairs[["s1_id", "cand_id", "p"]].to_parquet(os.path.join(args.work, "test_pairs.parquet"))
    tpred = apply_rule(tpairs, best_t, one_to_one)

    t_ids = t1["entity_id"].tolist()
    cand_map = tcand.groupby("s1_id")["cand_id"].agg(set).to_dict()
    write_id_lists(os.path.join(args.out, "candidate_pairs.tsv"), t_ids, cand_map, "candidate_entity_ids")
    write_id_lists(os.path.join(args.out, "matching_results.tsv"), t_ids, tpred, "matched_entity_ids")
    report["test"] = {
        "s1": len(t_ids), "avg_cands_per_s1": len(tcand) / max(len(t_ids), 1),
        "pred_nonempty_share": sum(1 for s in t_ids if tpred.get(s)) / max(len(t_ids), 1),
        "oof_pred_nonempty_share": sum(1 for s in s1_ids if pred.get(s)) / len(s1_ids),
        "countries": t1["country"].value_counts().to_dict(),
    }
    log(f"test: {report['test']}")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"report written to {report_path}")

    if smoke:
        log("smoke test: skipping the official validator (a sampled output covers only "
            "some test S1 entities, so it would fail the 'every entity present' rule by design)")
        return

    validator = os.path.join(os.path.dirname(args.data.rstrip("/\\")), "utils", "validate_submission.py")
    if os.path.exists(validator):
        log("running official validator")
        subprocess.run([sys.executable, validator, "--matching", os.path.join(args.out, "matching_results.tsv"),
                        "--candidate", os.path.join(args.out, "candidate_pairs.tsv"),
                        "--test-dir", os.path.join(args.data, "test"), "--check-ids"])
    else:
        log(f"validator not found at {validator}; run utils/validate_submission.py manually")


if __name__ == "__main__":
    main()
