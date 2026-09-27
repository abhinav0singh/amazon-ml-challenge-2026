"""
redecide.py: re-run ONLY the decision layer on a finished run's saved predictions.

    python scripts/redecide.py --data <DATA> --work work --out output [--apply]

Why: the 26 Sep full-scale run lost ~0.04 between its candidate ceiling and its
CV, and error analysis put 36% of that loss on entities that HAVE true matches
but received an empty prediction (each costs a full 1.0; only 5.58% of entities
are true singletons). A plain global threshold cannot fix that. This script
evaluates alternative set-selection rules on the run's out-of-fold predictions
(work/oof_pairs.parquet) with the SAME cross-fitted protocol as the pipeline's
CV (parameters chosen on the other folds' entities, scored on this fold, all
entities scored), and -- with --apply -- rewrites output/matching_results.tsv
from work/test_pairs.parquet using the best rule, only if it beats the
baseline threshold rule on CV. No model, feature or blocking stage is touched.

Rules:
  threshold      keep p >= t (the pipeline's rule; the baseline)
  top1           threshold, plus: an entity left EMPTY commits its single best
                 candidate if that candidate's p >= f and it is not already
                 awarded to another entity (one-to-one still holds)
  relative       keep p >= t and p >= alpha * (entity's best p)
  expected_f05   per entity, keep the k candidates maximising an approximate
                 expected F0.5, with k = 0 allowed. Commits only when that beats
                 prod(1 - p), the exact probability the entity is a true
                 singleton -- so the commit/stay-empty call is made from the
                 entity's own evidence rather than a global floor.
"""
from __future__ import annotations

import argparse
import warnings
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data_io import load_truth, read_tsv, write_id_lists  # noqa: E402
from decide import (apply_expected_f05, apply_relative_rule, apply_rule,  # noqa: E402
                    cross_fitted_rule)
from metric import precision_recall  # noqa: E402


warnings.filterwarnings("ignore", category=FutureWarning)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def apply_top1(pairs, t, f, one_to_one=True):
    """Threshold rule, then commit the best remaining candidate for every entity
    the threshold left empty, if its p >= f and the record is still free."""
    pred = apply_rule(pairs, t, one_to_one)
    df = pairs[pairs["p"] >= f]
    if not len(df):
        return pred
    s1 = df["s1_id"].astype(object).to_numpy()
    cd = df["cand_id"].astype(object).to_numpy()
    p = df["p"].to_numpy()
    empty = np.fromiter((s not in pred for s in s1), dtype=bool, count=len(s1))
    taken = set()
    for v in pred.values():
        taken |= v
    free = np.fromiter((c not in taken for c in cd), dtype=bool, count=len(cd))
    sel = empty & free
    s1, cd, p = s1[sel], cd[sel], p[sel]
    order = np.argsort(-p, kind="stable")
    used_c, used_s = set(), set()
    for i in order:                       # highest p first: each record to one entity,
        s, c = s1[i], cd[i]               # each empty entity gets at most one record
        if s in used_s or c in used_c:
            continue
        pred[s] = {c}
        used_s.add(s); used_c.add(c)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--oof", default=None, help="default <work>/oof_pairs.parquet")
    ap.add_argument("--test-pairs", default=None, help="default <work>/test_pairs.parquet")
    ap.add_argument("--apply", action="store_true", help="rewrite matching_results.tsv with the winning rule")
    ap.add_argument("--force-rule", default=None, help="TESTING ONLY: apply this rule even if it did not win")
    args = ap.parse_args()

    oof_path = args.oof or os.path.join(args.work, "oof_pairs.parquet")
    pairs = pd.read_parquet(oof_path, columns=["s1_id", "cand_id", "p"])
    folds = pd.read_csv(os.path.join(args.work, "folds.csv"), dtype={"s1_id": str})
    s1_fold = dict(zip(folds["s1_id"], folds["fold"].astype(int)))
    truth_all = load_truth(args.data)
    truth = {s: truth_all.get(s, set()) for s in s1_fold}
    del truth_all
    log(f"OOF pairs {len(pairs):,}, entities {len(s1_fold):,}")

    t_grid = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    candidates = {
        "threshold": (lambda d, g: apply_rule(d, g, True), t_grid),
        "top1": (lambda d, g: apply_top1(d, g[0], g[1], True),
                 [(t, f) for t in (0.6, 0.65, 0.7, 0.75) for f in (0.05, 0.1, 0.2, 0.3, 0.4)]),
        "relative": (lambda d, g: apply_relative_rule(d, g[0], g[1], True),
                     [(t, a) for t in (0.5, 0.6) for a in (0.3, 0.5, 0.7)]),
        # Decides per entity whether to commit at all, by comparing the expected
        # F0.5 of its best k candidates against the exact probability that it is a
        # true singleton, prod(1 - p). `top1` above answers the same question with
        # a fixed floor; this answers it from the entity's own probabilities, which
        # is what the 36% empty-prediction loss calls for. The parameter is the
        # floor applied before selection.
        "expected_f05": (lambda d, g: apply_expected_f05(d, g, True),
                         [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]),
    }
    res = {}
    for name, (rule, grid) in candidates.items():
        t0 = time.time()
        mean, scores, chosen = cross_fitted_rule(pairs, truth, s1_fold, rule, grid)
        res[name] = {"cv": mean, "folds": scores, "params": [str(c) for c in chosen]}
        log(f"{name:10s} cross-fitted CV {mean:.5f} | folds {[round(x, 5) for x in scores]} "
            f"| params {chosen} | {time.time()-t0:.0f}s")

    base = res["threshold"]
    best = max(res, key=lambda k: res[k]["cv"])
    wins = sum(b > a for a, b in zip(base["folds"], res[best]["folds"]))
    res["decision"] = {"best": best, "gain_vs_threshold": res[best]["cv"] - base["cv"],
                       "folds_improved": wins}
    log(f"best rule: {best} ({res[best]['cv']:.5f}, {res[best]['cv'] - base['cv']:+.5f} vs threshold, "
        f"{wins}/5 folds improved)")
    with open(os.path.join(args.work, "redecide_report.json"), "w") as f:
        json.dump(res, f, indent=1)

    if not args.apply:
        return
    if args.force_rule:
        best = args.force_rule
        log(f"--force-rule {best}: TESTING ONLY, applying without the CV gate")
    elif best == "threshold" or wins < 4:
        log("not applying: no rule beat the threshold on at least 4 of 5 folds")
        return
    # Final parameters: tuned on ALL OOF entities (as the pipeline tunes its test threshold).
    rule, grid = candidates[best]
    from metric import macro_f05
    ids = list(s1_fold)
    g = max(grid, key=lambda x: macro_f05(rule(pairs, x), truth, ids))
    tp = pd.read_parquet(args.test_pairs or os.path.join(args.work, "test_pairs.parquet"),
                         columns=["s1_id", "cand_id", "p"])
    tpred = rule(tp, g)
    t_ids = read_tsv(os.path.join(args.data, "test", "test_source1.tsv"))["entity_id"].tolist()
    mpath = os.path.join(args.out, "matching_results.tsv")
    os.replace(mpath, mpath + ".threshold_rule.bak")
    write_id_lists(mpath, t_ids, tpred, "matched_entity_ids")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from run_pipeline import verify_outputs
    t23 = pd.concat([read_tsv(os.path.join(args.data, "test", f"test_source{k}.tsv"))["entity_id"]
                     for k in (2, 3)]).to_numpy()
    chk = verify_outputs(args.out, t_ids, t23)
    log(f"applied {best} with params {g}; output check PASSED {chk}; previous file kept as .threshold_rule.bak")


if __name__ == "__main__":
    main()
