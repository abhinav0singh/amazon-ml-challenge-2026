"""
learn_k.py: train a model that decides HOW MANY candidates each entity keeps.

    python scripts/learn_k.py --data <DATA> --work work            # measure only
    python scripts/learn_k.py --data <DATA> --work work --apply    # and rewrite output

Why this and not another matcher: the run saves only (s1_id, cand_id, p, y, fold)
-- `to_pairs_frame` drops every pair feature -- so a second matcher cannot be
trained from a finished run without redoing blocking and features. What CAN be
trained is the stage after it. The 26 Sep full run lost ~0.04 between its
candidate ceiling and its CV, and 36% of that is entities that HAVE true matches
but were predicted empty. That is a decision-layer loss, and the decision layer
only needs probabilities.

The idea: for one entity, every "take the top k" choice has a TRUE F0.5 we can
compute on out-of-fold data, because the OOF frame carries `y`. So build one row
per (entity, k), let a model predict that score from the shape of the entity's
probability vector, and at inference pick the k with the highest prediction.

    k = 0 is included, so "predict nothing" competes on the same footing.

`decide.apply_expected_f05` answers the same question analytically, assuming the
candidates are independent. That assumption is wrong -- chain branches of one
business are strongly dependent -- so its estimate is biased. Its value is given
to the model as a feature (`exp_f05`), which means the model starts from the
analytic answer and can only improve on it where the data says it should.

Evaluation uses the same protocol as everything else: parameters and the model
itself are fitted on the OTHER folds' entities and scored on this fold, over the
FULL entity list including entities blocking found nothing for.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data_io import load_truth, read_tsv, write_id_lists  # noqa: E402
from decide import _award_to_best_s1, apply_rule  # noqa: E402
from metric import f05_single, macro_f05  # noqa: E402

FEATURES = ["k", "n_cands", "k_frac", "cum_p", "p_k", "p1", "gap_1k", "mean_top_k",
            "sum_p", "p_empty", "exp_f05", "n_above_50", "n_above_80"]
LGB_PARAMS = dict(objective="regression", learning_rate=0.08, num_leaves=63,
                  min_child_samples=100, feature_fraction=0.9, bagging_fraction=0.8,
                  bagging_freq=1, n_estimators=400, random_state=42, verbose=-1, n_jobs=-1)
K_MAX = 12          # entities rarely have more than this many true matches (mean 3.67)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def entity_blocks(pairs: pd.DataFrame):
    """Sort once, then yield the segment boundaries. Same integer-code approach the
    rest of decide.py uses -- a pandas group-by over millions of entities is the
    thing that made the CV stage take hours."""
    codes, _ = pd.factorize(pairs["s1_id"].astype(object).to_numpy())
    p = pairs["p"].to_numpy(dtype=np.float64)
    order = np.lexsort((-p, codes))
    codes, p = codes[order], p[order]
    ids = pairs["s1_id"].astype(object).to_numpy()[order]
    cands = pairs["cand_id"].astype(object).to_numpy()[order]
    y = pairs["y"].to_numpy(dtype=np.int8)[order] if "y" in pairs.columns else None
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    return ids, cands, p, y, starts, ends


def build_rows(pairs: pd.DataFrame, with_target: bool):
    """One row per (entity, k) for k = 0 .. min(n_cands, K_MAX).

    `with_target` adds the true F0.5 of taking that entity's top k, which is only
    available where the frame carries `y` (i.e. out-of-fold, never test).
    """
    ids, cands, p, y, starts, ends = entity_blocks(pairs)
    rows, ent, target = [], [], []
    for a, b in zip(starts, ends):
        pv = p[a:b]
        n = len(pv)
        total = float(pv.sum())
        p_empty = float(np.prod(1.0 - np.minimum(pv, 1 - 1e-12)))
        csum = np.cumsum(pv)
        n_true = int(y[a:b].sum()) if with_target else 0
        ytrue = y[a:b] if with_target else None
        kmax = min(n, K_MAX)
        for k in range(0, kmax + 1):
            cum = float(csum[k - 1]) if k else 0.0
            pk = float(pv[k - 1]) if k else 0.0
            exp_f05 = 1.25 * cum / (0.25 * total + k) if k else p_empty
            rows.append((k, n, k / n, cum, pk, float(pv[0]), float(pv[0]) - pk,
                         cum / k if k else 0.0, total, p_empty, exp_f05,
                         int((pv >= 0.5).sum()), int((pv >= 0.8).sum())))
            ent.append(ids[a])
            if with_target:
                tp = int(ytrue[:k].sum())
                if n_true == 0:
                    target.append(1.0 if k == 0 else 0.0)
                elif k == 0 or tp == 0:
                    target.append(0.0)
                else:
                    prec, rec = tp / k, tp / n_true
                    target.append(1.25 * prec * rec / (0.25 * prec + rec))
    X = pd.DataFrame(rows, columns=FEATURES)
    return X, np.asarray(ent, dtype=object), (np.asarray(target) if with_target else None)


def pick_k(model, pairs: pd.DataFrame) -> dict:
    """Predict the best k for every entity and return {s1_id: set(top-k cand_ids)}."""
    X, ent, _ = build_rows(pairs, with_target=False)
    pred = model.predict(X[FEATURES])
    ks = X["k"].to_numpy()
    # argmax per entity over the contiguous k-block
    starts = np.flatnonzero(np.r_[True, ent[1:] != ent[:-1]])
    ends = np.r_[starts[1:], len(ent)]
    best_k = {}
    for a, b in zip(starts, ends):
        best_k[ent[a]] = int(ks[a + int(np.argmax(pred[a:b]))])

    ids, cands, p, _, s2, e2 = entity_blocks(pairs)
    out = {}
    for a, b in zip(s2, e2):
        k = best_k.get(ids[a], 0)
        if k:
            out[ids[a]] = set(cands[a:a + k].tolist())
    return out


def cross_fitted_learned_k(pairs, truth, s1_fold, one_to_one=True):
    """Train on the other folds' entities, score on this one. Never on itself."""
    if one_to_one:
        pairs = _award_to_best_s1(pairs)
    fold_of = pairs["s1_id"].astype(str).map(s1_fold).to_numpy()
    folds = sorted(set(s1_fold.values()))
    scores, sizes = [], []
    for f in folds:
        t0 = time.time()
        tr = pairs[fold_of != f]
        va = pairs[fold_of == f]
        Xtr, _, ytr = build_rows(tr, with_target=True)
        model = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xtr[FEATURES], ytr)
        pred = pick_k(model, va)
        va_ids = [s for s, k in s1_fold.items() if k == f]
        scores.append(macro_f05(pred, truth, va_ids))
        sizes.append(len(Xtr))
        log(f"  fold {f}: trained on {len(Xtr):,} (entity,k) rows -> "
            f"{scores[-1]:.5f}  [{time.time()-t0:.0f}s]")
    return float(np.mean(scores)), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--oof", default=None)
    ap.add_argument("--test-pairs", default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--apply", action="store_true",
                    help="rewrite matching_results.tsv, only if this beats the "
                         "threshold rule on at least 4 of 5 folds")
    a = ap.parse_args()

    oof = a.oof or os.path.join(a.work, "oof_pairs.parquet")
    pairs = pd.read_parquet(oof, columns=["s1_id", "cand_id", "p", "y"])
    folds = pd.read_csv(a.folds or os.path.join(a.work, "folds.csv"), dtype={"s1_id": str})
    s1_fold = dict(zip(folds["s1_id"], folds["fold"].astype(int)))
    truth_all = load_truth(a.data)
    truth = {s: truth_all.get(s, set()) for s in s1_fold}
    del truth_all
    log(f"OOF {len(pairs):,} pairs, {len(s1_fold):,} entities")

    # Baseline, on exactly the same protocol.
    from decide import cross_fitted_rule
    t_grid = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    base_mean, base_folds, base_params = cross_fitted_rule(
        pairs[["s1_id", "cand_id", "p"]], truth, s1_fold,
        lambda d, g: apply_rule(d, g, True), t_grid)
    log(f"threshold  CV {base_mean:.5f} | folds {[round(x, 5) for x in base_folds]}")

    mean, fold_scores = cross_fitted_learned_k(pairs, truth, s1_fold)
    wins = sum(b > a_ for a_, b in zip(base_folds, fold_scores))
    log(f"learned_k  CV {mean:.5f} | folds {[round(x, 5) for x in fold_scores]} "
        f"| {mean - base_mean:+.5f} vs threshold, {wins}/5 folds improved")

    rep = {"threshold": {"cv": base_mean, "folds": base_folds,
                         "params": [str(p) for p in base_params]},
           "learned_k": {"cv": mean, "folds": fold_scores,
                         "gain": mean - base_mean, "folds_improved": wins}}
    with open(os.path.join(a.work, "learn_k_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    log(f"report -> {os.path.join(a.work, 'learn_k_report.json')}")

    if not a.apply:
        return 0
    if wins < 4 or mean <= base_mean:
        log("NOT applying: did not beat the threshold rule on at least 4 of 5 folds")
        return 0

    log("refitting on all out-of-fold entities and applying to test")
    X, _, y = build_rows(_award_to_best_s1(pairs), with_target=True)
    model = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[FEATURES], y)
    tp = pd.read_parquet(a.test_pairs or os.path.join(a.work, "test_pairs.parquet"),
                         columns=["s1_id", "cand_id", "p"])
    tpred = pick_k(model, _award_to_best_s1(tp))
    t_ids = read_tsv(os.path.join(a.data, "test", "test_source1.tsv"))["entity_id"].tolist()
    mpath = os.path.join(a.out, "matching_results.tsv")
    os.replace(mpath, mpath + ".before_learned_k.bak")
    write_id_lists(mpath, t_ids, tpred, "matched_entity_ids")
    from run_pipeline import verify_outputs
    t23 = pd.concat([read_tsv(os.path.join(a.data, "test", f"test_source{k}.tsv"))["entity_id"]
                     for k in (2, 3)]).to_numpy()
    log(f"applied; output check PASSED {verify_outputs(a.out, t_ids, t23)}; "
        f"previous file kept as .before_learned_k.bak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
