"""
decide.py: turns pair probabilities into final match lists, tuned for macro F0.5.

Decision rule (two knobs, both chosen on OUT-OF-FOLD predictions only):
  1. one_to_one: each S2/S3 record may be assigned to at most ONE S1 entity
     (its highest-probability S1). Turn it on only if training data confirms
     records rarely match two S1 entities (see check_one_to_one).
  2. threshold t: keep candidates with p >= t. Because a false match on a
     singleton costs a full 1.0, the best t is usually well above 0.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metric import macro_f05


def check_one_to_one(truth: dict) -> float:
    """Share of matched S2/S3 ids that belong to MORE than one S1 entity."""
    seen = {}
    for s, ms in truth.items():
        for m in ms:
            seen[m] = seen.get(m, 0) + 1
    return sum(v > 1 for v in seen.values()) / max(len(seen), 1)


def apply_rule(pairs: pd.DataFrame, t: float, one_to_one: bool) -> dict:
    """pairs: (s1_id, cand_id, p). Returns {s1_id: set(matched ids)}."""
    df = pairs[pairs["p"] >= t]
    if one_to_one and len(df):
        df = df.sort_values("p", ascending=False).drop_duplicates("cand_id")
    return df.groupby("s1_id")["cand_id"].agg(set).to_dict()


def tune(pairs: pd.DataFrame, truth: dict, s1_ids, one_to_one: bool, grid=None):
    """Grid-search t on OOF pairs. Returns (best_t, best_score, curve)."""
    grid = np.round(np.arange(0.20, 0.96, 0.025), 3) if grid is None else grid
    curve = [(t, macro_f05(apply_rule(pairs, t, one_to_one), truth, s1_ids)) for t in grid]
    best_t, best_s = max(curve, key=lambda x: x[1])
    return best_t, best_s, curve


def cross_fitted_score(pairs, truth, s1_fold: dict, one_to_one: bool):
    """Honest estimate: for each fold, pick t on the OTHER folds' S1 entities,
    score it on this fold. Returns (mean score, per-fold thresholds)."""
    folds = sorted(set(s1_fold.values()))
    scores, ts = [], []
    for f in folds:
        tr_ids = [s for s, k in s1_fold.items() if k != f]
        va_ids = [s for s, k in s1_fold.items() if k == f]
        t, _, _ = tune(pairs[pairs["s1_id"].isin(tr_ids)], truth, tr_ids, one_to_one)
        pred = apply_rule(pairs[pairs["s1_id"].isin(va_ids)], t, one_to_one)
        scores.append(macro_f05(pred, truth, va_ids))
        ts.append(t)
    return float(np.mean(scores)), ts
