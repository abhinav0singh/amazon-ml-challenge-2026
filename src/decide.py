"""
decide.py: turns pair probabilities into final match lists, tuned for macro F0.5.

The constraint the data gives us is one-to-MANY, and only in one direction.
Measured on train: ZERO S2/S3 records belong to more than one S1 entity -- exactly
zero across all 7,638,365 true pairs -- while a non-singleton S1 entity owns 3.67
matching records on average. So a candidate record has at most one rightful owner,
but an entity's match list has no size limit.

Decision rule (knobs chosen on OUT-OF-FOLD predictions only):
  1. one_to_one: each S2/S3 record is awarded to the single S1 entity that scores
     it highest (see check_one_to_one). Because entity capacity is unbounded,
     awarding a record to an entity never costs that entity a different candidate,
     so this greedy pass already maximises the total assigned probability. There is
     nothing a bipartite assignment solver could improve on here, and a solver that
     capped each entity at one record would destroy recall on the ~94% of entities
     that have several matches.
  2. threshold t: keep candidates with p >= t. A false match on a true singleton
     costs a full 1.0 -- but only 5.58% of train entities are singletons (123,247
     of 2,206,822), so predicting empty everywhere scores about 0.056, and
     predicting nothing on a non-singleton scores 0.0. The posture is high
     precision that still COMMITS. t is chosen by measurement; it is not pushed
     upward on the assumption that silence is safe.

Beyond the single global threshold, two set-level rules are available and are
measured the same way (see cross_fitted_rule):
  * apply_expected_f05 -- per entity, take the k highest-probability candidates
    that maximise an approximate expected F0.5, with k = 0 (predict nothing) a
    legitimate choice.
  * apply_relative_rule -- keep candidates within a factor alpha of the entity's
    own best probability, above an absolute floor t.
Both adapt to a measured property of the entity, never to its country.
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
    return _as_sets(df)


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
        score = macro_f05(pred, truth, va_ids)
        scores.append(score)
        print(f"  fold {f}: threshold={t:.3f}, F0.5={score:.4f}")
        ts.append(t)
    return float(np.mean(scores)), ts


# ---------------------------------------------------------------------------
# Set-level decision rules (issue #3). Everything below is additive: apply_rule,
# tune and cross_fitted_score above are unchanged and remain the baseline.
# ---------------------------------------------------------------------------


def _award_to_best_s1(df: pd.DataFrame) -> pd.DataFrame:
    """Award each candidate record to the single S1 entity that scores it highest.

    Same greedy pass apply_rule uses, factored out so the set-level rules below
    apply the constraint identically. It is exactly optimal for total assigned
    probability: entity capacity is unbounded, so an entity keeping record X never
    costs it a different candidate, and the problem separates per record.
    """
    return df.sort_values("p", ascending=False).drop_duplicates("cand_id")


def _as_sets(df: pd.DataFrame) -> dict:
    """{s1_id: set(cand_id)} from a frame of surviving pairs.

    Built from integer codes with one stable sort, not groupby().agg(set): on the
    full-scale OOF frame (~9M pairs, 2.2M entities, categorical ids) the group-by
    took ~86 s per call, and the CV stage makes ~200 calls -- ~5 h (measured
    26 Sep). This takes seconds. Same sets; the only difference is that entities
    with no surviving pair get no key instead of an empty set (a categorical
    group-by with observed=False emitted those), which every consumer already
    reads as an empty prediction via .get(s, set()).
    """
    if not len(df):
        return {}
    codes, uniq = pd.factorize(df["s1_id"].astype(object).to_numpy())
    cands = df["cand_id"].astype(object).to_numpy()
    order = np.argsort(codes, kind="stable")
    cs, cands = codes[order], cands[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], len(cs)]
    return {uniq[cs[a]]: set(cands[a:b].tolist()) for a, b in zip(starts, ends)}


def apply_expected_f05(pairs: pd.DataFrame, floor: float = 0.0,
                       one_to_one: bool = True) -> dict:
    """Per entity, keep the k candidates that maximise an approximate expected F0.5.

    For one entity with candidate probabilities p1 >= p2 >= ... >= pn, predicting
    the top k gives, exactly,

        F0.5 = 1.25 * TP / (0.25 * |truth| + k)

    so we only need estimates of TP and |truth|. Treating each candidate as
    independently true with probability p_i:

        E[TP]    ~= sum of the top k probabilities
        E[truth] ~= sum of ALL this entity's candidate probabilities

    and we take the ratio of those expectations. k = 0 is scored separately and
    exactly: predicting nothing is worth 1.0 precisely when the entity is a true
    singleton, which happens with probability prod(1 - p_i).

    Two approximations, both stated rather than hidden:
      * a ratio of expectations is not the expectation of the ratio;
      * |truth| is estimated from candidates only, so true matches that blocking
        never proposed are invisible and the level of every F0.5 here is
        optimistic. That bias is nearly constant across k within one entity, so
        it moves the estimated score far more than it moves the arg-max.

    `floor` drops candidates below a probability before selection, which both
    saves work and keeps a long tail of near-zero candidates out of the |truth|
    estimate. `floor = 0.0` considers every candidate.
    """
    df = pairs[pairs["p"] >= floor] if floor > 0 else pairs
    if not len(df):
        return {}
    if one_to_one:
        df = _award_to_best_s1(df)

    df = df.sort_values(["s1_id", "p"], ascending=[True, False])
    g = df.groupby("s1_id", sort=False)["p"]
    k = g.cumcount().to_numpy() + 1                      # 1-based rank within entity
    e_tp = g.cumsum().to_numpy()                         # E[TP] for the top-k set
    e_truth = g.transform("sum").to_numpy()              # E[|truth|] for the entity
    f = 1.25 * e_tp / (0.25 * e_truth + k)

    df = df.assign(_k=k, _f=f)
    # Best k per entity, and the exact value of predicting nothing.
    star = df.loc[df.groupby("s1_id", sort=False)["_f"].idxmax(), ["s1_id", "_k", "_f"]]
    empty = np.exp(np.log1p(-df["p"].clip(upper=1 - 1e-12))
                   .groupby(df["s1_id"], sort=False).sum())
    star = star.set_index("s1_id")
    commit = star["_f"] > empty.reindex(star.index)      # is any k >= 1 worth it?
    keep_k = star["_k"].where(commit, 0)

    df = df[df["_k"] <= df["s1_id"].map(keep_k).to_numpy()]
    return _as_sets(df)


def apply_relative_rule(pairs: pd.DataFrame, t: float, alpha: float,
                        one_to_one: bool = True) -> dict:
    """Keep candidates within a factor `alpha` of their entity's own best score.

    An entity's best probability says how confident the matcher is about that
    entity at all; `alpha` then keeps only the candidates that are competitive
    with it. The absolute floor `t` is what still allows an empty prediction --
    without it the rule would always keep at least the top candidate of every
    entity, and singletons would all score 0.

    alpha = 0 reduces exactly to the plain threshold rule at t. This adapts to a
    measured property of the entity, never to its country.
    """
    df = pairs[pairs["p"] >= t]
    if not len(df):
        return {}
    if one_to_one:
        df = _award_to_best_s1(df)
    best = df.groupby("s1_id")["p"].transform("max")
    return _as_sets(df[df["p"] >= alpha * best])


def contested_share(pairs: pd.DataFrame, t: float) -> dict:
    """How often two S1 entities want the same record above the threshold.

    Returns the share of candidate RECORDS that clear `t` for more than one S1
    entity ("contested"), plus the counts behind it. This is what decides whether
    a smarter conflict rule than "highest probability wins" can matter at all: if
    almost nothing is contested, there is nothing for it to fix.
    """
    df = pairs[pairs["p"] >= t]
    n_rec = df["cand_id"].nunique()
    if not n_rec:
        return {"threshold": float(t), "records_above_t": 0,
                "contested_records": 0, "contested_share": 0.0}
    per_record = df.groupby("cand_id")["s1_id"].nunique()
    n_contested = int((per_record > 1).sum())
    return {"threshold": float(t), "records_above_t": int(n_rec),
            "contested_records": n_contested,
            "contested_share": n_contested / n_rec}


def cross_fitted_rule(pairs: pd.DataFrame, truth: dict, s1_fold: dict, rule, grid):
    """Honest evaluation of any set-level rule, on the same protocol as
    cross_fitted_score(): for each fold, choose the rule's parameters on the OTHER
    folds' S1 entities and score those parameters on this fold, so nothing is ever
    tuned on the rows it scores.

    `rule(pairs, params) -> {s1_id: set(cand_id)}` and `grid` is the list of
    parameter values to search. Scoring always covers the fold's FULL entity list,
    including entities blocking found nothing for -- an entity with no candidates
    is a real prediction of "empty" and still scores.

    Returns (mean score, per-fold scores, per-fold chosen params).
    """
    folds = sorted(set(s1_fold.values()))
    scores, chosen = [], []
    for f in folds:
        tr_ids = [s for s, k in s1_fold.items() if k != f]
        va_ids = [s for s, k in s1_fold.items() if k == f]
        tr = pairs[pairs["s1_id"].isin(set(tr_ids))]
        best = max(grid, key=lambda g: macro_f05(rule(tr, g), truth, tr_ids))
        va = pairs[pairs["s1_id"].isin(set(va_ids))]
        scores.append(macro_f05(rule(va, best), truth, va_ids))
        chosen.append(best)
    return float(np.mean(scores)), scores, chosen
