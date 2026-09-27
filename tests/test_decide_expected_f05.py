"""Check the vectorised apply_expected_f05 against an obvious reference.

apply_expected_f05 is written with integer codes and one lexsort so it can run on
a full-scale out-of-fold frame (a pandas group-by version cost ~86 s per call on
9M pairs, and cross_fitted_rule calls it once per grid point per fold). That kind
of rewrite is exactly where a silent off-by-one hides, so every case below is
compared against a plain per-entity Python loop that is slow and obviously right.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from decide import apply_expected_f05, apply_rule  # noqa: E402


def reference(pairs: pd.DataFrame, floor: float = 0.0, one_to_one: bool = True) -> dict:
    """The same rule, written the slow obvious way: one entity at a time."""
    df = pairs[pairs["p"] >= floor] if floor > 0 else pairs
    if not len(df):
        return {}
    if one_to_one:
        df = df.sort_values("p", ascending=False).drop_duplicates("cand_id")
    out = {}
    for s1, grp in df.groupby(df["s1_id"].astype(object), sort=False):
        g = grp.sort_values("p", ascending=False)
        p = g["p"].to_numpy(dtype=np.float64)
        total = p.sum()
        best_f, best_k = -1.0, 0
        for k in range(1, len(p) + 1):
            f = 1.25 * p[:k].sum() / (0.25 * total + k)
            if f > best_f + 1e-12:          # first k attaining the max
                best_f, best_k = f, k
        empty = float(np.prod(1.0 - np.minimum(p, 1 - 1e-12)))
        if best_f <= empty:
            continue                        # predicting nothing is worth more
        out[s1] = set(g["cand_id"].to_numpy(dtype=object)[:best_k].tolist())
    return out


def frame(rng, n_entities, max_cands, dtype="object"):
    """A random candidate frame, with repeated cand_ids so one-to-one bites."""
    rows = []
    for e in range(n_entities):
        for _ in range(int(rng.integers(1, max_cands + 1))):
            rows.append((f"S1-{e:04d}",
                         f"S2-{int(rng.integers(0, n_entities * 2)):04d}",
                         float(rng.random())))
    df = pd.DataFrame(rows, columns=["s1_id", "cand_id", "p"])
    if dtype == "category":
        df["s1_id"] = df["s1_id"].astype("category")
        df["cand_id"] = df["cand_id"].astype("category")
    return df


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("one_to_one", [True, False])
def test_matches_reference(seed, one_to_one):
    rng = np.random.default_rng(seed)
    df = frame(rng, n_entities=40, max_cands=6)
    for floor in (0.0, 0.1, 0.5, 0.9):
        got = apply_expected_f05(df, floor, one_to_one)
        want = reference(df, floor, one_to_one)
        assert got == want, f"seed={seed} oto={one_to_one} floor={floor}"


def test_categorical_ids_behave_like_object():
    """The full-scale frame stores ids as categoricals; they must not change the answer."""
    rng = np.random.default_rng(7)
    obj = frame(rng, n_entities=30, max_cands=5, dtype="object")
    cat = obj.copy()
    cat["s1_id"] = cat["s1_id"].astype("category")
    cat["cand_id"] = cat["cand_id"].astype("category")
    assert apply_expected_f05(cat, 0.0, True) == apply_expected_f05(obj, 0.0, True)


def test_exact_ties_are_handled():
    """Every candidate identical: the reference and the vectorised version must
    agree on how many to keep, not merely on the score."""
    df = pd.DataFrame({"s1_id": ["A"] * 5, "cand_id": [f"S2-{i}" for i in range(5)],
                       "p": [0.5] * 5})
    assert apply_expected_f05(df, 0.0, True) == reference(df, 0.0, True)


def test_hand_computed_cases():
    """The worked examples this rule was designed against.

    A: p = .9,.8,.1  -> S = 1.8; k=1 .776, k=2 .867, k=3 .652; empty .018 -> keep 2
    B: p = .05,.02   -> S = .07;  k=1 .061, k=2 .043; empty .931          -> keep 0
    """
    df = pd.DataFrame({
        "s1_id": ["A", "A", "A", "B", "B"],
        "cand_id": ["x1", "x2", "x3", "y1", "y2"],
        "p": [0.90, 0.80, 0.10, 0.05, 0.02]})
    got = apply_expected_f05(df, 0.0, True)
    assert got == {"A": {"x1", "x2"}}, got


def test_certain_candidate_is_always_committed():
    """p = 1.0 makes the empty option worth exactly 0, so the entity must commit."""
    df = pd.DataFrame({"s1_id": ["A"], "cand_id": ["x"], "p": [1.0]})
    assert apply_expected_f05(df, 0.0, True) == {"A": {"x"}}


def test_empty_input_and_floor_that_removes_everything():
    df = pd.DataFrame({"s1_id": ["A"], "cand_id": ["x"], "p": [0.10]})
    assert apply_expected_f05(df.iloc[:0], 0.0, True) == {}
    assert apply_expected_f05(df, 0.99, True) == {}


def test_never_predicts_a_record_twice_under_one_to_one():
    """The one-to-one constraint must survive the set-selection step."""
    rng = np.random.default_rng(3)
    df = frame(rng, n_entities=60, max_cands=8)
    pred = apply_expected_f05(df, 0.0, True)
    seen = set()
    for v in pred.values():
        assert not (seen & v), "a record was awarded to two entities"
        seen |= v


def test_subset_of_candidates():
    """Whatever it predicts must come from the candidate set it was given."""
    rng = np.random.default_rng(11)
    df = frame(rng, n_entities=50, max_cands=6)
    allowed = {}
    for s, c in zip(df["s1_id"], df["cand_id"]):
        allowed.setdefault(s, set()).add(c)
    for s, v in apply_expected_f05(df, 0.0, True).items():
        assert v <= allowed[s]


def test_apply_rule_still_agrees_with_itself():
    """Guards the shared helpers: the baseline rule is untouched by this work."""
    rng = np.random.default_rng(5)
    df = frame(rng, n_entities=40, max_cands=5)
    for t in (0.2, 0.5, 0.8):
        a = apply_rule(df, t, True)
        b = {s: set(v) for s, v in
             df[df["p"] >= t].sort_values("p", ascending=False)
               .drop_duplicates("cand_id").groupby("s1_id")["cand_id"].agg(set).items() if v}
        assert a == b
