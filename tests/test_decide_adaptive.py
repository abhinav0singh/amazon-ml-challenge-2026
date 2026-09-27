"""Checks for the candidate-count adaptive threshold (issue #3 task B).

The safety property that matters most is that beta = 0 reproduces apply_rule
exactly: the baseline must sit inside the family being searched, so a grid search
can only move away from it by actually scoring better.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from decide import ADAPTIVE_REF_CANDS, apply_adaptive_rule, apply_rule  # noqa: E402


def frame(rng, n_entities=50, max_cands=12):
    """Entities with deliberately uneven candidate counts, so the adjustment bites."""
    rows = []
    for e in range(n_entities):
        for _ in range(int(rng.integers(1, max_cands + 1))):
            rows.append((f"S1-{e:04d}",
                         f"S2-{int(rng.integers(0, n_entities * 3)):04d}",
                         float(rng.random())))
    return pd.DataFrame(rows, columns=["s1_id", "cand_id", "p"])


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("one_to_one", [True, False])
@pytest.mark.parametrize("t", [0.2, 0.5, 0.8])
def test_beta_zero_is_exactly_the_baseline(seed, one_to_one, t):
    """The property the grid search depends on."""
    df = frame(np.random.default_rng(seed))
    assert apply_adaptive_rule(df, t, 0.0, one_to_one) == apply_rule(df, t, one_to_one)


def test_reference_count_gets_the_base_threshold():
    """An entity with exactly ADAPTIVE_REF_CANDS candidates is unaffected by beta."""
    n = ADAPTIVE_REF_CANDS
    df = pd.DataFrame({"s1_id": ["A"] * n,
                       "cand_id": [f"S2-{i}" for i in range(n)],
                       "p": np.linspace(0.30, 0.90, n)})
    base = apply_rule(df, 0.5, True)
    for beta in (0.05, 0.2, 1.0):
        assert apply_adaptive_rule(df, 0.5, beta, True) == base


def test_positive_beta_is_stricter_on_crowded_entities():
    """A crowded entity should keep no more than the baseline keeps; a sparse one
    no fewer. That is the whole point of the rule."""
    rng = np.random.default_rng(0)
    crowded = pd.DataFrame({"s1_id": ["A"] * 40,
                            "cand_id": [f"S2-{i}" for i in range(40)],
                            "p": rng.uniform(0.4, 0.9, 40)})
    sparse = pd.DataFrame({"s1_id": ["B"] * 2,
                           "cand_id": ["S3-1", "S3-2"],
                           "p": np.array([0.55, 0.45])})
    df = pd.concat([crowded, sparse], ignore_index=True)
    base = apply_rule(df, 0.5, True)
    adapt = apply_adaptive_rule(df, 0.5, 0.15, True)
    assert len(adapt.get("A", set())) <= len(base.get("A", set()))
    assert len(adapt.get("B", set())) >= len(base.get("B", set()))


def test_one_to_one_still_holds():
    rng = np.random.default_rng(4)
    df = frame(rng, n_entities=80, max_cands=10)
    seen = set()
    for v in apply_adaptive_rule(df, 0.4, 0.1, True).values():
        assert not (seen & v), "a record was awarded to two entities"
        seen |= v


def test_predictions_are_a_subset_of_candidates():
    rng = np.random.default_rng(9)
    df = frame(rng)
    allowed = {}
    for s, c in zip(df["s1_id"], df["cand_id"]):
        allowed.setdefault(s, set()).add(c)
    for s, v in apply_adaptive_rule(df, 0.3, 0.1, True).items():
        assert v <= allowed[s]


def test_candidate_count_is_taken_before_thresholding():
    """n_cands must describe the entity's full candidate list, not the survivors --
    otherwise it would depend on the threshold being searched and the rule would
    not be well defined."""
    # One entity, 10 candidates, only one of them above the base threshold.
    df = pd.DataFrame({"s1_id": ["A"] * 10,
                       "cand_id": [f"S2-{i}" for i in range(10)],
                       "p": [0.9, 0.85] + [0.01] * 8})
    # Counted BEFORE thresholding (correct): n = 10 > ref 8, so
    #   t_eff = 0.88 + 0.5*(log1p(10) - log1p(8)) = 0.980  -> nothing survives.
    # Counted AFTER thresholding (the bug this guards against): n would be 1,
    #   t_eff = 0.88 + 0.5*(log1p(1) - log1p(8)) = 0.128   -> all ten survive.
    # The two differ by the whole candidate list, so this pins the behaviour down.
    assert apply_adaptive_rule(df, 0.88, 0.5, True) == {}
    # At the threshold the buggy version would have computed, the two strong
    # candidates survive -- so the two readings differ by the entire prediction.
    assert apply_adaptive_rule(df, 0.128, 0.0, True) == {"A": {"S2-0", "S2-1"}}


def test_empty_input():
    df = pd.DataFrame({"s1_id": [], "cand_id": [], "p": []})
    assert apply_adaptive_rule(df, 0.5, 0.1, True) == {}


def test_categorical_ids():
    rng = np.random.default_rng(2)
    obj = frame(rng)
    cat = obj.copy()
    cat["s1_id"] = cat["s1_id"].astype("category")
    cat["cand_id"] = cat["cand_id"].astype("category")
    assert apply_adaptive_rule(cat, 0.4, 0.1, True) == apply_adaptive_rule(obj, 0.4, 0.1, True)
