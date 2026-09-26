"""blocking._cap_positions must keep exactly the rows the old pandas cap kept.

The old rule (_cap_per_entity): rank by score within each S1 entity, descending,
method="first" on a frame sorted by (ia, ib) -- ties go to the lower ib. The new
integer-position version runs before any string frame exists; this pins it to the
old behaviour with a BINDING cap and deliberate ties (the mini dataset never
binds the cap, so an end-to-end run cannot test this).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import blocking as B  # noqa: E402


def _old_keep(ia, ib, score, cap):
    df = pd.DataFrame({"ia": ia, "ib": ib, "score": score})
    return (df["score"].groupby(df["ia"]).rank(ascending=False, method="first") <= cap).to_numpy()


def test_cap_positions_matches_pandas_rank_with_ties():
    rng = np.random.default_rng(0)
    pairs = np.unique(np.stack([rng.integers(0, 300, 40_000), rng.integers(0, 5_000, 40_000)], 1), axis=0)
    ia, ib = pairs[:, 0], pairs[:, 1]
    score = rng.choice(np.linspace(0, 2, 21), size=len(ia)).astype(np.float32)   # many exact ties
    for cap in (1, 5, 20, 80, 10_000):
        assert np.array_equal(B._cap_positions(ia, ib, score, cap), _old_keep(ia, ib, score, cap)), cap


def test_cap_positions_keeps_at_most_cap_per_entity():
    rng = np.random.default_rng(1)
    pairs = np.unique(np.stack([rng.integers(0, 50, 5_000), rng.integers(0, 900, 5_000)], 1), axis=0)
    keep = B._cap_positions(pairs[:, 0], pairs[:, 1], rng.random(len(pairs)).astype(np.float32), 7)
    assert np.bincount(pairs[keep, 0]).max() <= 7
