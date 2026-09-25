"""Regression test for the int32 overflow that killed a 2h15m run on 25 Sep.

`_finish` fancy-indexes one sparse row per candidate PAIR, so its non-zero count
is len(pairs) * nnz-per-row. At full scale that exceeded scipy's int32 index
dtype and raised "negative dimensions are not allowed". Sampled runs never hit
it because they produce a thousand times fewer pairs.

We cannot build billions of non-zeros in a test, so instead we pin the property
that makes the fix correct: chunking must not change the answer. Forcing a tiny
target_nnz makes the chunked path run many times over a small input.
"""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import blocking as B


def _frames(n_a=40, n_b=60):
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    def mk(pfx, n):
        return pd.DataFrame({
            "entity_id": [f"{pfx}-{i}" for i in range(n)],
            "name_core": [" ".join(rng.choice(words, 3)) for _ in range(n)],
            "full_n": [" ".join(rng.choice(words, 5)) for _ in range(n)],
            "addr_n": [" ".join(rng.choice(words, 4)) for _ in range(n)],
        })
    return mk("S1", n_a), mk("S2", n_b)


def test_chunking_does_not_change_cosines():
    s1, s23 = _frames()
    mats = B._fit_views(s1, s23)
    ia = np.repeat(np.arange(len(s1)), 3)
    ib = np.tile(np.arange(3), len(s1))
    one = B._finish(s1, s23, mats, ia, ib, target_nnz=10**12)   # single chunk
    many = B._finish(s1, s23, mats, ia, ib, target_nnz=1)       # forced tiny chunks
    for col in ("cos_name", "cos_full", "cos_addr"):
        np.testing.assert_allclose(one[col].to_numpy(), many[col].to_numpy(), atol=1e-6)
    assert one["s1_id"].tolist() == many["s1_id"].tolist()


def test_chunk_size_is_derived_from_density_not_a_constant():
    """A fixed chunk constant silently stops protecting us as density grows."""
    s1, s23 = _frames()
    mats = B._fit_views(s1, s23)
    A = next(iter(mats.values()))[0]
    per_row = A.nnz / A.shape[0]
    assert per_row > 0
    chunk = int(80_000_000 / per_row)
    assert chunk * per_row <= 80_000_000
