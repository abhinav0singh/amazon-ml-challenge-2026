"""decide._as_sets / apply_rule must give the same predictions as the old
groupby().agg(set) implementation (replaced 26 Sep for speed: ~86 s -> seconds
per call on the full OOF frame). Compared as predictions: entities with no
surviving pair may be absent in the new dict but held an empty set in the old
one, which every consumer reads identically via .get(s, set())."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from decide import apply_rule  # noqa: E402
from metric import macro_f05  # noqa: E402


def _old_apply_rule(pairs, t, one_to_one):
    df = pairs[pairs["p"] >= t]
    if one_to_one and len(df):
        df = df.sort_values("p", ascending=False).drop_duplicates("cand_id")
    return df.groupby("s1_id", observed=False)["cand_id"].agg(set).to_dict()


def _nonempty(d):
    return {k: v for k, v in d.items() if v}


def _frame(categorical, seed):
    rng = np.random.default_rng(seed)
    n = 20_000
    s1 = np.array([f"S1-{i}" for i in rng.integers(0, 3_000, n)], dtype=object)
    c = np.array([f"S2-{i}" for i in rng.integers(0, 8_000, n)], dtype=object)   # contested ids
    p = rng.choice(np.linspace(0.1, 1.0, 10), size=n).astype(np.float32)          # heavy ties
    df = pd.DataFrame({"s1_id": s1, "cand_id": c, "p": p}).drop_duplicates(["s1_id", "cand_id"])
    if categorical:
        df["s1_id"] = pd.Categorical(df["s1_id"])
        df["cand_id"] = pd.Categorical(df["cand_id"])
    return df


def test_apply_rule_matches_old_groupby():
    for categorical in (True, False):
        for seed in (0, 1):
            df = _frame(categorical, seed)
            ids = sorted(set(df["s1_id"].astype(str)))
            truth = {s: set() for s in ids}
            for s, c in zip(df["s1_id"].astype(str)[::3], df["cand_id"].astype(str)[::3]):
                truth[s].add(c)
            for t in (0.1, 0.5, 0.95):
                for o2o in (True, False):
                    new, old = apply_rule(df, t, o2o), _old_apply_rule(df, t, o2o)
                    assert _nonempty(new) == _nonempty(old), (categorical, seed, t, o2o)
                    assert macro_f05(new, truth, ids) == macro_f05(old, truth, ids)


def test_empty_frame():
    df = _frame(True, 0)
    assert apply_rule(df, 2.0, True) == {}
