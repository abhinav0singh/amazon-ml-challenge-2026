"""Regression tests for cv_folds.make_s1_folds.

The CV code iterates over the returned dict's keys, so it must contain exactly
the ids of the run -- found by a fresh-clone dry run on 26 Sep, where a mini
dataset next to the committed full folds.csv reported a cross-fitted CV of 1.0.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cv_folds import make_s1_folds  # noqa: E402


def test_generated_folds_cover_exactly_the_ids(tmp_path):
    ids = [f"S1-{i}" for i in range(100)]
    folds = make_s1_folds(ids, path=str(tmp_path / "folds.csv"))
    assert set(folds) == set(ids)
    assert set(folds.values()) == {0, 1, 2, 3, 4}


def test_loaded_superset_file_is_restricted_to_the_run(tmp_path):
    path = str(tmp_path / "folds.csv")
    full = make_s1_folds([f"S1-{i}" for i in range(1000)], path=path)
    subset = [f"S1-{i}" for i in range(0, 1000, 7)]
    got = make_s1_folds(subset, path=path)
    assert set(got) == set(subset)                       # no extra ids leak into CV
    assert all(got[s] == full[s] for s in subset)        # each keeps its locked fold


def test_loaded_file_missing_ids_is_an_error(tmp_path):
    path = str(tmp_path / "folds.csv")
    make_s1_folds(["S1-a", "S1-b"], path=path)
    with pytest.raises(ValueError):
        make_s1_folds(["S1-a", "S1-zzz"], path=path)


def test_reload_is_identical_to_generation(tmp_path):
    path = str(tmp_path / "folds.csv")
    ids = [f"S1-{i}" for i in range(500)]
    assert make_s1_folds(ids, path=path) == make_s1_folds(ids, path=path)
