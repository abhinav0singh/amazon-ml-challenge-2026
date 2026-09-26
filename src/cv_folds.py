"""
cv_folds.py: THE single source of truth for cross-validation folds.

Entity resolution version: the unit that must not leak across folds is the
Source-1 entity. All candidate pairs of one S1 entity live in the same fold,
and the metric is computed per S1 entity, so we split S1 ids with a seeded
shuffle into K folds.

Run once (or let run_pipeline.py create it) -> work/folds.csv (s1_id, fold).
Everyone reuses that file; never regenerate with a different seed mid-competition.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

RANDOM_SEED = 42  # fixed, never change mid-competition
N_SPLITS = 5


def make_s1_folds(s1_ids, path: str = "work/folds.csv", n_splits: int = N_SPLITS, seed: int = RANDOM_SEED) -> dict:
    """Return {s1_id: fold} for exactly the given `s1_ids`. If `path` exists it
    is LOADED (never silently regenerated), so every teammate gets identical folds.

    Only the requested ids are returned, each with its locked fold. Returning the
    whole file was a bug: the CV code iterates over this dict's keys, so a run on a
    subset of entities (e.g. a mini dry-run dataset next to the committed full
    folds.csv) scored ~441k absent ids per fold as perfect "singletons" and
    reported a cross-fitted CV of 1.0000. A full run is unaffected -- its ids are
    exactly the file's."""
    if os.path.exists(path):
        f = pd.read_csv(path, dtype={"s1_id": str})
        folds = dict(zip(f["s1_id"], f["fold"]))
        missing = set(s1_ids) - set(folds)
        if missing:
            raise ValueError(f"{len(missing)} S1 ids missing from {path}; delete it only with team agreement")
        return {s: int(folds[s]) for s in s1_ids}
    ids = np.array(sorted(s1_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    folds = {s: i % n_splits for i, s in enumerate(ids)}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame({"s1_id": list(folds), "fold": list(folds.values())}).to_csv(path, index=False)
    print(f"Saved {n_splits} folds for {len(folds)} S1 entities to {path} (seed {seed})")
    return folds


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_io import read_tsv
    s1 = read_tsv("data/dataset/train/train_source1.tsv")
    make_s1_folds(s1["entity_id"].tolist())
