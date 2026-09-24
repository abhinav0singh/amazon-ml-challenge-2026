"""
cv_folds.py — THE single source of truth for cross-validation folds.

Per the team's core rule: this file is written ONCE by the Team Leader /
Validation Lead right after problem formulation is locked, then every
teammate imports the saved fold indices from here. Nobody else regenerates
folds independently — inconsistent folds across teammates silently breaks
stacking later.

HOW TO USE THIS FILE (fill in after the problem statement drops):
1. Load your training data.
2. Pick ONE split strategy below based on what the test set predicts
   (see docs/problem_formulation.md — this must already say which one).
3. Run this script once. It saves fold indices to disk.
4. Everyone else loads fold_indices.pkl — never re-runs this with different
   parameters mid-competition without the whole team knowing.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

RANDOM_SEED = 42  # fixed, never change mid-competition
N_SPLITS = 5
FOLD_OUTPUT_PATH = "cv_fold_indices.pkl"


def make_stratified_folds(df: pd.DataFrame, target_col: str, n_splits: int = N_SPLITS):
    """Use when the target is imbalanced and there's no group/time structure."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(df, df[target_col]))
    return folds


def make_group_folds(df: pd.DataFrame, group_col: str, n_splits: int = N_SPLITS):
    """Use when the test set predicts behavior of UNSEEN entities (users,
    products, sellers, etc.) — group_col must be the entity ID. No entity
    should appear in both train and validation within a fold."""
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(df, groups=df[group_col]))
    return folds


def make_time_series_folds(df: pd.DataFrame, time_col: str, n_splits: int = N_SPLITS):
    """Use when the test set predicts FUTURE events. Sort by time_col first,
    then walk forward — never let a fold train on data from after its
    validation window."""
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    fold_size = len(df_sorted) // (n_splits + 1)
    folds = []
    for i in range(1, n_splits + 1):
        train_idx = df_sorted.index[: i * fold_size]
        val_idx = df_sorted.index[i * fold_size : (i + 1) * fold_size]
        folds.append((train_idx.to_numpy(), val_idx.to_numpy()))
    return folds


def save_folds(folds, path: str = FOLD_OUTPUT_PATH):
    with open(path, "wb") as f:
        pickle.dump(folds, f)
    print(f"Saved {len(folds)} folds to {path}. Random seed used: {RANDOM_SEED}")


def load_folds(path: str = FOLD_OUTPUT_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # ---- EDIT THIS SECTION ONCE THE PROBLEM DROPS, THEN DELETE THIS COMMENT ----
    # df = pd.read_csv("data/train.csv")
    #
    # Pick exactly ONE, matching docs/problem_formulation.md's decision:
    # folds = make_stratified_folds(df, target_col="target")
    # folds = make_group_folds(df, group_col="entity_id")
    # folds = make_time_series_folds(df, time_col="timestamp")
    #
    # save_folds(folds)
    # -----------------------------------------------------------------------
    raise NotImplementedError(
        "Fill in the data load + fold strategy above once the problem "
        "statement drops, per docs/problem_formulation.md. Do not skip "
        "straight to modeling without running this first."
    )
