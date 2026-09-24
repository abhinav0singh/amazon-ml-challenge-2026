"""
features.py — Data Analyst's feature pipeline.

CRITICAL RULE: every function here that "fits" something (an encoder, a
scaler, a statistic) must be fit ONLY on training-fold data and applied to
validation/test data afterward. Fitting on the full dataset before folds
are respected is the #1 way teams silently inflate their local CV score
and then fail on real test data. Every function below is written to take
train/val indices explicitly so this can't be gotten wrong by accident.
"""

import numpy as np
import pandas as pd


def drop_redundant_columns(df: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
    """Drop columns that are redundant/non-predictive (IDs, columns that are
    deterministic functions of other columns, obvious leakage columns).
    Fill in cols_to_drop once EDA identifies them — don't guess blindly."""
    return df.drop(columns=cols_to_drop, errors="ignore")


def fold_safe_target_encode(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cat_col: str,
    target_col: str,
    smoothing: float = 10.0,
) -> pd.Series:
    """
    Target-encode a high-cardinality categorical column WITHOUT leakage.
    Computes the encoding map using ONLY train_idx rows, applies it to
    val_idx rows. Unseen categories in val fall back to the global mean.

    This is the pattern from the "why teams win" analysis — naive label
    encoding or fitting on the full dataset both leak the target.
    """
    train_df = df.loc[train_idx]
    global_mean = train_df[target_col].mean()

    agg = train_df.groupby(cat_col)[target_col].agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + global_mean * smoothing) / (
        agg["count"] + smoothing
    )

    encoded_val = df.loc[val_idx, cat_col].map(smoothed).fillna(global_mean)
    return encoded_val


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-logic ratios/deltas/aggregations here once the problem
    statement is known — these are typically the highest-leverage features,
    per the "why teams win" analysis. Examples (edit for the real schema):

    df["feature_a_per_feature_b"] = df["feature_a"] / (df["feature_b"] + 1e-5)
    df["category_mean_price"] = df.groupby("category")["price"].transform("mean")
    """
    return df


def check_for_missing_informative(df: pd.DataFrame) -> pd.DataFrame:
    """Missingness is often informative, not noise. For any column with
    nulls, consider adding a `{col}_was_missing` binary flag before
    imputing, rather than just filling and discarding the signal."""
    for col in df.columns[df.isnull().any()]:
        df[f"{col}_was_missing"] = df[col].isnull().astype(int)
    return df


if __name__ == "__main__":
    print(
        "This module holds fold-safe feature functions. Import and call "
        "them from a notebook or pipeline script once cv_folds.py has "
        "produced fold_indices — do not fit anything here on the full "
        "dataset before folds are applied."
    )
