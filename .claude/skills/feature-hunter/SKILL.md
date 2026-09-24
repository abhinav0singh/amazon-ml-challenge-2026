---
name: feature-hunter
description: Proposes, implements and validates engineered features for tabular competitions — group statistics, entity-history aggregates, frequency and fold-safe target encoding, categorical combinations, ratios, datetime and domain features — with explicit leakage classification and batch ablation against a fixed CV. Use when the user asks for feature ideas, wants to break a score plateau, asks how to encode categoricals or build aggregations, or has candidate features to test. Needs frozen folds and fixed model settings to judge features; does not tune hyperparameters.
---

# Feature Hunter

In tabular competitions the biggest gains usually come from features: information a GBDT cannot easily build from raw columns. You generate features from hypotheses, write leakage-safe code, and accept only what beats noise on the frozen CV.

## When to use

- A baseline exists and the user wants a higher score.
- The score has plateaued and error analysis or EDA suggested a direction.
- The user asks how to encode categoricals, aggregate entity history, or whether a specific feature is safe.

**Prerequisite:** a fixed fold assignment (for example `folds.csv`) and a fixed model config to compare against. If they don't exist, create them first. Features judged on shifting folds are meaningless.

## What trees need, and what they don't

- **Help trees:** ratios and differences between related columns, values relative to a group (`x - mean_g(x)`, `x / mean_g(x)`, rank within group), counts and frequencies, entity-history aggregates, lags, time-since-event, domain formulas, categorical combinations.
- **Do nothing for trees:** monotonic transforms of one feature (log, sqrt, scaling), one-hot encoding of low-cardinality columns when native categorical handling exists. These matter only for linear or NN models.

## Workflow

1. **Collect leads** from data inspection, error-analysis slices, domain knowledge and train-vs-test shift findings. Write each candidate as *feature → hypothesis → expected effect*.
2. **Group candidates into themed batches** of 5–20 (for example "customer-level aggregates", "cat×cat counts"). Order batches by expected gain ÷ implementation cost.
3. **Classify each feature's leakage risk before writing code:**

   | Class | Examples | Rule |
   |---|---|---|
   | Row-local | ratios within a row, row-wise missing count | Safe |
   | Unsupervised, train-only | counts or group means of X over train | Safe. Test gets train statistics |
   | Unsupervised, train+test ("transductive") | frequency or group stats over the combined data | Usually allowed and often helps under shift. Pick one policy for the whole project, record it, and check the rules. For temporal problems, note that it uses future periods |
   | Target-derived | target/mean encoding, WoE | **Only out-of-fold**: inside each CV fold, fit on the training part only. Test gets encodings fit on all of train |
   | Temporal | lags, rolling windows | Shift so each row uses only strictly earlier data |

4. **Implement** as functions. Unsupervised features are computed once and cached. Target-derived ones are fit inside the fold loop.
5. **Evaluate each batch:** same folds, same params, same seed(s), same early-stopping rule as the reference run. Record the mean CV delta, the per-fold deltas, and the number of folds improved.
6. **Acceptance rule:** accept a batch if the gain exceeds the measured noise floor (default: gain > 2× the seed-to-seed std of CV **and** it improves at least 4 of 5 folds). Otherwise reject it, or park it if it is cheap and neutral. If a batch wins, ablate within it by sub-group, not one feature at a time, unless the batch is small.
7. **Prune periodically.** Once 100+ features exist, remove groups with near-zero gain or permutation importance on OOF and confirm on CV. Use null-importance (shuffled-target) selection only for very wide sets; it is expensive.
8. **Update the feature ledger** after every batch.

## Fold-safe target encoding

```python
import numpy as np, pandas as pd
from sklearn.model_selection import KFold

def te_fit_apply(tr_part, ap_part, col, target, smoothing=20, prior=None):
    prior = tr_part[target].mean() if prior is None else prior
    s = tr_part.groupby(col)[target].agg(["sum", "count"])
    enc = (s["sum"] + prior * smoothing) / (s["count"] + smoothing)
    return ap_part[col].map(enc).fillna(prior).values

def te_in_fold(trn_df, val_df, col, target, inner_k=5, seed=0):
    """Train-side values come from an inner OOF split, so the model never sees
    encodings built from the row's own target. Validation uses all of trn_df."""
    trn_enc = np.zeros(len(trn_df))
    for i_tr, i_va in KFold(inner_k, shuffle=True, random_state=seed).split(trn_df):
        trn_enc[i_va] = te_fit_apply(trn_df.iloc[i_tr], trn_df.iloc[i_va], col, target)
    val_enc = te_fit_apply(trn_df, val_df, col, target)
    return trn_enc, val_enc
# For test: train-side = inner OOF over all of train; test side = te_fit_apply(train, test, ...)
```

- Multiclass: one encoding per class. Regression: mean, optionally median or std.
- CatBoost already does ordered target statistics internally. Pass raw categoricals rather than adding your own target encoding on top, unless CV proves otherwise.

## High-yield feature families (tabular)

- **Categorical:** count/frequency encoding; pairwise combinations of 2–3 key categoricals, then count-encoded or target-encoded; rare-level grouping.
- **Group statistics:** for key numeric × key categorical: mean, std, min, max, `x - mean`, `x / mean`, rank within group, group size.
- **Entity history** (multiple rows per entity): count, sum, mean, last value, trend, days since previous event, share of rows with a flag. Build them fold-safe if they use the target.
- **Datetime:** year, month, day of week, hour, weekend, month-end, days since a reference date, Indian holidays or festivals if the domain is retail or finance in India. Use cyclic sin/cos only for linear or NN models.
- **Numeric:** domain ratios (debt/income, price/area), differences, decimal-part or rounding features (strong on synthetic data), missing indicators, row-wise missing count.
- **Text columns:** length, word count, keyword flags; TF-IDF → SVD (20–50 components) fit inside folds or on unlabeled combined text.
- **Original-source data** (if the competition data was generated from a public dataset and the rules allow it): append it with an `is_original` flag. Score CV **only on competition rows**.

## Output format

```
## Feature Ledger
batch | features | hypothesis | leak class | CV Δ (mean ± fold sd) | folds improved | decision

## Code
(ready-to-run functions for the accepted and next-to-test batches)

## Next Batch (and why)
```

## DO NOT

- Do not fit target encoding or any target-using transform on full train before cross-validation.
- Do not compare features using different folds, params, seeds or early-stopping settings from the reference run.
- Do not accept a feature on a single-run gain below the noise floor.
- Do not generate hundreds of blind interactions or polynomial features. Every batch needs a hypothesis.
- Do not log-transform or scale features for tree models expecting a gain.
- Do not score original-data augmentation on original rows.
- Do not add features that won't exist, or would be computed differently, for test rows.
- Do not let feature count outgrow RAM. Estimate `rows × features × 4 bytes` before building.

## Uncertainty

If a gain sits right at the noise floor, report it as "inconclusive". Re-run with 2 more seeds if that is cheap; otherwise park the feature. If it is unclear whether a feature is available at prediction time, treat it as a leak until shown otherwise.
