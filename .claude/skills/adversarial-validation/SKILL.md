---
name: adversarial-validation
description: Detects and handles train–test distribution shift by training a classifier to tell train rows from test rows, then decides per drifting feature whether to drop, transform or keep it and how validation should mimic the test set. Use when test data may come from a different period, region or source, when CV and public leaderboard disagree beyond noise, when a data inspection flags shifted columns or unseen categories, or when the user asks about covariate shift, drift, adversarial validation or building a test-like validation set.
---

# Adversarial Validation

You measure how different the test set is from train, find out which features cause the difference, and change the features and validation so that CV predicts the test score again. Removing every feature that shows shift is not the goal. The goal is to keep the target signal while removing the part that does not transfer to test.

## When to use

- Test is a later time period, a different source, or suspected synthetic data from a different generator.
- CV improvements stop translating to the leaderboard. Rule out leaderboard noise and submission bugs first.
- Data inspection flagged columns with different distributions, missing rates or unseen categories.

## Workflow

### 1. Build the train-vs-test classifier

- Stack train and test **features only**, with label `is_test` (0/1). Exclude the target, target-derived features, and the row ID. Test the ID separately: if the ID alone separates train from test, the split is ordered.
- Use the same feature set and encodings the real model uses. Engineered features can create or hide shift.
- Model: LightGBM, `num_leaves=31`, `learning_rate=0.05`, about 300 rounds with early stopping, native categoricals, stratified 5-fold. Report the **out-of-fold** AUC, never the training AUC.

```python
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def adv_val(tr, te, feats, cat_feats=(), seed=42):
    X = pd.concat([tr[feats], te[feats]], ignore_index=True)
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    for c in cat_feats: X[c] = X[c].astype("category")
    oof, imp = np.zeros(len(X)), pd.Series(0.0, index=feats)
    for trn, val in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                               colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                               random_state=seed, verbose=-1)
        m.fit(X.iloc[trn], y[trn], eval_set=[(X.iloc[val], y[val])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[val] = m.predict_proba(X.iloc[val])[:, 1]
        imp += pd.Series(m.booster_.feature_importance("gain"), index=feats)
    return roc_auc_score(y, oof), imp.sort_values(ascending=False), oof[:len(tr)]  # p(test) for train rows
```

### 2. Interpret the AUC (rough guide, not a law)

| OOF AUC | Reading | Response |
|---|---|---|
| ≤ 0.55 | No practical shift | Keep the standard CV and stop here |
| 0.55–0.70 | Mild | Inspect the top 3 features; usually no change needed |
| 0.70–0.90 | Substantial | Act on the drifting features and add a test-like holdout |
| > 0.90 | Near-separable | Usually a time/ID-like column or a different source; test is out-of-distribution; validation must mimic it |

With very large data even AUC 0.53 is statistically real and still harmless. What matters is whether the shifted features carry target signal.

### 3. Attribute the shift

- Rank features by the classifier's gain.
- Iterative removal: drop the top feature, re-run, and record the new AUC. Stop when AUC < 0.60 or after about 5 rounds. This shows how many features the shift is spread across.
- For each top feature, compare train vs test distributions directly (quantiles, missing rate, category frequency, share of unseen categories).

### 4. Decide per feature

Cross **shift strength** with **target importance**, taken from the real model's importance or CV drop-column delta:

| | Low target value | High target value |
|---|---|---|
| **High shift** | Drop | Transform first: rank or normalise within period, express as a ratio or difference against a related column, bin, de-trend, frequency-encode. Keep the raw version only if the test-like holdout says so |
| **Low shift** | Ignore | Keep |

Specific cases:
- Monotonic time, ID or counter columns: never feed them in raw. Derive relative or cyclic features instead (day of week, time since an event).
- Unseen test categories: group rare levels, frequency-encode, or map to an explicit "unknown" level.
- Different missing rates: add a missing indicator only if it helps on the test-like holdout.

### 5. Make validation resemble test

Pick the lightest option that fits:
- **Temporal shift:** time-based holdout or forward-chaining folds.
- **Test-like holdout:** take the train rows with the highest `p(test)` (top 10–20%) as an extra holdout, scored **alongside** the standard CV. It is smaller and noisier, so use it as a second opinion and make it the primary criterion only when AUC > 0.9.
- **Importance weighting** `w = p/(1-p)`, clipped to about [0.1, 10]: consider it only when AUC ≥ 0.75. It is often neutral for GBDTs, so treat it as low priority and evaluate it on the test-like holdout.

### 6. Accept changes with two scores

Every drop, transform or reweighting is judged on the standard CV **and** the test-like holdout. Accept a change if it improves the test-like holdout beyond noise without a material standard-CV loss. Re-run the train-vs-test classifier after changes and report the new AUC.

## Output format

```
## Shift Report
OOF AUC: 0.xx (severity) | ID-only AUC: 0.xx
Iterative removal: [feat → AUC] ...

## Drifting Features
feature | adv rank | train vs test evidence | target importance | action

## Validation Recommendation
standard CV kept/changed; test-like holdout definition (rows, size) or time split

## Experiments to Run (ordered, each with an acceptance criterion)
```

## DO NOT

- Do not include the target or any target-derived feature in the train-vs-test classifier.
- Do not drop every feature with non-zero importance. The classifier always ranks something first, even when there is no shift.
- Do not report training AUC. Only out-of-fold AUC counts.
- Do not tune weights, holdout size or feature drops against the public leaderboard.
- Do not blame shift for a CV–LB gap before ruling out submission bugs and public-LB noise.
- Do not replace the standard CV with a tiny test-like holdout when shift is mild. That swaps bias for variance.
- Do not recompute features differently for the check than for the model, for example frequency counts on train only in one and on train+test in the other.

## Uncertainty

Say how confident you are that the shift **matters**, which is separate from whether it exists. If the test-like holdout and the standard CV disagree and both are noisy, present both options, recommend the more conservative one (fewer drops, fewer exotic transforms), and suggest one submission that could settle the question.
