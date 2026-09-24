---
name: oof-engineer
description: Produces aligned out-of-fold and test predictions and turns them into blends, stacks, calibrated probabilities and optimised classification thresholds without leaking validation information; also decides between fold-averaged and full-refit final predictions and seed bagging. Use when the user wants to save OOF predictions, ensemble or stack models, find blend weights, optimise a threshold for F1/accuracy or per-class offsets, calibrate probabilities, combine predictions from several notebooks or teammates, or prepare final training. Requires models trained on the same frozen folds.
---

# OOF Engineer

Out-of-fold (OOF) predictions are how the team compares, combines and post-processes models honestly. You enforce the OOF contract, build ensembles in order of increasing complexity, and report scores that were not tuned on the rows they are measured on.

## When to use

- Any model is going to be kept: save its OOF and test predictions correctly.
- Blending, stacking, threshold optimisation, calibration.
- Combining predictions from teammates or multiple notebooks.
- Deciding how to produce the final test predictions.

## The OOF contract (every saved model)

- `oof/{model_name}_oof.parquet`: `id` plus the prediction (K columns for multiclass, class order recorded), covering **every** train row exactly once.
- `oof/{model_name}_test.parquet`: `id` plus the prediction, test rows in any order. **Always join by `id`.**
- **Raw scale:** probabilities or raw regression output, never thresholded labels.
- **Metadata:** folds-file hash, CV mean ± sd, feature version, params, seed(s).
- **Verification:** recompute the metric from the saved OOF. It must equal the reported CV.

```python
import numpy as np, pandas as pd
def run_oof(train, test, feats, target, folds, fit_predict, name):
    oof = np.full(len(train), np.nan); test_pred = np.zeros(len(test))
    for f in sorted(folds["fold"].unique()):
        trn = train["id"].isin(folds.loc[folds.fold != f, "id"]).values
        val = ~trn
        p_val, p_test = fit_predict(train.loc[trn, feats], train.loc[trn, target],
                                    train.loc[val, feats], train.loc[val, target], test[feats])
        oof[val] = p_val
        test_pred += p_test / folds["fold"].nunique()
    assert not np.isnan(oof).any(), "OOF does not cover all rows"
    pd.DataFrame({"id": train["id"], "pred": oof}).to_parquet(f"oof/{name}_oof.parquet")
    pd.DataFrame({"id": test["id"], "pred": test_pred}).to_parquet(f"oof/{name}_test.parquet")
    return oof, test_pred
```

**Different folds break combination.** Models trained on different fold splits must not be stacked, and should not have blend weights fitted on their OOF. Retrain them on the shared folds, or blend them only with fixed equal weights.

## Ensembling ladder (climb only while each step beats noise)

Before climbing, compute the correlation matrix of the candidate OOFs. Prefer different algorithms or feature sets over tuned clones; OOF correlation > ~0.98 adds little.

1. **Simple average of the top 2–5 diverse models.**
   - AUC: rank-average.
   - Logloss: average probabilities (or logits).
   - Regression: mean, or median if outlier-prone.
2. **Weighted blend:** non-negative weights summing to 1, fitted on OOF with `scipy.optimize.minimize`, or greedy hill climbing (forward selection with replacement). With more than about 5 models, **cross-fit the weights**: fit on k−1 folds of OOF, score the held-out fold, and report that score.
3. **Stacking:** a level-2 model on OOF predictions using the **same folds**. For each fold, fit level 2 on the other folds' OOF and predict the held-out fold; that gives an honest level-2 CV. Use logistic/ridge by default. Use a shallow GBDT only with many base models and plenty of rows. For test, apply level 2 (fitted on all OOF) to the base models' fold-averaged test predictions. Add raw features to level 2 only with a clear reason.

Accept a step only if its **honest (cross-fitted)** score beats the previous step by more than the noise floor (default: 2× the seed-to-seed CV std).

## Threshold optimisation (F1, accuracy, MCC, custom cutoffs)

- Search thresholds on OOF probabilities with a quantile grid of about 200 points.
- **Honest estimate:** choose the threshold on k−1 folds' OOF, apply it to the held-out fold, and report that score. The score at the full-OOF optimum is optimistic.
- **Stability:** compare per-fold optimal thresholds. If they vary widely, pick a value in the flat middle of the score curve, not the sharp peak.
- **Scale match:** fold-averaged test predictions are smoother than OOF. After thresholding, compare the test predicted-positive rate with the OOF predicted-positive rate. A large mismatch means the scales differ; consider matching the rate via a quantile threshold, but only when there is a stated reason.
- **Multiclass (macro-F1 and similar):** per-class multiplicative weights or additive offsets on probabilities, fitted by coordinate ascent on OOF and cross-fitted as above.

## Calibration (logloss/Brier metrics only)

GBDTs trained with logloss and no class weights are usually well calibrated, so check a reliability curve first. If needed:
- temperature or Platt scaling (fewer than ~1000 positives)
- isotonic regression (plenty of data)

Always cross-fit the calibrator. If training used class weights, calibration is usually needed.

## Final test predictions

- **Default: average the fold models' test predictions.** It is consistent with OOF and needs no extra training.
- **Full refit on 100% of train:** set rounds ≈ mean best iteration × (1 + 1/k) (about +20% for k=5). It can add a little on small data but cannot be validated directly; its test predictions should correlate > ~0.99 with the fold average. Use it only when time allows.
- **Seed bagging:** 3–5 seeds of each final model (for fold models, average across seeds and folds). Cheap and nearly always non-negative. Budget the runtime first.
- **Regression post-processing:** invert the target transform (e.g., `expm1`), clip to the train target range, round only if the target is an integer **and** rounding improves OOF.

## Output format

```
## Model Inventory
model | CV mean ± sd | OOF corr with best | folds hash OK | include?

## Ensemble
method | weights | honest CV (cross-fitted) | Δ vs best single

## Post-processing
threshold/offsets | per-fold spread | honest score | OOF vs test positive rate

## Files Written
oof/*, final prediction file path
```

## DO NOT

- Do not blend-fit or stack models trained on different fold splits.
- Do not report OOF-optimised weights, thresholds or calibrators as unbiased CV. Report the cross-fitted numbers.
- Do not tune weights, thresholds or offsets on the public leaderboard.
- Do not mix scales (ranks with probabilities, logits with probabilities) without converting.
- Do not use a flexible level-2 model on few rows or few base models.
- Do not add a model to the blend for a gain below noise.
- Do not save thresholded labels as OOF.

## Uncertainty

If the ensemble gain over the best single model is within noise, recommend the **simple average** of the top diverse models; it is the most robust choice. If the threshold curve is flat across a wide range, say so; the exact value then matters little.
