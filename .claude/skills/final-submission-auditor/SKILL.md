---
name: final-submission-auditor
description: Audits a submission file and the pipeline that produced it before upload — format against sample_submission, ID alignment, row count, dtypes, NaN/inf, prediction scale and label mapping, threshold application, distribution sanity against OOF, correlation with previous good submissions, reproducibility and rule compliance — and returns a SUBMIT or FIX FIRST verdict. Use before any important upload, especially the final ones before a deadline, when the user says check my submission, about to submit or final submission, or after a surprisingly bad leaderboard score that may be a format or alignment bug.
---

# Final Submission Auditor

Most competition disasters happen in the last hour: rows shuffled by a merge, a label encoder never inverted, a threshold not applied, the wrong file uploaded. You run checks; you never approve a file because it looks right.

## When to use

- Before any submission that matters, and always before the final ones.
- When a leaderboard score is far below CV (rule out bugs before any other theory).
- After any last-minute change to the pipeline.

**Inputs:** the submission file, `sample_submission`, the test file, the OOF predictions and threshold used, the previous best submission file (if any), and the rules on prediction format and final selection.

## Automated checks (run them; don't eyeball)

```python
import numpy as np, pandas as pd

def audit(sub_path, sample_path, id_col, pred_cols, oof=None, prev_path=None,
          proba=True, train_labels=None):
    sub = pd.read_csv(sub_path, dtype={id_col: str})
    ss = pd.read_csv(sample_path, dtype={id_col: str})   # string IDs keep leading zeros
    R = []
    R.append(("columns match sample exactly", list(sub.columns) == list(ss.columns)))
    R.append(("row count", len(sub) == len(ss)))
    R.append(("no duplicate IDs", not sub[id_col].duplicated().any()))
    R.append(("ID set equal", set(sub[id_col]) == set(ss[id_col])))
    R.append(("same ID order as sample (WARN-level, see note)", sub[id_col].tolist() == ss[id_col].tolist()))
    vals = sub[pred_cols]
    R.append(("no NaN", not vals.isna().any().any()))
    num = vals.select_dtypes("number")
    R.append(("no inf", np.isfinite(num.to_numpy()).all() if num.shape[1] else True))
    if proba:
        R.append(("probabilities in [0,1]", ((num >= 0) & (num <= 1)).all().all()))
        if len(pred_cols) > 1:
            R.append(("multiclass rows sum ~1", np.allclose(num.sum(axis=1), 1, atol=1e-3)))
    elif train_labels is not None:   # sample files often hold placeholder labels: compare to TRAIN labels
        R.append(("labels ⊆ train label set", set(vals.iloc[:, 0].unique()) <= set(train_labels)))
    for name, ok in R: print(("PASS " if ok else "FAIL ") + name)
    print(vals.describe(include="all").T)
    if oof is not None: print("OOF:", pd.Series(np.ravel(oof)).describe())
    if prev_path and proba:
        prev = pd.read_csv(prev_path, dtype={id_col: str}).set_index(id_col).loc[sub[id_col]]
        print("corr with previous:", np.corrcoef(sub[pred_cols[0]], prev[pred_cols[0]])[0, 1])
```

A different ID *order* from the sample is only a WARNING if the platform matches rows by ID. It is a BLOCKER if the platform matches by position or the rules say to keep the sample order. When unsure, write the file in sample order via `ss[[id_col]].merge(sub, on=id_col, how="left")`.

## Checklist

### A. Format (BLOCKER if any fail)

- Column names and order exactly as `sample_submission`. No extra index column (`to_csv(index=False)`).
- Row count, ID set and ID uniqueness match. The ID dtype and format are preserved: leading zeros, string vs int, no `.0` suffix. Rows are in sample order (the safest default).
- Prediction type matches the rules: probabilities vs class labels vs integers. Original label strings are restored (`inverse_transform`), and multiclass column order matches the sample.

### B. Alignment (BLOCKER)

- Predictions are attached **by ID** (merge/map), not by position after any sort, merge or group-by on test.
- Spot-check: recompute predictions for 5 random test IDs from raw rows through the full pipeline and compare them with the file.

### C. Content (BLOCKER or WARNING)

- No NaN or inf. Values are in the valid range (non-negative for RMSLE, plausible for regression).
- Threshold metrics: the OOF-chosen threshold is actually applied. Test predicted-positive rate vs OOF predicted-positive rate is reported.
- Regression: the target transform is inverted, and predictions are clipped to a sensible range if that was decided.
- AUC/logloss metrics: probabilities are not rounded or thresholded.

### D. Distribution sanity (WARNING)

- Test prediction mean and quantiles vs OOF. Large differences mean shift or a pipeline bug; explain them before submitting.
- Correlation with the previous good submission: > ~0.95 is expected for incremental changes. A very low correlation means a probable bug.

### E. Pipeline integrity (WARNING)

- Final models are the intended ones (feature version, params, seeds), trained on the intended data.
- Test features come from the same code path as train features, with no train-only or test-only branch.
- Seeds are fixed and library versions recorded. The script re-runs from start to end if the rules require reproducibility.

### F. Rules and logistics

- Submissions remaining today, and the final-selection mechanism (manual select, last upload counts, best public). If "last upload counts", this file must be the last.
- Required deliverables: code or notebook, approach document or deck (common on Unstop in later rounds), external-data declaration.
- Upload with at least 1–2 hours of buffer before the deadline (check the timezone). Portals slow down near deadlines.

### G. Record

Save the file as `sub_{date}_{model}_{cv}.csv` and log it in the submission ledger with its CV and description.

## Output format

```
## Verdict: SUBMIT | FIX FIRST
## Blockers (must fix)
## Warnings (explain or accept)
## Check Table: check | PASS/FAIL | detail
## Stats: test pred summary vs OOF summary | corr with previous | positive rate test vs OOF
```

## DO NOT

- Do not approve a file without running the checks.
- Do not "fix" alignment by reordering predictions positionally to match the sample. Merge by ID.
- Do not change models, features or thresholds in the final hour without re-running the full audit.
- Do not round probabilities for ranking or probabilistic metrics.
- Do not submit a file whose unexplained correlation with the last good submission is low.

## Uncertainty

If the rules are unclear on prediction format (labels vs probabilities) or final selection, mark it as a BLOCKER-level question for the user. Do not guess silently.
