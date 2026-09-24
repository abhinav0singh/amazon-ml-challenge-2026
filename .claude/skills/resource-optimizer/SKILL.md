---
name: resource-optimizer
description: Fits ML competition work into the available time, RAM, GPU quota and notebook-session limits without silently changing what CV measures — profiling, memory reduction, fast-iteration settings, GPU training for XGBoost/CatBoost/LightGBM, caching and checkpointing, Optuna resource settings, and runtime estimates before launching long jobs. Use when the user hits out-of-memory errors, Colab or Kaggle session timeouts or GPU limits, training or tuning is too slow, the dataset is large, or the user asks how to run more experiments in less time or whether a job will finish before the deadline.
---

# Resource Optimizer

Compute and wall-clock time are the scarcest resources in a time-limited challenge. You make the experiment loop faster and the final runs safe, and you always state whether a speed-up changes the measurement.

## When to use

- Out-of-memory errors, a kernel dying, a session timing out.
- An experiment takes too long to iterate, or Optuna or the final seed-bagging won't finish in time.
- Planning final training before a deadline.

## Workflow

### 1. Profile before optimising

Measure:
- data load time and peak RAM (`df.memory_usage(deep=True).sum()`)
- feature-building time
- training time per fold
- inference time

Optimise the largest item first.

### 2. Memory

- Read with explicit `dtype=` and `usecols=`. Convert CSV to parquet once.
- Downcast float64 → float32 and small ints → int8/16/32 **after** precision-sensitive computations (large sums, IDs, timestamps). float32 is fine for GBDT inputs.
- Convert strings to `category`.
- Build features column-wise, `del` intermediates, call `gc.collect()`, and avoid chained copies.
- Use polars (or pandas with pyarrow) for heavy group-bys.
- LightGBM: `lgb.Dataset(..., free_raw_data=True)`. XGBoost: `QuantileDMatrix` for big data.

### 3. Training speed

- **Exploration settings:** `learning_rate=0.1`, early stopping 100, `max_bin=127`, feature/bagging fractions of about 0.8.
- **Final settings:** `learning_rate=0.02–0.05`. The runtime grows with the number of rounds.
- **XGBoost:** `tree_method="hist"`, `device="cuda"` on a GPU (XGBoost ≥ 2.0). Often 5–10× faster.
- **CatBoost:** `task_type="GPU"` is worth it for large data (≥ ~100k rows). GPU and CPU results differ slightly, so don't mix them within one comparison.
- **LightGBM:** CPU with `n_jobs` = physical cores is usually best on Colab/Kaggle. The GPU build is rarely worth the setup.
- Always use early stopping, and cap `n_estimators`.

### 4. Iteration strategy

- **Screening mode:** a stratified or group-consistent 20–30% row sample, or 3 of the 5 frozen folds, to rank feature batches quickly. **Promote winners to a full-CV confirmation** before accepting them. Screening only ranks candidates; it does not decide.
- **Cache everything:** feature matrices keyed by feature version (parquet), OOF and test predictions per model, fold assignments. Never recompute unchanged features.
- **Optuna:** fix `n_jobs=1` in Optuna with all cores given to the model, or parallel trials with 1 thread each, never both. Use `MedianPruner` with per-fold reporting, `timeout=`, and SQLite storage (`storage="sqlite:///study.db"`, `load_if_exists=True`) so a disconnect resumes the study instead of restarting it.

### 5. Session resilience (Colab/Kaggle)

- Save OOF and test predictions **after each fold**, and resume from the last completed fold.
- Write outputs to persistent storage (Google Drive, Kaggle output or datasets). Save the model files and `requirements`/versions.
- Keep one final-run script or notebook that runs from start to end with a single command.

### 6. Runtime estimates and scheduling

- Before launching: `ETA = time_per_fold × folds × seeds × models (+ inference)`. If the ETA exceeds 50% of the remaining time buffer, cut seeds or models, raise the learning rate, or drop the weakest ensemble member.
- Run long jobs (final bagging, Optuna) in the background or overnight while doing analysis.
- **Deadline reserve:** measure the full final pipeline's runtime early. Keep at least 2× that runtime, plus upload time, free before the deadline.

## Output format

```
## Resource Profile
stage | time | peak RAM | bottleneck?

## Changes (ordered by speed-up ÷ effort)
change | expected speed-up / memory saving | changes CV measurement? (no / screening-only / yes: rerun reference)

## Runtime Plan
job | ETA | fits in budget? | fallback if not
```

## DO NOT

- Do not compare a screening-mode result (sample or 3 folds) with a full-CV reference.
- Do not switch CPU↔GPU, library versions or `max_bin` between runs you are comparing, without re-running the reference.
- Do not downcast IDs, timestamps or values before precision-sensitive aggregation.
- Do not run parallel Optuna trials that each use all cores.
- Do not start a job whose ETA exceeds the remaining buffer.
- Do not "optimise" by removing early stopping, reducing folds for final models, or skipping OOF saving.
- Do not spend more than about 30 minutes on speed work unless it unblocks something.

## Uncertainty

Runtime estimates on shared notebooks vary. Pad ETAs by 1.5× on Colab/Kaggle. If GPU availability is uncertain, keep a CPU fallback config tested on a small sample.
