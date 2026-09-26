# Business Entity Resolution — runnable pipeline

> This file is the source of `code/business_entity_resolution/README.md` inside
> `<team_name>_submission.zip`. `scripts/make_package.py` copies it there under that name.
> Everything below is written for someone starting from the zip on a clean machine, with no
> knowledge of our repository.

This folder reproduces both submission files end to end from the challenge data alone:

```
output/matching_results.tsv     the final matches (this is the file uploaded to the leaderboard)
output/candidate_pairs.tsv      the candidate set the matcher runs inference over
```

Nothing here reaches the network. Every file read is under the data folder you point it at, or an
artefact the pipeline itself wrote.

---

## 1. Requirements

### Hardware — check this before starting

| | Minimum | Recommended |
|---|---|---|
| RAM | **32 GB** | 64 GB |
| CPU cores | 8 | 8–16 |
| Free disk | 50 GB | 100 GB |

**Do not attempt a full run on a 16 GB machine.** It does not fail — it swaps, and a stage that
needs minutes of CPU takes hours. On 25 September a 16 GB laptop spent 6.5 hours on a step that is
about 6 minutes of actual computation. The pipeline prints a loud warning below 30 GB.

A full run takes roughly **17–19 hours** serially, or **11–12 hours** with parallel blocking on
16 cores. See §3 for how to resume it if it stops.

### Software

**Python 3.12.** The pinned versions below do not build on Python 3.13 or 3.14; use 3.12.

```bash
python --version          # must report 3.12.x
```

Create an isolated environment and install the pinned dependencies:

```bash
python -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

On Windows PowerShell the second command is:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` pins exactly the seven packages the pipeline imports, directly or transitively:
`numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `rapidfuzz`, `pyarrow`.

Verify the scoring metric before trusting any number the pipeline prints:

```bash
.venv/bin/python src/metric.py
```

It must print exactly `metric OK: example = 0.714`. That is the worked example from the problem
statement. If it prints anything else, stop.

---

## 2. Where the data goes

The pipeline takes a single `--data` argument: the folder that contains `train/` and `test/`. Unzip
the organisers' `student_resource` archive somewhere local (not inside a cloud-synced folder — it is
about 2.5 GB and a sync lock mid-run will kill the run) so that you have:

```
<somewhere>/student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
└── utils/
    └── validate_submission.py
```

Then `--data` is `<somewhere>/student_resource/dataset`. Nothing is copied into this folder; the
data is read in place and never modified.

---

## 3. Run the pipeline

From **this folder** (`code/business_entity_resolution/`):

```bash
.venv/bin/python src/run_pipeline.py --data <somewhere>/student_resource/dataset --out output --work work --loco
```

Windows PowerShell:

```powershell
.venv\Scripts\python src\run_pipeline.py --data <somewhere>\student_resource\dataset --out output --work work --loco
```

What it does, in order:

1. Loads and normalises the train split (`src/normalize.py`).
2. Generates candidates by multi-key blocking and reports the recall ceiling (`src/blocking.py`).
3. Builds 28 features per candidate pair and labels them from `train_ground_truth.tsv`
   (`src/pair_features.py`).
4. Splits Source 1 entities into 5 folds (seed 42, written to `work/folds.csv`) and trains one
   LightGBM model per fold, producing out-of-fold probabilities.
5. Chooses the decision threshold on out-of-fold predictions and reports the honest **cross-fitted**
   macro-F0.5 (`src/decide.py`).
6. `--loco` only: retrains leaving one country out and scores on exactly that country — the proxy
   measurement for France, which appears in test but not in train.
7. Applies the same normalisation, blocking, features, the average of the five fold models and the
   same decision rule to the test split, then writes both TSVs.
8. Runs the organisers' validator automatically if it can find it (see §5).

### Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--data` | `data/dataset` | Folder containing `train/` and `test/`. |
| `--out` | `output` | Where `matching_results.tsv` and `candidate_pairs.tsv` are written. |
| `--work` | `work` | Scratch folder: `folds.csv`, `report.json`, parquet caches. |
| `--loco` | off | Run the leave-one-country-out generalisation check. |
| `--no-one-to-one` | off | Disable the one-record-to-one-entity assignment step. |
| `--sample N` | `0` (off) | **Smoke test only.** Run on N Source 1 entities. Writes `folds_sampleN.csv` and `report_sampleN.json` instead of the locked artefacts, and skips the validator. Scores from a sampled run are optimistic and are not cross-validation. |
| `--sample-seed` | `42` | Seed for `--sample`. |
| `--skip-test` | off | Stop after cross-validation. Produces no submission files. |
| `--block-workers N` | `1` (serial) | Blocking parallelism across country groups, using **processes** (threads segfault rapidfuzz). The count is capped by the number of groups and by free RAM. |
| `--fresh` | off | Ignore saved stages and recompute everything. See §3.1 — you rarely want this. |
| `--log FILE` | none | Append everything printed, including tracebacks, to this file as well. |

### 3.1 If it stops, run exactly the same command again

Each finished stage — blocking, models, out-of-fold prediction, CV, test blocking, test prediction —
is saved together with a signature of the code and the data that produced it. Re-running the same
command **resumes** from the last completed stage; you lose at most the stage that was in flight.

This is deliberate and it is the difference between a crash costing minutes and costing a day:

- **Do not delete `work/` between attempts.** That is where the resumable state lives.
- **Do not pass `--fresh`** unless you actually intend to recompute everything from scratch. It
  exists for the case where you want to be certain nothing is reused.
- Editing a source file or changing the input data changes the signature, so the affected stages
  recompute automatically. A stale cache cannot silently survive a code change.

`work/folds.csv` — the locked 5-fold split — is committed to the repository and is read if present.
If it is absent it regenerates deterministically from sorted entity ids and a fixed seed, so an
independent machine reproduces the same folds. Do not delete it: every cross-validation number ever
quoted by this project is tied to that file.

### A 3-minute smoke test first

Before committing to a full run, prove the whole path works:

```bash
.venv/bin/python src/run_pipeline.py --data <somewhere>/student_resource/dataset --out output_smoke --work work_smoke --sample 2000
```

This writes well-formed output for 2,000 entities into `output_smoke/`. **Do not upload it and do not
quote its scores** — the haystack is smaller, so they are optimistic by construction, and the test
side of a sampled run is a blind random sample with most true matches simply absent.

### Empty-prediction baseline

`src/make_empty_submission.py` writes a correctly formatted file that predicts no matches for every
entity. It is useful as a format check against the validator and the portal:

```bash
.venv/bin/python src/make_empty_submission.py --data <somewhere>/student_resource/dataset --out output_empty
```

---

## 4. Outputs

| Path | Contents |
|---|---|
| `output/matching_results.tsv` | One row per test Source 1 entity: `source1_entity_id` TAB `matched_entity_ids`. The id list is comma-joined, de-duplicated and sorted; the field is empty for entities predicted to have no matches. **This is the file uploaded to the leaderboard.** |
| `output/candidate_pairs.tsv` | One row per test Source 1 entity: `source1_entity_id` TAB `candidate_entity_ids`. The exact candidate set the matcher ran inference over, before thresholding. Every id in `matching_results.tsv` also appears here. **Approximately 1.5 GB uncompressed** at the shipped cap of 80 candidates per entity. |
| `work/report.json` | Blocking recall ceiling, entity cover, candidates per entity, cross-fitted CV macro-F0.5, per-fold thresholds, LOCO scores and test prediction rates. |

Both TSVs are written by hand with a literal tab separator, `\n` line endings, UTF-8 and no quoting,
because business addresses and the id lists both contain commas.

---

## 5. Validate before submitting

The organisers ship a dependency-free checker. Run it from the `student_resource/` directory, as the
problem statement specifies:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

It prints `PASS` (exit 0) when the files are safe to submit, or a numbered list of issues (exit 1).
It only reads your output files and the test source files; it does not compute your score.

Adding `--check-ids` makes it confirm that every id in your lists actually exists in the test set. It
costs several GB of RAM and some time, and it is worth it before every upload:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

`run_pipeline.py` invokes the validator with `--check-ids` automatically at the end of a full
(non-sampled) run if it finds it at `<parent of --data>/utils/validate_submission.py`. If it prints
`validator not found`, run one of the commands above by hand.

---

## 6. What is in `src/`

| File | Role |
|---|---|
| `metric.py` | The official F0.5 macro metric. Single implementation, unit-tested against the worked example. Run it standalone as a self-test. |
| `data_io.py` | TSV reading with a strict tab separator and everything as string, and the hand-written submission writer. |
| `normalize.py` | Country-agnostic text normalisation: accent stripping, abbreviation expansion, core-name extraction, postal-code and number extraction. |
| `blocking.py` | Candidate generation: multi-key blocking (rare-token inverted index, name prefix, postal key) with three TF-IDF cosine views reranking inside each block. |
| `pair_features.py` | The 28 pair features: string similarities, blocking cosines, postal/number agreement, and rank/competition context. |
| `cv_folds.py` | The locked 5-fold split over Source 1 entities, seed 42. Loads `work/folds.csv` if it exists and never silently regenerates it. |
| `decide.py` | Threshold and one-to-one assignment; `cross_fitted_score` is the honest evaluation. |
| `run_pipeline.py` | The end-to-end driver described in §3. |
| `make_empty_submission.py` | The all-empty format-check submission. |
| `sweep_blocking.py` | A cost/recall sweep over blocking settings. Measurement tool; not part of a submission run. |

---

## 7. Reproducibility notes

- One seed (`42`) drives fold assignment, LightGBM's `random_state` and its bagging, and `--sample`.
- `work/folds.csv` is written once and reloaded on every subsequent run, so folds never drift between
  runs or machines. Deleting it and regenerating it makes every previously recorded CV number
  incomparable.
- Every dependency is pinned to an exact version against Python 3.12.
- No network access, no external data, no pretrained model. See the methodology document at the root
  of this package for the full compliance reasoning.
