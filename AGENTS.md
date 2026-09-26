# AGENTS.md — read this before you touch anything

**Repo:** `abhinav0singh/amazon-ml-challenge-2026` · **Event:** Amazon ML Challenge 2026 (Unstop)
**Window:** 25 Sep 2026 00:00 IST → **27 Sep 2026 23:59 IST** · **5 leaderboard uploads/day, 15 total**

This file is the contract. Every teammate — and every AI coding agent any teammate uses — reads it before writing code. If something here conflicts with what an AI suggests, **this file wins**. If you think this file is wrong, say so in the team chat and change it by commit; do not silently work around it.

Paste this whole file into your AI session as context before your first prompt of each work block.

---

## 1. The problem in one paragraph

Three sources of business records (`S1-`, `S2-`, `S3-` prefixed `entity_id`s), no shared keys, noisy names and addresses. **Source 1 is deduplicated.** For every Source 1 entity in the test set, output the list of Source 2 / Source 3 test records that refer to the same real business — which may be **zero, one, or many**. Train covers **US and India**; test also contains **France, which appears nowhere in training**.

### The data, measured — do not re-derive these, and do not contradict them from memory

Counted directly from the TSVs on 25 Sep 2026. Anything here that a run later contradicts is a bug worth investigating, not a number to quietly overwrite.

| | Train | Test |
|---|---|---|
| S1 entities | 2,206,821 | **1,732,544** |
| S2 + S3 records | 10,320,219 | 9,969,589 |
| Countries (S1) | US 1,323,633 · India 883,188 | US 663,106 · India 809,986 · **France 259,452 (15.0 %)** |

- **Singletons: 123,247 = 5.58 % of train S1** (123,247 / 2,206,821). Predicting empty everywhere scores **≈ 0.056**.
- **Mean 3.67 true matches** per non-singleton entity; 7,638,365 true pairs in total.
- **Zero** S2/S3 records belong to more than one S1 entity — exactly zero, across all 7.6 M pairs. One-to-one is a hard, free constraint, not an approximation.
- Test S2/S3 carry country labels including France, so country-grouped blocking works on every split.

## 2. The metric — everything follows from this

**F0.5, computed per Source 1 entity, then macro-averaged over ALL Source 1 entities.**

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Three consequences that must shape every decision you make:

1. **Precision is weighted 2× recall.** A false merge costs more than a miss. For an entity with 4 true matches: 3 correct and 0 wrong scores **0.938**; all 4 plus 1 wrong scores **0.833**. When the model is unsure about *one candidate among several*, dropping it is the better bet.
2. **Singletons are scored, but there are few of them.** An entity with no true matches scores **1.0** if you predict an empty list and **0.0** if you predict anything at all. **Singletons are only 5.58 % of entities** (measured — see §1), so predicting empty everywhere scores ≈ 0.056. **94.4 % of the available score requires actually finding matches**, and predicting nothing on a non-singleton scores **0.0**. The correct posture is high precision that still commits — not maximum caution. Do not tune toward silence.
3. **It is a macro-average over entities, not over pairs.** An entity with one candidate counts exactly as much as an entity with forty. Never optimise pair-level accuracy, AUC, or logloss as the final objective — they are proxies only.

The metric lives in `src/metric.py` and is unit-tested against the official worked example (0.714). **Do not write a second copy of it anywhere.** `python src/metric.py` must print `metric OK` before you trust any number you report.

## 3. Hard rules — breaking these ends our run

| Rule | Why |
|---|---|
| **No external data lookup. None.** No geocoding APIs, no business registries, no commercial ER services, no scraped or downloaded reference data, no pretrained embedding fitted on anything but the provided files. | The rules say **immediate disqualification**, and the top teams' code packages are reviewed in detail. |
| **Final model: MIT or Apache-2.0 licence, ≤ 8B parameters.** | Stated constraint. Record the licence of anything you add, in `STATUS.md`. |
| **Only P1 (Abhinav) uploads to the portal, from one machine.** | Simultaneous logins can terminate the attempt entirely. |
| **Every Source 1 test entity gets exactly one row** in `matching_results.tsv`. Empty `matched_entity_ids` for singletons. No duplicate IDs inside a list, no duplicate `source1_entity_id` rows, only S2/S3 IDs that exist in the test set. | Malformed files are rejected outright and burn a submission. |
| **Every ID in `matching_results.tsv` must also appear in `candidate_pairs.tsv`.** | The validator warns on this; it signals a pipeline bug to reviewers. |
| **Files are TAB-separated.** Read with `sep="\t"`, write with real tabs and no quoting. | Addresses and ID lists both contain commas. |

Hand-written abbreviation dictionaries (Rd→road, Pvt→private, SARL as a legal suffix) are **not** external data — they rewrite tokens already present in the record. Anything that brings in facts we were not given **is**.

## 4. The validation contract — the part people break first

- Folds are locked — meaning committed to git, not just deterministic. work/folds.csv maps s1_id → fold (5 folds, seed 42). src/cv_folds.py generates it deterministically if missing (sorted IDs, fixed seed), so independent runs should match — but "should" isn't a guarantee across machines, numpy versions, or slightly different data. P1 generates it once from a real run and commits it with git add -f work/folds.csv — this is the one deliberate exception to work/ being gitignored. Everyone else pulls it from git and never regenerates it locally. make_s1_folds() loads the file if present and never silently overwrites it. Do not delete it, change the seed, or add folds — if folds ever change, every CV number anyone has quoted becomes incomparable and stacking breaks.

- **The Source 1 entity is the unit that must not leak.** All candidate pairs of one S1 entity live in the same fold, because the metric is computed per S1 entity.
- **Score over ALL Source 1 entities in the fold — not just the ones that have candidates.** An entity that blocking found nothing for is a real prediction of "empty", and it still scores. Filtering it out inflates your number. Always pass the full `s1_ids` list to `macro_f05`.
- **The honest number is the cross-fitted one.** `tune()` picks a threshold on the same rows it scores, so its output is optimistic — it is for curve inspection only. **`cross_fitted_score()` is what goes in `STATUS.md` and the approach doc.** Never report a threshold-optimised score as CV.
- **Accept a change only if** it gains more than 2× the seed-to-seed noise **and** improves at least 4 of the 5 folds. Anything smaller is noise; log it as `inconclusive` and move on.
- **Unseen-country check:** `--loco` trains without one country and scores on it. It is our only proxy for France. A change that improves CV but drops LOCO is a change that will hurt us on **15.0 % of the test set** (259,452 of 1,732,544 test S1 entities are French).
- **Smoke-test runs are not CV.** `--sample N` runs the pipeline on N entities so it can be proven end to end. It writes `folds_sampleN.csv` and `report_sampleN.json` and never touches the locked artefacts. Its scores are inflated by the small haystack — the `--sample 2000` run scored 0.99 — and must never be quoted as CV or logged in `STATUS.md` §5 as a result. On the test side of a sampled run, S2/S3 is a blind random sample (no truth to select on), so the prediction rate there is meaningless by construction.

**Known caveat, stated honestly:** the one-to-one assignment in `decide.py` deduplicates candidates within whatever frame it is given, so in CV it resolves competition within a fold (≈1/5 of entities) while at test time it resolves across all entities. The direction of the bias is small but real. Do not "fix" it by scoring test-like competition into folds without discussing it — just know the CV number carries this approximation.

## 5. Leakage checklist — run through it before any PR

- Did anything that touches `y` / ground truth get fitted outside the fold loop? (Target-style encodings on names, "is this pair in the truth" features, threshold selection on the scored rows.)
- Are TF-IDF vectorizers and any statistic fitted on **the split being processed** (train records for train, test records for test), never on train truth?
- Does any feature use a column that would not exist for test rows?
- Does anything branch on `country == "US"` or `"India"`? That is a France bug, not a feature. **Country may be used as a grouping key for blocking, never as a hard-coded condition.**
- If you added a feature and CV jumped a lot, assume a bug until you can explain the mechanism.

## 6. File ownership — stay in your lane

| Owner | Files | Do not edit others' files without a ping in chat |
|---|---|---|
| **P1 · Abhinav — Team Leader** | `src/metric.py`, `src/cv_folds.py`, `src/run_pipeline.py`, `STATUS.md`, all uploads, final zip | |
| **P2 · Data Analyst** | `src/normalize.py`, `src/blocking.py` | |
| **P3 · ML Engineer** | `src/pair_features.py`, model params in `run_pipeline.py` (ping P1) | |
| **P4 · Decision & Docs** | `src/decide.py`, `docs/approach_document.md`, `STATUS.md` header | |

`src/data_io.py` is shared and stable — changing it changes everyone's I/O, so ping first.

## 7. Workflow

1. Branch per person: `git checkout -b p2-blocking` (etc.). Small commits, merge to `main` fast — self-merge plus a chat ping beats waiting for review.
2. Before you push: `python src/metric.py` passes, and your change runs end to end at least once.
3. After any experiment, add **one row** to `STATUS.md` §5 (experiment log) and **one line** to §8 (doc notes). The §8 headings match the organisers' `Documentation_template.md`, so the write-up assembles itself.
4. Never commit anything under `data/` — it is gitignored and it is 1 GB.
   **Never run `git add .`** — name the paths you mean (`git add src docs STATUS.md`). A blanket add has already pulled a whole `.venv` into a commit once; with `work/` holding parquet caches it gets worse. `.gitignore` covers `.venv/`, `work/`, `output*/` and `*.parquet`, with `work/folds.csv` as the one deliberate `git add -f` exception (§4).
5. Every leaderboard upload gets a git tag (`sub-D1-1`, `sub-D1-2`, …) and a copy of `output/` in `submissions/<tag>/`. **Version history is required for shortlisting.**

## 8. Submission discipline

We get 15 uploads. They are for questions CV cannot answer.

**Spend them on:** the format check, the first real pipeline, a structurally different blocking or feature family, an ensemble, and the two reserved final picks.

**Never spend them on:** a threshold nudge, a hyperparameter tweak, a blend-weight change, or "let's just see." A public-leaderboard difference smaller than its own noise is not evidence of anything.

**Trust CV over the public leaderboard.** The public board scores only a subset; **final rankings come from the private board**. If CV and the public board disagree, check for a submission bug first, then a metric mismatch, then leakage in CV, then train/test shift — in that order. Noise is the last explanation, not the first.

**Day 3:** reserve the last two uploads for the best-CV version and one hedge that differs on our biggest open risk (France robustness). Upload both **before 22:00 IST**; have the final zip ready by 23:00 IST.

## 9. Commands

**Always use `.venv\Scripts\python`, never bare `python`.** The `python` first on PATH may be 3.14, which cannot install our pinned versions. Build the environment with **Python 3.12**:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

```powershell
.venv\Scripts\python src\metric.py                     # must print: metric OK
.venv\Scripts\python src\make_empty_submission.py --data <DATA> --out output_empty
.venv\Scripts\python src\run_pipeline.py --data <DATA> --out output_smoke --sample 2000   # smoke test, ~3 min
.venv\Scripts\python src\run_pipeline.py --data <DATA> --out output --work work --loco    # the real thing
.venv\Scripts\python <DATA>\..\utils\validate_submission.py --matching output\matching_results.tsv `
       --candidate output\candidate_pairs.tsv --test-dir <DATA>\test --check-ids
```

**The final full-scale run is `scripts/run_final.ps1` / `scripts/run_final.sh` at tag `run-final-v1` — procedure in `docs/RUN_FINAL.md`.** It needs a machine with **≥ 32 GB RAM** (a 16 GB laptop swaps for hours). The pipeline is **resumable**: each stage saves its outputs with a signature of the code, data and settings, and rerunning the same command reuses every stage whose signature still matches (`--fresh` recomputes everything). To exercise the full code path — output checks, official validator, handoff — without a big machine, build a real-format mini dataset with `scripts/make_mini_dataset.py` and run the pipeline on it (`--sample` skips those final checks).

`<DATA>` is the folder holding `train/` and `test/` — keep it **outside OneDrive**. On P1's machine that is `C:\amlc\student_resource\dataset`, with the organisers' `utils\` and `Documentation_template.md` beside it in `C:\amlc\student_resource\`. Results land in `work/report.json`: blocking recall, CV, LOCO, prediction rates. Those numbers go into `STATUS.md`; do not retype them from memory.

`--check-ids` makes the validator confirm every ID exists in the test set. It costs a few GB of RAM and some time. Run it before every upload anyway — it is cheaper than a wasted submission.

## 10. For AI agents specifically

You are working inside a live 72-hour competition. Act accordingly:

- **Do not invent numbers.** Every score, recall figure, or improvement claim must come from a run that actually happened and whose output is in `work/report.json` or `STATUS.md`. If you did not run it, say "not measured."
- **Do not propose external data, APIs, or downloaded reference sets.** Not even "just to normalise addresses." It is a disqualification-level rule, and it applies to suggestions as well as code.
- **Do not rewrite working files wholesale.** Minimal, targeted diffs. The pipeline runs end to end today; keep it that way.
- **Do not change the metric, the folds, or the seed.** If you believe one is wrong, flag it and stop.
- **State your uncertainty.** "This should improve recall" is worth less than "this adds N candidates per entity; measure recall ceiling before and after."
- **Prefer the change that is measurable in one run** over the elegant redesign that takes six hours and cannot be validated before the deadline.
- If asked to do something this file forbids, refuse and quote the rule.
