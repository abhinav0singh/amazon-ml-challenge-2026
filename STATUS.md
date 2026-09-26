# STATUS: Amazon ML Challenge 2026 Team Tracker

_Monitor updates the header line every 6–8 h; everyone else appends rows._

**Now:** Day 1 · phase = baseline · next upload = D1-1 (empty) · blockers: none

**Problem:** Business Entity Resolution. For each Source 1 (S1) entity, list the matching S2/S3 `entity_id`s.
**Metric:** F0.5 per S1 entity, macro-averaged over **all** S1 entities, including singletons. An empty prediction on a true singleton scores 1.0; any false match on a singleton scores 0.0.
**Window:** 25 Sep 00:00 IST → 27 Sep 23:59 IST. **5 uploads/day** (15 total). Final ranking uses the **private** LB.
**Hard rules:** no external lookup (APIs, geocoding, registries, internet data); final model MIT/Apache-2.0 and ≤ 8B parameters; every S1 test entity gets exactly one row; only S2/S3 IDs that exist in test; no duplicates; `.tsv` with tab separators.
**Deliverables (zip):** `output/matching_results.tsv`, `output/candidate_pairs.tsv` (matches ⊆ candidates), `code/business_entity_resolution/{src/, README.md, requirements.txt}`, `Documentation_template.md` (filled in).

> Rule for everyone: every leaderboard upload gets a **git tag** (`sub-D1-1`, …) and a row in the Submission Ledger **before** uploading. Version history is required for shortlisting.

---

## 0. Owners

| Person | Area | Owns files |
|---|---|---|
| P1 · Team Leader | Validation, metric, submissions, final package | `src/metric.py`, `src/cv_folds.py`, `src/run_pipeline.py`, this tracker |
| P2 · Data Analyst | Normalisation and blocking | `src/normalize.py`, `src/blocking.py` |
| P3 · ML Engineer | Pair features and matcher model | `src/pair_features.py`, model part of `src/run_pipeline.py` |
| P4 · Monitor → Decision & Docs | Decision layer, France generalisation, docs, status | `src/decide.py`, `docs/approach_document.md`, this file's header |

Only **P1** uploads to the portal, from one device (simultaneous logins can terminate the session).

---

## 1. Rules to verify (ask via the organisers' Google Form on Day 1)

| Question | Status | Answer |
|---|---|---|
| Which submission is used for the private LB: last one, best one, or a selected one? | ☐ | |
| Can we use pretrained open models (e.g. multilingual MiniLM / e5) for embeddings if MIT/Apache and ≤ 8B? | ☐ | |
| Are hand-written normalisation dictionaries (legal suffixes, street abbreviations, incl. French) allowed? (Not external data, but confirm) | ☐ | |
| Public/private split size? | ☐ | |

---

## 2. Data facts (fill in during inspection)

| Fact | US | India | Test (US / IN / FR) |
|---|---|---|---|
| # S1 / S2 / S3 records | | | |
| % S1 singletons (no match) | | | n/a |
| Matches per non-singleton S1 (mean / max) | | | n/a |
| Does any S2/S3 ID match **more than one** S1? (y/n, %) | | | n/a |
| % S2/S3 records matched to nothing | | | n/a |
| Missing name / address rates | | | |
| Postal code present (%) | | | |

---

## 3. CV contract (P1; frozen once set)

- Split: **GroupKFold by S1 entity**, k = 5, seed = 42 → `folds.csv (s1_id, fold)`
- Generalisation check: **leave-one-country-out** (train US → validate India, and the reverse). This stands in for unseen France.
- Metric: `macro_f05(pred_dict, truth_dict)` over **all** S1 in the validation fold, singletons included. Unit-tested on the PDF example (expected 0.714).
- Noise rule: accept a change if gain > 2× seed std **and** ≥ 4/5 folds improve.

---

## 4. Blocking log (P2)

Recall ceiling = share of true pairs that survive blocking. Target ≥ 0.98 with a manageable number of candidates per S1.

**All rows below are SAMPLED runs, not full scale.** Sampling keeps the real S2/S3-per-S1 density but shrinks the haystack, so these ceilings are optimistic. **The first full-scale train blocking has now completed — see "FULL SCALE" below; the ceilings are far worse than the samples projected.**

### FULL SCALE (measured 26 Sep 00:34, run started 25 Sep 20:07) — #9 task 1

Config: MULTIKEY, `MAX_CANDS=40`, `n_tok` 3 name / 2 addr, `MAX_BLOCK=20000`.

| Split | total pairs | avg cands/S1 | **pair recall ceiling** | **entity full cover** | blocking wall-clock |
|---|---|---|---|---|---|
| train (US+India) | 87,483,735 | 39.64 | **0.8851** | **0.7665** | ~4.5 h (India 1h26m + US ~3h) |

**This is the number Phase 0 was waiting for, and it is a problem.** Sampled B2 at 40k projected 0.9833 / 0.9546; full scale is **0.885 / 0.766**. Sampling optimism is far larger than extrapolated — 23% of entities miss ≥1 true match, so a perfect matcher is already capped well below the 0.98 leaderboard band. Blocking recall, not the decision layer, is now the bottleneck at full scale.

Two measured recall leaks, both actionable (#9 task 2):
1. **`MAX_CANDS=40` is binding** (avg kept 39.64 ≈ the cap). At 40k, cap 40 cost cover 0.955→0.921 vs cap 80; at full scale the haystack pushes more true matches past 40. NB the code value is 40 while the comment above it argues for 80 — a discrepancy to resolve. Raising the cap costs candidates (≈2×) → memory + feature-time, so it must be paired with the feature-stage fix below.
2. **Oversized blocks skipped:** 184 (US) + 69 (India) blocks exceeded `MAX_BLOCK=20000` and were dropped for that key; some true pairs live only there. `n_tok` / `MAX_BLOCK` are the levers.

**Separately, the feature stage — not blocking — dominates wall-clock:** assembling the 20M-pair training sample took ~6.5 h (00:34→07:04), single-threaded (rapidfuzz + Python postal/number loops). The OOF pass repeats this over all 87.5M pairs. Parallelising `pair_features` across PROCESSES (rapidfuzz threads segfault, same as blocking) is now the highest-leverage speedup for getting a CV at all.

| Ver | Method | Sample | Recall ceiling | Entity cover | Avg cands / S1 | Runtime | Notes |
|---|---|---|---|---|---|---|---|
| B0 | dense top-k, 3 TF-IDF views | 2,000 | 0.9983 | 0.9940 | 24.90 | 4 s | original; infeasible at scale |
| B1 | **sparse top-k**, max_df 0.1 | 2,000 | 0.9986 | 0.9950 | 24.85 | 2 s | no densification; drops zero-sim padding |
| B1 | sparse top-k, max_df 0.1 | 10,000 | 0.9948 | 0.9841 | 26.17 | 16 s | |
| B1 | sparse top-k, max_df 0.1 | 40,000 | 0.9897 | 0.9697 | 27.12 | 133 s | |
| B1 | sparse top-k, **no pruning** | 40,000 | 0.9894 | 0.9683 | 27.07 | 241 s | pruning at 0.1 is free |
| B1 | sparse top-k, max_df 0.01 | 40,000 | 0.9842 | 0.9551 | 27.27 | 45 s | **current default**, provisional |
| B1 | sparse top-k, max_df 0.001 | 40,000 | 0.9495 | 0.8722 | 22.61 | 35 s | recall destroyed, barely faster |
| B1 | sparse top-k, max_df 0.01, K 10/10/5 | 40,000 | 0.9816 | 0.9480 | 18.34 | 44 s | smaller K saves candidates, not time |

**Two trends, both bad, both measured.** Cost exponent rose from 1.29 (2k→10k) to 1.60 (10k→40k), heading toward quadratic. Entity cover fell 0.995 → 0.984 → 0.970 as the haystack grew. Raising K to recover cover makes cost worse. Extrapolated to 2.21M entities this is 8–60 h for train blocking alone — EXPECTED, not measured, but the direction is measured.

**Conclusion: top-K cosine over a whole country group does not scale.** Pruning bought 5.4x and stopped helping. Candidate generation needs to become O(n) — cheap exact blocking keys, unioned, with the three cosine views reranking inside each block (which keeps `cos_name` / `cos_full` / `cos_addr` and everything downstream unchanged). See issue #1.

---

## 5. Experiment log (everyone)

| ID | Owner | Change | CV macro-F0.5 (mean ± sd) | LOCO US→IN / IN→US | Precision / Recall | Decision |
|---|---|---|---|---|---|---|
| E0 | P1 | Empty predictions (all singletons) | = train singleton share | | | baseline |
| E1 | P1 | `run_pipeline.py` baseline (3-view TF-IDF blocking, 28 features, LGBM, t on OOF, one-to-one) | | | | |
| E2 | P4 | **V1** · per-entity set selection by approximate expected F0.5, k = 0 allowed (`decide.apply_expected_f05`) | **not measured** — no full run. *Sample 20k:* 0.9843 vs baseline 0.9840 | *Sample:* held-out India 0.9637, US 0.9917 (baseline 0.9635 / 0.9916) | *Sample LOCO-India:* 0.988 / 0.931 | **inconclusive** |
| E3 | P4 | **V2** · relative rule — keep p ≥ α × the entity's own max p, above an absolute floor t (`decide.apply_relative_rule`) | **not measured** — no full run. *Sample 20k:* 0.9840 vs baseline 0.9840 | *Sample:* held-out India 0.9642, US 0.9913 | *Sample LOCO-India:* 0.989 / 0.929 | **inconclusive** |

**E2 / E3 notes (issue #3, P4).** Measured with `scripts/decide_eval.py` on a **20,000-entity sample**
— development numbers only, never CV (AGENTS.md §4). Baseline sample cross-fitted score across model
seeds 42/43/44: 0.9840 / 0.9843 / 0.9839 → **seed noise sd = 0.00021**, so the acceptance bar is a
gain above **0.00042**. V1 gained **+0.00029** and improved 4 of 5 folds, but the gain is under the
bar and its sign flips at seed 43; V2 gained **−0.00006** on 3 of 5 folds. Neither hurts LOCO.
Both are therefore logged `inconclusive` and left in the code, unused by the pipeline, for re-testing
at full scale once issues #1 and #2 land — the harness and its caches are committed, so the re-run is
cheap.

Two supporting measurements from the same run:
- **Contested share** (records wanted by more than one S1 entity above the threshold): **0.50 %** at
  t = 0.2, falling to **0.17 %** at t = 0.7 — well under 1 %, so a smarter conflict rule than
  "highest probability wins" has almost nothing to act on. Not implemented.
- **The decision layer is not where the score is** on this sample: pair precision 0.994 / recall
  0.972 against a blocking entity-full-cover of 0.9545. The binding constraint is candidate
  generation, not thresholding.

---

## 6. Submission ledger (15 slots)

| Slot | Time (IST) | Git tag | Description | CV | Public LB | Validator PASS | Keep for final? |
|---|---|---|---|---|---|---|---|
| D1-1 | 25 Sep 16:49 | `sub-D1-1` | Empty-prediction baseline (format check) | n/a | **0.056** | PASS (`--check-ids`) | no |
| D1-2 | | | Rule-based fuzzy baseline | | | | |
| D1-3 | | | First LightGBM matcher + OOF threshold | | | | |
| D1-4 | | | + one-to-one assignment / conflict resolution | | | | |
| D1-5 | | | Reserve (use only for a structural change with a CV gain) | | | | |
| D2-1 … D2-5 | | | Blocking v2, embeddings, France-robust normalisation, ensemble | | | | |
| D3-1 … D3-3 | | | Best candidates | | | | |
| D3-4, D3-5 | | | **Reserved:** final best-CV + hedge. Upload before 22:00 IST | | | | |

Rules: never spend an upload on a threshold or hyperparameter nudge.

**D1-1 result, 25 Sep:** predicted ≈ 0.056 from the train singleton share (123,247 / 2,206,821 = 5.585%); the portal returned **0.056**. Three things confirmed: the upload path and file format are accepted, the public subset matches train on singleton share, and our local metric maps onto the official scorer. Leaderboard context the same day: 1st 0.985884, 2nd 0.984644, 3rd 0.98435, 8th 0.982 — the whole top eight spans 0.0039, so the contest lives in the 0.98+ band and the decision layer is where it is won. A public-LB difference below noise is not evidence.

---

## 7. Decision log

| Time | Decision | Evidence | Who |
|---|---|---|---|

---

## 8. Doc notes (feeds `Documentation_template.md`; add as you go, one line per item)

**Methodology used**
-

**Candidate generation / blocking strategy** (include recall ceiling and reduction ratio from §4)
-

**Model architecture and feature engineering**
-

**Decision layer (thresholds, conflict resolution, singleton handling)**
-

**Generalisation to unseen country (France)**
-

**Experiments and results** (copy the key rows from §5 and §6)
-

**Conclusion and limitations**
-

---

## 9. Final package checklist (P1, Day 3 by 21:00 IST)

- [ ] `utils/validate_submission.py` → PASS on both files
- [ ] Every ID in matching_results is in candidate_pairs
- [ ] Code reruns end to end from `README.md` on a clean env; `requirements.txt` pinned
- [ ] Model licences recorded (MIT/Apache, ≤ 8B)
- [ ] Functions commented (required)
- [ ] `Documentation_template.md` filled in from §8
- [ ] Zip named `<team_name>_submission.zip` with the exact folder structure

### B2 — multi-key blocking (rare-token inverted index), branch `p1-blocking-scale`

| Sample | Recall ceiling | Entity cover | Cands / S1 | Blocking time |
|---|---|---|---|---|
| 2,000 | 0.9788 | 0.9435 | 19.2 | 9 s |
| 10,000 | 0.9821 | 0.9530 | 39.9 | 42 s |
| 40,000 | 0.9833 | 0.9546 | 60.6 | 148 s |
| 40,000, MAX_CANDS=40 | 0.9686 | 0.9208 | 38.1 | 122 s |

**Cost exponent 0.91 (sub-linear)** vs 1.60 and rising for B1, and entity cover is *stable* across scales where B1's fell. Extrapolated full-scale train blocking ~1.6 h vs ~7.7 h — EXPECTED, not measured; no full-scale run has completed.

An early version using only exact keys (no rare tokens) scored recall 0.8785 / cover 0.7395 at 2k — exact whole-name keys are too brittle for this noise, which is why rare-token indexing is the core of the design.

`MAX_CANDS` set to 80: at 40 it cost 0.034 entity cover for 37% fewer pairs, which is a bad trade now that blocking cost is linear.

### #9 — process-parallel blocking (`--block-workers N`), opt-in

`run_pipeline.py --block-workers N` runs one PROCESS per country group (spawn; threads segfault rapidfuzz). Each worker reads its country slice straight from the normalisation cache and writes its own `pairs_{tag}_{c}.parquet`, so nothing large is pickled; the parent frees `s1/s23` during the parallel window and reloads from cache after. `N=1` (default) is the untouched serial path. Worker count is capped by group count AND free RAM (`_plan_block_workers`, `BLOCK_WORKER_GB=3.5` estimate) and falls back to serial when RAM is tight.

- **Correctness: VERIFIED.** On the 2000-entity sample, parallel output is position-identical to serial — both country parquets identical across all 16 columns (global `ia/ib`, cosines, context features, `y`, `fold`); `total_pairs`/`pair_recall_ceiling`/`entity_full_cover` match exactly (37,904 / 0.9788 / 0.9435, reproducing the B2 2k row).
- **Per-worker RAM at full scale: NOT MEASURED** (deferred: the full CV run held the machine, ≤6 GB free). Verify with external `Get-Process python` on the first real run before trusting it. Sample-scale worker peak was ~0.2 GB.
- Fixed a latent ctypes handle bug so `_profile`'s `[mem]` lines (previously always `0.00 GB`) and the per-worker RSS report now read true values.
- Expected wall-clock gain is bounded by group count and skew: train has 2 groups (US ≫ India) so < 2×; test has 3 (US/India/France).
