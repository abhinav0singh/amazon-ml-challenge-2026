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

| Ver | Method | Recall ceiling | Avg cands / S1 | Total pairs | Runtime | Notes |
|---|---|---|---|---|---|---|
| B0 | | | | | | |

---

## 5. Experiment log (everyone)

| ID | Owner | Change | CV macro-F0.5 (mean ± sd) | LOCO US→IN / IN→US | Precision / Recall | Decision |
|---|---|---|---|---|---|---|
| E0 | P1 | Empty predictions (all singletons) | = train singleton share | | | baseline |
| E1 | P1 | `run_pipeline.py` baseline (3-view TF-IDF blocking, 28 features, LGBM, t on OOF, one-to-one) | | | | |

---

## 6. Submission ledger (15 slots)

| Slot | Time (IST) | Git tag | Description | CV | Public LB | Validator PASS | Keep for final? |
|---|---|---|---|---|---|---|---|
| D1-1 | | | Empty-prediction baseline (format check; public LB ≈ public singleton share) | | | | |
| D1-2 | | | Rule-based fuzzy baseline | | | | |
| D1-3 | | | First LightGBM matcher + OOF threshold | | | | |
| D1-4 | | | + one-to-one assignment / conflict resolution | | | | |
| D1-5 | | | Reserve (use only for a structural change with a CV gain) | | | | |
| D2-1 … D2-5 | | | Blocking v2, embeddings, France-robust normalisation, ensemble | | | | |
| D3-1 … D3-3 | | | Best candidates | | | | |
| D3-4, D3-5 | | | **Reserved:** final best-CV + hedge. Upload before 22:00 IST | | | | |

Rules: never spend an upload on a threshold or hyperparameter nudge. A public-LB difference below noise is not evidence.

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
