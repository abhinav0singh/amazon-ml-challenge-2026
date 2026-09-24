# Problem Formulation

**Status: LOCKED at hour 1 (25 Sep 2026, IST).** Change it only with whole-team agreement, and log the change in `STATUS.md` → Decision log.

## 1. The exact evaluation metric

**F0.5, per Source-1 entity, macro-averaged over ALL Source-1 entities.**
- Per entity: `F0.5 = 1.25·P·R / (0.25·P + R)` over its predicted vs true set of S2/S3 ids.
- **Singletons count:** true set empty + predicted empty = **1.0**; true set empty + any prediction = **0.0**.
- Precision weighs 2× recall, so a wrong match hurts more than a missed one. When unsure, predict nothing.
- Implemented in `src/metric.py` (unit-tested on the official example → 0.714).

## 2. What the test set predicts

- For every S1 test entity, which S2/S3 test records refer to the same real business (0, 1 or many).
- **Unseen entities:** test businesses are new, so no id-level memorisation is possible.
- **Unseen country:** train = US + India; **test adds France**. Features must be country-agnostic, and normalisation must handle accents and French legal forms (SARL, SAS, SA, EURL) and street words (rue, av., bd).
- Two outputs: `output/matching_results.tsv` (scored) and `output/candidate_pairs.tsv` (blocking audit). Every match must also be a candidate.

## 3. Locked CV strategy

- **Split S1 entities into 5 folds** (seeded shuffle, seed 42) → `work/folds.csv`. All candidate pairs of an S1 entity stay in its fold.
- The threshold and decision rule are tuned on OOF predictions. The **reported CV is cross-fitted** (threshold picked on 4 folds, scored on the 5th).
- **Generalisation check:** leave-one-country-out (train without US → score US; train without India → score India), a proxy for France. Run with `--loco`.
- Decision rule: change accepted if CV gain > 2× seed noise and ≥ 4/5 folds improve. The public LB is a sanity check only.

## 4. Realistic scope for 72 hours

In scope:
- Normalisation → multi-view TF-IDF blocking → pair features (rapidfuzz, cosines, postal/number agreement, rank/competition context) → LightGBM matcher → threshold + one-to-one assignment.
- Then: blocking recall improvements, error analysis, French/India-specific normalisation rules (hand-written, country-agnostic application), optionally multilingual sentence embeddings (MIT/Apache, ≤ 8B) as extra blocking + similarity features, and a small ensemble.

Out of scope:
- Any external lookup (APIs, geocoding, registries, web data). Breaking this means disqualification.
- Training a large model from scratch. LLM-based pairwise matching unless a small open-licence model clearly beats LightGBM on CV within budget.

## 5. Data we actually have vs. what we'd ideally want

- Have: names, addresses, country label per record; ground-truth match lists for train.
- Missing: no phone, website or geo fields; no French training examples; noisy addresses (landmarks, missing PIN/ZIP).
- Unknowns to measure at hour 1 (fill in `STATUS.md` §2): singleton share, matches per entity, whether one S2/S3 id can match more than one S1 entity, data sizes.

## 6. Task assignments

| Person | Responsibility | Deliverable | Deadline (IST) |
|---|---|---|---|
| Team Leader (P1) | Metric, folds, run pipeline, all portal uploads, git tags, final zip | Baseline upload D1-2 | 25 Sep 06:00 |
| Data Analyst (P2) | Normalisation + blocking recall (≥ 0.98 target) | Blocking v2 with recall report | 25 Sep 14:00 |
| ML Engineer (P3) | Pair features, LightGBM, error analysis | Feature batch 1 + error slices | 25 Sep 16:00 |
| Monitor → Decision & Docs (P4) | Threshold/one-to-one/LOCO, France robustness, approach doc | Decision layer v2 + doc draft | 25 Sep 20:00 |

## 7. Approach document ownership

P4 owns `docs/approach_document.md`; everyone adds notes under `STATUS.md` → §8 Doc notes after each experiment. The final version must follow the organisers' `Documentation_template.md` headings: methodology, candidate generation/blocking, model architecture and features, other relevant information.
