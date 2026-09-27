# Approach Document: Business Entity Resolution

> **SUPERSEDED.** This was the working draft. The finished methodology, with every
> number filled in from the final run, is [`Documentation_template.md`](../Documentation_template.md)
> in the repository root. Kept here for history.

> Draft kept in sync with the code. The final version is copied into the organisers' `Documentation_template.md` (same headings) and zipped. Every number must come from `work/report.json` or `STATUS.md`. **[TBD]** = fill in from a real run.

## 1. Methodology used

A three-stage pipeline: **normalise → block (candidate generation) → pairwise match → set-level decision**.
- The problem is framed as binary classification over (S1 entity, S2/S3 candidate) pairs, followed by a decision layer tuned directly for the official metric (macro F0.5 per S1 entity, singletons included).
- Validation: 5 folds over S1 entities (all pairs of an entity in one fold), OOF predictions, and a cross-fitted threshold, so the reported CV is not tuned on the rows it scores.
- **Leave-one-country-out** runs estimate how well the model transfers to the unseen test country (France).
- No external data or lookup services are used. Every signal comes from the provided records.

## 2. Candidate generation / blocking strategy

- **Normalisation (country-agnostic):** accent stripping, lower-casing, punctuation removal, expansion of name abbreviations (Corp→corporation, Pvt→private, &→and) and address abbreviations (Rd→road, Blvd/Bd→boulevard, Nr→near). A "core name" drops legal forms (Inc, LLC, Pvt Ltd, SARL, SAS, …).
- **Three blocking views,** each giving top-K nearest neighbours by TF-IDF cosine within the record's country (falling back to all records if the country is absent):
  1. character 2–4-grams of the core name, K = 15
  2. character 3–4-grams of core name + address, K = 15
  3. word 1–2-grams of the address, K = 5
  
  The union of the three is the candidate set.
- Recall ceiling on train: **[TBD]**. Average candidates per S1: **[TBD]**. Reduction ratio vs all pairs: **[TBD]**.

## 3. Model architecture and feature engineering

- **28 pair features** (`src/pair_features.py`):
  - string similarity (rapidfuzz ratio / token-sort / token-set / partial / Jaro-Winkler) on the full and core name and on the address
  - the three blocking cosines
  - postal code and street-number agreement (missing is kept distinct from "different")
  - whether the candidate comes from Source 3
  - **context features:** the candidate's rank and score gap within its S1 entity's list, and how many S1 entities compete for the same candidate
- **Matcher:** LightGBM binary classifier (lr 0.05, 63 leaves, early stopping); test predictions = average of the 5 fold models.
- **Decision layer** (`src/decide.py`):
  - **one-to-one assignment**, where each S2/S3 record goes only to its highest-probability S1 entity (enabled because only **[TBD]**% of training records match more than one S1 entity)
  - probability threshold chosen on OOF predictions for macro F0.5 (t = **[TBD]**)

## 4. Experiments and results

| Version | Change | CV macro-F0.5 (cross-fitted) | LOCO (US / India) | Public LB |
|---|---|---|---|---|
| E0 | All-empty baseline | [TBD] | — | [TBD] |
| E1 | Baseline pipeline | [TBD] | [TBD] | [TBD] |

## 5. Error analysis and other relevant information

[TBD after the first error-analysis pass. The largest slices of false matches and misses, and what was changed.]

## 6. Conclusion and limitations

[TBD]
