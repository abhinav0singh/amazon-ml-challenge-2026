# Business Entity Resolution at 10M-record scale

**Amazon ML Challenge 2026 · Team `epoch` · 72-hour hackathon**

Match business records across three noisy, unlinked data sources. For each of **1,732,544**
Source-1 entities, find every Source-2/3 record describing the same real business — zero, one,
or many — out of **9,969,589** candidates, with no shared keys and no ground truth for the test
set.

| | |
|---|---|
| **Final score** | **0.9491** cross-fitted CV · **0.937** public leaderboard |
| Baseline (predict nothing) | 0.0558 |
| Search space reduced | **1 in 141,288** — 70.56 candidates scored per entity, not 9,969,589 |
| Blocking cost growth | **0.91 exponent — sub-linear in corpus size** |
| Data | 2.2M × 10.3M train pairs · 1.7M × 10.0M test · 2.3 GB |
| Full pipeline runtime | 19.8 h on 8 vCPU / 67 GB |

Scored by **F0.5 per entity, macro-averaged** — precision weighted 2× recall, and an entity with
no true matches scores 1.0 only if you correctly predict nothing.

---

## The interesting part: where the score actually goes

Most write-ups list what worked. The useful finding here came from measuring what *didn't*.

We decomposed all **7,638,365** true training pairs against the shipped model:

| Fate of a true pair | Pairs | Share | Cause |
|---|---|---|---|
| Never became a candidate | 246,719 | 3.2% | blocking |
| Candidate, scored p < 0.15 | 147,860 | 1.9% | matcher ranked it near zero |
| Kept, but below the threshold | 355,508 | 4.7% | F0.5 forces a high cutoff |
| Correctly predicted | 6,888,278 | 90.2% | — |

**Blocking was the smallest of the three causes**, which contradicted our working hypothesis all
the way to hour 65. A perfect decision layer over the pairs the matcher already kept would score
**0.9814** against our **0.9500** on the same frame — so the headroom was in the *ranking*, not
the cutoff.

We tested that directly with a second full-scale run: widen candidate generation until the
blocking oracle rises (India 0.979 → 0.9864), producing 40% more candidates per entity.
**Cross-fitted CV moved from 0.9491 to 0.9489 — not at all.** More candidates did not help,
because the matcher does not score them above threshold. We kept the smaller candidate set.

**A named blind spot.** Inspecting a 24-entity sample by hand found the matcher scoring **0.000**
on pairs like `Creative Global Limited` ↔ `ક્રિએટિવ ગ્લોબલ લિમિટેડ`, and the same for Devanagari and
Bengali. Every string feature we use is a character n-gram or an edit distance, and a Gujarati
string shares **no characters** with its Latin form — so all of them read zero. Transliteration is
invisible to this feature set *by construction*. That is the single highest-value extension of
this work, and we can say exactly why.

---

## Approach

**normalise → block → pairwise classify → set-level decision**

**Candidate generation** is the part that has to scale, so it got the most attention. Rather than
rank each entity against its whole country group — which measured a **1.60 cost exponent, rising**,
with coverage *falling* as the corpus grew — records are indexed under several complementary keys:
their 3 rarest name tokens, 2 rarest address tokens, a name prefix, and a postal key. A true match
survives if it shares *any one* of them, so the schemes fail independently: rare tokens survive
reordering and suffix noise, the prefix survives a mangled interior, the address key survives a
renamed business.

Exact whole-name keys were tried first and scored 0.879 recall — any single typo breaks the whole
key, and typos are what this dataset is made of. Rare-token indexing lifted that to 0.979.

Three TF-IDF cosine views (char n-grams on name, on name+address, word n-grams on address) rerank
*inside* each block, fitted once per country group so scores stay comparable.

**Matcher:** LightGBM over 28 similarity and context features — string distances, TF-IDF cosines,
postal/number agreement, and each candidate's rank among its entity's alternatives.

**Decision layer:** threshold tuned for macro-F0.5 on out-of-fold predictions, plus one-to-one
assignment. Training data confirmed *exactly zero* Source-2/3 records belong to two entities
across all 7.6M pairs, so the constraint is hard rather than approximate.

---

## Engineering

The pipeline had to run on a 16 GB laptop before it ever saw a cloud VM. Four things made that
possible, each found by measurement after a run died:

**Memory: 5.4 GB → 0.35 GB.** Normalisation gave every record a Python `set` for postal codes and
another for address numbers. At ~216 bytes per empty set that is 5.4 GB across 12.5M records —
enough to exhaust the machine before blocking started. Packed as flat int64 arrays plus offsets:
0.35 GB.

**An int32 overflow that killed a 2h15m run.** Attaching cosines to candidates fancy-indexed one
sparse row *per pair* — billions of non-zeros, past SciPy's int32 index limit, surfacing as
`negative dimensions are not allowed`. Sampled runs never reached it because 40k entities produce
a thousand times fewer pairs. Fixed with density-derived chunking and a regression test that pins
the property rather than the symptom.

**Non-determinism in blocking.** Identical runs produced 38,319 / 38,325 / 38,331 candidate pairs.
Rare-token selection broke frequency ties by `set` iteration order, which varies with Python's
string hash randomisation. Every before/after comparison was meaningless until it was fixed.

**Streaming throughout.** Country groups processed one at a time and spilled to parquet, records
addressed by integer position, features computed in chunks and discarded after prediction, and
pairs below the threshold grid's floor dropped at prediction time — which removes 82% of them.

---

## Validation discipline

- **Folds locked and committed** — 5 folds, seed 42, grouped by Source-1 entity, so all of an
  entity's pairs stay together. Committed to git so every number anyone quoted stays comparable.
- **Cross-fitted scoring only.** The threshold is chosen on four folds and scored on the fifth.
  The threshold-optimised number is never reported as CV.
- **A measured noise floor.** Changes are accepted only above 2× the seed-to-seed standard
  deviation *and* on ≥4 of 5 folds. Several promising ideas were rejected by this rule.
- **Leave-one-country-out** as the only available proxy for France, which is 15% of the test set
  and absent from training. It predicted the CV-to-leaderboard gap almost exactly.

Experiments that were tested and **rejected** are logged with the numbers that rejected them —
decision-layer rules (below the noise floor), an exact-key join (a no-op: the pairs it found were
already predicted), and wider candidate generation (no CV movement). The negative results localise
the remaining loss, which is why they are kept.

---

## Repository

```
src/          pipeline: normalize, blocking, pair_features, decide, metric, cv_folds, run_pipeline
scripts/      packaging, auditing, blocking diagnostics, decision-layer evaluation
docs/         RUN_FINAL.md (how to run at scale), COMPLIANCE_AUDIT.md (rules audit)
tests/        normalisation and blocking regression tests
Documentation_template.md   full methodology write-up
STATUS.md     experiment log, submission ledger, measured data facts
AGENTS.md     the team contract — metric, locked folds, ownership, hard rules
```

**Reproduce:**
```bash
py -3.12 -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python src/metric.py                       # must print: metric OK
.venv/Scripts/python src/run_pipeline.py --data <DATA> --out output --work work --loco
```
Needs ≥32 GB RAM. `docs/RUN_FINAL.md` has the full procedure and stage timings. The pipeline is
resumable — each stage caches against a signature of the code, data and settings.

---

## Honest assessment

We finished well outside the leading cluster, which sat at 0.991. That gap is not thresholding or
tuning — it is **355,508 true pairs our matcher ranks below false positives**, and the
transliteration finding above explains a real share of them.

What this repository does have: a candidate generator with measured sub-linear scaling and a
141,288× reduction ratio, a validation setup whose CV predicted the leaderboard to within the
France gap it also predicted, a full-scale run that is reproducible from a clean clone, and a set
of negative results that say precisely where the remaining loss lives.

---

## The team

Four people, three days. Credited by what they carried, not by commit count.

**Abhinav Singh** — Team lead. Pipeline architecture, the multi-key blocking redesign, the
validation contract (locked folds, cross-fitted scoring, the noise-floor rule), the memory and
scaling work, and every leaderboard upload.

**Heeda Hurain Siddiqui** — Ran the 19.8-hour full-scale job on GCP that produced the submitted
result, and handed back the validated output. Also wrote the France-focused normalisation tests
and found the single-letter address mapping (`n` → `north`) that fires on 8.5% of Indian addresses
but only 0.12% of French ones.

**Anshika Mishra** — The methodology document, the packaging and pre-upload audit tooling, and
the decision-layer evaluation harness. Also measured and then **rejected** the exact-key join:
it looked like a clear win at 99.98% precision, and she proved on real out-of-fold data that
every pair it found was already predicted — a no-op. That measurement stopped an untested change
being pasted over a validated submission in the last hours.

**Pragathi Kharvi** — Vectorised the pair-feature computation, independently reaching the same
`rapidfuzz.cpdist` solution that landed on main, and correctly leaving the worker count at its
default — the threaded path segfaults on this data.

Built in 72 hours, on a 16 GB laptop and rented compute.

---

## Licence and data

Code is MIT (see `LICENSE`). **The dataset is not ours and is not included** — it belongs to the
challenge organisers, is excluded by `.gitignore`, and nothing in this repository reproduces it.
The pipeline reads it from a path you supply.
