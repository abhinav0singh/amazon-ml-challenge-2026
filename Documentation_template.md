# Business Entity Resolution Challenge — Methodology Document

**Team:** `epoch` · **Repository:** `abhinav0singh/amazon-ml-challenge-2026`
**Challenge:** Amazon ML Challenge 2026 (Unstop) — Business Entity Resolution
**Document status:** Day 1 draft. Every number below is either **measured** (with the run that
produced it named) or marked **[TBD]**. Nothing is estimated.

> **Note on this file.** The organisers' blank `Documentation_template.md` ships inside
> `student_resource/`. That folder is not present on the machine this document was written on, so the
> headings below were taken verbatim from the problem statement PDF (*"Your methodology document must
> describe: Methodology used / Candidate generation/blocking strategy / Model architecture and
> feature engineering / Any other relevant information about the approach"*) and from the guidelines
> PDF (*"1-2 page document explaining the ML approach, ML models used, experiments and conclusion"*).
> **Before submitting, copy the organisers' blank template over this file's headings if they differ.**

---

## 0. Summary (the 1–2 page overview)

### 0.1 The problem

Three sources of business records (`S1-`, `S2-`, `S3-` prefixed `entity_id`s) with no shared keys and
noisy names and addresses. Source 1 is the deduplicated reference source. For every Source 1 entity
in the test set we must output the list of Source 2 / Source 3 test records that refer to the same
real business — which may be **zero, one, or many** records.

Scoring is **F0.5 computed per Source 1 entity and then macro-averaged over all Source 1 entities**,
singletons included. Precision is weighted twice as heavily as recall, and an entity with no true
matches scores 1.0 for an empty prediction and 0.0 for any prediction at all.

Training data covers **US and India**. The test set adds **France**, which appears nowhere in
training — 259,452 of 1,732,544 test Source 1 entities (14.98 %, measured directly from
`test_source1.tsv`).

### 0.2 The approach in one paragraph

A four-stage pipeline: **normalise → block (candidate generation) → pairwise classification →
set-level decision**. The problem is framed as binary classification over
(Source 1 entity, Source 2/3 candidate) pairs. Candidate generation is a multi-key blocking scheme
(rare-token inverted index, name prefix, postal key) with three TF-IDF cosine views reranking inside
each block. A LightGBM classifier scores each surviving pair on 28 similarity and context features.
A decision layer converts probabilities into per-entity sets using a global threshold and a
one-to-one assignment step, and is tuned directly for macro F0.5 on out-of-fold predictions.
Everything is fitted on the provided files only.

### 0.3 ML models used

| Component | Model / method | Library | Licence | Parameter count |
|---|---|---|---|---|
| Candidate generation | TF-IDF (char and word n-grams) + sparse top-k cosine | scikit-learn | BSD-3-Clause | n/a (no learned model; vocabulary is fitted per split) |
| Pairwise matcher | LightGBM gradient-boosted decision trees (`LGBMClassifier`, binary objective) | LightGBM | **MIT** | Upper bound **630,000** leaf values (5 folds × ≤2,000 trees × 63 leaves), six orders of magnitude below the 8 B limit |
| Decision layer | Threshold + greedy one-to-one assignment (no learned parameters) | — | — | 1 scalar threshold `t` |

**No pretrained model of any kind is used.** Nothing is downloaded, and no embedding is fitted on
anything but the provided files. See §4.2 for the full library licence list and the open question it
raises.

### 0.4 Experiments

| ID | Change | CV macro-F0.5 (cross-fitted) | LOCO | Decision |
|---|---|---|---|---|
| E0 | Empty prediction for every entity (format check) | ≈ **0.0558** (= measured train singleton share) | — | baseline |
| E1 | v1 full-scale pipeline (`run-final-v1`) | **0.9491** cross-fitted | India held out **0.9295**, US held out **0.9629** | **submitted** — public LB **0.937** |
| B0/B1 | Blocking: dense → sparse top-k over a whole country group | recall ceiling 0.9842–0.9986 on **sampled** runs (see §2.5) | — | rejected: cost exponent rose to 1.60, does not scale |
| B2 | Blocking: multi-key rare-token inverted index | recall ceiling 0.9788–0.9833 on **sampled** runs (see §2.5) | — | **current default**; cost exponent 0.91 (sub-linear) |

**The full-scale run completed on 27 Sep 05:28 (19.78 h, 8 vCPU / 67.4 GB).** Every CV, LOCO, precision, recall and
runtime number below now comes from that run. The earlier sampled blocking numbers in
§2.5 are from deliberately sub-sampled runs and are optimistic by construction — they are reported
as cost/ceiling measurements, never as CV.

### 0.5 Conclusion

The decisive properties of this problem are (a) the metric is a macro-average over entities, so an
entity with two candidates counts exactly as much as one with forty, and (b) precision is weighted
2×, so a false merge costs more than a miss. Both push the design toward a high-recall candidate
generator followed by a precision-oriented decision layer, rather than toward a single similarity
threshold. The largest remaining risks are blocking cost at full scale (2.2 M × 10.3 M records) and
generalisation to France, for which our only proxy is the leave-one-country-out measurement. Full
The full-scale run settles it. Cross-fitted CV is **0.9491** and the public leaderboard returned
**0.937**. The candidate set's oracle ceiling is ~0.989, so **~0.040 of the score is lost between
what blocking makes reachable and what the matcher actually finds** — and that loss is
discrimination, not thresholding: 348,000 true pairs are scored below the threshold, and about
88,000 entities have no candidate above 0.15 at all. Decision-layer rules were measured on the real
out-of-fold predictions and rejected (§4.8). The remaining work is a better matcher or better
candidates, not a better rule for turning probabilities into sets.

---

## 1. Methodology used

### 1.1 Problem framing

The task is framed as **binary classification over candidate pairs**, followed by a **set-level
decision** stage:

1. For each Source 1 entity, candidate generation proposes a small set of plausible Source 2 / 3
   records. This caps recall: a true match dropped here can never be predicted.
2. A classifier assigns each (S1, candidate) pair a probability of being a true match.
3. A decision layer turns those probabilities into the final set per entity, optimising the official
   metric directly rather than a pair-level proxy.

Pair-level accuracy, AUC and log-loss are used only as diagnostics. The objective that decides
anything is `macro_f05`, implemented once in `src/metric.py` and unit-tested against the worked
example in the problem statement (`python src/metric.py` prints `metric OK: example = 0.714`).

### 1.2 Normalisation (`src/normalize.py`)

Applied identically to every record of every source and every country. **Nothing branches on the
country label.** The steps, in order:

| Step | Function | What it does |
|---|---|---|
| 1 | `strip_accents` | Unicode NFKD decomposition, drop combining marks. `Société Générale` → `Societe Generale`. |
| 2 | `basic_clean` | Lower-case; `&` → ` and `; remove dots occurring between letters so `p.v.t.` → `pvt`; replace every character outside `[a-z0-9 ]` with a space; collapse runs of whitespace. |
| 3 | `norm_name` | `basic_clean`, then token-wise rewrite through `NAME_MAP`. Legal-form tokens are **kept**. |
| 4 | `core_name` | `norm_name` with every token in `LEGAL_TOKENS` removed. `ABC Pvt. Ltd.` → `abc`. If every token was a legal form, falls back to the full normalised name so the field is never empty. |
| 5 | `norm_addr` | `basic_clean`, then token-wise rewrite through `ADDR_MAP`. |
| 6 | `postal_codes` | The set of 5–6 digit runs found in the **raw** address (`\b(\d{5,6})\b`). Covers US ZIP, French code postal and Indian PIN without knowing which is which. |
| 7 | `numbers` | The set of all digit runs in the raw address (house numbers, plot numbers, postal codes). |

`add_normalized_columns` writes the derived columns `name_n`, `name_core`, `addr_n`,
`full_n` (= `name_core` + `" | "` + `addr_n`), `postal`, `nums` and `country_n`.

### 1.3 Validation design (`src/cv_folds.py`)

- **The Source 1 entity is the unit that must not leak.** Because the metric is computed per S1
  entity, all candidate pairs belonging to one S1 entity are assigned to the same fold.
- 5 folds, seed 42: sorted S1 ids, one seeded `numpy` shuffle, fold = position mod 5. Written once to
  `work/folds.csv` and **loaded, never silently regenerated**, so every teammate and every run scores
  on identical folds.
- **Every Source 1 entity in the fold is scored, not only the ones blocking found candidates for.**
  An entity with no candidates is a real prediction of "empty" and still contributes to the average.
  The full `s1_ids` list is always passed to `macro_f05`.
- **The reported CV is cross-fitted.** `decide.tune()` grid-searches the threshold on the same rows
  it scores, so its output is optimistic and is used for curve inspection only.
  `decide.cross_fitted_score()` picks the threshold on four folds and scores on the fifth; that is
  the number reported here and in the tracker.
- **Acceptance rule:** a change is adopted only if it gains more than 2× the seed-to-seed noise
  **and** improves at least 4 of the 5 folds. Anything smaller is logged as inconclusive and dropped.
- **Unseen-country check:** `--loco` trains without one country and scores on it — see §4.1.

### 1.4 Reproducibility

Single seed (42) for fold assignment, LightGBM `random_state`, bagging and any sub-sampling. Folds
are committed to git rather than merely deterministic, so a numpy or platform difference cannot
silently change them. Dependencies are pinned in `requirements.txt` against **Python 3.12**. The end
to end command is in `code/business_entity_resolution/README.md`.

**One reproducibility defect was found and fixed, and it is worth recording because it was invisible
to every test we had.** Candidate generation used not to be bit-reproducible across processes.
`blocking._rare_tokens` selects a record's three rarest tokens with
`sorted(set(tokens), key=...)`; `sorted` is stable, so ties on document frequency were broken by the
iteration order of a *set of strings*, which depends on Python's per-process string-hash
randomisation. Ties are the common case, not the rare one, because most tokens share a low document
frequency. Measured on a 20,000-entity sample: two runs of the same command on the same data produced
candidate sets differing by **0.16 %** of pairs (1,581 of 997,187 present in only one of them), and
the same input under six values of `PYTHONHASHSEED` produced six different key sets.

The fix is a deterministic tie-break, `key=lambda x: (doc_freq.get(x, 0), x)` — rarest first, then
alphabetical — which is stable across processes and costs nothing. It is in the shipped code. A
reviewer re-running the pipeline now gets the same candidate set we submitted. Figures in this
document that were measured before the fix are flagged in §4.6.

---

## 2. Candidate generation / blocking strategy

### 2.1 Why blocking is the ceiling

Exhaustively comparing every Source 1 entity with every Source 2/3 record is 2,206,822 × 10,320,219
≈ 2.3 × 10¹³ pairs on train alone. Blocking reduces that to a shortlist per entity, and the share of
true pairs that survive it — the **recall ceiling** — is a hard cap on everything downstream. The
recall ceiling and the reduction ratio are therefore measured, not assumed.

### 2.2 Country grouping — and why it is not a France bug

`generate_candidates` groups Source 1 records by the normalised country label and searches the
Source 2/3 records carrying the **same** label. If a country label is absent from the Source 2/3
side, or the resulting group is empty, the search falls back to **all** Source 2/3 records, so an
unseen or mislabelled country never loses its candidates.

Country is used **only as a grouping key**. There is no branch anywhere in `src/` of the form
`if country == "US"`. See §4.1 for the France discussion and the verification.

### 2.3 The three similarity views

Three TF-IDF views are fitted **once per country group**, on the concatenation of that group's
Source 1 and Source 2/3 text, and used to rerank inside every block of that group. Fitting per block
would put cosines from different blocks on different scales and would silently corrupt the `cos_*`
features and every rank/gap feature derived from them.

| View | Field | Analyzer | n-grams | Top-K |
|---|---|---|---|---|
| `cos_name` | `name_core` | `char_wb` | 2–4 | 15 |
| `cos_full` | `full_n` (core name + address) | `char_wb` | 3–4 | 15 |
| `cos_addr` | `addr_n` | `word` | 1–2 | 5 |

All three use `sublinear_tf=True`, `max_df=0.01`, `min_df=2`. Vocabulary pruning by document
frequency is the main lever on runtime: an n-gram present in a large share of records carries almost
no IDF weight but sits in a huge posting list, so it dominates the sparse product's non-zeros while
barely moving the cosine. If pruning empties the vocabulary of a small group, the code falls back to
an unpruned vectoriser for that group — small groups are cheap to match exhaustively.

The product is kept **sparse end to end** (`_topk_sparse`): only pairs sharing at least one n-gram
are ever materialised, and a row with fewer than K non-zero similarities genuinely yields fewer than
K candidates rather than being padded with arbitrary zero-similarity records.

### 2.4 Multi-key blocking (the current default)

Ranking each Source 1 entity against a whole country group is roughly quadratic and does not scale
(see §2.5). The current design partitions both sides into many small blocks by four complementary
keys and runs the three-view top-k matcher **inside each block**:

| Key | Definition | Failure mode it survives |
|---|---|---|
| `pfx5` | First 5 characters of `name_core` with spaces removed | A mangled or reordered name interior |
| `postal` | The smallest postal code in the address + `\|` + the first character of the space-stripped core name; only emitted when the record has a postal code | A completely different name spelling, when a code is present |
| `nametok` | The **3 rarest** tokens of `name_core` (by document frequency over both sides), prefixed `n:` | Word-order transposition, legal-suffix noise, partial typos — only **one** distinctive word has to survive |
| `addrtok` | The **2 rarest** tokens of `addr_n`, prefixed `a:` | A renamed business or a DBA/trade name |

A true match survives if it shares **any** key, so the keys are chosen to fail independently. Rare
tokens are both the most identifying and the cheapest to look up (short posting lists); common
tokens are skipped precisely because they would form huge blocks and carry little information.

Two caps bound the worst case:

- `MAX_BLOCK = 20000` — a key whose Source 2/3 block exceeds this is skipped **for that key**. These
  are the low-information keys (a very common prefix); they would dominate runtime and the other keys
  still cover those records.
- `MAX_CANDS = 80`, ranked by **`cos_full + cos_addr`** (`CAP_SCORE` in `blocking.py`) — after the
  keys are unioned, each Source 1 entity keeps its best 80 candidates by that sum.

  **Both the ranking rule and the cap value were changed after a full-scale measurement, and the
  change is the single largest score improvement in this project.** The original rule ranked by the
  *maximum* of the three cosines with a cap of 40. That looks reasonable and is badly wrong on this
  data: a business with many branches produces candidates whose names are *identical*, so every
  branch scores `cos_name = 1.0`. The maximum therefore ties across all of them, the tie is broken
  arbitrarily, and the cap discards the one branch that is actually the right answer — which differs
  from its siblings only in the address. Summing `cos_full + cos_addr` makes the address the
  tie-breaker exactly where the name has stopped discriminating. §2.5 has the numbers.

`candidate_pairs.tsv` is written from this final, capped candidate frame — it is the exact set the
matcher runs inference over, as the problem statement requires.

An earlier variant using only exact whole-name keys (no rare tokens) scored recall 0.8785 and entity
cover 0.7395 at the 2,000-entity sample — exact keys are too brittle for this noise, which is why
rare-token indexing is the core of the design.

### 2.5 Measured blocking results

#### Full scale — and why every sampled number above was misleading

First full-scale train blocking, measured 26 Sep 00:34 (config: multi-key, `MAX_CANDS=40` ranked by
`max`, 3 name / 2 address rare tokens, `MAX_BLOCK=20000`):

| Split | Total pairs | Cands / S1 | **Pair recall ceiling** | **Entity full cover** | Wall-clock |
|---|---|---|---|---|---|
| train (US + India) | 87,483,735 | 39.64 | **0.8851** | **0.7665** | ~4.5 h |

The 40,000-entity sample had projected **0.9833 / 0.9546**. The full-scale reality was
**0.885 / 0.766** — 23 % of entities missing at least one true match, capping a *perfect* matcher
far below the achievable band. **Sampling optimism was much larger than extrapolation from the
sample sizes predicted**, which is the single most important methodological lesson of this project:
a blocking ceiling measured on a shrunken haystack is not a forecast of the real one, because the
haystack is precisely what makes blocking hard.

#### Root cause, and the fix

`scripts/diag_blocking_recall.py` measures recall for a *subset* of Source 1 entities against the
**full** Source 2/3 haystack, so block sizes and cap competition are realistic while the run stays
affordable. That diagnostic localised the loss to the cap's ranking rule, not to the keys.

India — 10,000 Source 1 entities against all 4,133,346 India Source 2/3 records (pair recall / entity
full cover):

| Rank the cap by | cap 40 | cap 60 | cap 80 | no cap |
|---|---|---|---|---|
| `max` of the three cosines (old) | 0.823 / 0.656 | 0.926 / 0.815 | 0.941 / 0.847 | 0.943 / 0.853 |
| **`cos_full + cos_addr` (new)** | 0.934 / 0.829 | 0.942 / 0.848 | **0.943 / 0.852** | — |

US — 10,000 Source 1 entities against all 6,186,873 US records, with the **oracle macro-F0.5**: the
score a *perfect* matcher would achieve on the candidate set, i.e. the ceiling blocking imposes on
the metric itself.

| US setting | Recall | Cover | Oracle F0.5 |
|---|---|---|---|
| `max` rank, cap 40 (old) | 0.929 | 0.836 | 0.9713 |
| `cos_full + cos_addr`, cap 60 | 0.984 | 0.952 | 0.9947 |
| **`cos_full + cos_addr`, cap 80 (adopted)** | **0.985** | **0.955** | **0.9952** |
| no cap (ceiling) | 0.985 | 0.956 | 0.9953 |
| + a 3rd address rare token, cap 80 | 0.987 | 0.959 | 0.9961 |

**On US the old ranking rule alone cost 0.024 of the achievable score** — larger than any modelling
change measured in this project. At cap 80 the new rule reaches the no-cap ceiling to within 0.0001
of oracle F0.5, so the cap is no longer binding in any measurable way.

A third address rare token buys a further +0.0009 oracle F0.5 for +19 % pre-cap pairs and was
**deferred** as a poor trade against blocking cost; a fourth *name* rare token gave no gain at all.

**Which keys earn their place** (India): address rare tokens are the workhorse — 0.839 recall on
their own, and they are the only key that finds 6,451 true pairs. The postal key contributes
approximately nothing in India, where only 0.3 % of records carry a code; it remains because it is
decisive where a code *is* present and costs almost nothing.

**Remaining recall lives in India.** Its cover ceiling (0.853) is far below US's (0.956), so India,
not France, is where candidate generation is still weakest.

#### Earlier sampled runs — kept for the record

> **Every row below is from a deliberately sub-sampled run** (`src/sweep_blocking.py` and
> `run_pipeline.py --sample`). Sampling preserves the Source 2/3-records-per-entity density but
> shrinks the haystack. Compare them against the full-scale figures above to see how far optimistic
> they are; they are reported here as the cost/ceiling measurements that drove the design, never as
> estimates of real performance. The full-run figures are in the table above and in §2.6.

**B1 — sparse top-k over a whole country group (rejected):**

| Sample | Setting | Recall ceiling | Entity full cover | Cands / S1 | Blocking time |
|---|---|---|---|---|---|
| 2,000 | max_df 0.1 | 0.9986 | 0.9950 | 24.85 | 2 s |
| 10,000 | max_df 0.1 | 0.9948 | 0.9841 | 26.17 | 16 s |
| 40,000 | max_df 1.0 (no pruning) | 0.9894 | 0.9683 | 27.07 | 241 s |
| 40,000 | max_df 0.1 | 0.9897 | 0.9697 | 27.12 | 133 s |
| 40,000 | max_df 0.01 | 0.9842 | 0.9551 | 27.27 | 45 s |
| 40,000 | max_df 0.001 | 0.9495 | 0.8722 | 22.61 | 35 s |
| 40,000 | max_df 0.01, K 10/10/5 | 0.9816 | 0.9480 | 18.34 | 44 s |

Two measured trends condemned this design: the cost exponent rose from 1.29 (2k→10k) to **1.60**
(10k→40k), heading toward quadratic, and entity cover **fell** 0.9950 → 0.9841 → 0.9697 as the
haystack grew. Raising K to recover cover makes cost worse. Pruning bought 5.4× and then stopped
helping, which shows the remaining cost is not in long posting lists — so pruning alone cannot make
this design scale.

**B2 — multi-key blocking (current default):**

| Sample | Recall ceiling | Entity full cover | Cands / S1 | Blocking time |
|---|---|---|---|---|
| 2,000 | 0.9788 | 0.9435 | 19.2 | 9 s |
| 10,000 | 0.9821 | 0.9530 | 39.9 | 42 s |
| 40,000 | 0.9833 | 0.9546 | 60.6 | 148 s |
| 40,000, `MAX_CANDS` = 40 | 0.9686 | 0.9208 | 38.1 | 122 s |

Cost exponent **0.91 — sub-linear** — against 1.60 and rising for B1, and entity cover is *stable*
across scales where B1's fell. `MAX_CANDS` is set to 80 because at 40 it cost 0.034 of entity cover
for 37 % fewer pairs, a bad trade now that blocking cost is linear.

**Reduction ratio, measured on the submitted run.** Test blocking produced **122,251,746** candidate
pairs (**70.6 per Source 1 entity**) where the full cross-product of 1,732,544 Source 1 entities
against 9,969,589 Source 2/3 records is 1.727 × 10¹³ pairs. That is **1 pair kept in every 141,000**,
a reduction of **99.99929 %**, while still covering 96.77 % of true pairs on train.

### 2.6 The submitted run, end to end

All figures below are read from `submissions/sub-D2-1/RUN_SUMMARY.txt` and `report.json`, the
artefacts the run itself wrote. Nothing here is retyped from memory or estimated.

| | |
|---|---|
| Release tag | `run-final-v1` (git `180614f`) |
| Machine | 8 vCPU, 67.4 GB RAM, Python 3.12.3 |
| Wall-clock | **19.78 h** |
| Train blocking recall ceiling / entity cover | **0.9677 / 0.9121** (71.2 candidates per S1) |
| Test candidate pairs | **122,251,746** (70.6 per S1) |
| CV macro-F0.5, cross-fitted | **0.9491** — per fold 0.9491 / 0.9488 / 0.9492 / 0.9490 / 0.9493 |
| Threshold chosen | **0.65**, independently on every fold |
| OOF pair precision / recall | **0.9833 / 0.9018** |
| LOCO — hold out India / hold out US | **0.9295 / 0.9629** |
| Test rows written | 1,732,544 (1,626,387 non-empty, 5,579,264 matched ids) |
| Non-empty share, test vs OOF | 0.939 vs 0.937 |
| Official validator | exit code **0 (PASS)** |
| Public leaderboard | **0.937** |

The test non-empty share (0.939) sits within 0.002 of the out-of-fold share (0.937), which is the
cheapest available evidence that the test half behaved like cross-validation rather than diverging.

**CV 0.9491 against a public leaderboard of 0.937** is a gap of 0.012. Checked in the order
`AGENTS.md` §8 prescribes: a submission bug is ruled out (validator PASS, audit PASS, hashes match);
a metric mismatch is ruled out (the empty baseline predicted 0.056 and scored exactly 0.056);
leakage would push CV *above* the leaderboard rather than below. What remains is train/test shift —
France is 15.0 % of the test set and absent from training, and LOCO independently measures an unseen
country at 0.9295 against 0.9491 in-distribution. The direction and rough size agree.

### 2.7 Candidate-set efficiency

Blocking is judged not only by what it keeps but by how little it keeps. A candidate set that is
larger than it needs to be costs inference time at every later stage and, at Amazon's scale, is the
difference between a feasible system and an infeasible one.

The submitted run produced **70.6 candidates per Source 1 entity** (122,251,746 pairs over 1,732,544
entities), capped at 80 and ranked by `cos_full + cos_addr`. We measured how much of that is actually
load-bearing by re-ranking a full pair frame with the same score and truncating at each cap:

| Cap | Candidates / S1 | True pairs retained (vs cap 80) |
|---|---|---|
| 80 (shipped) | 58.4 | 1.00000 |
| 60 | 51.7 | 0.99965 |
| 40 | 38.1 | 0.99840 |
| 30 | 29.3 | 0.99703 |
| 20 | 19.8 | 0.99369 |
| 10 | 10.0 | 0.98342 |

**Cap 20 discards two thirds of the candidate set and 0.63 % of the true pairs. Cap 30 halves it for
0.30 %.** Those losses have to be read against the matcher, which already fails to rank 10 % of the
true pairs it is given above the threshold (pair recall 0.9018). The candidates removed by a tighter
cap are, overwhelmingly, ones the model was never going to select: of 122,251,746 candidates in the
submitted run, only 5,579,264 ids — **4.6 %** — appear in any prediction.

So the shipped cap of 80 is conservative, and deliberately so: it was chosen when the measured risk
was losing recall, before the cost of a large candidate set was part of the evaluation. On this
evidence a cap in the 20–30 range is the better operating point, and we record it here as the
change we would make with more runway, rather than one we can claim to have measured end to end.

**Caveat, stated rather than buried:** the table above comes from a 40,000-entity run, so its
haystack is smaller than the real one and the cap binds less (58.4 candidates per entity against
70.6 at full scale). At full scale a tighter cap would cut more, and would also cost somewhat more
recall than shown. The direction is solid; the exact figures are not a full-scale measurement.

---

## 3. Model architecture and feature engineering

### 3.1 The 28 pair features (`src/pair_features.py`)

Every feature is a **similarity or a relative rank**. No raw country value and no country-specific
rule enters the model, so a matcher trained on US and India data transfers to France.

**String similarity on the name (rapidfuzz), 5 features**
`nm_ratio`, `nm_tsort` (token-sort), `nm_tset` (token-set), `nm_partial`, `nm_jw` (Jaro-Winkler),
all computed on the normalised full name `name_n`.

**String similarity on the core name, 5 features**
`core_ratio`, `core_tset`, `core_jw`, `core_first_tok_eq` (do the first tokens match exactly),
`core_len_diff` (absolute character-length difference).

**String similarity on the address, 3 features**
`ad_ratio`, `ad_tset`, `ad_partial`. When either address is empty these are set to **−1**, which says
"unknown" rather than "different" — an important distinction given how often addresses are missing.

**Blocking cosines, 3 features**
`cos_name`, `cos_full`, `cos_addr`, carried over from candidate generation. All three are attached to
every pair, not only to the view that proposed it.

**Postal and number agreement, 3 features**
`postal_match` (1 if the postal-code sets intersect, 0 if both sides have codes that do not
intersect, **−1** if either side has none), `num_jacc` (Jaccard of the digit-run sets, −1 if both
empty), `num_overlap` (size of the intersection).

**Source, 1 feature**
`is_s3` — whether the candidate comes from Source 3.

**Context within the entity's candidate list, 5 features**
`n_cands` (how many candidates this entity has at all), `rank_name_in_s1`, `gap_name_to_best`,
`rank_full_in_s1`, `gap_full_to_best`. These let the model express "this is the best of forty
candidates" versus "this is the fourth of five", which a raw similarity cannot.

**Competition between entities for the same candidate, 3 features**
`rank_s1_for_cand`, `n_s1_for_cand`, `gap_to_best_s1_for_cand`. Because Source 1 is deduplicated, a
Source 2/3 record usually belongs to at most one Source 1 entity, so a candidate that several
entities want is evidence against most of them.

### 3.2 The matcher

LightGBM `LGBMClassifier`, binary objective:

```python
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, n_estimators=2000,
                  random_state=42, verbose=-1, n_jobs=-1)
```

One model per fold (5 total), each trained on the other four folds with **early stopping after 100
rounds without improvement** on the held-out fold. Out-of-fold probabilities come from the model that
did not see the fold. Test probabilities are the **arithmetic mean of the five fold models'**
`predict_proba` outputs.

Labels: a pair is positive if the candidate id appears in that Source 1 entity's ground-truth set.

**Licence and size:** LightGBM is **MIT**. A gradient-boosted tree ensemble has no "parameter count"
in the sense the 8-billion-parameter rule uses; the closest analogue is the number of leaf values,
bounded above by 5 folds × 2,000 trees × 63 leaves = **630,000**, six orders of magnitude below the
limit. Early stopping sets the realised tree count per fold; the submitted run's models are the five
saved by `run-final-v1`.

### 3.3 Decision layer (`src/decide.py`)

The decision layer is the only stage that directly sets precision, and F0.5 weights precision 2×.
Two illustrative cases for an entity with 4 true matches: predicting 3 correct and 0 wrong scores
**0.938**, while predicting all 4 plus 1 wrong scores **0.833** — a miss beats a false merge. But
predicting nothing on that entity scores **0.0**, and only **5.58 %** of training entities are
singletons (123,247 of 2,206,822, measured), so predicting empty everywhere scores about **0.056**.
The posture is therefore high precision that still commits, not maximum caution.

**Step 1 — threshold.** Keep candidates with `p >= t`. `t` is searched on a grid from 0.20 to 0.95 in
steps of 0.025.

**Step 2 — one-to-one assignment.** Measured on train: **zero** Source 2/3 records belong to more
than one Source 1 entity — exactly zero, across all **7,638,365** true pairs. `check_one_to_one`
recomputes this share at run time and the assignment step is enabled only when it is below 1 %. The
implementation sorts surviving pairs by probability descending and keeps the first occurrence of each
candidate id, so each Source 2/3 record is awarded to the single Source 1 entity that scores it
highest.

Note that the constraint is **one-to-many in the other direction**: a Source 1 entity may legitimately
own many records (mean **3.67** true matches per non-singleton entity), and the assignment step does
not cap that.

**Threshold selection is cross-fitted.** `tune()` picks `t` on the rows it scores and is therefore
optimistic — it exists for curve inspection. `cross_fitted_score()` picks `t` on four folds and scores
the fifth, and its output is the only score quoted as CV. On the submitted run every fold
independently chose the same threshold, **t = 0.65**, and the cross-fitted CV was **0.9491**
(per fold 0.9491 / 0.9488 / 0.9492 / 0.9490 / 0.9493 — a spread of 0.0005, so the decision is stable
across folds). Out-of-fold pair precision was **0.9833** and pair recall **0.9018**.

### 3.4 Output

`data_io.write_id_lists` writes both TSVs by hand — no pandas quoting — with a literal tab separator,
`\n` line endings and UTF-8. One row per test Source 1 entity, in the order they appear in
`test_source1.tsv`, with a comma-joined, de-duplicated, sorted id list, and an empty field for
entities predicted to have no matches.

- `output/candidate_pairs.tsv` — written from the final capped candidate frame.
- `output/matching_results.tsv` — written from the thresholded, assigned predictions, which are a
  subset of that same frame by construction.

---

## 4. Any other relevant information about the approach

### 4.1 How France is handled

France is 259,452 of 1,732,544 test Source 1 entities — **14.98 %** (counted directly from
`test_source1.tsv`; `entity_id` is unique across all 1,732,544 rows). It appears nowhere in training.
Four design commitments follow:

1. **Country is a grouping key, never a condition.** `generate_candidates` groups by the normalised
   country label to keep blocks small, and falls back to searching all Source 2/3 records when a
   label is missing from the other side. There is no `if country == …` branch anywhere in `src/`
   (verified by grep across the whole package; the string literals `"US"`, `"India"` and `"France"`
   do not occur in any source file).
2. **No country value reaches the model.** None of the 28 features is the country, or derived from
   it. Every feature is a similarity or a relative rank.
3. **Normalisation is country-agnostic and already covers French forms.** Accents are stripped
   generically, so `Société` and `Societe` compare equal. The legal-form list already contains the
   French company forms (`sarl`, `sas`, `sasu`, `sa`, `eurl`, `sci`, `snc`), and the address list
   already contains French street abbreviations (`bd` → boulevard, `chs` → chemin, `rte` → route,
   `av` → avenue). Postal codes are extracted by pattern (5–6 digits), which matches the French code
   postal as readily as a US ZIP or an Indian PIN.
4. **`--loco` is the proxy measurement.** Leave-one-country-out trains the matcher without one
   country's entities and scores on exactly those entities, at the globally chosen threshold. It is
   the only evidence available before the leaderboard about how the model behaves on a country it has
   never seen. A change that improves CV but drops LOCO is a change that hurts us on 15 % of the test
   set. **LOCO result, measured on the submitted run: holding out India scores 0.9295, holding out
   the US scores 0.9629**, against an in-distribution CV of 0.9491. So transferring to a country the
   matcher has never seen costs roughly **0.02–0.03**, and India is the harder direction. France is
   15.0 % of the test set, and this is the only evidence we have about it.

Per-country thresholds are deliberately **not** used: they cannot be set for France, and a
hard-coded country branch is the exact failure this design guards against. A threshold that adapts to
a *measured property of the entity* — its candidate count, its top-probability gap — is country-blind
and is a legitimate avenue.

### 4.2 Why our normalisation dictionaries are not external data

`src/normalize.py` contains three hand-written dictionaries: `LEGAL_TOKENS` (34 company-form tokens),
`NAME_MAP` (17 name abbreviations) and `ADDR_MAP` (40 street and landmark abbreviations). They are
listed in full in §4.3 so a reviewer can check every entry. **They are not external data**, for five
reasons:

1. **The rule prohibits a *lookup*** — going outside the provided data to learn a fact *about a
   specific entity in the dataset*. Every prohibited example has that shape: an entity-resolution
   service tells you "these two records are the same business"; a registry tells you "this business
   exists at this address"; a geocoder tells you "this address is at these coordinates". Each returns
   a fact about a record that was not in the record.
2. **These dictionaries return no facts about any record.** `rd → road` is a statement about the
   English language, applied blindly to every row. It cannot distinguish one business from another
   and adds no information that was not already in the string being rewritten. Replacing the
   dictionary with forty hand-written `if` statements would be identical in effect.
3. **They are not data in the operative sense.** They were typed by a person — not fetched, scraped,
   downloaded, or derived from any corpus. There is no source to cite because there is no source.
4. **The distinguishing test:** could this dictionary, applied to a record, tell us anything about
   *that particular business* that the record did not already say? No. A geocoding API can. That is
   the line, and these dictionaries sit clearly on the safe side of it.
5. **Precedent in the rules themselves:** the problem statement's *Tips for Success* advises teams to
   "pay attention to country-specific address patterns", and its *Noise Patterns to Expect* section
   lists exactly these pairs (Corp vs. Corporation, Pvt vs. Private, Rd vs. Road, St vs. Street) as
   the noise to handle. The organisers describe this technique as the intended solution.

**The boundary we enforce in review:** the dictionaries stay **linguistic**. A French street-word
list (rue, avenue, boulevard, chemin) is still linguistic and is fine. The moment a dictionary
contains a list of real place names, real company names, real postal-code ranges, or anything one
could only produce by consulting an outside source, it changes category and becomes a violation — a
list of French *communes* would be a gazetteer, i.e. external data.

**Independent verification.** A grep across the whole of `src/` for `requests`, `urllib`, `http`,
`socket`, `boto3`, `openai`, `anthropic`, `api_key`, `download`, `urlretrieve` and `os.environ`
returns **zero hits**. No module in the package can open a network connection. Every file the
pipeline reads is either under the `--data` folder or an artefact the pipeline itself wrote.

### 4.3 The dictionaries in full

**`LEGAL_TOKENS` (34)** — removed from the name to form the "core" name:

```
ag, bv, cie, co, company, corp, corporation, dba, eurl, gmbh, inc, incorporated, limited, llc,
llp, lp, ltd, nv, opc, pc, plc, pllc, private, pty, pvt, sa, sarl, sas, sasu, sci, snc, spa,
srl, the
```

**`NAME_MAP` (17)** — token rewrites applied to names:

```
& → and            assoc → associates    bros → brothers      centre → center
ctr → center       dept → department     ent → enterprises    intl → international
mfg → manufacturing mgmt → management    st → saint           ste → sainte
svc → services     svcs → services       sys → systems        tech → technologies
techs → technologies
```

**`ADDR_MAP` (40)** — token rewrites applied to addresses:

```
apt → apartment    av → avenue        ave → avenue       bd → boulevard     bld → building
bldg → building    blvd → boulevard   chs → chemin       cir → circle       clny → colony
ct → court         dr → drive         e → east           fl → floor         flr → floor
fwy → freeway      hwy → highway      ln → lane          mkt → market       n → north
ne → northeast     ngr → nagar        no → number        nr → near          nw → northwest
opp → opposite     ph → phase         pkwy → parkway     pl → place         rd → road
rte → route        s → south          se → southeast     sec → sector       sq → square
st → street        ste → suite        str → street       sw → southwest     w → west
```

### 4.4 Library licences

| Library | Version | Licence | Used for |
|---|---|---|---|
| numpy | 1.26.4 | **BSD-3-Clause** | arrays, RNG, fold assignment |
| pandas | 2.2.2 | **BSD-3-Clause** | TSV I/O, joins, group-by |
| scipy | 1.13.0 | **BSD-3-Clause** | sparse matrices behind the TF-IDF product (transitive via scikit-learn) |
| scikit-learn | 1.5.0 | **BSD-3-Clause** | `TfidfVectorizer` only |
| lightgbm | 4.3.0 | **MIT** | the pairwise matcher |
| rapidfuzz | 3.9.6 | **MIT** | string-similarity features |
| pyarrow | 16.1.0 | **Apache-2.0** | parquet caches for OOF and test pair frames |

**Open question for the team, not decided here.** The constraint in the problem statement reads
*"Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters"*. On its face
it constrains the **model**, and our only model — LightGBM — is MIT. But four of the seven libraries
above are **BSD-3-Clause**, not MIT or Apache-2.0. BSD-3-Clause is a permissive licence of the same
family, and numpy/pandas/scikit-learn are near-universal in ML submissions, so the intended reading
is almost certainly that the rule targets the model rather than the toolchain. **This should be
confirmed with the organisers via the query form rather than assumed.** It is recorded here so a
reviewer sees that we identified it rather than overlooked it.

No pretrained model is used. If one is ever added, its exact model id, licence and parameter count
must be recorded **before** it is used, and it must be MIT or Apache-2.0 and at most 8 B parameters.

### 4.5 Engineering: what it took to run at full scale

Getting this pipeline through 2.2 M × 10.3 M records was a larger problem than any modelling
decision, and the diagnosis is worth recording because the obvious answer was wrong.

**The 6.5-hour stage that looked CPU-bound was memory swapping.** Assembling the ~20 M-pair training
sample took about 6.5 hours on a 16 GB laptop, which invited the conclusion that feature computation
needed parallelising. Measured on 26 Sep, that conclusion was **false**: all 16 string features run
at **55,000 pairs/s on a single core** (1 M pairs in 18 s; the slowest scorers are `ad_partial` at
237k/s and `ad_tset` at 348k/s), so 20 M pairs is roughly **6 minutes** of actual CPU. The machine
had 0.8–3 GB free while holding the full frames plus a whole 52 M-row group read in one go. The time
went to the page file, not the processor. **The earlier recommendation to parallelise the feature
stage was withdrawn on this evidence.**

The fixes were all about peak memory, in `run_pipeline.py`:

- drop the raw text columns once normalisation has produced the derived ones;
- drop `full_n` after blocking, which is the last stage that needs it;
- stream every pair parquet in 1 M-row batches instead of reading a group whole;
- preallocate the training matrix rather than growing it;
- derive blocking recall statistics from the labels instead of building per-entity Python sets.

**A full run needs ≥ 32 GB RAM** (64 GB recommended), 8+ cores and ~50 GB free disk. The run prints a
loud warning below 30 GB. `docs/RUN_FINAL.md` is the runbook.

**Resumability.** Each finished stage — blocking, models, out-of-fold prediction, CV, test blocking,
test prediction — is saved with a signature of both the code and the data that produced it. Re-running
the same command resumes from the last completed stage, losing at most the stage that was running, and
a changed input or a changed source file invalidates the affected stage rather than silently reusing a
stale cache. `--fresh` forces a full recompute. On a run measured in double-digit hours against a hard
deadline, this is what makes a crash survivable rather than fatal.

**Verification is part of the run, not a manual step afterwards.** The pipeline checks its own output
and then shells out to the organisers' `validate_submission.py` with `--check-ids`, so a malformed
submission is caught by the run that produced it. `scripts/make_handoff.py` refuses to build the
handoff bundle at all unless the run finished and every check passed.

### 4.6 Honest limitations

1. **One-to-one resolves within a fold during CV, but across all entities at test time.** The
   assignment step in `decide.py` de-duplicates candidates within whatever frame it is given. In
   cross-validation that frame is one fold, roughly a fifth of the entities, so it resolves less
   competition than it will at test time, when every entity competes at once. The direction of the
   bias is small but real, and the CV number carries this approximation. We have deliberately **not**
   "fixed" it by scoring test-like competition into folds, because doing so would change the meaning
   of every CV number already recorded.
2. **One full-scale run stands behind every number here, not several.** The figures come from a
   single 19.78-hour run (`run-final-v1`). There is no seed-to-seed repetition at full scale, so the
   run-to-run variability of the reported CV is unmeasured.
3. **Early stopping uses the fold it is scored on.** Each fold model early-stops on the same held-out
   fold whose out-of-fold probabilities it produces. The number of boosting rounds is therefore
   mildly optimistic with respect to that fold. The effect on a 2,000-tree ceiling with 100-round
   patience is expected to be small, but it is not zero and it is not measured.
4. **France has no direct measurement.** `--loco` between US and India is a proxy. France differs
   from both in language, address grammar and legal forms, and a US↔India transfer result may over-
   or under-state French performance in either direction.
5. **The recall ceiling caps everything, and India is the weak country.** Entity full cover is the
   share of entities whose *entire* true set survives blocking; an entity only partly covered cannot
   reach F0.5 = 1.0 however good the matcher is. After the cap fix this is 0.955 on US but only
   0.852 on India, so the largest remaining headroom in the whole system is India's candidate
   generation — not the model, and not France.
6. **The improved blocking configuration has not itself been re-run at full scale.** The
   `cos_full + cos_addr` / cap-80 numbers in §2.5 come from the full-haystack diagnostic
   (10,000 Source 1 entities against the complete Source 2/3 side), which makes block sizes and cap
   competition realistic but is still not a whole-corpus run. The 0.8851 / 0.7665 figures are from the
   full run of the *old* configuration.
7. **A single global threshold may be the wrong shape.** An entity with forty candidates faces more
   chances to err than one with two, so the best threshold may depend on candidate count. This is
   under investigation; nothing is measured yet.
8. **Two changes that are in the shipped pipeline were never A/B-measured at full scale.** The
   `n`/`s`/`e`/`w` → north/south/east/west address normalisation merged in PR #12 was adopted on
   reasoning, not on a measured before/after at scale; and the learning rate was chosen, not tuned
   against alternatives. Both are plausible and neither is validated, which is stated here rather
   than implied by silence.
9. **Early figures in this document were produced before the blocking determinism fix.** Candidate
   generation used to depend on Python's per-process string-hash order (§1.4); any number in this
   document measured before that fix carries roughly 0.16 % of run-to-run candidate churn beneath
   it. The fix is in; the older measurements were not redone.

### 4.7 What we deliberately did not do

- **No external data of any kind** — no geocoding, no business registries, no commercial
  entity-resolution services, no scraped or downloaded reference lists, no pretrained embedding
  fitted on anything but the provided files.
- **No per-country rules.** Every rule applies to every row.
- **No second copy of the metric.** `src/metric.py` is the single implementation, unit-tested against
  the official worked example.
- **No threshold tuned on the rows it scores** in any reported number.

### 4.8 Where the remaining loss actually is

This is the most useful thing the full run told us, and it contradicts what we assumed early on.

The candidate set the matcher scored has an oracle macro-F0.5 of about **0.989** — that is what a
*perfect* matcher would score on it. The run achieved **0.9491**. So roughly **0.040 of the score is
lost between what candidate generation makes reachable and what the matcher actually finds.**

That loss is **discrimination, not thresholding**:

- **348,000 true pairs are scored below the chosen threshold.** They are present in the candidate
  set and the matcher simply does not rank them highly.
- **About 88,000 entities have no candidate scoring above 0.15 at all.** For those entities no
  decision rule can help, because there is nothing to promote.

We measured this rather than assuming it. Alternative decision rules were run on the real
out-of-fold predictions with the same cross-fitted protocol (`scripts/redecide.py`):

| Rule | Cross-fitted CV | Folds improved |
|---|---|---|
| Global threshold (shipped) | 0.94910 | — |
| Commit the best candidate for entities left empty | 0.94834 | 0 / 5 |
| Relative rule, `alpha` = 0.7, `t` = 0.6 | 0.94912 | 3 / 5 |

Both were rejected: one is worse, the other is +0.00002, which is far inside noise. Notably,
committing the best remaining candidate for empty entities made things **worse** even with floors as
low as 0.05 — which says those low-probability candidates are predominantly wrong, and that the
88,000 abandoned entities are not recoverable by being braver.

**The honest conclusion: the decision layer is not where this competition is won, and we have the
measurement to say so.** The remaining headroom is in the matcher's ability to rank true pairs
highly (pair recall 0.9018) and in candidate generation. A later blocking revision (v2, asymmetric
address keys, cap 100) lifts the oracle ceiling from ~0.989 to ~0.990, but a higher ceiling only
helps a matcher able to reach it.

---

### 4.9 Two further results, added after the run above

**A control experiment that falsified our own hypothesis.** Section 4.8 concludes the loss is
discrimination rather than candidate generation. We tested that directly with a second full-scale
run (`run-final-v2b`, completed 27 Sep 16:45): asymmetric address keys that widened candidate
generation until the blocking oracle rose measurably — India 0.979 to 0.9864 — producing 40% more
candidates per entity (98.76 against 70.56).

**Cross-fitted CV moved from 0.9491 to 0.9489. It did not move.**

More candidates genuinely reached the matcher and it still did not score them above threshold. We
kept the smaller candidate set, which is also the better answer under the organisers' candidate-set
criterion. This is the strongest evidence in the document for section 4.8's conclusion, because it
is the experiment that could have overturned it.

**A structural blind spot we can name.** Inspecting a 24-entity sample by hand found the matcher
scoring **0.000** on pairs such as `Creative Global Limited` against its Gujarati rendering, and
likewise for Devanagari and Bengali. Every string feature in section 3.1 is a character n-gram or an
edit distance, and an Indic-script string shares **no characters** with its Latin form, so all of
them read approximately zero. Transliteration is invisible to this feature set by construction, not
by accident.

On that sample a human reading the raw text scored 0.9465 against the pipeline's 0.9214 — winning on
recall (0.974 vs 0.922) and *losing* on precision (0.938 vs 0.973). The gap was not better judgement;
it was reading scripts the features cannot see. A transliteration-aware feature — phonetic keying, or
a script-normalising map built only from the provided files — is the single highest-value extension
of this work.

## 5. Reproducing this submission

See `code/business_entity_resolution/README.md` in this package for the exact commands from a fresh
machine: Python 3.12, `pip install -r requirements.txt`, where the data goes, the pipeline command,
the official validator command, and where both TSVs are written.
