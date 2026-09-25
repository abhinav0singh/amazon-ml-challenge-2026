# Business Entity Resolution Challenge — Methodology Document

**Team:** `<team_name>` · **Repository:** `abhinav0singh/amazon-ml-challenge-2026`
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
| Pairwise matcher | LightGBM gradient-boosted decision trees (`LGBMClassifier`, binary objective) | LightGBM | **MIT** | Upper bound **630,000** leaf values (5 folds × ≤2,000 trees × 63 leaves). Actual tree count is set by early stopping: **[TBD]** |
| Decision layer | Threshold + greedy one-to-one assignment (no learned parameters) | — | — | 1 scalar threshold `t` |

**No pretrained model of any kind is used.** Nothing is downloaded, and no embedding is fitted on
anything but the provided files. See §4.2 for the full library licence list and the open question it
raises.

### 0.4 Experiments

| ID | Change | CV macro-F0.5 (cross-fitted) | LOCO | Decision |
|---|---|---|---|---|
| E0 | Empty prediction for every entity (format check) | ≈ **0.0558** (= measured train singleton share) | — | baseline |
| E1 | Baseline pipeline (multi-key blocking, 28 features, LightGBM, OOF threshold, one-to-one) | **[TBD]** | **[TBD]** | pending |
| B0/B1 | Blocking: dense → sparse top-k over a whole country group | recall ceiling 0.9842–0.9986 on **sampled** runs (see §2.5) | — | rejected: cost exponent rose to 1.60, does not scale |
| B2 | Blocking: multi-key rare-token inverted index | recall ceiling 0.9788–0.9833 on **sampled** runs (see §2.5) | — | **current default**; cost exponent 0.91 (sub-linear) |

**No full-scale run of the pipeline has completed yet.** Every CV, LOCO, precision, recall and
runtime number for the full dataset is therefore **[TBD]** in this document. The blocking numbers in
§2.5 are from deliberately sub-sampled runs and are optimistic by construction — they are reported
as cost/ceiling measurements, never as CV.

### 0.5 Conclusion

The decisive properties of this problem are (a) the metric is a macro-average over entities, so an
entity with two candidates counts exactly as much as one with forty, and (b) precision is weighted
2×, so a false merge costs more than a miss. Both push the design toward a high-recall candidate
generator followed by a precision-oriented decision layer, rather than toward a single similarity
threshold. The largest remaining risks are blocking cost at full scale (2.2 M × 10.3 M records) and
generalisation to France, for which our only proxy is the leave-one-country-out measurement. Full
conclusions await the first full-scale run — **[TBD]**.

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
- `MAX_CANDS = 80` — after the keys are unioned, each Source 1 entity keeps at most 80 candidates,
  ranked by the **maximum** of the three cosines. Ranking on the maximum rather than on one view
  avoids discarding a candidate that only the address view liked, which is exactly the
  renamed-business case the address view exists to catch.

`candidate_pairs.tsv` is written from this final, capped candidate frame — it is the exact set the
matcher runs inference over, as the problem statement requires.

An earlier variant using only exact whole-name keys (no rare tokens) scored recall 0.8785 and entity
cover 0.7395 at the 2,000-entity sample — exact keys are too brittle for this noise, which is why
rare-token indexing is the core of the design.

### 2.5 Measured blocking results

> **All rows below are from deliberately sub-sampled runs** (`src/sweep_blocking.py` and
> `run_pipeline.py --sample`). Sampling preserves the Source 2/3-records-per-Source 1-entity density
> but shrinks the haystack, so these ceilings are **optimistic**. **No full-scale blocking run has
> completed.** Full-scale recall ceiling, entity cover, candidates per entity, reduction ratio and
> runtime are all **[TBD]**.

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

**Reduction ratio: [TBD].** It is reported by `blocking_report` as candidate volume relative to the
full cross-product, but only a full-scale run gives the figure that belongs in this document.

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
limit. The realised tree count is set by early stopping and is **[TBD]**.

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
the fifth, and its output is the only score quoted as CV. The chosen threshold is **[TBD]**.

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
   set. **LOCO result: [TBD].**

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

### 4.5 Honest limitations

1. **One-to-one resolves within a fold during CV, but across all entities at test time.** The
   assignment step in `decide.py` de-duplicates candidates within whatever frame it is given. In
   cross-validation that frame is one fold, roughly a fifth of the entities, so it resolves less
   competition than it will at test time, when every entity competes at once. The direction of the
   bias is small but real, and the CV number carries this approximation. We have deliberately **not**
   "fixed" it by scoring test-like competition into folds, because doing so would change the meaning
   of every CV number already recorded.
2. **No full-scale run has completed.** Every CV, LOCO, precision, recall and runtime figure for the
   full dataset is [TBD]. The blocking figures that do exist come from sub-sampled runs whose smaller
   haystack makes both recall ceiling and precision optimistic.
3. **Early stopping uses the fold it is scored on.** Each fold model early-stops on the same held-out
   fold whose out-of-fold probabilities it produces. The number of boosting rounds is therefore
   mildly optimistic with respect to that fold. The effect on a 2,000-tree ceiling with 100-round
   patience is expected to be small, but it is not zero and it is not measured.
4. **France has no direct measurement.** `--loco` between US and India is a proxy. France differs
   from both in language, address grammar and legal forms, and a US↔India transfer result may over-
   or under-state French performance in either direction.
5. **The recall ceiling caps everything.** Entity full cover — the share of entities whose *entire*
   true set survives blocking — was 0.9546 at the 40,000-entity sample. Any entity whose true set is
   only partly covered cannot reach F0.5 = 1.0 no matter how good the matcher is.
6. **Blocking cost at full scale is extrapolated, not measured.** The sub-linear cost exponent of
   0.91 is measured across 2k → 40k; the projected full-scale runtime that follows from it is an
   extrapolation and is marked as such wherever it appears.
7. **A single global threshold may be the wrong shape.** An entity with forty candidates faces more
   chances to err than one with two, so the best threshold may depend on candidate count. This is
   under investigation; nothing is measured yet.

### 4.6 What we deliberately did not do

- **No external data of any kind** — no geocoding, no business registries, no commercial
  entity-resolution services, no scraped or downloaded reference lists, no pretrained embedding
  fitted on anything but the provided files.
- **No per-country rules.** Every rule applies to every row.
- **No second copy of the metric.** `src/metric.py` is the single implementation, unit-tested against
  the official worked example.
- **No threshold tuned on the rows it scores** in any reported number.

---

## 5. Reproducing this submission

See `code/business_entity_resolution/README.md` in this package for the exact commands from a fresh
machine: Python 3.12, `pip install -r requirements.txt`, where the data goes, the pipeline command,
the official validator command, and where both TSVs are written.
