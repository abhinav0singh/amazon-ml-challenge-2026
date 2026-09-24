# Compliance Audit — Amazon ML Challenge 2026

**Audit date:** 25 Sep 2026 (Day 1) · **Auditor:** AI session working from `AGENTS.md` + the official documents
**Repo state audited:** `main` @ `6cdf368`, clean working tree
**Re-run this audit before every leaderboard upload and again before the final zip.**

## VERDICT

**FIX THESE FIRST:**

1. **The pipeline has never been executed.** No Python environment on this machine had `numpy`, `pandas`, `scikit-learn`, `rapidfuzz`, `lightgbm` or `pyarrow` installed before this session. `src/run_pipeline.py` cannot have been run. Every claim of a "working, tested baseline" is unverified.
2. **The pipeline is algorithmically infeasible at the real data scale** (2.2 M train S1 × 10.3 M S2/S3 records; 1.7 M test S1 × 10.0 M). Blocking and pair-feature construction as written would take weeks and exhaust RAM. This is the single biggest risk to the whole run, and it is not a compliance problem — it is an "we have no submission at all" problem.
3. **`AGENTS.md` §2 consequence 2 is factually wrong for this dataset** and is steering strategy in the wrong direction. Singletons are **5.58 %** of train S1, not "a large share". The all-empty submission scores ≈ **0.0558**, not something near 0.5.
4. **Housekeeping:** root `.gitignore` does not exclude `work/` or `output/`; `submissions/submission_v{1,2}.csv` are 0-byte files with the wrong extension; the pinned `requirements.txt` cannot install on the Python 3.14 that is first on `PATH`.

**No academic-integrity violation was found.** Nothing in `src/` touches the network, an API, or any file outside the organisers' dataset. On the rules that can cause disqualification, the repo is clean.

---

## 0. Sources used for this audit

| Document | Location | Status |
|---|---|---|
| Guidelines & key instructions | `~/Downloads/6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf` | **Read this session, in full** |
| Problem statement (authoritative text) | `C:\amlc\student_resource\README.md` (shipped with the dataset) | **Read this session, in full** |
| Problem statement PDF | `~/Downloads/ml unstop.pdf` | **Not read.** 8 pages, image-only scan, no extractable text layer, and no PDF rasteriser on this machine. Its content is assumed identical to the dataset `README.md`; if anyone can read it, diff it against §1 below. |
| Official validator | `C:\amlc\student_resource\utils\validate_submission.py` | Read this session |
| Team contract | `AGENTS.md` | Read; conflicts flagged in §4 |

`AGENTS.md` is our own convention layer. Where it disagrees with the two official documents, the official documents win — §4 lists every disagreement found.

---

## 1. Rule compliance audit

Every row was checked against the actual file contents this session, not against memory or `AGENTS.md`'s claims.

### 1.1 Academic integrity — the disqualification rules

| # | Rule (official wording) | What the code actually does | Verdict |
|---|---|---|---|
| A1 | No commercial entity-resolution APIs or services | `grep` over all of `src/` for `requests`, `urllib`, `http`, `socket`, `boto3`, `openai`, `anthropic`, `api_key`, `os.environ` → **zero hits**. No network-capable import anywhere. | **PASS** |
| A2 | No government business-registry lookups | No registry file, no lookup table of business identities, no ID cross-reference of any kind. | **PASS** |
| A3 | No geocoding APIs to normalise addresses | `normalize.py` is pure `re` + `unicodedata` (stdlib). No geocoder, no gazetteer, no place-name list. Postal codes are *extracted* from the record's own address text (`_POSTAL = \b(\d{5,6})\b`), never looked up. | **PASS** |
| A4 | No external data augmentation from internet sources | Every file read by the pipeline is under the `--data` folder (`data_io.py` lines 25–34) or is our own derived artefact (`work/folds.csv`, `work/*.parquet`). `requirements.txt` pins only computation libraries. No dataset, no embedding file, no checkpoint is downloaded. | **PASS** |
| A5 | Hand-written abbreviation dictionaries — allowed or not? | See the reasoned answer in §1.2. | **PASS (with reasoning; confirm via the query form as a belt-and-braces measure)** |
| A6 | Final model MIT / Apache-2.0, ≤ 8 B parameters | Currently the only model is **LightGBM** (MIT, gradient-boosted trees, no "parameters" in the LLM sense — comfortably inside any reading of the limit). `scikit-learn` TF-IDF is BSD-3 (a library, not a "model"). **No pretrained model is used anywhere in the repo today.** `requirements.txt` has `sentence-transformers` and `torch` **commented out**, and `STATUS.md` §1 lists "can we use MiniLM/e5" as an open question. | **PASS today · MUST BE LOCKED BEFORE USE** (see §1.3) |
| A7 | Reviewed by humans, code and methodology | `src/` is commented, every function has a docstring, and the pipeline is deterministic (`SEED = 42`). The audit trail requirement is met by design. | **PASS** |
| A8 | No cheating / plagiarism / multi-ID registration | Out of scope for code review; team behaviour. Only P1 uploads, from one device, per `AGENTS.md`. | **PASS (by policy, not verifiable here)** |

### 1.2 Are the normalisation dictionaries "external data"? — the reasoning

`normalize.py` contains three hand-written dictionaries: `LEGAL_TOKENS` (28 company-form tokens: pvt, ltd, llc, gmbh, sarl, sas…), `NAME_MAP` (18 name abbreviations: intl→international, mfg→manufacturing…) and `ADDR_MAP` (36 street abbreviations: rd→road, blvd→boulevard, ngr→nagar…).

**Conclusion: these are allowed.** The reasoning, which should also go into the methodology document so a reviewer reaches the same conclusion without having to ask:

1. **What the rule prohibits is a *lookup*** — going outside the provided data to learn a fact *about a specific entity in the dataset*. The listed examples are all of that shape: an ER service tells you "these two records are the same business"; a registry tells you "this business exists at this address"; a geocoder tells you "this address is at these coordinates". Each returns a fact about a record that was not in the record.
2. **These dictionaries return no facts about any record.** `rd → road` is a statement about the English language, applied blindly to every row. It cannot distinguish one business from another, and it adds no information that was not already in the string being rewritten. Deleting the dictionary and writing 36 `if` statements would be identical in effect.
3. **They are not data in the operative sense** — they were typed by a person, not fetched, scraped, downloaded, or derived from any corpus. There is no source to cite because there is no source.
4. **The distinguishing test:** could this dictionary, applied to a record, tell us anything about *that particular business* that the record did not already say? No. A geocoding API can. That is the line, and the dictionaries sit clearly on the safe side of it.
5. **Precedent inside the rules themselves:** the problem statement's own "Tips for Success" says to "pay attention to country specific address patterns" and lists the exact abbreviation pairs (Rd vs. Road, Pvt vs. Private) as the noise to handle. The organisers are describing this technique as the intended solution.

**Residual caution:** keep the dictionaries *linguistic*. The moment a dictionary contains a list of real place names, real company names, real postal-code ranges, or anything one could only produce by consulting an outside source, it changes category and becomes a violation. A French street-word list (rue, avenue, boulevard, chemin) is still linguistic and fine. A list of French *communes* would not be — that is a gazetteer, i.e. external data. **This boundary must be stated in the methodology document and enforced in review.**

### 1.3 Model licence — action required before Day 2

Nothing is violated today, but the team is one commit away from a problem. `STATUS.md` §1 and the commented-out `requirements.txt` lines show embeddings are being considered. **Before any pretrained model is imported:**

- Record the exact model ID, licence and parameter count in `STATUS.md` §9.
- MIT / Apache-2.0 only. (For reference: the `sentence-transformers` *library* is Apache-2.0, but a **model's** licence is separate and must be checked per model — several popular multilingual encoders are **not** Apache-2.0.)
- Any model is downloaded from its hub **as a model artefact**, which is the ordinary meaning of "using a pretrained model" and is contemplated by rule 5 of the Constraints section. It is **not** the same as looking up entity data — but note the model must be **pretrained on general text, never fitted on anything outside the provided files**, and the download must happen before/outside the scored pipeline so the pipeline itself never touches the network.
- **`STATUS.md` §1 already flags this as an open question for the organisers.** It is still unanswered. Until it is answered, prefer the TF-IDF path, which is unambiguous.

### 1.4 Output format rules

| # | Rule | What the code actually does | Verdict |
|---|---|---|---|
| F1 | Tab-separated, exact column names | `data_io.write_id_lists` writes `source1_entity_id\t<col>\n` by hand, no pandas quoting, `newline="\n"`, UTF-8. Column names passed by `run_pipeline.py` lines 153–154 are `candidate_entity_ids` and `matched_entity_ids` — both correct. | **PASS (by inspection)** |
| F2 | Exactly one row per Source-1 test entity, France included | `write_id_lists` iterates `t_ids = t1["entity_id"].tolist()`, i.e. every row of `test_source1.tsv`, in file order. No filtering by country anywhere. | **PASS (by inspection)** |
| F3 | Empty `matched_entity_ids` for singletons | `f.write(f"{s}\t{','.join(ids)}\n")` with `ids = []` → a bare trailing tab. Correct. | **PASS (by inspection)** |
| F4 | No duplicate IDs within a list | `sorted(set(...))` per row. | **PASS (by inspection)** |
| F5 | No duplicate `source1_entity_id` rows | One row per element of `t_ids`. Depends on `test_source1.tsv` having unique IDs — not yet verified on the real file. | **PASS (pending run)** |
| F6 | Only S2/S3 IDs that exist in the test set; no self-matches to S1 | Candidates come only from `t23` (= test S2 + test S3 concatenated), so S1 IDs can never enter the list and non-existent IDs cannot be invented. | **PASS (by inspection)** |
| F7 | Every ID in `matching_results.tsv` also in `candidate_pairs.tsv` | `tpred` is produced by thresholding `tpairs`, which is built from `tcand`; `candidate_pairs.tsv` is written from the same `tcand`. Matches are a subset by construction. | **PASS (by inspection)** |
| F8 | `candidate_pairs.tsv` must be the *final* candidate list fed to the model | It is: `tcand` is exactly what `build_pair_features` consumes and the model scores. No later filtering stage exists. | **PASS** |
| F9 | Validator run before submitting | `run_pipeline.py` lines 165–172 shell out to the official validator automatically. **But it does not pass `--check-ids`**, so the ID-existence check is silently skipped (it is off by default). | **PARTIAL — fix: add `--check-ids`** |

**Every F-row above is marked "by inspection" because there is no `output/` directory in the repo and the pipeline has never produced one.** No validator run has ever happened. Rows F1–F8 will be re-verified against real files as soon as the first output exists; **this audit must be re-run at that point before anything is uploaded.**

### 1.5 Competition mechanics

| # | Rule | Reality | Verdict |
|---|---|---|---|
| M1 | 5 uploads/day, 15 total | `STATUS.md` §6 has a 15-slot ledger. Zero used so far. | **PASS** |
| M2 | Maintain version history of submissions | `AGENTS.md` §7 requires a git tag + a copy in `submissions/<tag>/` per upload. The mechanism exists; nothing has used it yet. `submissions/` currently holds two **0-byte `.csv` placeholders** from an earlier scaffold, which contradict the required `.tsv` format and should be deleted before a reviewer sees them. | **PARTIAL — delete the placeholders** |
| M3 | Desktop/laptop only, one login per participant | Policy: only P1 uploads, from one machine. | **PASS (by policy)** |
| M4 | Both leaderboards used for evaluation/shortlisting; final decision from private | Understood; strategy in §3 follows it. **Note the official wording is "Evaluation and shortlisting will be based on performance across *both* leaderboards", while the problem statement says "The final decision will be based on the private leaderboard."** These are not identical. See §4. | **AMBIGUOUS — see §4** |

---

## 2. Submission package readiness

Required layout (from the problem statement, *Final Submission Package*):

| Required path | Exists? | State |
|---|---|---|
| `output/matching_results.tsv` | ❌ | Never generated. |
| `output/candidate_pairs.tsv` | ❌ | Never generated. |
| `code/business_entity_resolution/src/` | ❌ | Code lives at `src/`; the packaging step that copies it into this layout does not exist. |
| `code/business_entity_resolution/README.md` | ❌ | Repo `README.md` exists but is not a reproduce-end-to-end guide. |
| `code/business_entity_resolution/requirements.txt` | ⚠️ | `requirements.txt` exists and is pinned, but **includes packages the pipeline does not import** (xgboost, catboost, optuna, imbalanced-learn, jupyter, matplotlib, seaborn) and **cannot install on Python 3.14**, which is the `python` first on `PATH` on this machine. A reviewer who runs `pip install -r requirements.txt` on a modern Python gets a build failure. Must be trimmed to what is actually imported and pinned to a stated Python version. |
| `Documentation_template.md` (filled) | ❌ | The blank template is at `C:\amlc\student_resource\Documentation_template.md`; it has not been copied into the repo or filled in. `STATUS.md` §8 (which feeds it) is entirely empty. |
| `<team_name>_submission.zip` | ❌ | No packaging script exists. |

**Also required by the guidelines PDF, and not yet started:** a 1–2 page document covering methodology, ML models used, experiments and conclusion. `docs/approach_document.md` exists — verify whether it is a stub.

**Readiness: 0 of 7 deliverables complete.** None of this is hard, but it is all serial work at the end, and "we ran out of time to write the doc" is a recurring way teams lose a shortlisting they had earned on score. Assign it to P4 on Day 1, not Day 3.

---

## 3. What the data actually says — and why it changes the plan

These numbers were measured this session directly from the dataset files (`awk` over the raw TSVs). They are counts, not model output, so they are reliable without running the pipeline.

| Fact | Value | Why it matters |
|---|---|---|
| Train S1 entities | 2,206,822 | |
| Train S2 + S3 records | 10,320,219 | |
| Test S1 entities | 1,732,545 | Every one needs a row. |
| Test S2 + S3 records | 9,969,589 | |
| Train countries | US 1,323,633 · India 883,188 | No France, as stated. |
| **Test countries (S1)** | **US 663,106 · India 809,986 · France 259,452** | **France is 15.0 % of test S1, not a third.** LOCO still matters, but it is worth 15 % of the score, not 33 %. |
| Test S2/S3 carry country labels incl. France | S2: FR 703,378 / IN 2,312,565 / US 1,871,330 · S3: FR 731,615 / IN 2,405,000 / US 1,945,701 | **Important:** country-grouped blocking works on test. `blocking.py`'s "fall back to all S2/S3" branch will never fire, which would have been catastrophic at this scale. |
| **Train singletons** | **123,247 / 2,206,822 = 5.58 %** | **The all-empty submission scores ≈ 0.0558.** |
| Mean true matches per non-singleton S1 | 3.67 | Most entities have several matches. |
| Total true pairs in train | 7,638,365 | |
| **S2/S3 IDs matched to more than one S1** | **0 (exactly zero, across all 7.6 M pairs)** | The one-to-one assumption is not approximately true — it is exactly true in training. `decide.py`'s `one_to_one` will switch on. This is a strong, free precision constraint. |

### 3.1 The two findings that change strategy

**(a) `AGENTS.md` §2 consequence 2 is wrong, and it is pointing us the wrong way.**
The contract says: *"Singletons are scored… A large share of the total score is just 'correctly say nothing'."* At 5.58 %, saying nothing everywhere earns 0.0558. **94.4 % of the available score requires finding matches.** Precision is still weighted 2× — that part of the contract is correct and still governs the threshold — but a strategy tuned around "when unsure, predict nothing" will leave most of the score on the table. The optimal threshold will be high, but not nearly as high as a 50 %-singleton world would demand. **Recommend: fix §2 by commit, with these counts cited.**

**(b) The pipeline cannot run at this scale. This is the emergency.**
Three concrete blockers in the current code, in order of severity:

1. **`blocking._topk_sparse` is a dense chunked matmul.** With `max_cells = 4e7` and the India group's 4.72 M S2/S3 records, `step = 4e7 / 4.72e6 = 8 rows per chunk`. India alone (809,986 S1 rows) is ~101,000 chunk iterations, each materialising an 8 × 4.72 M dense float32 array (151 MB) and running `argpartition` over it — and that is **one of three TF-IDF views**, on **one of three countries**, on **one of two splits**. Order of magnitude: weeks of wall-clock on 8 cores. It must be replaced with sparse top-k (an inverted-index / `sparse_dot_topn`-style approach, or NN-Descent over the TF-IDF vectors) that never densifies.
2. **`pair_features.build_pair_features` loops over pairs in Python.** At 35 candidates/S1 × 2.2 M S1 ≈ 77 M pairs, with 16 rapidfuzz calls each, this is ~10¹⁰ operations in a Python `for` loop — days. rapidfuzz's `process.cdist` with `workers=-1` does the same work in C across all cores; the loop must be rewritten to batch.
3. **`a = s1.set_index("entity_id").loc[cand["s1_id"]]` materialises 77 M rows × 7 object-dtype columns** — hundreds of gigabytes. This OOMs before it computes anything. Features must be computed per candidate block against integer-indexed numpy arrays, never by reindexing a string DataFrame to pair length.

**None of this is a criticism of the design** — the pipeline's structure (blocking → pair features → LightGBM → threshold on OOF → one-to-one) is exactly right, and the CV discipline in it is better than most teams will have. It was written against an assumed dataset one to two orders of magnitude smaller. **The architecture survives; three implementations must be replaced.**

### 3.2 Prioritised plan — Day 1 remaining → 27 Sep 23:59 IST

Ranked by (expected gain × likelihood) / hours. **Step 0 is not optional: every number below the line is currently unmeasured, and this document will not speculate about any of them.**

| # | Action | Owner | Hours | Why this rank |
|---|---|---|---|---|
| **0a** | **Finish the environment.** `.venv` on Python 3.12 (3.14 cannot install the pinned versions) — in progress this session. | P1 | 0.3 | Nothing else can start. |
| **0b** | **Ship the empty submission (D1-1) and validate it.** `make_empty_submission.py` is pure pandas I/O and runs at this scale. It proves the upload path, proves the format, and its public score tells us the **public** singleton share — which is a genuinely useful number we cannot otherwise get. Expected ≈ 0.056 if the public split matches train. | P1 | 0.5 | Cheapest possible information; the only submission worth spending on a non-model. |
| **0c** | **Run the pipeline on a 1 % subsample** (≈ 22 k S1, sampled S2/S3) to prove it works end to end and to get the first real `work/report.json`. | P1 | 1 | Converts "untested code" into "tested code" and gives the first honest blocking-recall and CV numbers. Do **not** quote these as final CV — they are a smoke test. |
| **1** | **Rewrite `_topk_sparse` as true sparse top-k.** No densification. Keep the same three-view union so the design and the docs stay valid. | P2 | 3–5 | **Highest leverage by a wide margin.** Without it there is no submission at all. |
| **2** | **Vectorise `build_pair_features`** with `rapidfuzz.process.cdist(workers=-1)` and integer-indexed numpy arrays; drop the pair-length DataFrame reindex. | P3 | 3–4 | Second hard blocker. Same feature set, same semantics — a pure speed change, so it needs no CV justification. |
| **3** | **Full-scale run → real `report.json`** → first model submission (D1-3 or D2-1). | P1 | 2–4 run | The first number anyone is allowed to quote as CV. |
| **4** | **Tune the threshold properly on OOF, and trust `cross_fitted_score`.** Given (a) above, expect the optimal `t` to be high but not extreme. Never report `tune()`'s own score as CV. | P4 | 1 | Free score; the decision layer is where F0.5 is won. |
| **5** | **Exploit the exact one-to-one constraint.** Zero S2/S3 IDs serve two S1 entities in 7.6 M training pairs. `decide.py` already does greedy assignment by probability; a proper assignment (or at least a mutual-best-match rule) is strictly better and cheap. | P4 | 2 | Pure precision gain, which F0.5 rewards double. |
| **6** | **France robustness (`--loco`).** Worth 15 % of the test set. Check that no feature degrades on the held-out country; the pipeline is already country-agnostic by construction, which is the right starting point. | P4 | 2 | Real but bounded; do it after there is a working full-scale model. |
| **7** | **Documentation + final zip.** Fill `Documentation_template.md` from `STATUS.md` §8 as work happens, not at the end. | P4 | 3 | Required for shortlisting. Zero score, total gate. |

**The single highest-leverage thing to fix first**, given F0.5's precision weighting and the singleton rule: **it is not a modelling choice at all — it is item 1, making blocking run.** Blocking sets the recall ceiling, and with 94.4 % of the score behind non-singleton entities, a recall ceiling we never measure is the difference between a competitive score and nothing. After that, the highest-leverage *modelling* decision is the threshold + one-to-one assignment in `decide.py` (items 4–5), because that is where precision — the doubly-weighted term — is actually set.

---

## 4. Open questions and conflicts

### 4.1 Conflicts between `AGENTS.md` and the official documents — resolve by commit

| # | `AGENTS.md` says | Official documents / measured data say | Recommendation |
|---|---|---|---|
| C1 | §2: "Singletons are scored… **a large share of the total score** is just 'correctly say nothing'." | Singletons are **5.58 %** of train S1. | **Fix §2.** The precision-2× point is right; the "large share" claim is wrong and is steering us toward an over-conservative threshold. |
| C2 | §8: "**Final rankings come from the private board.**" | The problem statement agrees ("The final decision will be based on the private leaderboard"), but the **guidelines PDF** says "Evaluation and shortlisting will be based on performance **across both leaderboards**." | Not a contradiction we can resolve ourselves. Behave as if private is what counts (it is the stricter, safer target) **and** ask via the form. Does not change our strategy either way. |
| C3 | §4: "Folds are **locked** in `work/folds.csv`." | `work/` is not committed, and the root `.gitignore` does not even mention it. Each teammate will generate their own copy on first run. | The generation is deterministic (sorted IDs, seed 42), so copies *should* match — but "should" is not "locked". **P1: after the first real run, commit the file with `git add -f work/folds.csv`** and update §4 and `docs/SETUP.md` to say so. |
| C4 | User brief (not `AGENTS.md`): "test includes France… a third of the test set". | France is **15.0 %** of test S1. | Correct the working assumption. LOCO is still worth doing; it is worth half what we thought. |
| C5 | §9 command: `--data <DATA>` where `<DATA>` holds `train/` and `test/`. | The extracted path is `C:\amlc\student_resource\dataset`, and `utils/` sits at `C:\amlc\student_resource\utils`. | The validator auto-discovery in `run_pipeline.py:165` derives `<DATA>/../utils` — correct for this layout. Update §9 and `SETUP.md` to show the real path, since the docs currently say `C:\amlc\dataset`. |

### 4.2 Questions for the organisers' Google Form

`STATUS.md` §1 already lists four questions, **all still unanswered**. Check whether the form query P1 sent has a reply before re-asking.

| Already in `STATUS.md` §1 | Status | Note |
|---|---|---|
| Which submission is used for the private LB: last, best, or selected? | ☐ unanswered | **Highest-value question on the list.** It determines whether the Day-3 endgame is "upload the best and stop" or "upload the best last". Neither official document states it. Until answered: **make the final upload the one with the best CV**, which is safe under every interpretation. |
| Pretrained open models (MiniLM / e5) allowed if MIT/Apache and ≤ 8 B? | ☐ unanswered | Constraint 5 says the final model must be MIT/Apache ≤ 8 B, which strongly implies pretrained models are expected. Still worth confirming, since it interacts with the external-data rule. |
| Hand-written normalisation dictionaries allowed? | ☐ unanswered | Our reasoned answer is **yes** (§1.2). Asking costs nothing and produces a citable answer for the methodology doc. |
| Public/private split size? | ☐ unanswered | Low stakes. Affects only how much we read into public-LB movement. |

**New questions to add:**

| New question | Why |
|---|---|
| Is "evaluation and shortlisting based on performance across both leaderboards" (guidelines) consistent with "final decision based on the private leaderboard" (problem statement)? | Conflict C2. |
| Does the 8 B parameter limit apply to the final scoring model only, or to anything used anywhere in the pipeline (e.g. an encoder used only for blocking)? | Determines whether an embedding model in the blocking stage is in scope. |

---

## 5. Re-audit checklist

Before **every** leaderboard upload:

- [ ] `python src/metric.py` prints `metric OK: example = 0.714`
- [ ] The official validator passes on the actual files being uploaded, **with `--check-ids`**
- [ ] Row count of `matching_results.tsv` = 1,732,546 (1,732,545 entities + header)
- [ ] The number quoted as "CV" came from `cross_fitted_score`, not `tune`
- [ ] Ledger row added to `STATUS.md` §6 **and** the git tag created, **before** uploading
- [ ] A copy of `output/` saved to `submissions/<tag>/`

Before the **final zip**:

- [ ] Everything above, plus every row of §2 of this document ticked
- [ ] `requirements.txt` trimmed to what is imported, and installable from clean on a stated Python version
- [ ] §1.2's dictionary reasoning reproduced in the methodology document
- [ ] Every model's licence and parameter count recorded
- [ ] This audit re-run and re-dated
