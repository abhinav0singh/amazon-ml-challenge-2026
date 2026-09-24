# Amazon ML Challenge 2026 — Team Repo

72-hour ML hackathon. Team of 4: Team Leader, Data Analyst, ML Engineer, 
Monitor. This repo is our single source of truth — if it's not committed 
here, it doesn't count as done.

## Start here (everyone, before touching any code)

1. Read `prompts/master_prompt.md` and paste it into a fresh AI chat.
2. It'll ask which role you are and for `docs/problem_formulation.md` — 
   have that ready (Team Leader fills it in first, right after the problem 
   drops).
3. Once your AI session plays back a correct understanding of the metric, 
   CV strategy, and your role, paste in your role prompt from `prompts/` 
   and start working.

Don't skip the master prompt, even if you're in a hurry — it's one extra 
message and it's what stops four independent AI sessions from quietly 
working off four different assumptions.

## Repo structure

```
repo/
├── README.md                    ← you are here
├── requirements.txt              pinned deps — install with pip install -r requirements.txt
├── src/
│   ├── cv_folds.py               locked S1-entity folds -> work/folds.csv
│   ├── pair_features.py / decide.py / run_pipeline.py / make_empty_submission.py
│   ├── metric.py / data_io.py / normalize.py / blocking.py
├── docs/
│   ├── problem_formulation.md    filled by Team Leader, hour 0 — locked after that
│   └── approach_document.md      REQUIRED official submission — see below, not internal notes
├── prompts/                      one AI prompt per role, paste into your own chat session
│   ├── master_prompt.md          run this FIRST, always
│   ├── team_leader_prompt.md
│   ├── data_analyst_prompt.md
│   ├── ml_engineer_prompt.md
│   ├── monitor_prompt.md
│   └── validation_lead_prompt.md optional 5th role if we split it out
├── notebooks/                    EDA and scratch work
├── data/                         gitignored — see "Getting the data" below
├── models/                       saved model artifacts
└── submissions/                  every submission CSV, versioned — never overwrite one
```

## Non-negotiable rules

- **CV folds are locked once, in `src/cv_folds.py`, right after 
  `docs/problem_formulation.md` is filled.** Everyone imports the saved 
  fold indices from there. Nobody regenerates folds independently — 
  inconsistent folds across teammates silently breaks stacking later.
- **No leakage.** Any scaler, encoder, or imputer gets fit on training-fold 
  data only, never the full dataset before splitting. `src/features.py` has 
  fold-safe patterns built in — use them.
- **Trust local CV over the public leaderboard.** The public leaderboard 
  only scores a fraction of the test set. A private leaderboard, scored on 
  the full test set, decides real placement. Chasing small public-board 
  bumps at the cost of local CV is how teams get blindsided.
- **`docs/approach_document.md` is a required, scored submission**, not 
  internal notes — per the official rules, Top 10 selection depends on the 
  leaderboard result AND this document. Update it incrementally through the 
  72 hours, not in the last exhausted hour.
- **Every submission gets versioned** in `submissions/` with local CV score 
  noted — this is what lets us roll back if something breaks near the 
  deadline.
- **Fixed random seeds, pinned `requirements.txt`, from commit one.** We 
  need to be able to reproduce our best submission in the final 2 hours, 
  not scramble for it.

## Getting the data

```bash
pip install -r requirements.txt
```

Data is gitignored — download it from the competition portal into `data/` 
locally. Do not commit raw data to the repo.

## Workflow during the 72 hours

1. Team Leader fills `docs/problem_formulation.md` and locks it.
2. Team Leader or Validation Lead runs `src/cv_folds.py` once, commits the 
   saved fold indices.
3. Data Analyst builds features in `src/features.py`, respecting the locked 
   folds, hands off to ML Engineer via the repo (push + a short ping, not a 
   verbal explanation).
4. ML Engineer builds models, runs error analysis, ensembles if time 
   allows, saves submissions with version numbers.
5. Monitor keeps `STATUS.md` (create this once work starts) updated every 
   6-8 hours — who's doing what, what's blocked, are we on pace.
6. Everyone contributes to `docs/approach_document.md` as their piece lands — 
   validation approach after step 1-2, features after step 3, modeling and 
   error analysis after step 4.

## Git workflow

One branch per person, frequent small commits, merge to `main` fast (a 
self-merge + a chat ping beats waiting on formal review — we don't have 
review-cycle time to spare). Never let a long-lived branch sit unmerged for 
more than a few hours.

## Timeline (from the official guidelines PDF, 25 Sep 2026)

| When | What |
|---|---|
| **25 Sep 12:00 AM IST** | Challenge window opens; problem and data released |
| Each day | **Max 5 leaderboard uploads/day** (15 total). Only the Team Leader uploads, from one device |
| **27 Sep 11:59 PM IST** | Challenge closes. Plan the final uploads by **22:00 IST** and the zip by **23:00 IST** |
| After close | Top **100** announced (private + public LB + artefacts), then documents requested |

> The earlier version of this table (close "Sep 27 ~12:29 AM IST", "Top 50/Top 10") came from an old blog post and contradicts the official PDF. Confirm via the organisers' Google Form; until then we keep a submittable best version ready at all times.

## Run the pipeline (Business Entity Resolution)

```bash
pip install -r requirements.txt
# put the organisers' student_resource/ folder at data/ -> data/dataset/{train,test}, data/utils/
python src/metric.py                                   # metric self-test (0.714)
python src/make_empty_submission.py --data data/dataset --out output_empty   # upload D1-1
python src/run_pipeline.py --data data/dataset --out output --work work --loco
python data/utils/validate_submission.py --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv --test-dir data/dataset/test
```
Outputs: `output/matching_results.tsv` (upload this), `output/candidate_pairs.tsv`, and `work/report.json` (blocking recall, CV, LOCO; copy these numbers into `STATUS.md`).
Versioning: copy each uploaded `output/` into `submissions/<tag>/` and `git tag <tag>`. The old `submissions/*.csv` placeholders are empty and the wrong format (the portal needs `.tsv`), so delete them.

## Setup checklist (do this before the clock starts)

- [ ] Everyone has repo access
- [ ] Everyone has run `prompts/master_prompt.md` and confirmed their role
- [ ] AWS Builder Center profile created + Student Rewards verified (all 4 — 
  see the AWS prep blog's action plan)
- [ ] `pip install -r requirements.txt` runs clean for everyone
- [ ] Group chat channel confirmed (GitHub = code/record, chat = fast 
  coordination — don't use one for the other's job)
