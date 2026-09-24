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
│   ├── cv_folds.py               THE locked CV fold generator — run once, everyone imports the output
│   └── features.py               fold-safe feature engineering functions
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

## Timeline (verified against official rules, not the loose blog dates)

| When | What |
|---|---|
| **Sep 25, ~12:00 AM IST** | Hackathon starts (24 Sep 2:30 PM EDT) |
| Throughout 72 hrs | Build, and keep `docs/approach_document.md` updated as we go |
| **Sep 27, ~12:29 AM IST** | Submission closes (27 Sep 2:29 PM EDT) — code/notebook **and** the approach doc, zipped |
| Oct 2 | Top 50 announced |
| Oct 7 | Grand Finale — Top 10 present live to Amazon Scientists |

## Setup checklist (do this before the clock starts)

- [ ] Everyone has repo access
- [ ] Everyone has run `prompts/master_prompt.md` and confirmed their role
- [ ] AWS Builder Center profile created + Student Rewards verified (all 4 — 
  see the AWS prep blog's action plan)
- [ ] `pip install -r requirements.txt` runs clean for everyone
- [ ] Group chat channel confirmed (GitHub = code/record, chat = fast 
  coordination — don't use one for the other's job)
