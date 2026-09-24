# One-time setup (every teammate, ~10 minutes)

Windows PowerShell. Mac/Linux: same commands, swap `\` for `/` and `python` for `python3`.

## Step 1 — clone and branch

```powershell
cd $HOME\Desktop
git clone https://github.com/abhinav0singh/amazon-ml-challenge-2026.git
cd amazon-ml-challenge-2026

git config user.name  "<Your Name>"
git config user.email "<your github email>"

# your own branch — pick the one for your role
git checkout -b <BRANCH>
```

| Role | `<BRANCH>` | You own |
|---|---|---|
| P1 Abhinav — Team Leader | `p1-lead` | metric, folds, pipeline runs, uploads, final zip |
| P2 — Data Analyst | `p2-blocking` | `src/normalize.py`, `src/blocking.py` |
| P3 — ML Engineer | `p3-features` | `src/pair_features.py`, model params |
| P4 — Decision & Docs | `p4-decide-docs` | `src/decide.py`, `docs/approach_document.md` |

## Step 2 — environment

```powershell
pip install -r requirements.txt
python src\metric.py          # must print: metric OK: example = 0.714
```

If it prints anything else, stop and tell P1. Every number the team reports depends on this file.

## Step 3 — data (keep it OUT of OneDrive and out of git)

Unzip the organisers' `student_resource` to a short local path, e.g. `C:\amlc`, so you have:

```
C:\amlc\dataset\train\train_source1.tsv   (…source2, source3, train_ground_truth)
C:\amlc\dataset\test\test_source1.tsv     (…source2, source3)
C:\amlc\utils\validate_submission.py
C:\amlc\Documentation_template.md
```

```powershell
$env:AMLC = "C:\amlc\dataset"     # set it once per terminal session
python src\run_pipeline.py --data $env:AMLC --out output --work work --loco
```

OneDrive will sync and lock a 1 GB folder mid-run. Do not keep the data on the Desktop.

## Step 4 — read the contract

Read `AGENTS.md` top to bottom. Paste it into your AI session before your first prompt of every work block.

## Step 5 — your first push (proves the whole loop works)

```powershell
# make any small real change, e.g. add your name to the owners table in STATUS.md
git add -A
git commit -m "p2: setup verified, claim blocking ownership"
git push -u origin <BRANCH>
```

Then on GitHub: open a PR to `main`, self-merge, ping the group chat. Do not let a branch sit unmerged for more than a few hours.

## Daily rhythm

```powershell
git checkout main; git pull            # start of every work block
git checkout <BRANCH>; git merge main  # take everyone else's work
# ... work, run, log to STATUS.md ...
git add -A; git commit -m "..."; git push
```

## Before every push

- [ ] `python src\metric.py` passes
- [ ] your change ran end to end at least once
- [ ] one row added to `STATUS.md` §5 (experiment log)
- [ ] one line added to `STATUS.md` §8 (doc notes)
- [ ] nothing under `data/`, `work/`, or `output/` is staged
