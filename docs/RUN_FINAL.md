# RUN_FINAL — the one full-scale run (for Heeda)

**Your only job:** run one command on a big enough machine, then send Abhinav one zip.
You do **not** upload anything to the Unstop portal. You do **not** change any code.
If anything fails: stop, and send `work/final_run.log` to Abhinav. Do not improvise.

---

## 1. Machine — check this first

| | Minimum | Recommended |
|---|---|---|
| RAM | **32 GB** | 64 GB |
| CPU cores | 8 | 8–16 |
| Free disk | 50 GB | 100 GB |
| OS | Windows 10/11 or Linux (Ubuntu 24.04 ships Python 3.12) | |
| Python | **3.12 exactly** | |

**Do not run this on a 16 GB laptop.** On 25 Sep a 16 GB laptop spent 6.5 hours swapping
on a step that takes minutes of CPU. The run prints a loud warning below 30 GB.

If your own machine is too small, use a cloud VM (e.g. 8 vCPU / 64 GB, Ubuntu 24.04).
That is rented compute for our own code and the organisers' data — not external data.
Keep the data private to the VM, and delete the VM afterwards.

## 2. One-time setup

1. Get the data from Abhinav: `student_resource.zip`. Unzip it **outside OneDrive**
   (e.g. `D:\amlc\` or `~/amlc/`). You need the folder that contains `train/` and `test/`,
   e.g. `D:\amlc\student_resource\dataset`. Keep `utils/` next to `dataset/` — the run uses
   the organisers' validator from there.
2. Get the code at the exact release tag (not whatever `main` is later):
   ```
   git clone https://github.com/abhinav0singh/amazon-ml-challenge-2026.git
   cd amazon-ml-challenge-2026
   git checkout run-final-v1
   ```
   Do not clone into OneDrive either.

## 3. Run it

**Windows (PowerShell):**
```
powershell -ExecutionPolicy Bypass -File scripts\run_final.ps1 -Data D:\amlc\student_resource\dataset
```

**Linux / cloud VM** (inside `tmux` so a dropped SSH session does not kill it):
```
tmux new -s run
bash scripts/run_final.sh ~/amlc/student_resource/dataset
```
(Detach with `Ctrl-b d`, re-attach with `tmux attach -t run`.)

The first run creates `.venv`, installs pinned packages, runs the self-tests, then the
pipeline. Keep the machine awake and plugged in (Windows: set sleep to *Never*).

## 4. If it stops — just run the same command again

The pipeline saves each finished stage (blocking → models → out-of-fold → CV → test
blocking → test predictions) with a signature of the code and data. Re-running the same
command **resumes** from the last finished stage — you lose at most the stage that was
running. Never delete `work/` between attempts. Never pass `--fresh` unless Abhinav asks.

## 5. How long it takes

Estimated from rates measured on 26 Sep (not a timed full run — there has not been one with this code):

| Stage | 8 cores / 32 GB (serial blocking) | 16 cores / 64 GB (parallel blocking) |
|---|---|---|
| normalise train + test (first run only) | ~0.6 h | ~0.6 h |
| train blocking (US group is the long pole) | ~4.5 h | ~3 h |
| training sample + 5 fold models | ~2.5 h | ~1.5 h |
| out-of-fold prediction + CV + LOCO | ~2.5 h | ~2 h |
| **→ CV line appears after** | **~10 h** | **~7 h** |
| test blocking | ~3.5 h | ~1.5 h |
| test prediction (5 models averaged) + checks | ~4 h | ~2.5 h |
| **total** | **~17–19 h** | **~11–12 h** |

**Start as early as possible.** Upload deadline is 27 Sep, before 22:00 IST. Use the biggest
machine you can get — it is the difference between finishing tonight and finishing tomorrow
morning.

## 6. What you should see (progress markers in the log)

```
pre-flight: {'ram_total_gb': ..., 'cpus': ...}
blocking train ...                         <- the longest part
blocking: {'total_pairs': ..., 'pair_recall_ceiling': ..., 'entity_full_cover': ...}
assembling training sample ...
fold 0: best_iter=... ... fold 4: best_iter=...
out-of-fold prediction ...
CV macro-F0.5 (cross-fitted) = 0.9xxx      <- report this number to Abhinav immediately
LOCO: hold out 'india' ... / 'us' ...
CV report written -> work/report.json
blocking test ...
test prediction ...
output check PASSED
official validator exit code 0 (PASS)
DONE.
handoff ready: handoff/handoff_<git>_<time>.zip
ALL DONE.
```

**As soon as the `CV macro-F0.5 (cross-fitted)` line appears, message Abhinav that number**
— the test half still takes hours, and he needs the CV to plan.

## 7. Hand off

Send Abhinav the single zip from `handoff/` (Google Drive link is fine). It contains
`matching_results.tsv`, `candidate_pairs.tsv`, `RUN_SUMMARY.txt`, `report.json`, `final_run.log`.
`make_handoff.py` refuses to build the zip unless the run finished and every check passed —
if it refuses, send the log instead.

## 8. For Abhinav, on receipt (before uploading)

1. Unzip; compare `sha256` of both TSVs with `RUN_SUMMARY.txt`
   (`certutil -hashfile matching_results.tsv SHA256`).
2. Run the official validator locally:
   `.venv\Scripts\python C:\amlc\student_resource\utils\validate_submission.py --matching matching_results.tsv --candidate candidate_pairs.tsv --test-dir C:\amlc\student_resource\dataset\test --check-ids`
3. Tag + archive per AGENTS.md §7: `git tag sub-D2-N`, copy the files to `submissions/sub-D2-N/`.
4. Upload **only `matching_results.tsv`** to the portal — per the organisers' README it is
   the only file the leaderboard scores. `candidate_pairs.tsv` (~1.5 GB at cap 80) goes
   into the final submission zip, not the portal (Anshika's packaging issue).
