#!/usr/bin/env bash
# run_final.sh -- the ONE command for the final full-scale run (Linux / cloud VM).
#
#     bash scripts/run_final.sh /path/to/student_resource/dataset
#
# The argument is the folder that holds train/ and test/.
# Resumable: if it stops for ANY reason (crash, reboot, lost SSH session), run the
# exact same command again. Finished stages are reused; nothing is recomputed.
# Run it inside tmux or screen so a dropped SSH connection does not kill it.
# See docs/RUN_FINAL.md.
set -uo pipefail
DATA="${1:?usage: bash scripts/run_final.sh /path/to/dataset [workers]}"
WORKERS="${2:-3}"
cd "$(dirname "$0")/.."
PY=.venv/bin/python

fail() { echo "FAILED: $1. Do NOT improvise -- send work/final_run.log to Abhinav." >&2; exit 1; }

[ -f "$DATA/train/train_source1.tsv" ] && [ -f "$DATA/test/test_source1.tsv" ] \
    || fail "wrong data folder: '$DATA' must contain train/ and test/"

if [ ! -x "$PY" ]; then
    echo "Creating .venv with Python 3.12 (first time only) ..."
    python3.12 -m venv .venv || fail "create venv -- install python3.12 and python3.12-venv"
    "$PY" -m pip install --upgrade pip || fail "pip upgrade"
    "$PY" -m pip install -r requirements.txt || fail "pip install -r requirements.txt"
fi
"$PY" -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version" || fail "the venv must be Python 3.12"
"$PY" src/metric.py || fail "metric self-test"
"$PY" -m pytest -q tests || fail "unit tests"

"$PY" -u src/run_pipeline.py --data "$DATA" --out output --work work --loco \
    --block-workers "$WORKERS" --log work/final_run.log || fail "pipeline"
"$PY" scripts/make_handoff.py || fail "handoff bundle"
echo "ALL DONE. Send the zip in handoff/ to Abhinav. Do not upload it yourself."
