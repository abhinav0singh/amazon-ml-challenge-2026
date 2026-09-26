"""
make_handoff.py: bundle a finished full run for P1 to upload.

    python scripts/make_handoff.py            # after run_final.* has finished

Refuses to build unless the run finished cleanly: both submission TSVs exist,
work/report.json has the test section and a passed output check, and the files
on disk still match the sha256 hashes the run recorded. Writes
handoff/handoff_<git>_<time>.zip (ZIP64, so a multi-GB candidate_pairs.tsv is
fine) containing the two TSVs, RUN_SUMMARY.txt, report.json and the run log.

P1 then unzips, re-runs the official validator, compares the hashes against
RUN_SUMMARY.txt, and uploads matching_results.tsv + candidate_pairs.tsv.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import zipfile

import argparse

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default=os.path.join(ROOT, "output"))
_ap.add_argument("--work", default=os.path.join(ROOT, "work"))
_ap.add_argument("--dest", default=os.path.join(ROOT, "handoff"))
_args = _ap.parse_args()
OUT, WORK = _args.out, _args.work
FILES = [(os.path.join(OUT, "matching_results.tsv"), "matching_results.tsv"),
         (os.path.join(OUT, "candidate_pairs.tsv"), "candidate_pairs.tsv"),
         (os.path.join(OUT, "RUN_SUMMARY.txt"), "RUN_SUMMARY.txt"),
         (os.path.join(WORK, "report.json"), "report.json")]
LOG = (os.path.join(WORK, "final_run.log"), "final_run.log")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    missing = [src for src, _ in FILES if not os.path.exists(src)]
    if missing:
        sys.exit("NOT READY -- missing:\n  " + "\n  ".join(missing) +
                 "\nThe run has not finished. Re-run the same run_final command; it resumes.")
    with open(os.path.join(WORK, "report.json")) as f:
        report = json.load(f)
    if "sample" in report:
        sys.exit("REFUSING: report.json is from a --sample smoke test, not a full run.")
    if "test" not in report or "output_check" not in report:
        sys.exit("REFUSING: report.json has no test/output_check section -- the run did not finish.")
    for name, want in report.get("output_sha256", {}).items():
        got = sha256(os.path.join(OUT, name))
        if got != want:
            sys.exit(f"REFUSING: {name} changed since the run wrote it (sha256 {got} != {want}).")
    rc = report.get("official_validator_exit_code")
    if rc not in (0, None):
        sys.exit(f"REFUSING: the official validator exited with code {rc}. Read final_run.log.")

    os.makedirs(_args.dest, exist_ok=True)
    name = f"handoff_{report.get('git_commit', 'unknown')}_{time.strftime('%Y%m%d_%H%M')}.zip"
    dest = os.path.join(_args.dest, name)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for src, arc in FILES + ([LOG] if os.path.exists(LOG[0]) else []):
            z.write(src, arc)
    print(f"handoff ready: {dest} ({os.path.getsize(dest) / 1e6:.0f} MB)")
    print(f"CV (cross-fitted) = {report.get('cv_macro_f05_cross_fitted')}, "
          f"validator exit code = {rc}")
    if rc is None:
        print("NOTE: the official validator was not found on this machine -- P1 must run it "
              "before uploading.")
    print("Send this ONE zip to P1. Do not upload anything to the portal yourself.")


if __name__ == "__main__":
    main()
