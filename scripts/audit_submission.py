"""
audit_submission.py: the last check before P1 uploads (#22 part 4).

    python scripts/audit_submission.py --handoff handoff/handoff_<git>_<time>.zip \
                                       --test-dir C:/amlc/dataset/test
    python scripts/audit_submission.py --dir output --test-dir C:/amlc/dataset/test

Prints a line per check and exits 0 only if every one passes.

Why this exists even though the run already checks itself: `run_pipeline.verify_outputs`
runs on the machine that produced the files, before they are zipped and moved. This runs
on the files P1 is actually about to upload, after transfer, and it adds the two checks
nothing else makes:

  * the per-country prediction rate. France is 15.0% of test Source 1 and appears nowhere
    in training. If the model quietly collapses on it -- predicting empty far more often
    than for US and India -- every other check still passes and we lose a seventh of the
    score. Nothing upstream looks for this.
  * the test prediction rate against the out-of-fold rate from the same run. A large gap
    means the test half behaved differently from cross-validation, which is a pipeline bug
    far more often than it is a real effect.

Reads the big TSVs line by line, so a 1.5 GB candidate_pairs.tsv costs no memory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile

EXPECTED_TEST_S1 = 1_732_544          # measured from test_source1.tsv, see AGENTS.md section 1
M_HEADER = "source1_entity_id\tmatched_entity_ids"
C_HEADER = "source1_entity_id\tcandidate_entity_ids"
SHARE_GAP_LIMIT = 0.05                # test vs OOF non-empty share, per #22 part 4
FRANCE_LOW_LIMIT = 0.80    # France rate must be >= this x the weakest seen country
FRANCE_HIGH_LIMIT = 1.30   # ... and <= this x the strongest, see the two-sided check below

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(ok: bool, label: str, detail: str = "", warn_only: bool = False) -> bool:
    """Record and print one check. `warn_only` flags a judgement call rather than a rule."""
    if ok:
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    elif warn_only:
        WARNINGS.append(f"{label}: {detail}")
        print(f"  WARN  {label}  ({detail})")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL  {label}  ({detail})")
    return ok


class Source:
    """Read the bundle either from a handoff zip or from a plain directory,
    without unpacking multi-GB files to disk."""

    def __init__(self, zip_path: str | None, dir_path: str | None):
        self.zf = zipfile.ZipFile(zip_path) if zip_path else None
        self.dir = dir_path

    def names(self) -> list[str]:
        return self.zf.namelist() if self.zf else sorted(os.listdir(self.dir))

    def open_text(self, name: str):
        """A text handle on one member, streamed."""
        if self.zf:
            return io.TextIOWrapper(self.zf.open(name), encoding="utf-8", newline="")
        return open(os.path.join(self.dir, name), encoding="utf-8", newline="")

    def open_binary(self, name: str):
        return self.zf.open(name) if self.zf else open(os.path.join(self.dir, name), "rb")

    def sha256(self, name: str) -> str:
        """Hash a member in 16 MB chunks."""
        h = hashlib.sha256()
        with self.open_binary(name) as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()


def load_test_s1(test_dir: str):
    """(ids in file order, country per id) for every test Source 1 entity.

    Read by hand rather than with pandas: this runs next to a 1.5 GB file on a machine
    that may already be short of memory, and we only need two columns.
    """
    path = os.path.join(test_dir, "test_source1.tsv")
    ids, countries = [], []
    with open(path, encoding="utf-8", newline="") as f:
        header = next(f).rstrip("\n").split("\t")
        i_id, i_country = header.index("entity_id"), header.index("country")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[i_id])
            countries.append(parts[i_country])
    return ids, countries


def audit_rows(src: Source, t_ids, t_countries, check_ids: bool, test_dir: str):
    """Stream both TSVs once, row-aligned with test_source1.tsv, and collect every
    per-row statistic the checks below need."""
    s23 = set()
    if check_ids:
        print("  ...  loading test S2/S3 ids (this costs a few GB of RAM)")
        for split in ("test_source2.tsv", "test_source3.tsv"):
            with open(os.path.join(test_dir, split), encoding="utf-8", newline="") as f:
                next(f)
                for line in f:
                    s23.add(line.split("\t", 1)[0])

    stats = {"rows": 0, "nonempty": 0, "matched": 0, "cands": 0,
             "by_country": {}, "bad_prefix": 0, "dup_in_list": 0,
             "not_in_cands": 0, "unknown_id": 0, "misaligned": 0}
    seen_s1 = set()
    dup_s1 = 0

    with src.open_text("matching_results.tsv") as fm, src.open_text("candidate_pairs.tsv") as fc:
        hm = next(fm).rstrip("\r\n")
        hc = next(fc).rstrip("\r\n")
        check(hm == M_HEADER, "matching_results.tsv header", repr(hm))
        check(hc == C_HEADER, "candidate_pairs.tsv header", repr(hc))

        for s1, country, lm, lc in zip(t_ids, t_countries, fm, fc):
            sm, _, ml = lm.rstrip("\r\n").partition("\t")
            sc, _, cl = lc.rstrip("\r\n").partition("\t")
            if sm != s1 or sc != s1:
                stats["misaligned"] += 1
            if sm in seen_s1:
                dup_s1 += 1
            else:
                seen_s1.add(sm)

            m = ml.split(",") if ml else []
            c = cl.split(",") if cl else []
            mset = set(m)
            if len(mset) != len(m):
                stats["dup_in_list"] += 1
            if not mset <= set(c):
                stats["not_in_cands"] += 1
            for i in m:
                if not (i.startswith("S2-") or i.startswith("S3-")):
                    stats["bad_prefix"] += 1
                if check_ids and i not in s23:
                    stats["unknown_id"] += 1

            b = stats["by_country"].setdefault(country, {"n": 0, "nonempty": 0, "matched": 0})
            b["n"] += 1
            b["nonempty"] += bool(m)
            b["matched"] += len(m)
            stats["rows"] += 1
            stats["nonempty"] += bool(m)
            stats["matched"] += len(m)
            stats["cands"] += len(c)

        stats["extra_rows"] = (next(fm, None) is not None) or (next(fc, None) is not None)
    stats["dup_s1"] = dup_s1
    return stats


def main():
    ap = argparse.ArgumentParser(description="Audit the submission before upload.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--handoff", help="handoff zip from scripts/make_handoff.py")
    g.add_argument("--dir", help="a directory holding the outputs instead")
    ap.add_argument("--test-dir", required=True, help="dataset/test (for ids and countries)")
    ap.add_argument("--check-ids", action="store_true",
                    help="also confirm every matched id exists in test S2/S3 (several GB of RAM)")
    a = ap.parse_args()

    src = Source(a.handoff, a.dir)
    print(f"\nAuditing {a.handoff or a.dir}\n")

    print("Bundle contents")
    names = src.names()
    for want in ("matching_results.tsv", "candidate_pairs.tsv"):
        check(want in names, f"{want} present")
    have_report = "report.json" in names
    check(have_report, "report.json present", "no report -> hash and OOF checks skipped",
          warn_only=not have_report)
    if FAILURES:
        print("\nAUDIT FAILED -- the bundle is incomplete.")
        return 1

    report = json.load(src.open_text("report.json")) if have_report else {}

    print("\nProvenance")
    if report:
        check("sample" not in report, "report is from a full run, not --sample")
        check(report.get("official_validator_exit_code") in (0, None),
              "official validator exit code",
              str(report.get("official_validator_exit_code")))
        if report.get("official_validator_exit_code") is None:
            check(False, "official validator was not run on the producing machine",
                  "P1 must run it locally before uploading", warn_only=True)
        for name, want in (report.get("output_sha256") or {}).items():
            if name in names:
                got = src.sha256(name)
                check(got == want, f"sha256 {name}", f"{got[:16]}... vs recorded {want[:16]}...")
        check(bool(report.get("output_check")), "run recorded a passed output check",
              str(report.get("output_check")))

    print("\nFormat")
    t_ids, t_countries = load_test_s1(a.test_dir)
    check(len(t_ids) == EXPECTED_TEST_S1, "test_source1.tsv entity count",
          f"{len(t_ids)} (expected {EXPECTED_TEST_S1})")
    st = audit_rows(src, t_ids, t_countries, a.check_ids, a.test_dir)
    check(st["rows"] == len(t_ids), "one row per test S1 entity",
          f"{st['rows']} rows vs {len(t_ids)} entities")
    check(not st["extra_rows"], "no extra rows after the last entity")
    check(st["misaligned"] == 0, "row order matches test_source1.tsv", f"{st['misaligned']} off")
    check(st["dup_s1"] == 0, "no duplicate source1_entity_id", f"{st['dup_s1']} duplicated")
    check(st["dup_in_list"] == 0, "no duplicate ids within a list", f"{st['dup_in_list']} rows")
    check(st["bad_prefix"] == 0, "matched ids are S2-/S3- only", f"{st['bad_prefix']} bad")
    check(st["not_in_cands"] == 0, "every matched id is also a candidate",
          f"{st['not_in_cands']} rows")
    if a.check_ids:
        check(st["unknown_id"] == 0, "every matched id exists in test S2/S3",
              f"{st['unknown_id']} unknown")

    print("\nPrediction behaviour")
    share = st["nonempty"] / max(st["rows"], 1)
    print(f"  ...  non-empty predictions: {st['nonempty']:,} / {st['rows']:,} = {share:.4f}")
    print(f"  ...  {st['matched']:,} matched ids, {st['cands']:,} candidate ids "
          f"({st['matched'] / max(st['nonempty'], 1):.2f} per predicting entity)")
    oof = (report.get("test") or {}).get("oof_pred_nonempty_share")
    if oof is not None:
        check(abs(share - oof) <= SHARE_GAP_LIMIT, "test vs OOF non-empty share",
              f"test {share:.4f} vs OOF {oof:.4f}, gap {abs(share - oof):.4f} "
              f"(limit {SHARE_GAP_LIMIT})")

    print("\nUnseen-country check (France)")
    rates = {}
    for country, b in sorted(st["by_country"].items()):
        r = b["nonempty"] / max(b["n"], 1)
        rates[country] = r
        print(f"  ...  {country:<10} {b['n']:>9,} entities, non-empty {r:.4f}, "
              f"{b['matched'] / max(b['nonempty'], 1):.2f} matches per predicting entity")
    fr = next((c for c in rates if c.strip().lower().startswith("fr")), None)
    if fr is None:
        check(False, "France present in the test set", "no France rows found", warn_only=True)
    else:
        others = [r for c, r in rates.items() if c != fr]
        # Two-sided, because an unseen country can fail in either direction and the
        # two failures look nothing alike:
        #   too LOW  -> the model has no confidence on France and leaves entities
        #               empty. Every non-singleton it abandons scores a flat 0.0.
        #   too HIGH -> it is over-committing on a country it never trained on.
        #               Those extra predictions are false merges, and F0.5 weights
        #               precision twice, so this is not the safe direction either.
        # Measured on a 40k sampled run (27 Sep): France 0.2694 vs US 0.1341 and
        # India 0.1682 -- i.e. the HIGH side is the one that actually showed up,
        # and a one-sided check would have passed it without a word.
        floor = FRANCE_LOW_LIMIT * min(others) if others else 0.0
        ceil_ = FRANCE_HIGH_LIMIT * max(others) if others else 1.0
        check(rates[fr] >= floor, "France prediction rate not collapsed",
              f"France {rates[fr]:.4f}, weakest other {min(others):.4f} "
              f"-> floor {floor:.4f}. Below this means unseen-country collapse "
              f"on 15% of the test set")
        check(rates[fr] <= ceil_, "France prediction rate not inflated",
              f"France {rates[fr]:.4f}, strongest other {max(others):.4f} "
              f"-> ceiling {ceil_:.4f}. Above this the model is over-committing on "
              f"a country it never trained on, and every extra prediction is a "
              f"false merge against a precision-weighted metric",
              warn_only=True)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"AUDIT FAILED -- {len(FAILURES)} problem(s). DO NOT UPLOAD:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    if WARNINGS:
        print(f"AUDIT PASS, with {len(WARNINGS)} thing(s) to look at:")
        for w in WARNINGS:
            print(f"  - {w}")
    else:
        print("AUDIT PASS -- safe to upload.")
    if report.get("cv_macro_f05_cross_fitted") is not None:
        print(f"CV macro-F0.5 (cross-fitted) = {report['cv_macro_f05_cross_fitted']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
