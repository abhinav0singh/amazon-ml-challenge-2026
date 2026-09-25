"""
make_package.py: assemble <team_name>_submission.zip in the exact layout the
problem statement requires (Final Submission Package, page 5).

    python scripts/make_package.py --team <team_name> --output-dir output

Layout produced, and nothing else:

    <team_name>_submission.zip
    |-- output/
    |   |-- matching_results.tsv
    |   `-- candidate_pairs.tsv
    |-- code/
    |   `-- business_entity_resolution/
    |       |-- src/                 every .py in the repo's src/
    |       |-- README.md            from docs/package_README.md
    |       `-- requirements.txt     from the repo root, pinned
    `-- Documentation_template.md    from the repo root, filled in

Design rule: the contents are an explicit allow-list, never a walk of the repo.
The zip must never contain the dataset, work/, caches, notebooks, .git or a
virtualenv -- a reviewer unzips this and it has to be small, clean and runnable.
The build REFUSES to run if either output TSV is missing, so a half-finished
package can never be handed over by accident.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

# Repo root = the parent of the folder holding this script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE_DIR = "code/business_entity_resolution"          # inside the zip
TSVS = ("matching_results.tsv", "candidate_pairs.tsv")  # both are mandatory


def repo(*parts: str) -> str:
    """An absolute path inside the repository."""
    return os.path.join(ROOT, *parts)


def collect(output_dir: str) -> list[tuple[str, str]]:
    """Return the full manifest as (source path on disk, path inside the zip).

    Raises SystemExit with a readable message if anything mandatory is missing,
    so the failure happens here rather than as a half-built zip.
    """
    missing, items = [], []

    # 1. output/ -- the two scored deliverables. Both mandatory.
    for name in TSVS:
        src = os.path.join(output_dir, name)
        if os.path.isfile(src):
            items.append((src, f"output/{name}"))
        else:
            missing.append(src)

    # 2. code/business_entity_resolution/src/ -- every .py, no caches.
    src_dir = repo("src")
    if not os.path.isdir(src_dir):
        missing.append(src_dir)
    else:
        pys = sorted(f for f in os.listdir(src_dir) if f.endswith(".py"))
        if not pys:
            missing.append(os.path.join(src_dir, "*.py"))
        items += [(os.path.join(src_dir, f), f"{CODE_DIR}/src/{f}") for f in pys]

    # 3. The two files that make the code folder self-contained.
    for src, dst in [(repo("docs", "package_README.md"), f"{CODE_DIR}/README.md"),
                     (repo("requirements.txt"), f"{CODE_DIR}/requirements.txt"),
                     (repo("Documentation_template.md"), "Documentation_template.md")]:
        if os.path.isfile(src):
            items.append((src, dst))
        else:
            missing.append(src)

    if missing:
        print("REFUSING TO BUILD -- these required files do not exist:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("\nThe two output TSVs come from a full run:\n"
              "  python src/run_pipeline.py --data <DATA> --out output --work work",
              file=sys.stderr)
        raise SystemExit(1)
    return items


def build(team: str, output_dir: str, dest_dir: str) -> str:
    """Write the zip and return its path. Deterministic order, no extras."""
    items = collect(output_dir)
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, f"{team}_submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in items:
            z.write(src, arc)
    return zip_path


def main():
    ap = argparse.ArgumentParser(description="Build the final submission zip.")
    ap.add_argument("--team", required=True,
                    help="team name; the zip is named <team>_submission.zip")
    ap.add_argument("--output-dir", default="output",
                    help="folder holding matching_results.tsv and candidate_pairs.tsv "
                         "(default: output)")
    ap.add_argument("--dest", default=ROOT,
                    help="where to write the zip (default: the repository root)")
    a = ap.parse_args()

    team = a.team.strip()
    if not team or any(c in team for c in r'/\:*?"<>|'):
        raise SystemExit(f"--team must be a plain name, got {a.team!r}")

    zip_path = build(team, a.output_dir, a.dest)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        total = sum(i.file_size for i in z.infolist())
    print(f"wrote {zip_path}  ({len(names)} files, {total / 1e6:.1f} MB uncompressed)")
    for n in names:
        print(f"  {n}")


if __name__ == "__main__":
    main()
