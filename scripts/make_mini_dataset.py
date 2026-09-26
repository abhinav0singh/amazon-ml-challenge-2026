"""
make_mini_dataset.py: a small, real-format copy of the dataset for dry runs.

    python scripts/make_mini_dataset.py --data <DATA> --dest <MINI>/dataset [--n-train 3000 --n-test 2000]

`--sample` smoke tests skip the parts of a full run that matter most at the end:
the output verification, the official validator, RUN_SUMMARY.txt and the handoff
bundle. This builds a folder with the exact organisers' layout (train/, test/,
same file names and columns) so the FULL code path can be exercised in minutes:

    python src/run_pipeline.py --data <MINI>/dataset --work work_mini --out output_mini --loco

Train: n S1 entities, every true match of theirs, plus random S2/S3 fill at the
real S2/S3-per-S1 density. Test: n random S1 plus random S2/S3 at the same density
(no truth exists for test). Scores on it mean nothing -- it is a plumbing test.

Copies utils/ from beside the source dataset (if present) so the run can call the
official validator. Reads the large files in chunks to stay light on RAM.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data_io import read_tsv  # noqa: E402


def filter_copy(src, dst, keep_ids, chunk=1_000_000):
    """Stream `src` and write only rows whose entity_id is in keep_ids."""
    first = True
    for part in pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False, quoting=3,
                            chunksize=chunk):
        part = part[part["entity_id"].isin(keep_ids)]
        part.to_csv(dst, sep="\t", index=False, mode="w" if first else "a",
                    header=first, quoting=3)
        first = False


def ids_of(path, chunk=2_000_000):
    out = []
    for part in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3,
                            usecols=["entity_id"], chunksize=chunk):
        out.append(part["entity_id"].to_numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for split in ("train", "test"):
        os.makedirs(os.path.join(args.dest, split), exist_ok=True)

    for split, n in (("train", args.n_train), ("test", args.n_test)):
        d = os.path.join(args.data, split)
        s1 = read_tsv(os.path.join(d, f"{split}_source1.tsv"))
        keep = s1.iloc[np.sort(rng.choice(len(s1), size=min(n, len(s1)), replace=False))]
        keep.to_csv(os.path.join(args.dest, split, f"{split}_source1.tsv"), sep="\t",
                    index=False, quoting=3)
        wanted = set()
        if split == "train":
            gt = read_tsv(os.path.join(d, "train_ground_truth.tsv"))
            gt = gt[gt["source1_entity_id"].isin(set(keep["entity_id"]))]
            gt.to_csv(os.path.join(args.dest, split, "train_ground_truth.tsv"), sep="\t",
                      index=False, quoting=3)
            for m in gt["matched_entity_ids"]:
                wanted |= {x.strip() for x in m.split(",") if x.strip()}
        pool = np.concatenate([ids_of(os.path.join(d, f"{split}_source{k}.tsv")) for k in (2, 3)])
        density = len(pool) / len(s1)
        short = int(round(len(keep) * density)) - len(wanted)
        if short > 0:
            wanted |= set(rng.choice(pool, size=min(short, len(pool)), replace=False).tolist())
        for k in (2, 3):
            filter_copy(os.path.join(d, f"{split}_source{k}.tsv"),
                        os.path.join(args.dest, split, f"{split}_source{k}.tsv"), wanted)
        print(f"{split}: {len(keep)} S1, {len(wanted)} S2/S3 kept")
        del s1, pool

    utils = os.path.join(os.path.dirname(os.path.abspath(args.data.rstrip("/\\"))), "utils")
    if os.path.isdir(utils):
        dst = os.path.join(os.path.dirname(os.path.abspath(args.dest.rstrip("/\\"))), "utils")
        shutil.copytree(utils, dst, dirs_exist_ok=True)
        print(f"copied official utils -> {dst}")
    print(f"mini dataset ready at {args.dest}")


if __name__ == "__main__":
    main()
