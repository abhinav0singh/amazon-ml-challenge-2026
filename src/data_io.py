"""
data_io.py: loading the challenge .tsv files and writing submission files.

All files are TAB-separated. Everything is read as string with
keep_default_na=False, so an empty cell stays "" and a business literally
named "NA" is not turned into NaN.
"""
from __future__ import annotations

import os
import pandas as pd

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str) -> pd.DataFrame:
    """Read one challenge TSV with a strict tab separator, all columns as str."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def load_split(data_dir: str, split: str):
    """Load <split>_source{1,2,3}.tsv from data_dir/<split>/.
    Returns (s1, s23): S1 records and the concatenated S2+S3 records."""
    d = os.path.join(data_dir, split)
    s1 = read_tsv(os.path.join(d, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(d, f"{split}_source2.tsv"))
    s3 = read_tsv(os.path.join(d, f"{split}_source3.tsv"))
    s23 = pd.concat([s2, s3], ignore_index=True)
    return s1, s23


def load_truth(data_dir: str) -> dict:
    """Ground truth as {s1_id: set(matched ids)}. An empty list means a singleton."""
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    out = {}
    for s1, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[s1] = {x.strip() for x in m.split(",") if x.strip()}
    return out


def write_id_lists(path: str, s1_ids, mapping: dict, col: str):
    """Write one row per S1 id: `source1_entity_id <TAB> <col>` with a
    comma-joined, de-duplicated, sorted id list (empty string for none).
    Written by hand (no pandas quoting) so ids are never wrapped in quotes."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in s1_ids:
            ids = sorted(set(mapping.get(s, ())))
            f.write(f"{s}\t{','.join(ids)}\n")
