"""
exact_key_join.py: recover true pairs that blocking never proposed, with a hash join.

    # measure on train first -- precision/recall of the keys, nothing written
    python scripts/exact_key_join.py --data <DATA> --measure

    # then apply to a finished run's outputs
    python scripts/exact_key_join.py --data <DATA> --out output --apply

Why: the submitted run's blocking has a recall ceiling of 0.9677 and entity cover
of 0.9121, so roughly 9% of entities never get their full true set in front of the
matcher. The public leaderboard's top seven teams sit at 0.991, tightly bunched,
which is the signature of a largely DETERMINISTIC solution: most true matches agree
exactly on normalised fields, and are recoverable without a model at all.

The keys, both derived only from the records themselves (no external data):

    name_core + "|" + smallest postal code in the address
    name_core + "|" + normalised address

Two records sharing one of these are near-certain to be the same business. F0.5
weights precision twice, so near-certain pairs are exactly what it rewards, and
adding them costs nothing when they are already predicted.

Two rules this respects, because getting either wrong invalidates the submission:

  * Every id written to matching_results.tsv must also appear in
    candidate_pairs.tsv. So --apply writes BOTH files: the join's pairs are added
    to the candidate set as well as to the matches.
  * One S2/S3 record belongs to at most one S1 entity (measured: exactly zero
    exceptions across 7,638,365 train pairs). Records already matched by the model
    keep their assignment; a contested new record goes to a single entity, chosen
    deterministically so the result does not depend on dict ordering.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data_io import load_truth, read_tsv, write_id_lists  # noqa: E402
from normalize import core_name, norm_addr, postal_codes  # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def build_keys(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Both exact keys for one side. Blank where the key cannot be formed."""
    log(f"normalising {tag} ({len(df):,} records)")
    nc = df["business_name"].map(core_name).to_numpy()
    ad = df["business_address"].map(norm_addr).to_numpy()
    pcs = df["business_address"].map(postal_codes)
    pc = pcs.map(lambda s: min(s) if s else "").to_numpy()
    return pd.DataFrame({
        "id": df["entity_id"].to_numpy(),
        "k_pc": np.where((pc != "") & (nc != ""), np.char.add(np.char.add(nc.astype(str), "|"), pc.astype(str)), ""),
        "k_ad": np.where((ad != "") & (nc != ""), np.char.add(np.char.add(nc.astype(str), "|"), ad.astype(str)), ""),
    })


def join(k1: pd.DataFrame, k2: pd.DataFrame, col: str, max_block: int) -> pd.DataFrame:
    """Inner join on one key. Keys shared by more than `max_block` records on
    either side are dropped: those are degenerate (an empty or generic name) and
    would produce a combinatorial blow-up of low-quality pairs."""
    a = k1.loc[k1[col] != "", ["id", col]]
    b = k2.loc[k2[col] != "", ["id", col]]
    va = a[col].value_counts()
    vb = b[col].value_counts()
    bad = set(va[va > max_block].index) | set(vb[vb > max_block].index)
    if bad:
        a = a[~a[col].isin(bad)]
        b = b[~b[col].isin(bad)]
        log(f"  {col}: dropped {len(bad):,} oversized keys (> {max_block} on a side)")
    m = a.merge(b, on=col, suffixes=("_s1", "_s23"))
    log(f"  {col}: {len(m):,} joined pairs from {a['id'].nunique():,} S1 entities")
    return m[["id_s1", "id_s23"]]


def measure(data: str, max_block: int):
    """Precision and recall of each key against the training ground truth."""
    s1 = read_tsv(os.path.join(data, "train", "train_source1.tsv"))
    s23 = pd.concat([read_tsv(os.path.join(data, "train", f"train_source{k}.tsv"))
                     for k in (2, 3)], ignore_index=True)
    k1, k2 = build_keys(s1, "train S1"), build_keys(s23, "train S2/S3")
    del s1, s23
    truth = load_truth(data)
    total_true = sum(len(v) for v in truth.values())
    log(f"truth: {len(truth):,} entities, {total_true:,} true pairs")

    for col, name in (("k_pc", "name_core + postal"), ("k_ad", "name_core + addr_n")):
        m = join(k1, k2, col, max_block)
        if not len(m):
            continue
        hit = np.fromiter((c in truth.get(s, ()) for s, c in zip(m["id_s1"], m["id_s23"])),
                          dtype=bool, count=len(m))
        tp = int(hit.sum())
        print(f"\n  {name}")
        print(f"    precision            {tp / len(m):.5f}   ({tp:,} true of {len(m):,})")
        print(f"    recall of all pairs  {tp / total_true:.5f}   ({tp:,} of {total_true:,})")
        print(f"    entities with a hit  {m.loc[hit, 'id_s1'].nunique():,}", flush=True)


def apply(data: str, out: str, key: str, max_block: int, pairs_file: str | None = None):
    """Add the join's pairs to BOTH output files, respecting one-to-one.

    `pairs_file` skips the join and uses a precomputed two-column TSV
    (source1_entity_id, matched_entity_id). Normalising 11.7M test records takes
    about ten minutes, so on a deadline it is worth computing once and reusing.
    """
    t1 = read_tsv(os.path.join(data, "test", "test_source1.tsv"))
    if pairs_file:
        pre = read_tsv(pairs_file)
        new = pre.rename(columns={pre.columns[0]: "id_s1", pre.columns[1]: "id_s23"})[["id_s1", "id_s23"]]
        log(f"loaded {len(new):,} precomputed pairs from {pairs_file}")
    else:
        t23 = pd.concat([read_tsv(os.path.join(data, "test", f"test_source{k}.tsv"))
                         for k in (2, 3)], ignore_index=True)
        k1, k2 = build_keys(t1, "test S1"), build_keys(t23, "test S2/S3")
        del t23
        cols = ["k_pc", "k_ad"] if key == "both" else [key]
        new = pd.concat([join(k1, k2, c, max_block) for c in cols], ignore_index=True).drop_duplicates()
    log(f"join proposes {len(new):,} pairs over {new['id_s1'].nunique():,} entities")

    mpath = os.path.join(out, "matching_results.tsv")
    cpath = os.path.join(out, "candidate_pairs.tsv")
    matches = {r.source1_entity_id: (set(r.matched_entity_ids.split(",")) if r.matched_entity_ids else set())
               for r in read_tsv(mpath).itertuples()}
    cands = {r.source1_entity_id: (set(r.candidate_entity_ids.split(",")) if r.candidate_entity_ids else set())
             for r in read_tsv(cpath).itertuples()}
    taken = {c for v in matches.values() for c in v}
    log(f"existing: {sum(map(len, matches.values())):,} matched ids, "
        f"{sum(map(len, cands.values())):,} candidate ids")

    # Deterministic order so the result never depends on dict iteration order.
    new = new.sort_values(["id_s23", "id_s1"], kind="stable")
    added = 0
    for s1_id, c_id in zip(new["id_s1"].to_numpy(), new["id_s23"].to_numpy()):
        if c_id in taken:            # already assigned to some entity -- one-to-one
            continue
        matches.setdefault(s1_id, set()).add(c_id)
        cands.setdefault(s1_id, set()).add(c_id)
        taken.add(c_id)
        added += 1
    log(f"added {added:,} new matched ids (skipped {len(new) - added:,} already-taken records)")

    t_ids = t1["entity_id"].tolist()
    os.replace(mpath, mpath + ".before_exact_key.bak")
    os.replace(cpath, cpath + ".before_exact_key.bak")
    write_id_lists(mpath, t_ids, matches, "matched_entity_ids")
    write_id_lists(cpath, t_ids, cands, "candidate_entity_ids")
    log("rewrote both files; previous versions kept as .before_exact_key.bak")
    log("RUN scripts/audit_submission.py BEFORE UPLOADING")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--measure", action="store_true", help="train-set precision/recall only")
    ap.add_argument("--apply", action="store_true", help="rewrite the test output files")
    ap.add_argument("--key", default="both", choices=["k_pc", "k_ad", "both"])
    ap.add_argument("--max-block", type=int, default=50,
                    help="drop keys shared by more than this many records on a side")
    ap.add_argument("--pairs", default=None,
                    help="use a precomputed two-column TSV of pairs instead of joining "
                         "(saves ~10 min of normalising 11.7M test records)")
    a = ap.parse_args()
    if a.measure:
        measure(a.data, a.max_block)
    if a.apply:
        apply(a.data, a.out, a.key, a.max_block, a.pairs)
    if not (a.measure or a.apply):
        ap.error("pass --measure or --apply")


if __name__ == "__main__":
    main()
