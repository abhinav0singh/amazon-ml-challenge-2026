"""
sweep_blocking.py: measure the blocking cost/recall trade-off before committing
to a setting for the full-scale run.

The sparse top-k cost is driven by how many non-zero similarities the product
has, which is the sum of the posting-list lengths of a query's n-grams. An
n-gram appearing in a large share of records carries almost no IDF weight but a
huge posting list, so pruning the vocabulary by document frequency is the main
lever on runtime. It is also the main risk to the recall ceiling, which caps
the whole pipeline -- so it must be measured, not guessed.

    python src/sweep_blocking.py --data C:\\amlc\\student_resource\\dataset --n 40000

Prints one row per setting: recall ceiling, entity cover, candidates per S1,
and blocking wall-clock. Nothing here is CV; it is a cost/ceiling measurement.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocking  # noqa: E402  (mutated below to test settings)
from blocking import blocking_report, generate_candidates  # noqa: E402
from data_io import load_split, load_truth  # noqa: E402
from normalize import add_normalized_columns  # noqa: E402
from run_pipeline import subsample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=40000, help="S1 entities to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"loading train, sampling {args.n} S1 entities", flush=True)
    s1, s23 = load_split(args.data, "train")
    truth = load_truth(args.data)
    s1, s23 = subsample(s1, s23, args.n, args.seed, truth)
    s1, s23 = add_normalized_columns(s1), add_normalized_columns(s23)
    s1_ids = s1["entity_id"].tolist()
    truth = {s: truth.get(s, set()) for s in s1_ids}
    print(f"sampled: {len(s1)} S1, {len(s23)} S2/S3", flush=True)

    # (max_df, min_df, K_NAME, K_FULL, K_ADDR)
    settings = [
        (1.0, 1, 15, 15, 5),     # no pruning -- the original behaviour
        (0.1, 2, 15, 15, 5),     # current default
        (0.01, 2, 15, 15, 5),
        (0.001, 2, 15, 15, 5),
        (0.01, 2, 10, 10, 5),    # pruning plus a smaller K
    ]
    print(f"\n{'max_df':>8} {'min_df':>6} {'K':>10} {'recall':>8} {'cover':>7} "
          f"{'cands/S1':>9} {'secs':>7}", flush=True)
    for max_df, min_df, kn, kf, ka in settings:
        blocking.MAX_DF, blocking.MIN_DF = max_df, min_df
        blocking.K_NAME, blocking.K_FULL, blocking.K_ADDR = kn, kf, ka
        t0 = time.time()
        cand = generate_candidates(s1, s23, by_country=True)
        secs = time.time() - t0
        r = blocking_report(cand, truth, s1_ids)
        print(f"{max_df:>8} {min_df:>6} {f'{kn}/{kf}/{ka}':>10} "
              f"{r['pair_recall_ceiling']:>8.4f} {r['entity_full_cover']:>7.4f} "
              f"{r['avg_cands_per_s1']:>9.2f} {secs:>7.1f}", flush=True)

    print("\nRecall ceiling caps every downstream score. Prefer the cheapest "
          "setting whose ceiling is within noise of the unpruned one.", flush=True)


if __name__ == "__main__":
    main()
