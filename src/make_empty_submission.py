"""
make_empty_submission.py: submission D1-1, "every S1 entity is a singleton".

Why: it tests the whole upload path, and its public score is roughly the
share of singletons in the public test subset, which tells us how conservative
the matching threshold must be.

    python src/make_empty_submission.py --data data/dataset --out output_empty
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_io import read_tsv, write_id_lists  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/dataset")
ap.add_argument("--out", default="output_empty")
a = ap.parse_args()
ids = read_tsv(os.path.join(a.data, "test", "test_source1.tsv"))["entity_id"].tolist()
write_id_lists(os.path.join(a.out, "matching_results.tsv"), ids, {}, "matched_entity_ids")
write_id_lists(os.path.join(a.out, "candidate_pairs.tsv"), ids, {}, "candidate_entity_ids")
print(f"wrote {len(ids)} empty rows to {a.out}/")
