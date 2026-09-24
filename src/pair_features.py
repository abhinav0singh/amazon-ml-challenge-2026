"""
pair_features.py: features for each (S1 record, candidate S2/S3 record) pair.

Every feature is a SIMILARITY or a RELATIVE rank. No raw country, and no
country-specific rule, so the matcher trained on US/India transfers to France.
Groups of features:
  * string similarity on name (full and core) and address (rapidfuzz)
  * TF-IDF cosines carried over from blocking
  * postal code / number agreement
  * context: how this candidate ranks among the S1's candidates, and how this
    S1 ranks among all S1s that proposed the same candidate (competition)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURES = [
    "cos_name", "cos_full", "cos_addr",
    "nm_ratio", "nm_tsort", "nm_tset", "nm_partial", "nm_jw",
    "core_ratio", "core_tset", "core_jw", "core_first_tok_eq", "core_len_diff",
    "ad_ratio", "ad_tset", "ad_partial",
    "postal_match", "num_jacc", "num_overlap",
    "is_s3", "n_cands",
    "rank_name_in_s1", "gap_name_to_best", "rank_full_in_s1", "gap_full_to_best",
    "rank_s1_for_cand", "n_s1_for_cand", "gap_to_best_s1_for_cand",
]


def _jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else -1.0


def build_pair_features(cand: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
    """cand: output of blocking (s1_id, cand_id, cos_*). s1/s23: normalised records.
    Returns cand with every column in FEATURES added."""
    a = s1.set_index("entity_id").loc[cand["s1_id"]]
    b = s23.set_index("entity_id").loc[cand["cand_id"]]
    df = cand.reset_index(drop=True).copy()

    an, bn = a["name_n"].to_numpy(), b["name_n"].to_numpy()
    ac, bc = a["name_core"].to_numpy(), b["name_core"].to_numpy()
    aa, ba = a["addr_n"].to_numpy(), b["addr_n"].to_numpy()
    ap, bp = a["postal"].to_numpy(), b["postal"].to_numpy()
    au, bu = a["nums"].to_numpy(), b["nums"].to_numpy()

    n = len(df)
    cols = {c: np.empty(n, dtype=np.float32) for c in [
        "nm_ratio", "nm_tsort", "nm_tset", "nm_partial", "nm_jw", "core_ratio", "core_tset",
        "core_jw", "core_first_tok_eq", "core_len_diff", "ad_ratio", "ad_tset", "ad_partial",
        "postal_match", "num_jacc", "num_overlap"]}
    for i in range(n):
        x, y = an[i], bn[i]
        cols["nm_ratio"][i] = fuzz.ratio(x, y)
        cols["nm_tsort"][i] = fuzz.token_sort_ratio(x, y)
        cols["nm_tset"][i] = fuzz.token_set_ratio(x, y)
        cols["nm_partial"][i] = fuzz.partial_ratio(x, y)
        cols["nm_jw"][i] = JaroWinkler.similarity(x, y)
        x, y = ac[i], bc[i]
        cols["core_ratio"][i] = fuzz.ratio(x, y)
        cols["core_tset"][i] = fuzz.token_set_ratio(x, y)
        cols["core_jw"][i] = JaroWinkler.similarity(x, y)
        xs, ys = x.split(), y.split()
        cols["core_first_tok_eq"][i] = float(bool(xs) and bool(ys) and xs[0] == ys[0])
        cols["core_len_diff"][i] = abs(len(x) - len(y))
        x, y = aa[i], ba[i]
        if x and y:
            cols["ad_ratio"][i] = fuzz.ratio(x, y)
            cols["ad_tset"][i] = fuzz.token_set_ratio(x, y)
            cols["ad_partial"][i] = fuzz.partial_ratio(x, y)
        else:  # one address missing: say "unknown" rather than "different"
            cols["ad_ratio"][i] = cols["ad_tset"][i] = cols["ad_partial"][i] = -1
        pa, pb = ap[i], bp[i]
        cols["postal_match"][i] = -1.0 if not (pa and pb) else float(bool(pa & pb))
        cols["num_jacc"][i] = _jacc(au[i], bu[i])
        cols["num_overlap"][i] = len(au[i] & bu[i])
    for k, v in cols.items():
        df[k] = v

    df["is_s3"] = df["cand_id"].str.startswith("S3").astype(np.float32)
    g = df.groupby("s1_id")
    df["n_cands"] = g["cand_id"].transform("size").astype(np.float32)
    for c, short in [("cos_name", "name"), ("cos_full", "full")]:
        df[f"rank_{short}_in_s1"] = g[c].rank(ascending=False, method="min").astype(np.float32)
        df[f"gap_{short}_to_best"] = (g[c].transform("max") - df[c]).astype(np.float32)
    # Competition between S1 records for the same candidate (Source 1 is deduplicated,
    # so a record usually belongs to at most one S1 entity).
    gc = df.groupby("cand_id")
    df["rank_s1_for_cand"] = gc["cos_full"].rank(ascending=False, method="min").astype(np.float32)
    df["n_s1_for_cand"] = gc["s1_id"].transform("size").astype(np.float32)
    df["gap_to_best_s1_for_cand"] = (gc["cos_full"].transform("max") - df["cos_full"]).astype(np.float32)
    return df
