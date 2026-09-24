"""
blocking.py: candidate generation (S1 -> plausible S2/S3 records).

The candidate set caps recall: a true match dropped here can never be predicted.
Strategy (all unsupervised, fitted on the records of the split being processed):
  1. Group by country label when that country exists in S2/S3; otherwise search
     all S2/S3 records (so an unseen or mislabelled country never loses everything).
  2. Char n-gram TF-IDF on the CORE name   -> top-K nearest S2/S3 per S1 (cosine).
  3. Char n-gram TF-IDF on name+address    -> top-K nearest.
  4. Word TF-IDF on the address             -> top-K nearest (catches renamed / DBA names).
  Union of the three lists = candidate set. The cosine scores are kept as features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

K_NAME, K_FULL, K_ADDR = 15, 15, 5

# Rows of S1 per sparse matmul chunk. The cost driver is the number of non-zero
# similarities produced, not the chunk's row count, so this trades peak memory
# against Python-loop overhead.
CHUNK_ROWS = 2048

# Vocabulary pruning. An n-gram present in a large share of records carries
# almost no IDF weight but sits in a huge posting list, so it dominates the
# product's non-zeros while barely moving the cosine. Dropping those terms is
# the difference between a feasible and an infeasible matmul at 4.7M records.
# max_df is a fraction of documents; min_df drops one-off noise.
#
# MEASURED on a 40,000-entity train sample (src/sweep_blocking.py, 25 Sep):
#   max_df  recall  cover   cands/S1  secs
#   1.0     0.9894  0.9683  27.07     241.0
#   0.1     0.9897  0.9697  27.12     133.1
#   0.01    0.9842  0.9551  27.27      44.8
#   0.001   0.9495  0.8722  22.61      35.0
# 0.1 is free (recall equal to unpruned, 1.8x faster). 0.01 costs 0.005 recall
# for 5.4x. 0.001 destroys recall for almost no further gain, which shows the
# remaining cost is NOT in long posting lists -- so pruning alone cannot make
# this design scale. 0.01 is provisional, chosen to keep experiments fast while
# the candidate-generation design is settled; revisit once it is.
MAX_DF, MIN_DF = 0.01, 2


def _topk_sparse(A, B, k, chunk_rows=CHUNK_ROWS):
    """For each row of A, the indices of its top-k most similar rows in B.

    A and B are L2-normalised sparse TF-IDF matrices, so the product is the
    cosine. The product is kept SPARSE end to end: only pairs sharing at least
    one n-gram are ever materialised, which at this data scale is a tiny
    fraction of A.shape[0] * B.shape[0].

    Returns flat (rows, cols) index arrays rather than a fixed (n_a, k) block,
    because a row with fewer than k non-zero similarities genuinely has fewer
    than k candidates. The old dense version padded those rows out to k with
    arbitrary zero-similarity records; dropping them is both cheaper and
    cleaner, and it is the one behavioural difference from that version.
    """
    Bt = B.T.tocsr()
    rows_out, cols_out = [], []
    for start in range(0, A.shape[0], chunk_rows):
        S = (A[start:start + chunk_rows] @ Bt).tocsr()
        indptr, indices, data = S.indptr, S.indices, S.data
        for i in range(S.shape[0]):
            lo, hi = indptr[i], indptr[i + 1]
            n = hi - lo
            if n == 0:
                continue
            if n > k:
                sel = lo + np.argpartition(-data[lo:hi], k - 1)[:k]
            else:
                sel = np.arange(lo, hi)
            cols_out.append(indices[sel])
            rows_out.append(np.full(sel.size, start + i, dtype=np.int64))
    if not rows_out:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(rows_out), np.concatenate(cols_out)


def _block_group(s1g, s23g):
    """Candidates for one country group. Returns a long DataFrame of
    (s1_id, cand_id, cos_name, cos_full, cos_addr)."""
    specs = [
        ("cos_name", "name_core", dict(analyzer="char_wb", ngram_range=(2, 4)), K_NAME),
        ("cos_full", "full_n", dict(analyzer="char_wb", ngram_range=(3, 4)), K_FULL),
        ("cos_addr", "addr_n", dict(analyzer="word", ngram_range=(1, 2)), K_ADDR),
    ]
    pieces, mats = [], {}
    for name, col, vec_kw, k in specs:
        text = pd.concat([s1g[col], s23g[col]])
        vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32,
                              max_df=MAX_DF, min_df=MIN_DF, **vec_kw)
        try:
            vec.fit(text)
        except ValueError:
            # Pruning can empty the vocabulary on a small group (every term is
            # either too rare or, in a handful of records, too common). Fall
            # back to no pruning: small groups are cheap to match exhaustively.
            vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, min_df=1, **vec_kw)
            vec.fit(text)
        A, B = vec.transform(s1g[col]), vec.transform(s23g[col])
        mats[name] = (A, B)
        rows, cols = _topk_sparse(A, B, k)
        pieces.append(pd.DataFrame({
            "s1_id": s1g["entity_id"].to_numpy()[rows],
            "cand_id": s23g["entity_id"].to_numpy()[cols],
        }))
    cand = pd.concat(pieces, ignore_index=True).drop_duplicates().reset_index(drop=True)

    # All three cosines for EVERY candidate pair (not only the list that proposed it)
    ia = pd.Series(np.arange(len(s1g)), index=s1g["entity_id"].to_numpy()).loc[cand["s1_id"]].to_numpy()
    ib = pd.Series(np.arange(len(s23g)), index=s23g["entity_id"].to_numpy()).loc[cand["cand_id"]].to_numpy()
    for name, (A, B) in mats.items():
        cand[name] = np.asarray(A[ia].multiply(B[ib]).sum(axis=1)).ravel()
    return cand


def generate_candidates(s1: pd.DataFrame, s23: pd.DataFrame, by_country: bool = True) -> pd.DataFrame:
    """Run blocking for a whole split. s1/s23 must already have normalised columns."""
    if not by_country:
        return _block_group(s1, s23)
    out = []
    s23_countries = set(s23["country_n"])
    for c, s1g in s1.groupby("country_n"):
        s23g = s23[s23["country_n"] == c] if c in s23_countries else s23
        if len(s23g) == 0:
            s23g = s23
        out.append(_block_group(s1g, s23g))
    return pd.concat(out, ignore_index=True)


def blocking_report(cand: pd.DataFrame, truth: dict, s1_ids) -> dict:
    """Recall ceiling (share of true pairs present in candidates), the share of
    S1 entities whose FULL true set is covered, and candidate volume."""
    cset = cand.groupby("s1_id")["cand_id"].agg(set).to_dict()
    tot = hit = full = 0
    for s in s1_ids:
        t = truth.get(s, set())
        c = cset.get(s, set())
        tot += len(t)
        hit += len(t & c)
        full += t <= c
    n = len(list(s1_ids))
    return {
        "pair_recall_ceiling": hit / tot if tot else 1.0,
        "entity_full_cover": full / n if n else 1.0,
        "avg_cands_per_s1": len(cand) / max(n, 1),
        "total_pairs": len(cand),
    }
