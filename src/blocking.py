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


def _topk_sparse(A, B, k, max_cells=4e7):
    """For each row of A, return the indices and cosines of its top-k rows in B.
    A and B are L2-normalised sparse TF-IDF matrices. Computed in row chunks
    so peak memory stays about max_cells*4 bytes."""
    n_b = B.shape[0]
    k = min(k, n_b)
    step = max(1, int(max_cells // max(n_b, 1)))
    idx_out = np.zeros((A.shape[0], k), dtype=np.int64)
    sim_out = np.zeros((A.shape[0], k), dtype=np.float32)
    Bt = B.T.tocsr()
    for start in range(0, A.shape[0], step):
        S = (A[start:start + step] @ Bt).toarray().astype(np.float32)
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        rows = np.arange(S.shape[0])[:, None]
        idx_out[start:start + step] = part
        sim_out[start:start + step] = S[rows, part]
    return idx_out, sim_out


def _block_group(s1g, s23g):
    """Candidates for one country group. Returns a long DataFrame of
    (s1_id, cand_id, cos_name, cos_full, cos_addr)."""
    specs = [
        ("cos_name", "name_core", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=1), K_NAME),
        ("cos_full", "full_n", dict(analyzer="char_wb", ngram_range=(3, 4), min_df=1), K_FULL),
        ("cos_addr", "addr_n", dict(analyzer="word", ngram_range=(1, 2), min_df=1), K_ADDR),
    ]
    pieces, mats = [], {}
    for name, col, vec_kw, k in specs:
        vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **vec_kw)
        vec.fit(pd.concat([s1g[col], s23g[col]]))
        A, B = vec.transform(s1g[col]), vec.transform(s23g[col])
        mats[name] = (A, B)
        idx, _ = _topk_sparse(A, B, k)
        pieces.append(pd.DataFrame({
            "s1_id": np.repeat(s1g["entity_id"].to_numpy(), idx.shape[1]),
            "cand_id": s23g["entity_id"].to_numpy()[idx.ravel()],
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
