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

# Multi-key blocking: partition into many small blocks by several complementary
# keys and match inside each, instead of ranking against a whole country group.
# Turns the cost from roughly quadratic into roughly linear. MAX_BLOCK caps the
# S2/S3 side of any one block; oversized blocks come from low-information keys,
# would dominate the runtime, and are covered by the other keys.
MULTIKEY = True
MAX_BLOCK = 20000

# Cap on candidates kept per S1 entity after the key blocks are unioned.
# Unioning several keys makes candidates per entity grow with corpus size
# (19 -> 40 -> 61 at samples 2k -> 10k -> 40k), which is cost the feature stage
# has to absorb. This IS the final candidate list, so it is also what
# candidate_pairs.tsv must contain.
#
# MEASURED at 40k: cap 40 cut candidates 60.6 -> 38.1 per entity but cost
# entity cover 0.9546 -> 0.9208. That is far too expensive -- cover is close to
# a hard cap on macro F0.5, and now that blocking cost is sub-linear we are no
# longer desperate for the saving. 80 is set so the cap does not bind at any
# scale measured so far (so it costs nothing observed) while still bounding the
# worst case at full scale. Tune with real numbers, not intuition -- see the B
# blocking issue.
MAX_CANDS = 40


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


SPECS = [
    ("cos_name", "name_core", dict(analyzer="char_wb", ngram_range=(2, 4)), "K_NAME"),
    ("cos_full", "full_n", dict(analyzer="char_wb", ngram_range=(3, 4)), "K_FULL"),
    ("cos_addr", "addr_n", dict(analyzer="word", ngram_range=(1, 2)), "K_ADDR"),
]


def _fit_views(s1g, s23g):
    """Fit the three TF-IDF views ONCE for a country group and transform both
    sides. Fitting per block would make cosines from different blocks sit on
    different scales, which would silently corrupt the cos_* features and every
    rank_/gap_ feature derived from them."""
    mats = {}
    for name, col, vec_kw, kname in SPECS:
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
        mats[name] = (vec.transform(s1g[col]), vec.transform(s23g[col]), globals()[kname])
    return mats


def _finish(s1g, s23g, mats, ia, ib):
    """Build the candidate frame from global row positions and attach all three
    cosines to every pair, not only to the view that proposed it."""
    cand = pd.DataFrame({"s1_id": s1g["entity_id"].to_numpy()[ia],
                         "cand_id": s23g["entity_id"].to_numpy()[ib]})
    for name, (A, B, _) in mats.items():
        cand[name] = np.asarray(A[ia].multiply(B[ib]).sum(axis=1)).ravel()
    return cand


def _block_group(s1g, s23g):
    """Candidates for one country group, ranking against the WHOLE group.
    Correct but roughly quadratic; kept as the reference implementation and for
    small groups. Returns (s1_id, cand_id, cos_name, cos_full, cos_addr)."""
    mats = _fit_views(s1g, s23g)
    seen = []
    for name, (A, B, k) in mats.items():
        rows, cols = _topk_sparse(A, B, k)
        seen.append(np.stack([rows, cols], axis=1))
    pairs = np.unique(np.concatenate(seen), axis=0)
    return _finish(s1g, s23g, mats, pairs[:, 0], pairs[:, 1])


def _group_positions(rows, keys):
    """Map each key value to the array of row positions carrying it, via one
    sort rather than a pandas groupby (much cheaper at these sizes). `rows` and
    `keys` are parallel arrays, so a row may appear under several keys."""
    if len(rows) == 0:
        return {}
    order = np.argsort(keys, kind="stable")
    order, sk = np.asarray(rows)[order], np.asarray(keys)[order]
    bounds = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
    return {sk[bounds[i]]: order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)}


def _postal_key(df, core_ns):
    """Postal code + a name initial, from the compact int32 column when present."""
    init = core_ns.str[:1].to_numpy()
    if "postal1" in df.columns:
        p = df["postal1"].to_numpy()
        return np.where(p > 0, p.astype(str).astype(object) + "|" + init, "")
    p = df["postal"].map(lambda s: min(s) if s else "").to_numpy()
    return np.where(p != "", p.astype(object) + "|" + init, "")


def _single(keys):
    """One key per record: (row positions, key values), blanks dropped."""
    k = np.asarray(keys, dtype=object)
    r = np.flatnonzero(k != "")
    return r, k[r]


def _rare_tokens(series, doc_freq, n_tok, prefix):
    """Index each record under its `n_tok` RAREST tokens.

    This is the workhorse key. Exact whole-name keys are brittle against the
    noise this dataset is built from -- typos, reordering, abbreviation, DBA
    names -- because any one of those breaks the whole key. Indexing by the
    rarest individual tokens only needs ONE distinctive word to survive, and
    rare tokens are both the most identifying and the cheapest to look up
    (short posting lists). Common tokens are skipped precisely because they
    would form huge blocks and carry little information.
    """
    rows, keys = [], []
    for i, s in enumerate(series):
        t = s.split()
        if not t:
            continue
        # Tie-break on the token itself. Sorting on frequency alone left ties
        # broken by set iteration order, which varies run to run with Python's
        # string hash randomisation -- identical runs produced 38,319 / 38,325 /
        # 38,331 candidate pairs. Blocking must be reproducible or no
        # before/after comparison means anything.
        for tok in sorted(set(t), key=lambda x: (doc_freq.get(x, 0), x))[:n_tok]:
            rows.append(i)
            keys.append(prefix + tok)
    return np.asarray(rows, dtype=np.int64), np.asarray(keys, dtype=object)


def _doc_freq(*series_list):
    """Token document frequency over both sides, so 'rare' means rare in the
    corpus being matched rather than in one side of it."""
    c = {}
    for series in series_list:
        for s in series:
            for tok in set(s.split()):
                c[tok] = c.get(tok, 0) + 1
    return c


def blocking_keys(df: pd.DataFrame, dfreq_name=None, dfreq_addr=None) -> dict:
    """Complementary blocking schemes, all derived from the record itself
    (no external data, no country branching). Each returns (rows, keys), and a
    record may appear under several keys of the same scheme.

    A true match survives if it shares ANY key, so the schemes are chosen to
    fail independently: rare name tokens survive reordering and suffix noise,
    the name prefix survives a mangled interior, rare address tokens survive a
    renamed business, and the postal key is decisive when a code is present.
    """
    core_ns = df["name_core"].str.replace(" ", "", regex=False)
    out = {
        "pfx5": _single(core_ns.str[:5].to_numpy()),
        # `postal1` is one representative code as an int32, 0 meaning none.
        # It replaces a per-record set: 12.5M Python sets cost ~5.4 GB, which
        # exhausted a 16 GB machine before blocking could start. Fall back to
        # the old set column if a caller has not packed it (sampled runs).
        "postal": _single(_postal_key(df, core_ns)),
    }
    if dfreq_name is not None:
        out["nametok"] = _rare_tokens(df["name_core"].to_numpy(), dfreq_name, 3, "n:")
    if dfreq_addr is not None:
        out["addrtok"] = _rare_tokens(df["addr_n"].to_numpy(), dfreq_addr, 2, "a:")
    return out


def _block_by_keys(s1: pd.DataFrame, s23: pd.DataFrame, max_block: int = MAX_BLOCK) -> pd.DataFrame:
    """O(n) candidate generation: partition both sides by each key, run the
    top-k cosine matcher inside each small block, union the results.

    Cost is driven by the sum of (block s1 size x block s23 size) rather than
    the whole country group, so it grows roughly linearly with the data instead
    of quadratically. Blocks larger than `max_block` on the S2/S3 side are
    skipped for that key -- they are the low-information keys (a very common
    prefix), they would dominate the runtime, and the other keys still cover
    those records.
    """
    mats = _fit_views(s1, s23)          # fitted once, so cosines stay comparable
    dfn = _doc_freq(s1["name_core"].to_numpy(), s23["name_core"].to_numpy())
    dfa = _doc_freq(s1["addr_n"].to_numpy(), s23["addr_n"].to_numpy())
    k1, k23 = blocking_keys(s1, dfn, dfa), blocking_keys(s23, dfn, dfa)
    seen, skipped, blocks = [], 0, 0
    for name in k1:
        ga = _group_positions(*k1[name])
        gb = _group_positions(*k23[name])
        for key, pa in ga.items():
            pb = gb.get(key)
            if pb is None or pb.size > max_block:
                skipped += pb is not None
                continue
            blocks += 1
            for _, (A, B, k) in mats.items():
                r, c = _topk_sparse(A[pa], B[pb], k)
                if r.size:
                    seen.append(np.stack([pa[r], pb[c]], axis=1))
    if not seen:
        return pd.DataFrame(columns=["s1_id", "cand_id", "cos_name", "cos_full", "cos_addr"])
    pairs = np.unique(np.concatenate(seen), axis=0)   # same pair from several keys -> once
    cand = _finish(s1, s23, mats, pairs[:, 0], pairs[:, 1])
    n_before = len(cand)
    cand = _cap_per_entity(cand, MAX_CANDS)
    print(f"    [blocking] {blocks} blocks matched, {skipped} oversized skipped "
          f"(> {max_block}), {n_before} pairs -> {len(cand)} after cap", flush=True)
    return cand


def _cap_per_entity(cand: pd.DataFrame, max_cands: int) -> pd.DataFrame:
    """Keep the best `max_cands` candidates per S1 entity, ranked by the
    strongest of the three cosine views. Ranking on the max rather than on one
    view avoids discarding a candidate that only the address view liked, which
    is exactly the renamed-business case the address view exists to catch."""
    if max_cands <= 0 or cand.empty:
        return cand
    score = cand[["cos_name", "cos_full", "cos_addr"]].max(axis=1)
    keep = score.groupby(cand["s1_id"]).rank(ascending=False, method="first") <= max_cands
    return cand[keep].reset_index(drop=True)


def generate_candidates(s1: pd.DataFrame, s23: pd.DataFrame, by_country: bool = True,
                        multikey: bool = MULTIKEY) -> pd.DataFrame:
    """Run blocking for a whole split. s1/s23 must already have normalised columns."""
    block = _block_by_keys if multikey else _block_group
    if not by_country:
        return block(s1, s23)
    out = []
    s23_countries = set(s23["country_n"])
    for c, s1g in s1.groupby("country_n"):
        s23g = s23[s23["country_n"] == c] if c in s23_countries else s23
        if len(s23g) == 0:
            s23g = s23
        out.append(block(s1g, s23g))
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
