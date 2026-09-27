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

MEMORY AND SPEED
----------------
At full scale there are of order 10^8 candidate pairs, so this module never
builds a pair-length frame of strings. Two rules make that work:

  * Records are addressed by integer POSITION into the side arrays. The old
    `s1.set_index("entity_id").loc[cand["s1_id"]]` materialised one row of
    object-dtype columns per pair -- hundreds of GB at full scale.
  * String similarities are computed with `rapidfuzz.process.cpdist`, which
    compares aligned pairs element-wise in C across all cores, instead of a
    Python loop making 16 calls per pair.

Context features are computed once on the (comparatively small) candidate
frame; string features are computed per chunk so the caller can consume and
discard them. See `iter_string_features`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

# Context features come from the candidate frame; string features are computed
# per chunk. The model consumes them in this order, so the order is part of the
# contract between this module and run_pipeline.
CONTEXT_FEATURES = [
    "cos_name", "cos_full", "cos_addr",
    "is_s3", "n_cands",
    "rank_name_in_s1", "gap_name_to_best", "rank_full_in_s1", "gap_full_to_best",
    "rank_s1_for_cand", "n_s1_for_cand", "gap_to_best_s1_for_cand",
]
STRING_FEATURES = [
    "nm_ratio", "nm_tsort", "nm_tset", "nm_partial", "nm_jw",
    "core_ratio", "core_tset", "core_jw", "core_first_tok_eq", "core_len_diff",
    "ad_ratio", "ad_tset", "ad_partial",
    "postal_match", "num_jacc", "num_overlap",
]
FEATURES = CONTEXT_FEATURES + STRING_FEATURES

# (output name, source column, rapidfuzz scorer). Each becomes one cpdist call.
_SCORERS = [
    ("nm_ratio", "name_n", fuzz.ratio),
    ("nm_tsort", "name_n", fuzz.token_sort_ratio),
    ("nm_tset", "name_n", fuzz.token_set_ratio),
    ("nm_partial", "name_n", fuzz.partial_ratio),
    ("nm_jw", "name_n", JaroWinkler.similarity),
    ("core_ratio", "name_core", fuzz.ratio),
    ("core_tset", "name_core", fuzz.token_set_ratio),
    ("core_jw", "name_core", JaroWinkler.similarity),
    ("ad_ratio", "addr_n", fuzz.ratio),
    ("ad_tset", "addr_n", fuzz.token_set_ratio),
    ("ad_partial", "addr_n", fuzz.partial_ratio),
]


def ragged(sets) -> tuple:
    """Pack an iterable of small int sets into (flat values, offsets).

    A Python `set` costs about 216 bytes empty, so one per record for postal
    codes and one for address numbers is ~5.4 GB across the 12.5M training
    records -- measured, and enough on its own to exhaust a 16 GB laptop before
    blocking even starts. The same information as a flat int64 array plus
    offsets costs about 0.35 GB.
    """
    flat, off = [], [0]
    for s in sets:
        flat.extend(sorted(s))
        off.append(len(flat))
    return (np.asarray(flat, dtype=np.int64),
            np.asarray(off, dtype=np.int64))


def prepare_side(df: pd.DataFrame, postal=None, nums=None) -> dict:
    """Pack one side's normalised columns into position-indexed numpy arrays.

    Done once per split. Everything downstream indexes these by integer
    position, so no pair-length string structure is ever built.

    `postal` and `nums` are (flat, offsets) pairs from `ragged`. They may be
    left out, in which case the set-valued columns are read from `df` and
    packed here -- convenient for sampled runs, but it materialises every set
    at once, so full-scale callers should pass pre-packed arrays.
    """
    core = df["name_core"].to_numpy()
    if postal is None:
        postal = ragged(df["postal"])
    if nums is None:
        nums = ragged(df["nums"])
    return {
        "name_n": df["name_n"].to_numpy(),
        "name_core": core,
        "addr_n": df["addr_n"].to_numpy(),
        # first token and length are cheap scalars, precomputed per record
        "first_tok": np.array([c.split(" ", 1)[0] if c else "" for c in core], dtype=object),
        "core_len": np.array([len(c) for c in core], dtype=np.float32),
        "postal_flat": postal[0], "postal_off": postal[1],
        "nums_flat": nums[0], "nums_off": nums[1],
        "is_s3": df["entity_id"].str.startswith("S3").to_numpy().astype(np.float32),
    }


def add_context_features(cand: pd.DataFrame, b_is_s3: np.ndarray, ib: np.ndarray) -> pd.DataFrame:
    """Rank/competition features, computed on the candidate frame itself.

    These need a groupby over all of an entity's candidates, so they cannot be
    computed chunk-wise. They are cheap: no strings, just float32 columns.
    `cos_*` must already be present (blocking attaches them).
    """
    cand["is_s3"] = b_is_s3[ib]
    g = cand.groupby("s1_id", observed=True)
    cand["n_cands"] = g["cand_id"].transform("size").astype(np.float32)
    for c, short in [("cos_name", "name"), ("cos_full", "full")]:
        cand[f"rank_{short}_in_s1"] = g[c].rank(ascending=False, method="min").astype(np.float32)
        cand[f"gap_{short}_to_best"] = (g[c].transform("max") - cand[c]).astype(np.float32)
    # Competition between S1 records for the same candidate (Source 1 is
    # deduplicated, and measured on train NO S2/S3 record belongs to two S1
    # entities, so a candidate wanted by several entities is a contested one).
    gc = cand.groupby("cand_id", observed=True)
    cand["rank_s1_for_cand"] = gc["cos_full"].rank(ascending=False, method="min").astype(np.float32)
    cand["n_s1_for_cand"] = gc["s1_id"].transform("size").astype(np.float32)
    cand["gap_to_best_s1_for_cand"] = (gc["cos_full"].transform("max") - cand["cos_full"]).astype(np.float32)
    return cand


# Process-level parallelism for the string-feature stage. This is the single
# hottest thing in the pipeline -- every full run spends most of its wall-clock
# here -- and it was pinned to one core, so a 256-core machine ran it on one.
#
# Processes, not threads: rapidfuzz 3.9.6's threaded cpdist segfaults on this
# data (Windows access violation, reproducible at 60k real pairs). And fork, not
# spawn: the side dicts hold hundreds of MB of numpy arrays, which fork shares
# copy-on-write and spawn would pickle per task. On Windows there is no fork, so
# this degrades to the serial path rather than silently pickling.
PROC_WORKERS = 1
_MIN_PARALLEL = 200_000     # below this the fork overhead outweighs the work
_SHARED = None              # set in the parent before forking; children inherit


def set_proc_workers(n: int) -> None:
    """Set the worker count for the string-feature stage (1 = serial)."""
    global PROC_WORKERS
    PROC_WORKERS = max(1, int(n))


def _feature_shard(bounds):
    """Child side: compute one slice from the inherited arrays."""
    s, e = bounds
    ia, ib, a, b = _SHARED
    return _string_features_serial(ia[s:e], ib[s:e], a, b)


def string_features(ia: np.ndarray, ib: np.ndarray, a: dict, b: dict,
                    workers: int = 1) -> np.ndarray:
    """The 16 string features for aligned pairs, parallel when configured."""
    n = len(ia)
    if PROC_WORKERS <= 1 or n < _MIN_PARALLEL or not hasattr(os, "fork"):
        return _string_features_serial(ia, ib, a, b, workers)
    import multiprocessing as mp
    global _SHARED
    _SHARED = (ia, ib, a, b)
    step = max(1, -(-n // PROC_WORKERS))
    bounds = [(s, min(s + step, n)) for s in range(0, n, step)]
    try:
        with mp.get_context("fork").Pool(len(bounds)) as pool:
            parts = pool.map(_feature_shard, bounds)
        return np.vstack(parts)
    except Exception:
        # A pool failure must never lose the run; fall back and keep going.
        return _string_features_serial(ia, ib, a, b, workers)
    finally:
        _SHARED = None


def _string_features_serial(ia: np.ndarray, ib: np.ndarray, a: dict, b: dict,
                            workers: int = 1) -> np.ndarray:
    # workers=1 is deliberate. rapidfuzz 3.9.6's multi-threaded cpdist crashes
    # with a Windows access violation on this data (reproducible at 60k pairs;
    # it survives small toy inputs, which is why it is easy to miss). Single
    # threaded it still does 60k pairs per scorer in 0.05s -- roughly 27 min
    # for the full test set across all 11 scorers -- so the parallelism is not
    # worth a segfault mid-run. Revisit only with a newer rapidfuzz.
    """The 16 string features for one chunk of aligned pairs.

    `ia`/`ib` are integer positions into the side dicts from `prepare_side`.
    Returns a (len(ia), 16) float32 array, columns in STRING_FEATURES order.
    """
    n = len(ia)
    out = np.empty((n, len(STRING_FEATURES)), dtype=np.float32)
    col = {name: i for i, name in enumerate(STRING_FEATURES)}

    for name, src, scorer in _SCORERS:
        qa, qb = a[src][ia], b[src][ib]
        out[:, col[name]] = process.cpdist(qa, qb, scorer=scorer, workers=workers,
                                           dtype=np.float32)

    # An address missing on either side means "unknown", not "different".
    miss = (a["addr_n"][ia] == "") | (b["addr_n"][ib] == "")
    for name in ("ad_ratio", "ad_tset", "ad_partial"):
        out[miss, col[name]] = -1.0

    out[:, col["core_first_tok_eq"]] = (a["first_tok"][ia] == b["first_tok"][ib]).astype(np.float32)
    out[:, col["core_len_diff"]] = np.abs(a["core_len"][ia] - b["core_len"][ib])

    # Set-valued features, read out of the ragged arrays. Still a Python loop,
    # but over 3 cheap operations per pair rather than 16 string comparisons,
    # and each record's slice holds only a handful of integers.
    apf, apo, bpf, bpo = a["postal_flat"], a["postal_off"], b["postal_flat"], b["postal_off"]
    auf, auo, buf, buo = a["nums_flat"], a["nums_off"], b["nums_flat"], b["nums_off"]
    pm = out[:, col["postal_match"]]
    nj = out[:, col["num_jacc"]]
    no = out[:, col["num_overlap"]]
    for i in range(n):
        p, q = ia[i], ib[i]
        x = apf[apo[p]:apo[p + 1]]
        y = bpf[bpo[q]:bpo[q + 1]]
        if x.size == 0 or y.size == 0:
            pm[i] = -1.0                      # one side has no code: unknown, not different
        else:
            pm[i] = float(not set(x.tolist()).isdisjoint(y.tolist()))
        x = auf[auo[p]:auo[p + 1]]
        y = buf[buo[q]:buo[q + 1]]
        if x.size or y.size:
            sx, sy = set(x.tolist()), set(y.tolist())
            inter = len(sx & sy)
            nj[i] = inter / len(sx | sy)
            no[i] = inter
        else:
            nj[i] = -1.0
            no[i] = 0.0
    return out


def iter_string_features(ia: np.ndarray, ib: np.ndarray, a: dict, b: dict,
                         chunk: int = 4_000_000):
    """Yield (start, stop, features) so the caller can predict and discard.

    Holding every string feature for every pair is the single largest memory
    cost in the pipeline; streaming them is what keeps it inside a laptop.
    """
    for start in range(0, len(ia), chunk):
        stop = min(start + chunk, len(ia))
        yield start, stop, string_features(ia[start:stop], ib[start:stop], a, b)


def build_pair_features(cand: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
    """Full in-memory build. Convenient for sampled runs and for tests; at full
    scale use `add_context_features` plus `iter_string_features` instead."""
    a, b = prepare_side(s1), prepare_side(s23)
    pos1 = pd.Series(np.arange(len(s1)), index=s1["entity_id"].to_numpy())
    pos23 = pd.Series(np.arange(len(s23)), index=s23["entity_id"].to_numpy())
    ia = pos1.loc[cand["s1_id"]].to_numpy()
    ib = pos23.loc[cand["cand_id"]].to_numpy()
    df = cand.reset_index(drop=True).copy()
    add_context_features(df, b["is_s3"], ib)
    sf = string_features(ia, ib, a, b)
    for j, name in enumerate(STRING_FEATURES):
        df[name] = sf[:, j]
    return df
