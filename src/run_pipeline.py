"""
run_pipeline.py: end-to-end, data -> blocking -> matching -> output.

    python src/run_pipeline.py --data <DATA> --out output --work work [--loco]
    python src/run_pipeline.py --data <DATA> --out output_smoke --sample 2000

Steps
  1. Normalise train and test records (normalize.py).
  2. Blocking, one country group at a time (blocking.py); report the recall ceiling.
  3. Pair features (pair_features.py) and labels from train_ground_truth.tsv.
  4. 5-fold GroupKFold by S1 entity (work/folds.csv) -> LightGBM out-of-fold
     (OOF) pair probabilities.
  5. Decision rule tuned on OOF for macro F0.5; honest cross-fitted score reported.
  6. Optional --loco: train on one country, score on another (unseen-France proxy).
  7. Test: blocking -> features -> average of the 5 fold models -> decision rule
     -> output/matching_results.tsv + output/candidate_pairs.tsv.
Everything printed is also saved to work/report.json for the tracker and docs.

MEMORY
------
At full scale there are of order 10^8 candidate pairs, which cannot be held as
one feature matrix. Three things keep this inside a laptop:

  * One country group is processed at a time, and its candidate frame is spilled
    to parquet. Disk stands in for RAM.
  * Records are addressed by integer position into the split's arrays, never by
    a pair-length string column.
  * String features are streamed in chunks; we predict on a chunk and discard it.
    Only (s1, cand, p) survives, and only for pairs that could ever be selected.

`_profile()` prints the peak RSS after each stage so a run that is heading for
trouble says so early rather than dying at hour two.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import blocking_report, generate_candidates  # noqa: E402
from cv_folds import make_s1_folds  # noqa: E402
from data_io import load_split, load_truth, write_id_lists  # noqa: E402
from decide import apply_rule, check_one_to_one, cross_fitted_score, tune  # noqa: E402
from metric import macro_f05, precision_recall  # noqa: E402
from normalize import add_normalized_columns  # noqa: E402
from pair_features import (CONTEXT_FEATURES, FEATURES, STRING_FEATURES,  # noqa: E402
                           add_context_features, prepare_side, string_features)

SEED = 42
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
# decide.apply_rule groups a categorical without observed=; pandas 2.2 warns that the
# default will change. With the pinned pandas the result is unchanged (unobserved
# entities get an empty set = an empty prediction), and the warning fires on every
# threshold evaluation -- hundreds of lines that would bury the run log.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*observed=False.*")
# learning_rate 0.1, not 0.05 -- MEASURED 26 Sep on 3M real candidate pairs (fold 0
# held out): lr 0.1 stopped at 859 rounds, val logloss 0.01615, AP 0.99357; lr 0.05
# stopped at 1702 rounds, logloss 0.01602, AP 0.99367. Same quality within 0.0001 AP,
# half the trees -- and prediction cost is proportional to trees: the test side
# averages 5 fold models over ~125M pairs, ~5 h at 2000 trees on 8 cores. At 16M
# training rows lr 0.05 would likely hit the round cap. 2000 rounds leaves early
# stopping in charge (2.3x what lr 0.1 needed at 2.4M rows).
LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=2000, random_state=SEED, verbose=-1, n_jobs=-1)

# Uniformly sampled training pairs. Uniform, not class-balanced: resampling would
# shift the predicted probabilities, and the decision threshold is chosen on those
# probabilities. At ~5.8% positives a 20M sample still holds over a million of them.
MAX_TRAIN_PAIRS = 20_000_000
# Pairs below this can never be selected -- the threshold grid starts at 0.20 --
# so they are dropped at prediction time instead of being carried in memory.
P_KEEP = 0.15
PRED_CHUNK = 4_000_000

# Per-worker resident memory for process-parallel blocking (--block-workers), used
# only to cap the worker count so RAM is never oversubscribed. A worker holds one
# country's slice of the normalisation cache, the three TF-IDF views for that group
# (the US full-haystack diagnostic held ~6-8 GB with only 10k S1), ~110M pre-cap
# pairs for the US group, and the ~98M-row capped frame with context features.
# Estimated ~15-17 GB for US at cap 80, so 16 GB: a 32 GB machine then blocks
# SERIALLY (the proven path, ~21-23 GB peak) and only a ~64 GB machine runs two
# or three groups at once. The first parallel run logs each worker's true peak
# RSS; replace this with that measurement. AMLC_BLOCK_WORKER_GB overrides it (only
# for testing the parallel path on a small dataset -- never lower it for a full run).
BLOCK_WORKER_GB = float(os.environ.get("AMLC_BLOCK_WORKER_GB", "16.0"))

# Total RAM below which a full-scale run is expected to swap. The 25 Sep run on a
# 16 GB laptop spent ~6.5 h assembling a 20M-pair training sample that takes minutes
# of CPU (measured: 55k pairs/s/core for all 16 string features) -- the machine was
# paging. The pre-flight check warns loudly below this.
MIN_RAM_GB = 30.0

# Parquet row-group size for the spilled pair frames, so every later stage can
# stream them in bounded batches instead of loading a 50M-row group at once.
ROW_GROUP = 1_000_000

# Columns only the normaliser needs. Dropping them after normalisation frees about
# a third of the frame memory (~2.8 GB on train) -- measured: S1 1.55 GB -> ~1.05 GB.
RAW_COLS = ["business_name", "business_address", "country"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _profile(stage):
    """Peak resident memory so far, printed after each stage."""
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        # Declare the handle types: without restype/argtypes the -1 pseudo-handle
        # from GetCurrentProcess is marshalled as a truncated int, the call fails
        # silently, and the struct stays zero (which is why the [mem] lines read
        # 0.00 GB before this fix).
        k, ps = ctypes.windll.kernel32, ctypes.windll.psapi
        k.GetCurrentProcess.restype = wt.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
        log(f"  [mem] after {stage}: now {c.WorkingSetSize/1e9:.2f} GB, "
            f"peak {c.PeakWorkingSetSize/1e9:.2f} GB")
    except Exception:
        try:                                    # Linux / cloud VMs
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            log(f"  [mem] after {stage}: peak {peak:.2f} GB, free {_free_gb():.1f} GB")
        except Exception:
            pass


def normalise_compact(df, chunk=1_000_000):
    """Normalise in row chunks, packing the set-valued columns away as we go.

    `add_normalized_columns` gives each record a Python `set` for postal codes
    and one for address numbers. At 216 bytes per empty set that is ~5.4 GB
    across the 12.5M training records -- measured, and on its own enough to
    exhaust a 16 GB laptop before blocking starts. Here each chunk's sets are
    converted to ragged int arrays and dropped immediately, so at most `chunk`
    of them exist at once.

    Returns (frame without the set columns, postal ragged, nums ragged).
    """
    from pair_features import ragged
    parts, pflat, poff, nflat, noff = [], [], [0], [], [0]
    for i in range(0, len(df), chunk):
        d = add_normalized_columns(df.iloc[i:i + chunk])
        for s in d["postal"]:
            pflat.extend(sorted(int(v) for v in s))
            poff.append(len(pflat))
        for s in d["nums"]:
            nflat.extend(sorted(int(v) for v in s))
            noff.append(len(nflat))
        # blocking needs one representative postal code as a key; keep it as an
        # int32 (0 = none) rather than the set, which is what costs the memory
        d["postal1"] = d["postal"].map(lambda s: min((int(v) for v in s), default=0)).astype(np.int32)
        parts.append(d.drop(columns=["postal", "nums"]))
        del d
        gc.collect()
    out = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    del parts
    gc.collect()
    return (out,
            (np.asarray(pflat, dtype=np.int64), np.asarray(poff, dtype=np.int64)),
            (np.asarray(nflat, dtype=np.int64), np.asarray(noff, dtype=np.int64)))


def _normalize_version():
    """Hash of normalize.py, used to invalidate the normalisation cache.

    Without this a cache built by older normalisation code is reused silently,
    and every number after it is computed from text the current code would not
    produce. Nothing in the output would hint at it.
    """
    import hashlib
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalize.py")
    with open(src, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_or_normalise(df, work, tag, data_key=""):
    """Normalise, or reload a cached normalisation from a previous run.

    Normalising 12.5M records takes ~20 minutes and is entirely deterministic,
    so a crash later in the pipeline should not cost it twice. Two runs were
    already lost that way on 25 Sep.

    The cache is valid only for the same normalize.py AND the same input data
    (`data_key`): the cache name carries only the split and sample size, so
    without the data check a run on a different dataset (e.g. a mini dry-run
    copy) in the same --work folder would silently reuse the wrong frames.
    """
    fp = os.path.join(work, f"norm_{tag}.parquet")
    fz = os.path.join(work, f"norm_{tag}.npz")
    ver = _normalize_version() + "|" + data_key
    if os.path.exists(fp) and os.path.exists(fz):
        z = np.load(fz)
        cached = str(z["ver"]) if "ver" in z else "<none>"
        if cached == ver:
            log(f"  reusing cached normalisation for '{tag}'")
            # skip the raw text columns at read time: they are dropped right after
            # anyway, and reading them first is a ~2.8 GB transient on train
            cols = [c for c in pq.ParquetFile(fp).schema_arrow.names if c not in RAW_COLS]
            d = pd.read_parquet(fp, columns=cols)
            return d, (z["pf"], z["po"]), (z["nf"], z["no"])
        # A cache built by different normalisation code is silently wrong: every
        # number downstream would be computed from text the current code would
        # not produce, with nothing to show for it. Rebuild rather than reuse.
        log(f"  cache for '{tag}' is from different normalize.py or data; re-normalising")
    d, p, nm = normalise_compact(df)
    try:
        d.to_parquet(fp, index=False)
        np.savez(fz, pf=p[0], po=p[1], nf=nm[0], no=nm[1], ver=ver)
        log(f"  cached normalisation for '{tag}'")
    except Exception as e:      # a cache failure must never kill the run
        log(f"  could not cache normalisation ({e}); continuing")
    return d, p, nm


def fit_model(X_tr, y_tr, X_va, y_va):
    """One LightGBM matcher with early stopping on the validation fold."""
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def subsample(s1, s23, n, seed, truth=None):
    """SMOKE TEST ONLY. Cut the split down to `n` Source-1 entities.

    Keeps the S2/S3-records-per-S1-entity density of the full split, so the
    pipeline sees realistically sized candidate pools per entity:
      * train (truth given): keep every true match of the kept entities, then
        top up with random other S2/S3 records to reach the original density.
      * test (no truth): keep a random S1 sample and thin S2/S3 by the same rate.

    The resulting scores are NOT comparable to a full run and must never be
    quoted as CV. The haystack is smaller, so blocking recall and precision are
    both optimistic.
    """
    rng = np.random.default_rng(seed)
    ids = s1["entity_id"].to_numpy()
    if n >= len(ids):
        return s1, s23
    density = len(s23) / len(s1)
    keep = rng.choice(ids, size=n, replace=False)
    s1s = s1[s1["entity_id"].isin(set(keep))]
    if truth is None:
        pool = s23["entity_id"].to_numpy()
        take = min(len(pool), int(round(n * density)))
        wanted = set(rng.choice(pool, size=take, replace=False))
    else:
        wanted = set()
        for s in keep:
            wanted |= truth.get(s, set())
        short = int(round(n * density)) - len(wanted)
        if short > 0:
            others = s23.loc[~s23["entity_id"].isin(wanted), "entity_id"].to_numpy()
            take = min(len(others), short)
            wanted |= set(rng.choice(others, size=take, replace=False))
    return s1s, s23[s23["entity_id"].isin(wanted)]


def block_split(s1, s23, work, tag, truth=None, s1_fold=None):
    """Block one split country group by country group, spilling each group's
    candidates to parquet. Returns (parquet paths, blocking stats).

    Candidates carry integer positions into the split's own frames, so the
    string features can be recomputed later without re-blocking and without a
    pair-length string column ever existing.
    """
    idx1 = pd.Index(s1["entity_id"].to_numpy())
    idx23 = pd.Index(s23["entity_id"].to_numpy())
    b_is_s3 = s23["entity_id"].str.startswith("S3").to_numpy().astype(np.float32)
    s23_countries = set(s23["country_n"])
    # Per-POSITION fold and |truth|, so labels and recall stats never need a
    # per-pair string lookup beyond the one membership test for y.
    fold_by_pos = s1["entity_id"].map(s1_fold).to_numpy() if s1_fold is not None else None
    ntrue_by_pos = (np.fromiter((len(truth.get(s, ())) for s in s1["entity_id"]),
                                dtype=np.int64, count=len(s1)) if truth is not None else None)
    paths, n_pairs, cover_num, cover_den, hit, tot = [], 0, 0, 0, 0, 0

    for c, s1g in s1.groupby("country_n"):
        s23g = s23[s23["country_n"] == c] if c in s23_countries else s23
        if len(s23g) == 0:
            s23g = s23
        log(f"  blocking '{c}': {len(s1g)} S1 vs {len(s23g)} S2/S3")
        cand = generate_candidates(s1g, s23g, by_country=False)
        if cand.empty:
            continue
        # get_indexer returns positions only; pos.loc[labels] also built a label
        # index as long as the pair frame (~0.8 GB per 100M pairs).
        ia = idx1.get_indexer(cand["s1_id"]).astype(np.int32)
        ib = idx23.get_indexer(cand["cand_id"]).astype(np.int32)
        if (ia < 0).any() or (ib < 0).any():
            raise AssertionError(f"blocking '{c}': candidate id not found in its split")

        y = None
        if truth is not None:
            y = np.fromiter((cid in truth.get(sid, ()) for sid, cid
                             in zip(cand["s1_id"], cand["cand_id"])),
                            dtype=np.int8, count=len(cand))
            # Blocking quality, before anything is dropped. Candidate pairs are
            # unique, so an entity's labelled hits are exactly |truth & candidates|
            # -- the same numbers the old per-entity Python sets gave, without
            # holding every candidate id in a set.
            g1 = idx1.get_indexer(s1g["entity_id"])
            hits = np.bincount(ia[y == 1], minlength=len(s1))[g1]
            nt = ntrue_by_pos[g1]
            tot += int(nt.sum()); hit += int(hits.sum())
            cover_num += int((hits == nt).sum()); cover_den += len(s1g)
        cos = {k: cand[k].to_numpy(np.float32) for k in ("cos_name", "cos_full", "cos_addr")}
        del cand                # entity-id strings are not needed past this point
        gc.collect()
        # group-by keys are the integer positions: no strings from here on
        frame = pd.DataFrame({"s1_id": ia, "cand_id": ib, **cos})
        add_context_features(frame, b_is_s3, ib)
        frame = frame.rename(columns={"s1_id": "ia", "cand_id": "ib"})
        if y is not None:
            frame["y"] = y
            frame["fold"] = fold_by_pos[ia].astype(np.int8)
        n_pairs += len(frame)
        p = os.path.join(work, f"pairs_{tag}_{c or 'na'}.parquet")
        frame.to_parquet(p, index=False, row_group_size=ROW_GROUP)
        paths.append(p)
        del frame, ia, ib, y, cos
        gc.collect()
        _profile(f"blocking '{c}'")

    stats = {"total_pairs": int(n_pairs), "avg_cands_per_s1": n_pairs / max(len(s1), 1)}
    if truth is not None:
        stats["pair_recall_ceiling"] = hit / tot if tot else 1.0
        stats["entity_full_cover"] = cover_num / max(cover_den, 1)
    return paths, stats


def _free_gb():
    """Available physical RAM in GB (Windows GlobalMemoryStatusEx, no dependency)."""
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1e9
    except Exception:
        pass
    try:                        # Linux / cloud VMs
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 0.0                  # unknown -> assume tight, so the planner stays serial


def _peak_rss_gb():
    """This process's peak working set in GB, for the per-worker memory report."""
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        k, ps = ctypes.windll.kernel32, ctypes.windll.psapi
        k.GetCurrentProcess.restype = wt.HANDLE   # else the pseudo-handle truncates
        ps.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 1e9
    except Exception:
        pass
    try:                                        # Linux: ru_maxrss is in KB
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return 0.0


def _load_norm_cache(work, tag):
    """Read a normalisation cache back from disk (the cache-hit half of
    load_or_normalise, without the raw frame it would otherwise need). Used to
    reload the split frames in the PARENT after parallel blocking, once the
    workers -- which read their slices straight from the same cache -- are done."""
    d = _slim(pd.read_parquet(os.path.join(work, f"norm_{tag}.parquet")))
    z = np.load(os.path.join(work, f"norm_{tag}.npz"))
    return d, (z["pf"], z["po"]), (z["nf"], z["no"])


def _slim(df):
    """Drop the raw text columns once normalised: nothing downstream reads them,
    and they are about a third of the frame's memory."""
    return df.drop(columns=[c for c in RAW_COLS if c in df.columns])


def _block_group_worker(country, s1_pq, s23_pq, out_dir, tag, truth_sub, fold_sub):
    """Block ONE country group in a separate process and spill its pair frame.

    This reproduces exactly what block_split's per-group body does, but the
    worker owns no large parent state: it reads its own country slice from the
    normalisation cache parquet, so nothing large is ever pickled across the
    process boundary. The output parquet is byte-for-position identical to the
    serial path -- same GLOBAL integer positions, same columns, same order --
    which is the whole correctness requirement, since downstream stages index
    the full-split arrays by those positions.

    The one subtlety: the cache slice gives LOCAL row positions, but the stored
    ia/ib must be GLOBAL (positions into the full split, matching s1_ids_arr).
    gpos maps local -> global; context-feature VALUES are position-invariant
    (grouping is a bijection within a group and is_s3 is per-record), so they
    are computed locally and only ia/ib are remapped.

    Returns a stats dict; path is None when the group produced no candidates.
    """
    import pyarrow.parquet as pq

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from blocking import generate_candidates
    from pair_features import add_context_features

    want = ["entity_id", "country_n", "name_core", "full_n", "addr_n", "postal1"]
    have = set(pq.ParquetFile(s1_pq).schema.names)
    cols = [c for c in want if c in have]

    def slice_of(path, allow_fallback):
        cn = pd.read_parquet(path, columns=["country_n"])["country_n"].to_numpy()
        mask = cn == country
        if not mask.any() and allow_fallback:      # country absent on this side
            return pd.read_parquet(path, columns=cols).reset_index(drop=True), \
                   np.arange(len(cn), dtype=np.int64)
        g = pd.read_parquet(path, columns=cols,
                            filters=[("country_n", "==", country)]).reset_index(drop=True)
        return g, np.flatnonzero(mask).astype(np.int64)

    s1g, gpos1 = slice_of(s1_pq, allow_fallback=False)
    s23g, gpos23 = slice_of(s23_pq, allow_fallback=True)

    st = {"country": country, "path": None, "n_pairs": 0,
          "hit": 0, "tot": 0, "cover_num": 0, "cover_den": 0, "peak_gb": 0.0}

    cand = generate_candidates(s1g, s23g, by_country=False)
    if cand.empty:
        st["peak_gb"] = _peak_rss_gb()
        return st

    ia_l = pd.Index(s1g["entity_id"].to_numpy()).get_indexer(cand["s1_id"])
    ib_l = pd.Index(s23g["entity_id"].to_numpy()).get_indexer(cand["cand_id"])
    if (ia_l < 0).any() or (ib_l < 0).any():
        raise AssertionError(f"worker '{country}': candidate id not found in its slice")
    ia_g = gpos1[ia_l].astype(np.int32)     # int32 to match the serial path exactly
    ib_g = gpos23[ib_l].astype(np.int32)
    b_is_s3 = s23g["entity_id"].str.startswith("S3").to_numpy().astype(np.float32)

    y = None
    if truth_sub is not None:
        y = np.fromiter((cid in truth_sub.get(sid, ()) for sid, cid
                         in zip(cand["s1_id"], cand["cand_id"])), dtype=np.int8, count=len(cand))
        # blocking quality from the labels, exactly as in block_split
        hits = np.bincount(ia_l[y == 1], minlength=len(s1g))
        nt = np.fromiter((len(truth_sub.get(s, ())) for s in s1g["entity_id"]),
                         dtype=np.int64, count=len(s1g))
        st["cover_den"] = len(s1g)
        st["tot"] = int(nt.sum()); st["hit"] = int(hits.sum())
        st["cover_num"] = int((hits == nt).sum())
    cos = {k: cand[k].to_numpy(np.float32) for k in ("cos_name", "cos_full", "cos_addr")}
    del cand
    gc.collect()

    frame = pd.DataFrame({"s1_id": ia_g, "cand_id": ib_g, **cos})
    # is_s3 is looked up with LOCAL ib into a LOCAL flag array (same record, same
    # value); the rank/competition groupbys use the frame's own s1_id/cand_id,
    # which hold GLOBAL positions -- identical grouping to the serial path.
    add_context_features(frame, b_is_s3, ib_l)
    frame = frame.rename(columns={"s1_id": "ia", "cand_id": "ib"})
    if y is not None:
        frame["y"] = y
        fold_local = s1g["entity_id"].map(fold_sub).to_numpy()
        frame["fold"] = fold_local[ia_l].astype(np.int8)

    st["n_pairs"] = len(frame)
    p = os.path.join(out_dir, f"pairs_{tag}_{country or 'na'}.parquet")
    frame.to_parquet(p, index=False, row_group_size=ROW_GROUP)
    st["path"] = p
    st["peak_gb"] = _peak_rss_gb()
    return st


def _plan_block_workers(requested, n_groups):
    """Effective worker count: never more than groups, never more than RAM allows.

    Falls back to 1 (i.e. the serial block_split) when the machine is tight or
    there is only one group, so parallelism is opt-in AND self-limiting. The RAM
    cap uses BLOCK_WORKER_GB, a conservative estimate -- replace it with the
    measured per-worker peak once a real run has reported one."""
    if requested <= 1 or n_groups <= 1:
        return 1
    free = _free_gb()
    ram_cap = max(1, int(free / BLOCK_WORKER_GB))
    w = min(requested, n_groups, ram_cap)
    log(f"  block workers: requested {requested}, groups {n_groups}, free RAM "
        f"{free:.1f} GB (~{BLOCK_WORKER_GB} GB/worker -> cap {ram_cap}) -> using {w}")
    if w <= 1:
        log("  falling back to serial blocking (RAM too tight for >1 worker)")
    return w


def block_split_parallel(work, tag, s1_pq, s23_pq, countries, s1_len,
                         truth_by_country, fold_by_country, max_workers):
    """Process-parallel counterpart of block_split: one worker per country group.

    Each worker loads its slice from the cache and writes its own pairs_*.parquet
    (§9), so the parent holds no split frames during this window -- the caller is
    expected to have released them first and to reload from the same cache after.
    Stats are aggregated to match block_split's return shape exactly."""
    paths = []
    agg = {"n_pairs": 0, "hit": 0, "tot": 0, "cover_num": 0, "cover_den": 0}
    peak = 0.0
    # spawn on every platform: it is the mode tested on Windows, and Linux's default
    # fork can deadlock a child after the parent has initialised OpenMP (LightGBM).
    import multiprocessing
    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = {ex.submit(_block_group_worker, c, s1_pq, s23_pq, work, tag,
                          None if truth_by_country is None else truth_by_country.get(c),
                          None if fold_by_country is None else fold_by_country.get(c)): c
                for c in countries}
        for fut in as_completed(futs):
            st = fut.result()
            if st["path"]:
                paths.append(st["path"])
            for k in agg:
                agg[k] += st[k]
            peak = max(peak, st["peak_gb"])
            log(f"  [parallel] '{st['country']}' done: {st['n_pairs']:,} pairs, "
                f"worker peak {st['peak_gb']:.2f} GB")
    log(f"  [parallel] max worker peak RSS {peak:.2f} GB across {max_workers} workers "
        f"(x{max_workers} concurrent -> budget ~{peak * max_workers:.1f} GB)")
    stats = {"total_pairs": int(agg["n_pairs"]), "avg_cands_per_s1": agg["n_pairs"] / max(s1_len, 1)}
    if truth_by_country is not None:
        stats["pair_recall_ceiling"] = agg["hit"] / agg["tot"] if agg["tot"] else 1.0
        stats["entity_full_cover"] = agg["cover_num"] / max(agg["cover_den"], 1)
    return paths, stats


def predict_paths(paths, models, a, b, fold_col=None):
    """Stream every spilled pair frame through the model(s).

    Yields nothing; returns the surviving (ia, ib, p) plus y where available.
    Only pairs with p >= P_KEEP are kept: the threshold grid starts at 0.20, so
    the rest can never be selected and carrying them is pure memory cost.
    """
    keep_ia, keep_ib, keep_p, keep_y = [], [], [], []
    for path in paths:
        has_y = "y" in pq.ParquetFile(path).schema_arrow.names
        n_done, t0 = 0, time.time()
        # Streamed in bounded batches: the whole US group is ~50M rows, and loading
        # it at once is what pushed the 25 Sep run into swap.
        for df in _iter_frames(path):
            ia, ib = df["ia"].to_numpy(), df["ib"].to_numpy()
            X = np.hstack([df[CONTEXT_FEATURES].to_numpy(np.float32),
                           string_features(ia, ib, a, b)])
            if fold_col is None:              # test: average the fold models
                p_all = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0).astype(np.float32)
            else:                             # train: each pair scored out-of-fold
                p_all = np.zeros(len(df), dtype=np.float32)
                fb = df[fold_col].to_numpy()
                for f, m in enumerate(models):
                    sel = fb == f
                    if sel.any():
                        p_all[sel] = m.predict_proba(X[sel])[:, 1]
            keep = p_all >= P_KEEP
            keep_ia.append(ia[keep]); keep_ib.append(ib[keep]); keep_p.append(p_all[keep])
            if has_y:
                keep_y.append(df["y"].to_numpy()[keep])
            n_done += len(df)
            del df, X, p_all
        log(f"  predicted {os.path.basename(path)}: {n_done:,} pairs in {time.time()-t0:.0f}s")
        gc.collect()
        _profile(f"predict {os.path.basename(path)}")
    out = {"ia": np.concatenate(keep_ia), "ib": np.concatenate(keep_ib),
           "p": np.concatenate(keep_p)}
    if keep_y:
        out["y"] = np.concatenate(keep_y)
    return out


def _iter_frames(path, batch=PRED_CHUNK, columns=None):
    """Yield a spilled pair parquet as bounded pandas batches."""
    for rb in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=columns):
        yield rb.to_pandas()


def _n_rows(path):
    return pq.ParquetFile(path).metadata.num_rows


class SavedModel:
    """A fold model reloaded from disk, with the predict_proba the pipeline uses.

    Every model -- freshly trained or resumed -- is saved at its best iteration
    and predicted through this class, so a resumed run scores exactly as a
    fresh one would. Booster.predict on a binary objective returns P(match)."""

    def __init__(self, path):
        self.path = path
        self.booster = lgb.Booster(model_file=path)

    def predict_proba(self, X):
        p = self.booster.predict(X)
        return np.column_stack([1.0 - p, p])


# ---------------------------------------------------------------------------
# Resumable stages. A full run is many hours; a crash, an OOM or a cloud session
# timeout late in the run must not cost the stages before it. Each stage writes
# its outputs, then a small marker holding a SIGNATURE of the code and settings
# that produced them. Rerunning the same command reuses a stage only when its
# marker exists, the signature matches and every file is present -- so a stale
# artifact from different code or data is never reused silently.
# ---------------------------------------------------------------------------
# Bump when the logic in THIS file changes what a stage writes. The imported
# modules (normalize, blocking, pair_features) are hashed automatically.
STAGE_LOGIC_VERSION = "2026-09-26a"


def _src_hash(*names):
    """Content hash of source files in src/, so editing them invalidates stages."""
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for n in names:
        with open(os.path.join(here, n), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def _data_sig(data_dir, split):
    """Size plus a hash of the first and last MB of each input TSV: cheap on
    GB-sized files, and enough to notice a different or re-extracted dataset."""
    out = []
    d = os.path.join(data_dir, split)
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not name.endswith(".tsv"):
            continue
        size = os.path.getsize(p)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read(1 << 20))
            if size > (1 << 20):
                f.seek(max(size - (1 << 20), 0))
                h.update(f.read(1 << 20))
        out.append([name, size, h.hexdigest()[:12]])
    return out


def _sig(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _stage_file(work, name):
    return os.path.join(work, f"stage_{name}.json")


def _stage_load(work, name, sig):
    """The stage's saved marker if it can be trusted, else None."""
    p = _stage_file(work, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            m = json.load(f)
    except Exception:
        return None
    if m.get("sig") != sig:
        log(f"  stage '{name}': saved outputs are from different code/settings -> recomputing")
        return None
    missing = [x for x in m.get("files", []) if not os.path.exists(x)]
    if missing:
        log(f"  stage '{name}': {len(missing)} saved file(s) missing -> recomputing")
        return None
    log(f"  stage '{name}': RESUMING from outputs saved {m.get('when', '?')}")
    return m


def _stage_clear(work, name):
    """Remove a marker BEFORE recomputing, so a crash mid-stage can never leave a
    marker pointing at half-written files."""
    p = _stage_file(work, name)
    if os.path.exists(p):
        os.remove(p)


def _stage_save(work, name, sig, files, data):
    m = {"sig": sig, "files": list(files), "data": data,
         "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = _stage_file(work, name) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1, default=str)
    os.replace(tmp, _stage_file(work, name))


def _total_ram_gb():
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 1e9
    except Exception:
        try:                                   # Linux / cloud VMs
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        except Exception:
            return float("nan")


def train_fold_models(paths, a, b, s1_fold, work, tag):
    """Assemble the uniform training sample by STREAMING the spilled pair frames,
    then fit one LightGBM per fold and save each at its best iteration.

    X is preallocated for the expected sample size (+1% slack, ~50x the binomial
    standard deviation) rather than stacked from pieces, which would briefly hold
    it twice -- ~4.5 GB instead of ~2.2 GB at 20M rows."""
    total = sum(_n_rows(p) for p in paths)
    frac = min(1.0, MAX_TRAIN_PAIRS / max(total, 1))
    cap = total if frac >= 1.0 else int(total * frac * 1.01) + 10_000
    nctx = len(CONTEXT_FEATURES)
    X = np.empty((cap, len(FEATURES)), dtype=np.float32)
    y = np.empty(cap, dtype=np.int8)
    fold = np.empty(cap, dtype=np.int8)
    rng = np.random.default_rng(SEED)
    n, t0 = 0, time.time()
    log(f"assembling training sample: {total:,} pairs, sampling {frac:.1%}")
    for p in paths:
        for df in _iter_frames(p):
            sel = np.ones(len(df), bool) if frac >= 1.0 else rng.random(len(df)) < frac
            sub = df[sel]
            k = len(sub)
            if not k:
                continue
            if n + k > len(X):                  # practically unreachable; grow, never truncate
                extra = max(k, len(X) // 10)
                X = np.concatenate([X, np.empty((extra, X.shape[1]), np.float32)])
                y = np.concatenate([y, np.empty(extra, np.int8)])
                fold = np.concatenate([fold, np.empty(extra, np.int8)])
            X[n:n + k, :nctx] = sub[CONTEXT_FEATURES].to_numpy(np.float32)
            X[n:n + k, nctx:] = string_features(sub["ia"].to_numpy(), sub["ib"].to_numpy(), a, b)
            y[n:n + k] = sub["y"].to_numpy()
            fold[n:n + k] = sub["fold"].to_numpy()
            n += k
            del df, sub
    X, y, fold = X[:n], y[:n], fold[:n]
    log(f"training on {n:,} of {total:,} pairs ({frac:.1%}), positives {y.mean():.3%} "
        f"(features {time.time()-t0:.0f}s)")
    _profile("training sample")

    fold_paths, best = [], []
    for f in sorted(set(s1_fold.values())):
        tr, va = fold != f, fold == f
        if not va.any():                        # only possible in tiny smoke samples
            fold_paths.append(fold_paths[-1] if fold_paths else None)
            best.append(None)
            continue
        t1 = time.time()
        m = fit_model(X[tr], y[tr], X[va], y[va])
        mp = os.path.join(work, f"model_{tag}_fold{f}.txt")
        m.booster_.save_model(mp, num_iteration=m.best_iteration_ or None)
        fold_paths.append(mp)
        best.append(int(m.best_iteration_ or 0))
        log(f"fold {f}: best_iter={m.best_iteration_} ({time.time()-t1:.0f}s)")
        del m
        gc.collect()
    first = next(p for p in fold_paths if p)
    fold_paths = [p or first for p in fold_paths]
    info = {"fold_model_paths": fold_paths, "best_iterations": best,
            "train_pairs_total": int(total), "train_pairs_sampled_frac": float(frac),
            "train_rows": int(n), "train_positive_rate": float(y.mean())}
    del X, y, fold
    gc.collect()
    _profile("training")
    return info


def cv_stage(pairs, truth, s1_ids, s1_fold, one_to_one, country_map=None):
    """Threshold on OOF, the honest cross-fitted CV, and the LOCO check.

    The CV number is cross_fitted_score() -- the one AGENTS.md says to report.
    Per-fold scores are recovered by re-applying the thresholds it chose to each
    fold (no second copy of the tuning protocol), and checked to average back to
    its mean exactly, so the two can never silently disagree."""
    best_t, best_s, curve = tune(pairs, truth, s1_ids, one_to_one)
    honest, fold_ts = cross_fitted_score(pairs, truth, s1_fold, one_to_one)
    fold_scores = []
    for f, t in zip(sorted(set(s1_fold.values())), fold_ts):
        va = [s for s, k in s1_fold.items() if k == f]
        fold_scores.append(macro_f05(apply_rule(pairs[pairs["s1_id"].isin(va)], t, one_to_one),
                                     truth, va))
    if abs(float(np.mean(fold_scores)) - honest) > 1e-9:
        raise AssertionError(f"per-fold scores {fold_scores} do not average to CV {honest}")
    pred = apply_rule(pairs, best_t, one_to_one)
    P, R = precision_recall(pred, truth, s1_ids)
    out = dict(oof_best_t=float(best_t), oof_macro_f05_at_best_t=float(best_s),
               cv_macro_f05_cross_fitted=float(honest),
               fold_thresholds=[float(t) for t in fold_ts],
               fold_scores=[float(x) for x in fold_scores],
               oof_pair_precision=float(P), oof_pair_recall=float(R),
               empty_baseline_macro_f05=float(macro_f05({}, truth, s1_ids)),
               oof_pred_nonempty_share=sum(1 for s in s1_ids if pred.get(s)) / max(len(s1_ids), 1),
               threshold_curve=[[float(t), float(s)] for t, s in curve])

    # Leave-one-country-out: the only proxy we have for unseen France.
    if country_map is not None and country_map.nunique() > 1:
        c_of_pair = pairs["s1_id"].astype(str).map(country_map)
        out["loco"] = {}
        for c in sorted(c_of_pair.dropna().unique()):
            held = (c_of_pair == c).to_numpy()
            ids_c = country_map[country_map == c].index.tolist()
            # The threshold must come from the countries we kept. Tuning it on all
            # OOF pairs -- including the held-out country -- would contaminate the
            # one signal we have for unseen France and read optimistically.
            t_c, _, _ = tune(pairs[~held], truth, country_map[country_map != c].index.tolist(), one_to_one)
            sc = macro_f05(apply_rule(pairs[held], t_c, one_to_one), truth, ids_c)
            out["loco"][c] = {"score": float(sc), "threshold_from_other_countries": float(t_c)}
    return out


def write_candidates(path, t_ids, t23_ids, paths):
    """candidate_pairs.tsv, streamed from the spilled test pair frames.

    Same format as data_io.write_id_lists (one row per S1 id in order, sorted
    de-duplicated ids), but built by sorting integer positions instead of
    holding ~70M ids in Python sets (several GB)."""
    ia = np.concatenate([pd.read_parquet(p, columns=["ia"])["ia"].to_numpy() for p in paths])
    ib = np.concatenate([pd.read_parquet(p, columns=["ib"])["ib"].to_numpy() for p in paths])
    order = np.argsort(ia, kind="stable")
    ia, ib = ia[order], ib[order]
    bounds = np.searchsorted(ia, np.arange(len(t_ids) + 1))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for i, s in enumerate(t_ids):
            lo, hi = bounds[i], bounds[i + 1]
            ids = sorted(set(t23_ids[ib[lo:hi]].tolist())) if hi > lo else []
            f.write(f"{s}\t{','.join(ids)}\n")
    return int(len(ia))


def verify_outputs(out_dir, t_ids, t23_ids):
    """Hard checks of the two submission files against the rules in AGENTS.md
    section 3. Independent of the organisers' validator, so the run fails loudly
    here even on a machine that does not have it. Raises on any violation."""
    mpath = os.path.join(out_dir, "matching_results.tsv")
    cpath = os.path.join(out_dir, "candidate_pairs.tsv")
    s23 = set(t23_ids.tolist())
    rows = nonempty = matched = cands = 0
    with open(mpath, encoding="utf-8") as fm, open(cpath, encoding="utf-8") as fc:
        if next(fm) != "source1_entity_id\tmatched_entity_ids\n":
            raise AssertionError("matching_results.tsv: bad header")
        if next(fc) != "source1_entity_id\tcandidate_entity_ids\n":
            raise AssertionError("candidate_pairs.tsv: bad header")
        for s, lm, lc in zip(t_ids, fm, fc):
            sm, ml = lm.rstrip("\n").split("\t")
            sc, cl = lc.rstrip("\n").split("\t")
            if sm != s or sc != s:
                raise AssertionError(f"row {rows}: expected {s}, got {sm} / {sc}")
            m = ml.split(",") if ml else []
            c = cl.split(",") if cl else []
            if len(set(m)) != len(m):
                raise AssertionError(f"{s}: duplicate ids in matched list")
            cset = set(c)
            if not set(m) <= cset:
                raise AssertionError(f"{s}: matched ids not in candidate_pairs")
            if not cset <= s23:
                raise AssertionError(f"{s}: candidate id not in the test S2/S3 set")
            rows += 1; nonempty += bool(m); matched += len(m); cands += len(c)
        if next(fm, None) is not None or next(fc, None) is not None:
            raise AssertionError("extra rows after the last S1 id")
    if rows != len(t_ids):
        raise AssertionError(f"{rows} rows written, {len(t_ids)} S1 test ids expected")
    return {"rows": rows, "nonempty_rows": nonempty, "matched_ids": matched, "candidate_ids": cands}


class _Tee:
    """Write to the console and a log file at once (flushes every write, so a
    crash never loses the tail of the log)."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def __getattr__(self, name):            # encoding, isatty, fileno, ... -> the console
        return getattr(self.stream, name)


def _tee_output(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fh = open(path, "a", encoding="utf-8", buffering=1)
    fh.write(f"\n===== run started {time.strftime('%Y-%m-%d %H:%M:%S')} : {' '.join(sys.argv)} =====\n")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


def _git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True,
                               cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        return r.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(args, smoke):
    """Fail fast on anything that would waste a multi-hour run, and say loudly
    when the machine is too small. Returns a dict for the report."""
    problems = []
    for split, files in [("train", ["train_source1.tsv", "train_source2.tsv",
                                    "train_source3.tsv", "train_ground_truth.tsv"]),
                         ("test", ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"])]:
        for f in files:
            if not os.path.exists(os.path.join(args.data, split, f)):
                problems.append(f"missing {os.path.join(args.data, split, f)}")
    if problems:
        raise SystemExit("PRE-FLIGHT FAILED:\n  " + "\n  ".join(problems))
    import shutil
    ram = _total_ram_gb()
    disk = shutil.disk_usage(os.path.abspath(args.work)).free / 1e9
    info = {"ram_total_gb": round(ram, 1), "disk_free_gb": round(disk, 1),
            "cpus": os.cpu_count(), "python": sys.version.split()[0]}
    log(f"pre-flight: {info}")
    if not smoke and ram < MIN_RAM_GB:
        log(f"  !!! WARNING: {ram:.0f} GB RAM < {MIN_RAM_GB:.0f} GB. A full run on this machine "
            f"is expected to swap and take many times longer. Use a bigger machine. !!!")
    if not smoke and disk < 25:
        raise SystemExit(f"PRE-FLIGHT FAILED: only {disk:.0f} GB free in {args.work}; a full run "
                         f"writes ~15-20 GB of intermediate files. Free space or move --work.")
    if not smoke and not os.path.exists(os.path.join(args.work, "folds.csv")):
        log("  !!! WARNING: work/folds.csv not found -- it will be regenerated. It should be "
            "the committed file (git pull) so CV numbers stay comparable. !!!")
    return info


def to_pairs_frame(d, s1_ids, s23_ids, s1_fold=None):
    """Turn kept positions back into the (s1_id, cand_id, p) frame decide.py wants."""
    df = pd.DataFrame({"s1_id": pd.Categorical(s1_ids[d["ia"]]),
                       "cand_id": pd.Categorical(s23_ids[d["ib"]]),
                       "p": d["p"]})
    if "y" in d:
        df["y"] = d["y"]
    if s1_fold is not None:
        df["fold"] = df["s1_id"].astype(str).map(s1_fold).astype(np.int8)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset", help="folder containing train/ and test/")
    ap.add_argument("--out", default="output")
    ap.add_argument("--work", default="work")
    ap.add_argument("--loco", action="store_true", help="leave-one-country-out check")
    ap.add_argument("--no-one-to-one", action="store_true")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="SMOKE TEST: run on N Source-1 entities instead of the full split. "
                         "Scores from a sampled run are NOT CV and must not be reported as such.")
    ap.add_argument("--sample-seed", type=int, default=SEED)
    ap.add_argument("--skip-test", action="store_true",
                    help="stop after CV; produces no submission files")
    ap.add_argument("--block-workers", type=int, default=1, metavar="N",
                    help="blocking parallelism across country groups, using PROCESSES "
                         "(threads segfault rapidfuzz). 1 = serial (default). The count is "
                         "capped by group count and free RAM; see _plan_block_workers.")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore saved stages and recompute everything (default: resume "
                         "any stage whose saved outputs match the current code and data)")
    ap.add_argument("--log", default=None,
                    help="also append everything printed (including tracebacks) to this file")
    args = ap.parse_args()
    if args.log:
        _tee_output(args.log)
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    report = {}
    t_start = time.time()

    smoke = args.sample > 0
    # Sampled runs get their own artifact names, so a smoke test can never
    # overwrite -- or be mistaken for -- the full-scale stages.
    sfx = f"_sample{args.sample}" if smoke else ""
    folds_path = os.path.join(args.work, f"folds_sample{args.sample}.csv" if smoke else "folds.csv")
    report_path = os.path.join(args.work, f"report_sample{args.sample}.json" if smoke else "report.json")
    report["machine"] = preflight(args, smoke)
    report["git_commit"] = _git_commit()
    if args.fresh:
        for f in os.listdir(args.work):
            if f.startswith("stage_") and f.endswith(f"{sfx}.json") and (smoke or "_sample" not in f):
                os.remove(os.path.join(args.work, f))
        log("--fresh: cleared saved stages")
    code_sig = {n: _src_hash(n) for n in ("normalize.py", "blocking.py", "pair_features.py")}
    if smoke:
        log(f"*** SMOKE TEST: {args.sample} S1 entities, seed {args.sample_seed}. "
            f"Numbers are NOT comparable to a full run -- do not quote them as CV. ***")
        report["sample"] = {"n_s1": args.sample, "seed": args.sample_seed,
                            "warning": "subsampled smoke test; scores are optimistic and not CV"}

    # ---------- train ----------
    log("loading + normalising train")
    s1, s23 = load_split(args.data, "train")
    truth = load_truth(args.data)
    if smoke:
        s1, s23 = subsample(s1, s23, args.sample, args.sample_seed, truth)
        log(f"sampled train: {len(s1)} S1, {len(s23)} S2/S3")
    train_key = _sig(_data_sig(args.data, "train"), args.sample, args.sample_seed)
    s1, p1r, n1r = load_or_normalise(s1, args.work, f'train_s1{args.sample}', train_key)
    s23, p2r, n2r = load_or_normalise(s23, args.work, f'train_s23{args.sample}', train_key)
    s1, s23 = _slim(s1), _slim(s23)
    gc.collect()
    s1_ids_arr, s23_ids_arr = s1["entity_id"].to_numpy(), s23["entity_id"].to_numpy()
    s1_ids = s1_ids_arr.tolist()
    truth = {s: truth.get(s, set()) for s in s1_ids}
    report["train_sizes"] = {"s1": len(s1), "s23": len(s23)}
    report["train_singleton_share"] = float(np.mean([len(v) == 0 for v in truth.values()]))
    report["multi_s1_share"] = check_one_to_one(truth)
    one_to_one = (not args.no_one_to_one) and report["multi_s1_share"] < 0.01
    report["one_to_one_used"] = one_to_one
    log(f"singleton share {report['train_singleton_share']:.3f} | "
        f"ids matched to >1 S1: {report['multi_s1_share']:.4f} -> one_to_one={one_to_one}")
    _profile("load+normalise train")

    s1_fold = make_s1_folds(s1_ids, path=folds_path, seed=SEED)
    with open(folds_path, "rb") as f:
        folds_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    report["folds_sha256_16"] = folds_hash
    if len(s1_fold) != len(s1_ids):
        raise AssertionError(f"folds cover {len(s1_fold)} ids, the run has {len(s1_ids)}")
    log(f"folds: {len(s1_fold):,} S1 entities from {folds_path} (sha256 {folds_hash})")

    log("blocking train")
    tag = f"train{sfx}"
    sig_bt = _sig("block", tag, code_sig, STAGE_LOGIC_VERSION, _data_sig(args.data, "train"),
                  args.sample, args.sample_seed, folds_hash)
    saved = _stage_load(args.work, f"block_{tag}", sig_bt)
    workers = 1 if saved else _plan_block_workers(args.block_workers, s1["country_n"].nunique())
    if saved:
        paths, bstats = saved["files"], saved["data"]
    elif workers > 1:
        _stage_clear(args.work, f"block_{tag}")
        # Parallel: workers read their slices from the cache, so free the parent's
        # frames first (they would otherwise sit alongside every worker's slice and
        # blow the RAM budget), then reload from the same cache for the feature stage.
        s1_pq = os.path.join(args.work, f"norm_train_s1{args.sample}.parquet")
        s23_pq = os.path.join(args.work, f"norm_train_s23{args.sample}.parquet")
        tbc, fbc = {}, {}
        for c, g in s1.groupby("country_n"):
            ids = g["entity_id"].to_numpy()
            tbc[c] = {s: truth.get(s, set()) for s in ids}
            fbc[c] = {s: s1_fold.get(s) for s in ids}
        s1_len = len(s1)
        del s1, s23, p1r, n1r, p2r, n2r
        gc.collect()
        paths, bstats = block_split_parallel(args.work, tag, s1_pq, s23_pq,
                                             list(tbc.keys()), s1_len, tbc, fbc, workers)
        s1, p1r, n1r = _load_norm_cache(args.work, f"train_s1{args.sample}")
        s23, p2r, n2r = _load_norm_cache(args.work, f"train_s23{args.sample}")
        _stage_save(args.work, f"block_{tag}", sig_bt, paths, bstats)
    else:
        _stage_clear(args.work, f"block_{tag}")
        paths, bstats = block_split(s1, s23, args.work, tag, truth=truth, s1_fold=s1_fold)
        _stage_save(args.work, f"block_{tag}", sig_bt, paths, bstats)
    report["blocking_train"] = bstats
    log(f"blocking: {bstats}")
    _profile("blocking train")

    # full_n exists only for the blocking TF-IDF view; features never read it.
    s1, s23 = s1.drop(columns=["full_n"]), s23.drop(columns=["full_n"])
    gc.collect()
    a, b = prepare_side(s1, p1r, n1r), prepare_side(s23, p2r, n2r)

    # ---------- training sample + fold models ----------
    sig_m = _sig("models", sig_bt, LGB_PARAMS, MAX_TRAIN_PAIRS, SEED, FEATURES, STAGE_LOGIC_VERSION)
    saved = _stage_load(args.work, f"models_{tag}", sig_m)
    if saved:
        tinfo = saved["data"]
    else:
        _stage_clear(args.work, f"models_{tag}")
        tinfo = train_fold_models(paths, a, b, s1_fold, args.work, tag)
        _stage_save(args.work, f"models_{tag}", sig_m,
                    sorted(set(tinfo["fold_model_paths"])), tinfo)
    models = [SavedModel(p) for p in tinfo["fold_model_paths"]]
    report.update(train_pairs_total=tinfo["train_pairs_total"],
                  train_pairs_sampled_frac=tinfo["train_pairs_sampled_frac"],
                  train_rows=tinfo["train_rows"], train_positive_rate=tinfo["train_positive_rate"],
                  best_iterations=tinfo["best_iterations"])

    # ---------- out-of-fold prediction ----------
    sig_o = _sig("oof", sig_m, P_KEEP)
    oof_path = os.path.join(args.work, f"oof_pairs{sfx}.parquet")
    if _stage_load(args.work, f"oof_{tag}", sig_o):
        pairs = pd.read_parquet(oof_path)
    else:
        _stage_clear(args.work, f"oof_{tag}")
        log("out-of-fold prediction")
        oof = predict_paths(paths, models, a, b, fold_col="fold")
        pairs = to_pairs_frame(oof, s1_ids_arr, s23_ids_arr, s1_fold)
        del oof
        gc.collect()
        pairs.to_parquet(oof_path, index=False)
        _stage_save(args.work, f"oof_{tag}", sig_o, [oof_path], {"rows": int(len(pairs))})
    report["oof_pairs_kept"] = int(len(pairs))
    _profile("OOF")

    # ---------- CV: threshold, honest cross-fitted score, LOCO ----------
    sig_cv = _sig("cv", sig_o, one_to_one, bool(args.loco))
    saved = _stage_load(args.work, f"cv_{tag}", sig_cv)
    if saved:
        cv = saved["data"]
    else:
        _stage_clear(args.work, f"cv_{tag}")
        cv = cv_stage(pairs, truth, s1_ids, s1_fold, one_to_one,
                      s1.set_index("entity_id")["country_n"] if args.loco else None)
        _stage_save(args.work, f"cv_{tag}", sig_cv, [oof_path], cv)
    report.update(cv)
    best_t = cv["oof_best_t"]
    log(f"CV macro-F0.5 (cross-fitted) = {cv['cv_macro_f05_cross_fitted']:.4f} "
        f"| folds {[round(x, 4) for x in cv['fold_scores']]} "
        f"| at t={best_t}: {cv['oof_macro_f05_at_best_t']:.4f} "
        f"| P={cv['oof_pair_precision']:.3f} R={cv['oof_pair_recall']:.3f}")
    for c, v in cv.get("loco", {}).items():
        log(f"LOCO: hold out '{c}', threshold {v['threshold_from_other_countries']} "
            f"from the rest: {v['score']:.4f}")

    del pairs, a, b, s1, s23
    gc.collect()
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"CV report written -> {report_path}")
    if args.skip_test:
        log(f"--skip-test: stopping after CV. report -> {report_path}")
        return

    # ---------- test ----------
    log("test: load + normalise")
    t1, t23 = load_split(args.data, "test")
    if smoke:
        t1, t23 = subsample(t1, t23, args.sample, args.sample_seed)
        log(f"sampled test: {len(t1)} S1, {len(t23)} S2/S3 "
            f"(random sample -- test prediction rates below are NOT interpretable)")
        report["sample"]["test_caveat"] = (
            "test S2/S3 sampled without truth, so most real matches are absent; "
            "pred_nonempty_share is not comparable to oof_pred_nonempty_share")
    test_key = _sig(_data_sig(args.data, "test"), args.sample, args.sample_seed)
    t1, tp1r, tn1r = load_or_normalise(t1, args.work, f'test_s1{args.sample}', test_key)
    t23, tp2r, tn2r = load_or_normalise(t23, args.work, f'test_s23{args.sample}', test_key)
    t1, t23 = _slim(t1), _slim(t23)
    gc.collect()
    t1_ids_arr, t23_ids_arr = t1["entity_id"].to_numpy(), t23["entity_id"].to_numpy()
    t_ids = t1_ids_arr.tolist()
    test_countries = t1["country_n"].value_counts().to_dict()

    log("blocking test")
    ttag = f"test{sfx}"
    sig_btest = _sig("block", ttag, code_sig, STAGE_LOGIC_VERSION, _data_sig(args.data, "test"),
                     args.sample, args.sample_seed)
    saved = _stage_load(args.work, f"block_{ttag}", sig_btest)
    tworkers = 1 if saved else _plan_block_workers(args.block_workers, t1["country_n"].nunique())
    if saved:
        tpaths, tstats = saved["files"], saved["data"]
    elif tworkers > 1:
        _stage_clear(args.work, f"block_{ttag}")
        # Test has up to three groups (US / India / France), so it gains most from
        # parallelism. No truth on the test side -> no y, no fold, no recall stats.
        t1_pq = os.path.join(args.work, f"norm_test_s1{args.sample}.parquet")
        t23_pq = os.path.join(args.work, f"norm_test_s23{args.sample}.parquet")
        tcountries = list(t1["country_n"].unique())
        t1_len = len(t1)
        del t1, t23, tp1r, tn1r, tp2r, tn2r
        gc.collect()
        tpaths, tstats = block_split_parallel(args.work, ttag, t1_pq, t23_pq,
                                              tcountries, t1_len, None, None, tworkers)
        t1, tp1r, tn1r = _load_norm_cache(args.work, f"test_s1{args.sample}")
        t23, tp2r, tn2r = _load_norm_cache(args.work, f"test_s23{args.sample}")
        _stage_save(args.work, f"block_{ttag}", sig_btest, tpaths, tstats)
    else:
        _stage_clear(args.work, f"block_{ttag}")
        tpaths, tstats = block_split(t1, t23, args.work, ttag)
        _stage_save(args.work, f"block_{ttag}", sig_btest, tpaths, tstats)
    report["blocking_test"] = tstats
    _profile("blocking test")

    # candidate_pairs.tsv must be the FINAL candidate list the model scores, so it
    # is written from the same spilled frames the predictions come from.
    log("writing candidate_pairs.tsv")
    n_cand = write_candidates(os.path.join(args.out, "candidate_pairs.tsv"), t_ids, t23_ids_arr, tpaths)
    _profile("candidate_pairs")

    sig_tp = _sig("testpred", sig_btest, sig_m, P_KEEP)
    tp_path = os.path.join(args.work, f"test_pairs{sfx}.parquet")
    if _stage_load(args.work, f"testpred_{ttag}", sig_tp):
        tpairs = pd.read_parquet(tp_path)
    else:
        _stage_clear(args.work, f"testpred_{ttag}")
        log("test prediction")
        t1, t23 = t1.drop(columns=["full_n"]), t23.drop(columns=["full_n"])
        ta, tb = prepare_side(t1, tp1r, tn1r), prepare_side(t23, tp2r, tn2r)
        tp = predict_paths(tpaths, models, ta, tb)
        tpairs = to_pairs_frame(tp, t1_ids_arr, t23_ids_arr)
        del tp, ta, tb
        gc.collect()
        tpairs.to_parquet(tp_path, index=False)
        _stage_save(args.work, f"testpred_{ttag}", sig_tp, [tp_path], {"rows": int(len(tpairs))})
    tpred = apply_rule(tpairs, best_t, one_to_one)
    write_id_lists(os.path.join(args.out, "matching_results.tsv"), t_ids, tpred, "matched_entity_ids")

    report["test"] = {
        "s1": len(t_ids), "candidate_pairs": n_cand, "avg_cands_per_s1": tstats["avg_cands_per_s1"],
        "threshold": float(best_t),
        "pred_nonempty_share": sum(1 for s in t_ids if tpred.get(s)) / max(len(t_ids), 1),
        "oof_pred_nonempty_share": report["oof_pred_nonempty_share"],
        "countries": test_countries,
    }
    log(f"test: {report['test']}")

    if smoke:
        log("smoke test: output covers only the sampled test S1 entities, so the files are "
            "NOT a valid submission and the full-set checks are skipped by design")
        report["runtime_hours"] = round((time.time() - t_start) / 3600, 2)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return

    # ---------- hard checks + official validator + summary ----------
    log("verifying output files")
    report["output_check"] = verify_outputs(args.out, t_ids, t23_ids_arr)
    log(f"  output check PASSED: {report['output_check']}")
    validator = os.path.join(os.path.dirname(args.data.rstrip("/\\")), "utils", "validate_submission.py")
    if os.path.exists(validator):
        log("running official validator")
        rc = subprocess.run([sys.executable, validator,
                             "--matching", os.path.join(args.out, "matching_results.tsv"),
                             "--candidate", os.path.join(args.out, "candidate_pairs.tsv"),
                             "--test-dir", os.path.join(args.data, "test"), "--check-ids"]).returncode
        report["official_validator_exit_code"] = rc
        log(f"  official validator exit code {rc} ({'PASS' if rc == 0 else 'CHECK ITS OUTPUT ABOVE'})")
    else:
        report["official_validator_exit_code"] = None
        log(f"  official validator not found at {validator}; run it by hand before uploading")
    report["runtime_hours"] = round((time.time() - t_start) / 3600, 2)
    report["output_sha256"] = {n: _sha256(os.path.join(args.out, n))
                               for n in ("matching_results.tsv", "candidate_pairs.tsv")}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    write_summary(os.path.join(args.out, "RUN_SUMMARY.txt"), report)
    log(f"DONE. report -> {report_path}; summary -> {os.path.join(args.out, 'RUN_SUMMARY.txt')}")
    _profile("done")


def write_summary(path, r):
    """One human-readable page: what was run, what it scored, what to upload."""
    lines = [
        "RUN SUMMARY -- Amazon ML Challenge 2026, business entity resolution",
        f"written {time.strftime('%Y-%m-%d %H:%M:%S')} | git {r.get('git_commit')} | "
        f"runtime {r.get('runtime_hours')} h | machine {r.get('machine')}",
        "",
        f"CV macro-F0.5 (cross-fitted, the honest number) : {r.get('cv_macro_f05_cross_fitted'):.4f}",
        f"  per fold                                      : {[round(x, 4) for x in r.get('fold_scores', [])]}",
        f"  OOF pair precision / recall                   : {r.get('oof_pair_precision'):.4f} / "
        f"{r.get('oof_pair_recall'):.4f}",
        f"  threshold used for test                       : {r.get('oof_best_t')}",
    ]
    for c, v in (r.get("loco") or {}).items():
        lines.append(f"  LOCO hold-out {c:<8s}                         : {v['score']:.4f}")
    bt = r.get("blocking_train", {})
    lines += [
        f"train blocking recall ceiling / entity cover    : {bt.get('pair_recall_ceiling', float('nan')):.4f} / "
        f"{bt.get('entity_full_cover', float('nan')):.4f}  ({bt.get('avg_cands_per_s1', 0):.1f} cands/S1)",
        "",
        f"test S1 rows {r['test']['s1']:,} | candidate pairs {r['test']['candidate_pairs']:,} | "
        f"non-empty predictions {r['test']['pred_nonempty_share']:.3f} "
        f"(train OOF {r['test']['oof_pred_nonempty_share']:.3f})",
        f"test countries: {r['test']['countries']}",
        f"output check: {r.get('output_check')}",
        f"official validator exit code: {r.get('official_validator_exit_code')}",
        "",
        "FILES TO HAND TO P1 FOR UPLOAD (verify the hashes on the receiving machine):",
    ]
    for n, h in (r.get("output_sha256") or {}).items():
        lines.append(f"  {n}  sha256 {h}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
