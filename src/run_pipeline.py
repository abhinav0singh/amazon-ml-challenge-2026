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
import json
import os
import subprocess
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import blocking_report, generate_candidates  # noqa: E402
from cv_folds import make_s1_folds  # noqa: E402
from data_io import load_split, load_truth, write_id_lists  # noqa: E402
from decide import apply_rule, check_one_to_one, cross_fitted_score, tune  # noqa: E402
from metric import macro_f05, precision_recall  # noqa: E402
from normalize import add_normalized_columns  # noqa: E402
from pair_features import (CONTEXT_FEATURES, FEATURES, STRING_FEATURES,  # noqa: E402
                           add_context_features, iter_string_features, prepare_side)

SEED = 42
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50,
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
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        log(f"  [mem] after {stage}: now {c.WorkingSetSize/1e9:.2f} GB, "
            f"peak {c.PeakWorkingSetSize/1e9:.2f} GB")
    except Exception:
        pass


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
    pos1 = pd.Series(np.arange(len(s1), dtype=np.int32), index=s1["entity_id"].to_numpy())
    pos23 = pd.Series(np.arange(len(s23), dtype=np.int32), index=s23["entity_id"].to_numpy())
    b_is_s3 = s23["entity_id"].str.startswith("S3").to_numpy().astype(np.float32)
    s23_countries = set(s23["country_n"])
    paths, n_pairs, cover_num, cover_den, hit, tot = [], 0, 0, 0, 0, 0

    for c, s1g in s1.groupby("country_n"):
        s23g = s23[s23["country_n"] == c] if c in s23_countries else s23
        if len(s23g) == 0:
            s23g = s23
        log(f"  blocking '{c}': {len(s1g)} S1 vs {len(s23g)} S2/S3")
        cand = generate_candidates(s1g, s23g, by_country=False)
        if cand.empty:
            continue
        ia = pos1.loc[cand["s1_id"]].to_numpy(np.int32)
        ib = pos23.loc[cand["cand_id"]].to_numpy(np.int32)

        if truth is not None:   # blocking quality, measured before anything is dropped
            cset = cand.groupby("s1_id")["cand_id"].agg(set)
            for s in s1g["entity_id"]:
                t = truth.get(s, set())
                got = cset.get(s, set())
                tot += len(t); hit += len(t & got)
                cover_num += (t <= got); cover_den += 1

        y = None
        if truth is not None:
            y = np.fromiter((cid in truth.get(sid, ()) for sid, cid
                             in zip(cand["s1_id"], cand["cand_id"])),
                            dtype=np.int8, count=len(cand))
        # group-by keys become the integer positions: no strings from here on
        frame = pd.DataFrame({"ia": ia, "ib": ib,
                              "cos_name": cand["cos_name"].to_numpy(np.float32),
                              "cos_full": cand["cos_full"].to_numpy(np.float32),
                              "cos_addr": cand["cos_addr"].to_numpy(np.float32)})
        frame = frame.rename(columns={"ia": "s1_id", "ib": "cand_id"})
        add_context_features(frame, b_is_s3, ib)
        frame = frame.rename(columns={"s1_id": "ia", "cand_id": "ib"})
        if y is not None:
            frame["y"] = y
            frame["fold"] = pd.Series(cand["s1_id"].map(s1_fold).to_numpy()).astype(np.int8)
        n_pairs += len(frame)
        p = os.path.join(work, f"pairs_{tag}_{c or 'na'}.parquet")
        frame.to_parquet(p, index=False)
        paths.append(p)
        del cand, frame, ia, ib, y
        gc.collect()
        _profile(f"blocking '{c}'")

    stats = {"total_pairs": int(n_pairs), "avg_cands_per_s1": n_pairs / max(len(s1), 1)}
    if truth is not None:
        stats["pair_recall_ceiling"] = hit / tot if tot else 1.0
        stats["entity_full_cover"] = cover_num / max(cover_den, 1)
    return paths, stats


def predict_paths(paths, models, a, b, fold_col=None):
    """Stream every spilled pair frame through the model(s).

    Yields nothing; returns the surviving (ia, ib, p) plus y where available.
    Only pairs with p >= P_KEEP are kept: the threshold grid starts at 0.20, so
    the rest can never be selected and carrying them is pure memory cost.
    """
    keep_ia, keep_ib, keep_p, keep_y = [], [], [], []
    for path in paths:
        df = pd.read_parquet(path)
        ctx = df[CONTEXT_FEATURES].to_numpy(np.float32)
        ia, ib = df["ia"].to_numpy(), df["ib"].to_numpy()
        folds = df[fold_col].to_numpy() if fold_col else None
        p_all = np.empty(len(df), dtype=np.float32)
        for start, stop, sf in iter_string_features(ia, ib, a, b, chunk=PRED_CHUNK):
            X = np.hstack([ctx[start:stop], sf])
            if folds is None:                 # test: average the fold models
                p_all[start:stop] = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
            else:                             # train: each pair scored out-of-fold
                blk = np.zeros(stop - start, dtype=np.float32)
                fb = folds[start:stop]
                for f, m in enumerate(models):
                    sel = fb == f
                    if sel.any():
                        blk[sel] = m.predict_proba(X[sel])[:, 1]
                p_all[start:stop] = blk
            del X, sf
        m = p_all >= P_KEEP
        keep_ia.append(ia[m]); keep_ib.append(ib[m]); keep_p.append(p_all[m])
        if "y" in df:
            keep_y.append(df["y"].to_numpy()[m])
        del df, ctx, p_all
        gc.collect()
        _profile(f"predict {os.path.basename(path)}")
    out = {"ia": np.concatenate(keep_ia), "ib": np.concatenate(keep_ib),
           "p": np.concatenate(keep_p)}
    if keep_y:
        out["y"] = np.concatenate(keep_y)
    return out


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
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    report = {}

    smoke = args.sample > 0
    folds_path = os.path.join(args.work, f"folds_sample{args.sample}.csv" if smoke else "folds.csv")
    report_path = os.path.join(args.work, f"report_sample{args.sample}.json" if smoke else "report.json")
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
    s1, s23 = add_normalized_columns(s1), add_normalized_columns(s23)
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
    a, b = prepare_side(s1), prepare_side(s23)

    log("blocking train")
    paths, bstats = block_split(s1, s23, args.work, "train", truth=truth, s1_fold=s1_fold)
    report["blocking_train"] = bstats
    log(f"blocking: {bstats}")

    # ---------- training sample ----------
    log("assembling training sample")
    sizes = [pd.read_parquet(p, columns=["y"]).shape[0] for p in paths]
    total = sum(sizes)
    frac = min(1.0, MAX_TRAIN_PAIRS / max(total, 1))
    report["train_pairs_total"] = int(total)
    report["train_pairs_sampled_frac"] = float(frac)
    rng = np.random.default_rng(SEED)
    Xs, ys, fs = [], [], []
    for p in paths:
        df = pd.read_parquet(p)
        sel = np.ones(len(df), bool) if frac >= 1.0 else rng.random(len(df)) < frac
        sub = df[sel]
        ctx = sub[CONTEXT_FEATURES].to_numpy(np.float32)
        sf = np.vstack([s for _, _, s in iter_string_features(
            sub["ia"].to_numpy(), sub["ib"].to_numpy(), a, b, chunk=PRED_CHUNK)])
        Xs.append(np.hstack([ctx, sf])); ys.append(sub["y"].to_numpy()); fs.append(sub["fold"].to_numpy())
        del df, sub, ctx, sf
        gc.collect()
    X = np.vstack(Xs); y = np.concatenate(ys); fold = np.concatenate(fs)
    del Xs, ys, fs
    gc.collect()
    log(f"training on {len(X):,} of {total:,} pairs ({frac:.1%}), positives {y.mean():.3%}")
    _profile("training sample")

    models = []
    for f in sorted(set(s1_fold.values())):
        tr, va = fold != f, fold == f
        if not va.any():
            models.append(models[-1] if models else None)
            continue
        m = fit_model(X[tr], y[tr], X[va], y[va])
        models.append(m)
        log(f"fold {f}: best_iter={m.best_iteration_}")
    del X, y, fold
    gc.collect()
    _profile("training")

    log("out-of-fold prediction")
    oof = predict_paths(paths, models, a, b, fold_col="fold")
    pairs = to_pairs_frame(oof, s1_ids_arr, s23_ids_arr, s1_fold)
    del oof
    gc.collect()
    pairs.to_parquet(os.path.join(args.work, "oof_pairs.parquet"), index=False)
    report["oof_pairs_kept"] = int(len(pairs))
    _profile("OOF")

    best_t, best_s, curve = tune(pairs, truth, s1_ids, one_to_one)
    honest, fold_ts = cross_fitted_score(pairs, truth, s1_fold, one_to_one)
    pred = apply_rule(pairs, best_t, one_to_one)
    P, R = precision_recall(pred, truth, s1_ids)
    report.update(oof_best_t=float(best_t), oof_macro_f05_at_best_t=best_s,
                  cv_macro_f05_cross_fitted=honest, fold_thresholds=[float(t) for t in fold_ts],
                  oof_pair_precision=P, oof_pair_recall=R,
                  empty_baseline_macro_f05=macro_f05({}, truth, s1_ids))
    log(f"CV macro-F0.5 (cross-fitted) = {honest:.4f} | at t={best_t}: {best_s:.4f} "
        f"| P={P:.3f} R={R:.3f} | empty baseline = {report['empty_baseline_macro_f05']:.4f}")

    # ---------- leave-one-country-out (proxy for unseen France) ----------
    if args.loco and s1["country_n"].nunique() > 1:
        cmap = s1.set_index("entity_id")["country_n"]
        pairs["c"] = pairs["s1_id"].astype(str).map(cmap)
        report["loco"] = {}
        for c in sorted(pairs["c"].dropna().unique()):
            held = pairs["c"] == c
            ids_c = cmap[cmap == c].index.tolist()
            # The threshold must come from the countries we kept. Tuning it on all
            # OOF pairs -- including the held-out country -- would contaminate the
            # one signal we have for unseen France and read optimistically.
            t_c, _, _ = tune(pairs[~held], truth, cmap[cmap != c].index.tolist(), one_to_one)
            sc = macro_f05(apply_rule(pairs[held], t_c, one_to_one), truth, ids_c)
            report["loco"][c] = {"score": sc, "threshold_from_other_countries": float(t_c)}
            log(f"LOCO: hold out '{c}', threshold {t_c} from the rest: {sc:.4f}")
        pairs.drop(columns="c", inplace=True)

    del pairs, a, b, s1, s23
    gc.collect()
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
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
    t1, t23 = add_normalized_columns(t1), add_normalized_columns(t23)
    t1_ids_arr, t23_ids_arr = t1["entity_id"].to_numpy(), t23["entity_id"].to_numpy()
    t_ids = t1_ids_arr.tolist()
    ta, tb = prepare_side(t1), prepare_side(t23)

    log("blocking test")
    tpaths, tstats = block_split(t1, t23, args.work, "test")
    report["blocking_test"] = tstats

    # candidate_pairs.tsv must be the FINAL candidate list the model scores, so it
    # is written from the same spilled frames, streamed rather than held in memory.
    log("writing candidate_pairs.tsv")
    cand_map = {}
    for p in tpaths:
        d = pd.read_parquet(p, columns=["ia", "ib"])
        for ia_v, grp in d.groupby("ia")["ib"]:
            cand_map.setdefault(t_ids[ia_v], set()).update(t23_ids_arr[grp.to_numpy()])
        del d
        gc.collect()
    write_id_lists(os.path.join(args.out, "candidate_pairs.tsv"), t_ids, cand_map, "candidate_entity_ids")
    del cand_map
    gc.collect()
    _profile("candidate_pairs")

    log("test prediction")
    tp = predict_paths(tpaths, models, ta, tb)
    tpairs = to_pairs_frame(tp, t1_ids_arr, t23_ids_arr)
    del tp
    gc.collect()
    tpairs.to_parquet(os.path.join(args.work, "test_pairs.parquet"), index=False)
    tpred = apply_rule(tpairs, best_t, one_to_one)
    write_id_lists(os.path.join(args.out, "matching_results.tsv"), t_ids, tpred, "matched_entity_ids")

    report["test"] = {
        "s1": len(t_ids), "avg_cands_per_s1": tstats["avg_cands_per_s1"],
        "pred_nonempty_share": sum(1 for s in t_ids if tpred.get(s)) / max(len(t_ids), 1),
        "oof_pred_nonempty_share": sum(1 for s in s1_ids if pred.get(s)) / len(s1_ids),
        "countries": t1["country"].value_counts().to_dict(),
    }
    log(f"test: {report['test']}")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"report written to {report_path}")
    _profile("done")

    if smoke:
        log("smoke test: skipping the official validator (a sampled output covers only "
            "some test S1 entities, so it would fail the 'every entity present' rule by design)")
        return

    validator = os.path.join(os.path.dirname(args.data.rstrip("/\\")), "utils", "validate_submission.py")
    if os.path.exists(validator):
        log("running official validator")
        subprocess.run([sys.executable, validator, "--matching", os.path.join(args.out, "matching_results.tsv"),
                        "--candidate", os.path.join(args.out, "candidate_pairs.tsv"),
                        "--test-dir", os.path.join(args.data, "test"), "--check-ids"])
    else:
        log(f"validator not found at {validator}; run utils/validate_submission.py manually")


if __name__ == "__main__":
    main()
