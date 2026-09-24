"""
metric.py: the official scoring rule, re-implemented locally.

F0.5 is computed PER Source-1 entity and then macro-averaged over ALL Source-1
entities in the evaluation set, singletons included:
  - true set empty, predicted empty      -> 1.0
  - true set empty, predicted non-empty  -> 0.0
  - otherwise F0.5 = 1.25*P*R / (0.25*P + R)   (0 if P = R = 0)
"""
from __future__ import annotations


def f05_single(pred: set, truth: set) -> float:
    """F0.5 for one Source-1 entity, following the singleton convention above."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: dict, truth: dict, s1_ids=None) -> float:
    """Macro F0.5. `pred` / `truth` map s1_id -> set of matched ids.
    `s1_ids` = the entities to average over (default: every key in `truth`).
    Entities missing from `pred` count as an empty prediction."""
    ids = list(truth.keys()) if s1_ids is None else list(s1_ids)
    if not ids:
        return float("nan")
    return sum(f05_single(pred.get(i, set()), truth.get(i, set())) for i in ids) / len(ids)


def precision_recall(pred: dict, truth: dict, s1_ids=None):
    """Micro precision/recall over pairs; a diagnostic only, not the official metric."""
    ids = list(truth.keys()) if s1_ids is None else list(s1_ids)
    tp = sum(len(pred.get(i, set()) & truth.get(i, set())) for i in ids)
    npred = sum(len(pred.get(i, set())) for i in ids)
    ntrue = sum(len(truth.get(i, set())) for i in ids)
    return (tp / npred if npred else 1.0), (tp / ntrue if ntrue else 1.0)


if __name__ == "__main__":
    # Worked example from the problem statement: expected 0.714
    s = f05_single({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(s - 0.714) < 1e-3, s
    assert f05_single(set(), set()) == 1.0 and f05_single({"x"}, set()) == 0.0
    print(f"metric OK: example = {s:.3f}")
