"""Exact competition scoring metric (pipeline step 9).

The score is the **macro-averaged per-source1-entity F0.5**:

    F0.5 = (1.25 * P * R) / (0.25 * P + R)

with an entity-level convention:
  * entity has **no true matches**  ->  1.0 if nothing predicted, else 0.0
  * entity **has true matches**     ->  standard P/R/F computed on its own set
                                       (empty prediction gives F0.5 = 0).

Precision is weighted 2x (beta = 0.5), i.e. a false positive hurts twice as much
as a missed true match.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping


def f0_5(precision: float, recall: float) -> float:
    """F-beta with beta = 0.5 from scalar precision and recall."""
    p, r = float(precision), float(recall)
    if p <= 0.0 and r <= 0.0:
        return 0.0
    return (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0


def entity_f0_5(true_ids: Iterable, pred_ids: Iterable) -> float:
    """F0.5 for a single source-1 entity based on its true / predicted sets.

    Implements the competition rule for zero-true-match entities.
    """
    true_set = set(true_ids or ())
    pred_set = set(pred_ids or ())
    if not true_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    inter = len(true_set & pred_set)
    precision = inter / len(pred_set)
    recall = inter / len(true_set)
    return f0_5(precision, recall)


def macro_f0_5(true_by_id: Mapping, pred_by_id: Mapping) -> float:
    """Macro-averaged per-entity F0.5 over all source-1 entities.

    ``true_by_id`` / ``pred_by_id`` map source1 id -> iterable of ref ids.
    Every id in ``true_by_id`` is scored (missing in ``pred_by_id`` -> empty).
    """
    if not true_by_id:
        return 0.0
    total = 0.0
    for eid, true_ids in true_by_id.items():
        total += entity_f0_5(true_ids, pred_by_id.get(eid, ()))
    return total / len(true_by_id)


def aggregate_pr(true_by_id: Mapping, pred_by_id: Mapping) -> dict:
    """Macro-averaged precision / recall across entities (for reporting)."""
    p_sum = r_sum = 0.0
    n = 0
    for eid, t in true_by_id.items():
        t = set(t)
        p = set(pred_by_id.get(eid, ()))
        inter = len(t & p)
        p_sum += inter / len(p) if p else 0.0
        r_sum += inter / len(t) if t else 0.0
        n += 1
    return {"precision": p_sum / n, "recall": r_sum / n, "n_entities": n}