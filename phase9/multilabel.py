"""Multi-label emotion evaluation on GoEmotions.

GoEmotions instances have 1-5 emotion labels per text. Previous Phase 9
evaluations restricted to single-label instances. Multi-label evaluation
uses the full dataset and tests ranking capability.

Metrics per instance:
  - precision@k = |top-k predicted ∩ true EA set| / k
  - recall@k    = |top-k predicted ∩ true EA set| / |true EA set|
  - F1@k        = 2 * P * R / (P + R)
  - hit@k       = 1 if any top-k prediction is in true set else 0
"""
from __future__ import annotations

import numpy as np


def precision_at_k(ranked: list[str], true_set: set[str], k: int) -> float:
    top = set(ranked[:k])
    if not top:
        return 0.0
    return len(top & true_set) / k


def recall_at_k(ranked: list[str], true_set: set[str], k: int) -> float:
    if not true_set:
        return 0.0
    top = set(ranked[:k])
    return len(top & true_set) / len(true_set)


def f1_at_k(ranked: list[str], true_set: set[str], k: int) -> float:
    p = precision_at_k(ranked, true_set, k)
    r = recall_at_k(ranked, true_set, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def hit_at_k(ranked: list[str], true_set: set[str], k: int) -> int:
    return 1 if set(ranked[:k]) & true_set else 0


def jaccard(pred_set: set[str], true_set: set[str]) -> float:
    if not pred_set and not true_set:
        return 1.0
    union = pred_set | true_set
    if not union:
        return 0.0
    return len(pred_set & true_set) / len(union)


def aggregate_multilabel(ranked_list: list[list[str]],
                         true_sets: list[set[str]],
                         k_values: tuple[int, ...] = (1, 2, 3, 5)) -> dict:
    """Compute mean P@k / R@k / F1@k / hit@k over all instances."""
    results = {}
    for k in k_values:
        p_scores = [precision_at_k(r, t, k) for r, t in zip(ranked_list, true_sets)]
        r_scores = [recall_at_k(r, t, k) for r, t in zip(ranked_list, true_sets)]
        f_scores = [f1_at_k(r, t, k) for r, t in zip(ranked_list, true_sets)]
        h_scores = [hit_at_k(r, t, k) for r, t in zip(ranked_list, true_sets)]
        results[f"P@{k}"] = float(np.mean(p_scores))
        results[f"R@{k}"] = float(np.mean(r_scores))
        results[f"F1@{k}"] = float(np.mean(f_scores))
        results[f"hit@{k}"] = float(np.mean(h_scores))
    return results
