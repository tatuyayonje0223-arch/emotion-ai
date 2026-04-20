"""Majority-class baseline — the control missing from Phase 9.1-9.14.

For a given class distribution, always predicting the majority class gives
a non-trivial accuracy that any classifier must exceed to demonstrate value.

Phase 9 audit v4 (2026-04-20) found that on GoEmotions validation n=500:
- 10-way: majority (CARE) = 27.6%, keyword 28.0%, model 19.2%
  → model is 8.4% BELOW majority, keyword barely above.
- 6-way Ekman: majority (joy) = 53.2%, keyword 36.4%, model 36.4%
  → BOTH are 16.8% below majority.
"""
from __future__ import annotations

from collections import Counter


def majority_baseline_accuracy(instances, label_getter=lambda i: i.primary_ea) -> dict:
    """Compute majority-class accuracy.

    Returns: {"majority_label", "n_majority", "n_total", "accuracy"}
    """
    labels = [label_getter(i) for i in instances]
    labels = [l for l in labels if l is not None]
    if not labels:
        return {"majority_label": None, "accuracy": 0.0, "n_majority": 0, "n_total": 0}

    counter = Counter(labels)
    maj_label, n_maj = counter.most_common(1)[0]
    n_total = len(labels)
    return {
        "majority_label": maj_label,
        "n_majority": n_maj,
        "n_total": n_total,
        "accuracy": n_maj / n_total,
        "distribution": dict(counter),
    }


def make_majority_baseline(instances, label_getter=lambda i: i.primary_ea):
    """Create a function that always returns the majority label."""
    info = majority_baseline_accuracy(instances, label_getter)
    maj = info["majority_label"]
    def predict(text: str) -> str:
        return maj
    return predict, info


if __name__ == "__main__":
    # Demo: compute on validation n=500
    import sys
    sys.path.insert(0, ".")
    from phase9.dataset import load_goemotions_full, split_single_label
    from phase9.coarse_grained import EA_TO_EKMAN

    data = load_goemotions_full(split="validation")
    data = split_single_label(data)[:500]

    print("10-way EmotionAI majority baseline on n=500:")
    info = majority_baseline_accuracy(data)
    print(f"  majority label: {info['majority_label']}  count: {info['n_majority']}/{info['n_total']}  accuracy: {info['accuracy']:.3f}")

    print("\n6-way Ekman majority baseline on n=500:")
    info6 = majority_baseline_accuracy(
        data, label_getter=lambda i: EA_TO_EKMAN.get(i.primary_ea) if i.primary_ea else None)
    print(f"  majority label: {info6['majority_label']}  count: {info6['n_majority']}/{info6['n_total']}  accuracy: {info6['accuracy']:.3f}")
