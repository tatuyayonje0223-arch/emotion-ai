"""Specificity test via input-level lesioning.

For each emotion E in {FEAR, RAGE, SEEKING, SADNESS, DISGUST, CARE, PANIC_GRIEF,
PLAY, LUST, SURPRISE}, create a lesioned-E model (zero E's primary drives) and
evaluate per-class accuracy against ground truth.

**Specificity hypothesis**: lesioning E drops accuracy on true-E instances more
than on non-E instances.

Usage:
    PYTHONPATH=. python phase9/run_lesion_eval.py                # embedded sample
    PYTHONPATH=. python phase9/run_lesion_eval.py --full --limit 100
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

from phase9.baselines import model_rates_baseline, reset_model_cache
from phase9.dataset import load_sample, load_goemotions_full, split_single_label
from phase9.emotion_mapping import EMOTIONAI_LABELS
from phase9.lesioned import LESION_DRIVES, make_lesioned_baseline


def evaluate_on_subset(name: str, fn, instances) -> dict[str, dict]:
    """Return {true_label: {'correct': n, 'total': n}} for per-class accuracy."""
    by_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for inst in instances:
        true = inst.primary_ea
        if true is None:
            continue
        pred = fn(inst.text)
        by_class[true]["total"] += 1
        if pred == true:
            by_class[true]["correct"] += 1
    return dict(by_class)


def per_class_accuracy(results: dict) -> dict[str, float]:
    """Convert {label: {correct, total}} -> {label: acc_pct}."""
    return {
        label: (data["correct"] / data["total"] * 100) if data["total"] > 0 else None
        for label, data in results.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=100,
                        help="Cap instance count (default 100 for lesion eval time)")
    args = parser.parse_args()

    if args.full:
        print(f"Loading GoEmotions {args.split}, limit={args.limit}...")
        data = load_goemotions_full(split=args.split)
        data = split_single_label(data)
    else:
        print("Loading embedded sample...")
        data = load_sample()
        data = split_single_label(data)

    if args.limit:
        data = data[:args.limit]
    print(f"Instances: {len(data)}\n")

    # Collect baseline (unlesioned) per-class accuracy
    print("--- baseline (unlesioned model_rates) ---")
    reset_model_cache()
    t0 = time.time()
    baseline = evaluate_on_subset("baseline", model_rates_baseline, data)
    print(f"  {time.time() - t0:.1f}s")
    base_acc = per_class_accuracy(baseline)

    # Evaluate each lesion
    results_per_lesion = {}
    for emo in EMOTIONAI_LABELS:
        reset_model_cache()
        print(f"--- lesioned_{emo} (zeros: {LESION_DRIVES[emo]}) ---")
        fn = make_lesioned_baseline(emo)
        t0 = time.time()
        r = evaluate_on_subset(f"lesioned_{emo}", fn, data)
        print(f"  {time.time() - t0:.1f}s")
        results_per_lesion[emo] = per_class_accuracy(r)

    # Print specificity table
    print("\n" + "=" * 100)
    print(f"  Specificity table: per-class accuracy (%) under each lesion")
    print("=" * 100)
    header = f"  {'true label':<14s} {'baseline':>10s}  " + "  ".join(
        f"L-{e[:5]:>5s}" for e in EMOTIONAI_LABELS)
    print(header)
    for true_lbl in EMOTIONAI_LABELS:
        n = baseline.get(true_lbl, {}).get("total", 0)
        if n == 0:
            continue
        b_acc = base_acc.get(true_lbl, 0.0)
        row_parts = [f"{b_acc:>8.1f}%" if b_acc is not None else "    —  "]
        for lesion_emo in EMOTIONAI_LABELS:
            acc = results_per_lesion[lesion_emo].get(true_lbl)
            if acc is None:
                row_parts.append("   —  ")
            else:
                drop = b_acc - acc if b_acc is not None else 0
                marker = "*" if lesion_emo == true_lbl and drop > 0 else " "
                row_parts.append(f"{acc:>5.1f}%{marker}")
        print(f"  {true_lbl:<14s} n={n:<4d} {'  '.join(row_parts)}")

    print("\n* = lesion of own circuit dropped accuracy (specificity evidence)")


if __name__ == "__main__":
    main()
