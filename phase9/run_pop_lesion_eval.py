"""Population-level lesion specificity eval.

Like phase9/run_lesion_eval.py but uses pop_lesion (neuron-level silencing)
instead of input_lesion (text→drive zeroing).

Compared to input lesion, population lesion is the stronger neuroscience-
standard experiment. Tests whether the CIRCUIT (not just the input) is
causally necessary for emotion classification.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

from phase9.baselines import model_rates_baseline, reset_model_cache
from phase9.dataset import load_sample, load_goemotions_full, split_single_label
from phase9.emotion_mapping import EMOTIONAI_LABELS
from phase9.pop_lesion import POP_LESION_TARGETS, make_pop_lesioned_baseline


def evaluate_on_subset(name, fn, instances):
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


def per_class_accuracy(results):
    return {
        label: (data["correct"] / data["total"] * 100) if data["total"] > 0 else None
        for label, data in results.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=50,
                        help="n=50 default (10 lesions × 50 ≈ 20 min compute)")
    args = parser.parse_args()

    if args.full:
        data = load_goemotions_full(split=args.split)
        data = split_single_label(data)
    else:
        data = load_sample()
        data = split_single_label(data)

    if args.limit:
        data = data[:args.limit]
    print(f"Instances: {len(data)}\n")

    # Baseline unlesioned
    print("--- baseline (unlesioned) ---")
    reset_model_cache()
    t0 = time.time()
    baseline = evaluate_on_subset("baseline", model_rates_baseline, data)
    print(f"  {time.time() - t0:.1f}s")
    base_acc = per_class_accuracy(baseline)

    # Each lesion
    results = {}
    for emo in EMOTIONAI_LABELS:
        print(f"--- pop-lesion {emo} (pops: {POP_LESION_TARGETS[emo]}) ---")
        fn = make_pop_lesioned_baseline(emo)
        t0 = time.time()
        r = evaluate_on_subset(f"pop_{emo}", fn, data)
        print(f"  {time.time() - t0:.1f}s")
        results[emo] = per_class_accuracy(r)

    # Specificity table
    print("\n" + "=" * 100)
    print("  Population-level lesion specificity: per-class accuracy (%) under each lesion")
    print("=" * 100)
    header = f"  {'true':<13s} {'n':>3s}  {'base':>6s}  " + "  ".join(
        f"L-{e[:5]:>5s}" for e in EMOTIONAI_LABELS)
    print(header)
    for true_lbl in EMOTIONAI_LABELS:
        n = baseline.get(true_lbl, {}).get("total", 0)
        if n == 0:
            continue
        b_acc = base_acc.get(true_lbl)
        if b_acc is None:
            continue
        row = f"{b_acc:>5.1f}"
        for lesion_emo in EMOTIONAI_LABELS:
            acc = results[lesion_emo].get(true_lbl)
            if acc is None:
                row_s = "   —  "
            else:
                drop = b_acc - acc
                marker = "*" if lesion_emo == true_lbl and drop > 0 else " "
                row_s = f"{acc:>5.1f}{marker}"
            row += f"  {row_s}"
        print(f"  {true_lbl:<13s} {n:>3d}  {row}")
    print("\n* = own-lesion dropped accuracy (specificity evidence)")


if __name__ == "__main__":
    main()
