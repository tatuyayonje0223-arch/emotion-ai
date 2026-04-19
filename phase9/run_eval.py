"""Phase 9 eval runner — applies baselines to labeled dataset.

Usage:
    PYTHONPATH=. python phase9/run_eval.py                 # sample (offline)
    PYTHONPATH=. python phase9/run_eval.py --full          # full GoEmotions (needs datasets lib)

Reports: per-baseline accuracy + macro-F1 + per-class P/R/F1 + confusion.
Then McNemar test comparing model_rates vs keyword (primary hypothesis).
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

from phase9.baselines import BASELINES, reset_model_cache
from phase9.dataset import load_sample, load_goemotions_full, split_single_label
from phase9.metrics import evaluate, mcnemar_test, format_confusion
from phase9.emotion_mapping import EMOTIONAI_LABELS


def run_baseline_on_instances(name, fn, instances, verbose=False):
    preds = []
    trues = []
    t0 = time.time()
    for i, inst in enumerate(instances):
        true_label = inst.primary_ea
        if true_label is None:
            continue
        pred = fn(inst.text)
        preds.append(pred)
        trues.append(true_label)
        if verbose and (i % 10 == 0):
            print(f"    [{name}] {i+1}/{len(instances)}")
    elapsed = time.time() - t0
    print(f"  [{name}] {len(preds)} predictions in {elapsed:.1f}s")
    return preds, trues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Use full GoEmotions dataset (needs datasets lib)")
    parser.add_argument("--split", default="validation",
                        help="GoEmotions split (train/validation/test)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap instance count (for time-limited tests)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  EmotionAI Phase 9 — Behavioral Validation")
    print("=" * 70)

    # Load dataset
    if args.full:
        print(f"\nLoading full GoEmotions '{args.split}'...")
        data = load_goemotions_full(split=args.split)
        data = split_single_label(data)
    else:
        print("\nLoading embedded sample (offline, 40 instances)...")
        data = load_sample()
        data = split_single_label(data)

    if args.limit:
        data = data[:args.limit]

    print(f"Instances: {len(data)}")
    dist = Counter(i.primary_ea for i in data)
    print("Class distribution:")
    for label in EMOTIONAI_LABELS:
        n = dist.get(label, 0)
        if n > 0:
            print(f"  {label:<12s}: {n}")

    # Run each baseline
    results = {}
    for name, fn in BASELINES.items():
        print(f"\n--- Running {name} ---")
        if name == "model_rates":
            reset_model_cache()  # fresh brain
        preds, trues = run_baseline_on_instances(name, fn, data, verbose=args.verbose)
        results[name] = evaluate(name, preds, trues)

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    for name, res in results.items():
        print(f"\n{res.summary()}")

    # McNemar: model_rates vs keyword (the primary question)
    if "model_rates" in results and "keyword" in results:
        print("\n" + "=" * 70)
        print("  McNemar test: model_rates vs keyword (primary hypothesis)")
        print("=" * 70)
        mn = mcnemar_test(results["model_rates"], results["keyword"])
        print(f"  model correct / keyword wrong: {int(mn['b01'])}")
        print(f"  model wrong / keyword correct: {int(mn['b10'])}")
        print(f"  chi2 = {mn['chi2']:.3f}, p-value = {mn['p_value']:.4f}")
        if results["model_rates"].accuracy > results["keyword"].accuracy:
            diff = results["model_rates"].accuracy - results["keyword"].accuracy
            significant = "significant" if mn['p_value'] < 0.05 else "NOT significant"
            print(f"  model beats keyword by {diff:.3f} absolute accuracy; {significant} (p<0.05)")
        else:
            diff = results["keyword"].accuracy - results["model_rates"].accuracy
            print(f"  keyword beats model by {diff:.3f} — neural simulation adds no value here")

    print("\nConfusion matrix (model_rates):")
    if "model_rates" in results:
        print(format_confusion(results["model_rates"]))


if __name__ == "__main__":
    main()
