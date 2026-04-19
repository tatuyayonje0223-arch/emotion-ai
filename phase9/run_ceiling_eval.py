"""Ceiling evaluation — adds LR (trained ML) and Gemini (LLM) to Phase 9 picture.

Usage:
    PYTHONPATH=. python phase9/run_ceiling_eval.py --full --limit 200 \
        --lr-train-limit 5000 --skip-gemini
    PYTHONPATH=. python phase9/run_ceiling_eval.py --full --limit 100 --gemini-limit 100
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

from phase9.baselines import BASELINES, reset_model_cache
from phase9.ceiling_baselines import fit_lr_baseline, make_gemini_baseline
from phase9.dataset import load_sample, load_goemotions_full, split_single_label
from phase9.emotion_mapping import EMOTIONAI_LABELS
from phase9.metrics import evaluate, mcnemar_test


def run(name, fn, instances):
    preds, trues = [], []
    t0 = time.time()
    for i, inst in enumerate(instances):
        t = inst.primary_ea
        if t is None:
            continue
        preds.append(fn(inst.text))
        trues.append(t)
        if (i + 1) % 50 == 0:
            print(f"    [{name}] {i+1}/{len(instances)}")
    print(f"  [{name}] {len(preds)} preds in {time.time()-t0:.0f}s")
    return preds, trues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200,
                        help="Test set size per baseline")
    parser.add_argument("--lr-train-limit", type=int, default=5000)
    parser.add_argument("--gemini-limit", type=int, default=100,
                        help="Cap for Gemini (rate-limited)")
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-lr", action="store_true")
    args = parser.parse_args()

    if args.full:
        print(f"Loading GoEmotions {args.split}...")
        test = load_goemotions_full(split=args.split)
        test = split_single_label(test)
        if args.limit:
            test = test[:args.limit]

        train = None
        if not args.skip_lr:
            print(f"Loading GoEmotions train (limit {args.lr_train_limit})...")
            train = load_goemotions_full(split="train")
            train = split_single_label(train)
            if args.lr_train_limit:
                train = train[:args.lr_train_limit]
    else:
        test = load_sample()
        test = split_single_label(test)
        train = test  # overlap ok for demo

    print(f"Test instances: {len(test)}")
    if train is not None:
        print(f"Train instances: {len(train)}")

    results = {}

    # Existing baselines
    for name, fn in BASELINES.items():
        print(f"\n--- {name} ---")
        if name == "model_rates":
            reset_model_cache()
        preds, trues = run(name, fn, test)
        results[name] = evaluate(name, preds, trues)

    # LR trained ceiling
    if not args.skip_lr and train is not None:
        print(f"\n--- LR (trained on {len(train)} instances) ---")
        lr_fn = fit_lr_baseline(train)
        preds, trues = run("lr_trained", lr_fn, test)
        results["lr_trained"] = evaluate("lr_trained", preds, trues)

    # Gemini zero-shot ceiling
    if not args.skip_gemini:
        gtest = test[:args.gemini_limit]
        print(f"\n--- Gemini zero-shot (n={len(gtest)}) ---")
        g_fn = make_gemini_baseline(rate_limit_sec=4.5)
        preds, trues = run("gemini", g_fn, gtest)
        results["gemini"] = evaluate("gemini", preds, trues)

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'baseline':<15s} {'n':>5s} {'accuracy':>10s} {'macro-F1':>10s}")
    for name, res in results.items():
        n_eval = len(res.preds)
        print(f"  {name:<15s} {n_eval:>5d} {res.accuracy:>9.3f} {res.macro_f1:>10.3f}")

    # Ranking
    print("\n  Ranking (by accuracy, same-n only for fair comparison):")
    # Group by sample size
    by_n = {}
    for name, res in results.items():
        by_n.setdefault(len(res.preds), []).append((name, res.accuracy, res.macro_f1))
    for n, entries in sorted(by_n.items()):
        print(f"\n  At n={n}:")
        for name, acc, f1 in sorted(entries, key=lambda x: -x[1]):
            print(f"    {name:<15s} acc={acc:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
