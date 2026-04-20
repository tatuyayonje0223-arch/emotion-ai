"""Multi-label evaluation harness.

Uses ALL GoEmotions instances (not just single-label). Maps each instance's
multiple labels to EA labels (dedup). Computes P/R/F1@k for each baseline.
"""
from __future__ import annotations

import argparse
import time

from phase9.baselines import reset_model_cache
from phase9.dataset import load_goemotions_full
from phase9.topk import RANKERS
from phase9.multilabel import aggregate_multilabel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading GoEmotions {args.split} (all mappable instances)...")
    data = load_goemotions_full(split=args.split)
    # Don't restrict to single-label — use all mappable instances
    data = [inst for inst in data if inst.ea_labels]
    if args.limit:
        data = data[:args.limit]
    print(f"Instances: {len(data)}")

    # Compute multi-label stats
    from collections import Counter
    label_count_dist = Counter(len(inst.ea_labels) for inst in data)
    print("\nInstances by # of EA labels:")
    for nl, cnt in sorted(label_count_dist.items()):
        print(f"  {nl}-label: {cnt}")

    # Run each ranker
    ranked_results = {}
    for name, ranker in RANKERS.items():
        print(f"\n--- {name} ---")
        if name == "model":
            reset_model_cache()
        t0 = time.time()
        ranked_list = [ranker(inst.text) for inst in data]
        true_sets = [set(inst.ea_labels) for inst in data]
        print(f"  {len(ranked_list)} rankings in {time.time()-t0:.0f}s")

        results = aggregate_multilabel(ranked_list, true_sets)
        ranked_results[name] = results
        print(f"  {name}:")
        for metric, val in results.items():
            print(f"    {metric:<8s} {val:.3f}")

    # Comparison
    print("\n" + "=" * 60)
    print("  Summary: multi-label metrics (mean over instances)")
    print("=" * 60)
    header = f"  {'baseline':<10s}  " + "  ".join(f"{m:>8s}" for m in [
        "P@1", "R@1", "F1@1", "hit@1",
        "P@3", "R@3", "F1@3", "hit@3",
    ])
    print(header)
    for name, res in ranked_results.items():
        row = f"  {name:<10s}  " + "  ".join(
            f"{res.get(m, 0):>8.3f}" for m in [
                "P@1", "R@1", "F1@1", "hit@1",
                "P@3", "R@3", "F1@3", "hit@3",
            ]
        )
        print(row)

    # Deltas
    if "keyword" in ranked_results and "model" in ranked_results:
        print("\n  Δ (model - keyword):")
        for metric in ["P@1", "R@1", "F1@1", "hit@1", "P@3", "R@3", "F1@3", "hit@3", "F1@5"]:
            k = ranked_results["keyword"].get(metric, 0)
            m = ranked_results["model"].get(metric, 0)
            diff = m - k
            sign = "+" if diff >= 0 else ""
            print(f"    {metric:<8s}: {sign}{diff:.3f}  (model {m:.3f} - keyword {k:.3f})")


if __name__ == "__main__":
    main()
