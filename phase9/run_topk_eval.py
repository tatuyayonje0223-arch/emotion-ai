"""Run top-k accuracy evaluation on 10-way GoEmotions classification.

For each baseline, get ranked list of 10 EmotionAI labels. Compute top-1,
top-2, top-3, top-5 accuracy. Compare keyword vs model rankings.
"""
from __future__ import annotations

import argparse
import time

from phase9.baselines import reset_model_cache
from phase9.dataset import load_sample, load_goemotions_full, split_single_label
from phase9.topk import RANKERS, topk_accuracy
from phase9.emotion_mapping import EMOTIONAI_LABELS


def collect_rankings(name, ranker, instances):
    ranked_list = []
    trues = []
    t0 = time.time()
    for inst in instances:
        t = inst.primary_ea
        if t is None:
            continue
        ranked_list.append(ranker(inst.text))
        trues.append(t)
    print(f"  [{name}] {len(ranked_list)} rankings in {time.time()-t0:.0f}s")
    return ranked_list, trues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    if args.full:
        data = load_goemotions_full(split="validation")
        data = split_single_label(data)
    else:
        data = load_sample()
        data = split_single_label(data)
    if args.limit:
        data = data[:args.limit]

    print(f"Instances: {len(data)}\n")

    results = {}
    for name, ranker in RANKERS.items():
        print(f"--- {name} ---")
        if name == "model":
            reset_model_cache()
        results[name] = collect_rankings(name, ranker, data)

    # Compute top-k for each
    print("\n" + "=" * 60)
    print("  Top-k accuracy (10-way)")
    print("=" * 60)
    header = f"  {'baseline':<10s}  {'top-1':>7s}  {'top-2':>7s}  {'top-3':>7s}  {'top-5':>7s}"
    print(header)
    for name, (ranked, trues) in results.items():
        accs = [topk_accuracy(ranked, trues, k) for k in (1, 2, 3, 5)]
        print(f"  {name:<10s}  {accs[0]:>6.1%}  {accs[1]:>6.1%}  {accs[2]:>6.1%}  {accs[3]:>6.1%}")

    # Delta analysis
    if "keyword" in results and "model" in results:
        print("\n  Δ (model - keyword):")
        k_ranked, k_trues = results["keyword"]
        m_ranked, m_trues = results["model"]
        # Ensure same trues
        assert k_trues == m_trues
        for k in (1, 2, 3, 5):
            kacc = topk_accuracy(k_ranked, k_trues, k)
            macc = topk_accuracy(m_ranked, m_trues, k)
            diff = macc - kacc
            sign = "+" if diff >= 0 else ""
            print(f"    top-{k}: model {macc:.1%} - keyword {kacc:.1%} = {sign}{diff:.1%}")


if __name__ == "__main__":
    main()
