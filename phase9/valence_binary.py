"""Phase 9.15 Binary valence classification (positive vs negative).

v4 audit noted 2-way valence was never tested. Simpler task — maybe model
captures positive/negative better than fine distinctions.

Mappings:
  EmotionAI → binary valence:
    Positive: SEEKING, CARE, PLAY, LUST
    Negative: FEAR, RAGE, SADNESS, DISGUST, PANIC_GRIEF
    Ambiguous (skip): SURPRISE

  GoEmotions → binary:
    Positive: joy, excitement, gratitude, optimism, pride, amusement,
              love, admiration, approval, caring, desire, relief
    Negative: anger, annoyance, disapproval, fear, nervousness,
              sadness, disappointment, grief, remorse, disgust, embarrassment
    Skip: surprise, curiosity, confusion, realization, neutral
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

from phase9.baselines import keyword_baseline, model_rates_baseline, reset_model_cache
from phase9.dataset import load_goemotions_full, split_single_label
from phase9.majority_baseline import majority_baseline_accuracy


EA_TO_VALENCE: dict[str, str | None] = {
    "SEEKING": "positive", "CARE": "positive", "PLAY": "positive", "LUST": "positive",
    "FEAR": "negative", "RAGE": "negative", "SADNESS": "negative",
    "DISGUST": "negative", "PANIC_GRIEF": "negative",
    "SURPRISE": None,   # ambiguous
}

GO_TO_VALENCE: dict[str, str | None] = {
    # positive
    "joy": "positive", "excitement": "positive", "gratitude": "positive",
    "optimism": "positive", "pride": "positive", "amusement": "positive",
    "love": "positive", "admiration": "positive", "approval": "positive",
    "caring": "positive", "desire": "positive", "relief": "positive",
    # negative
    "anger": "negative", "annoyance": "negative", "disapproval": "negative",
    "fear": "negative", "nervousness": "negative",
    "sadness": "negative", "disappointment": "negative", "grief": "negative",
    "remorse": "negative", "disgust": "negative", "embarrassment": "negative",
    # ambiguous (drop)
    "surprise": None, "curiosity": None, "confusion": None,
    "realization": None, "neutral": None,
}


def go_inst_valence(inst) -> str | None:
    """Extract single binary valence from GoEmotions instance."""
    mapped = [GO_TO_VALENCE.get(l) for l in inst.go_labels]
    mapped = [m for m in mapped if m]
    if not mapped:
        return None
    c = Counter(mapped)
    # Return most common (no-conflict) — if positive and negative both, pick majority
    return c.most_common(1)[0][0]


def ea_to_valence(ea_label: str) -> str:
    v = EA_TO_VALENCE.get(ea_label)
    return v if v else "positive"  # surprise fallback to positive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    data = load_goemotions_full(split="validation")
    # use all (multi-label too, but with valence-consistent mapping)
    data = [inst for inst in data if inst.ea_labels]
    data = data[:args.limit]

    # Filter to valence-mappable instances (exclude pure surprise/neutral)
    filtered = []
    true_labels = []
    for inst in data:
        v = go_inst_valence(inst)
        if v:
            filtered.append(inst)
            true_labels.append(v)
    print(f"Instances: {len(data)} loaded, {len(filtered)} valence-mappable")

    dist = Counter(true_labels)
    print(f"\nValence distribution: {dict(dist)}")
    maj_label, maj_n = dist.most_common(1)[0]
    maj_acc = maj_n / len(filtered)
    print(f"Majority baseline: always-{maj_label} = {maj_acc:.3f}")

    # Random
    import random
    rng = random.Random(42)
    random_preds = [rng.choice(["positive", "negative"]) for _ in filtered]
    random_acc = sum(1 for p, t in zip(random_preds, true_labels) if p == t) / len(filtered)
    print(f"\nRandom baseline: {random_acc:.3f}")

    # Keyword argmax → binary
    print("\n--- keyword ---")
    t0 = time.time()
    kw_preds = [ea_to_valence(keyword_baseline(inst.text)) for inst in filtered]
    kw_acc = sum(1 for p, t in zip(kw_preds, true_labels) if p == t) / len(filtered)
    print(f"  {len(kw_preds)} preds in {time.time()-t0:.0f}s  accuracy: {kw_acc:.3f}")

    # Model_rates argmax → binary
    print("\n--- model_rates ---")
    reset_model_cache()
    t0 = time.time()
    model_preds = [ea_to_valence(model_rates_baseline(inst.text)) for inst in filtered]
    model_acc = sum(1 for p, t in zip(model_preds, true_labels) if p == t) / len(filtered)
    print(f"  {len(model_preds)} preds in {time.time()-t0:.0f}s  accuracy: {model_acc:.3f}")

    # Summary
    print("\n" + "=" * 50)
    print(f"  Binary valence (pos vs neg) on n={len(filtered)}")
    print("=" * 50)
    print(f"  majority:     {maj_acc:.3f}")
    print(f"  random:       {random_acc:.3f}")
    print(f"  keyword:      {kw_acc:.3f}")
    print(f"  model_rates:  {model_acc:.3f}")
    print(f"\n  Δ model - keyword:   {model_acc - kw_acc:+.3f}")
    print(f"  Δ model - majority:  {model_acc - maj_acc:+.3f}")
    print(f"  Δ keyword - majority: {kw_acc - maj_acc:+.3f}")


if __name__ == "__main__":
    main()
