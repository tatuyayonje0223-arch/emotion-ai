"""Phase 9.12 Coarse-grained Ekman 6-way eval.

Each instance classified into one of {anger, disgust, fear, joy, sadness, surprise}.
Baselines:
  random_coarse:   uniform 6-way
  keyword_coarse:  keyword argmax then EA → Ekman
  model_coarse:    IntegratedBrainV2 argmax then EA → Ekman
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from phase9.baselines import keyword_baseline, model_rates_baseline, reset_model_cache
from phase9.dataset import load_sample, load_goemotions_full, split_single_label
from phase9.coarse_grained import EKMAN_6, ea_to_ekman, GO_TO_EKMAN


def go_instance_ekman(inst) -> str | None:
    """Extract single Ekman label from GoEmotions labels (priority-based)."""
    mapped = [GO_TO_EKMAN.get(l) for l in inst.go_labels]
    mapped = [m for m in mapped if m]
    if not mapped:
        return None
    # Most common Ekman label (for multi-label; for single-label this is just that one)
    return Counter(mapped).most_common(1)[0][0]


def to_ekman_pred(ea_pred: str) -> str:
    return ea_to_ekman(ea_pred) or "joy"


def run_baseline(name, ea_fn, instances):
    preds = []
    trues = []
    t0 = time.time()
    for inst in instances:
        t = go_instance_ekman(inst)
        if t is None:
            continue
        ea_p = ea_fn(inst.text)
        preds.append(to_ekman_pred(ea_p))
        trues.append(t)
    print(f"  [{name}] {len(preds)} preds in {time.time()-t0:.0f}s")
    return preds, trues


def report(name, preds, trues):
    acc = accuracy_score(trues, preds)
    p, r, f, sup = precision_recall_fscore_support(trues, preds, labels=EKMAN_6, zero_division=0)
    macro_f1 = float(f.mean())
    print(f"\n  {name}: acc={acc:.3f}  macro-F1={macro_f1:.3f}")
    for i, lbl in enumerate(EKMAN_6):
        print(f"    {lbl:<10s} P={p[i]:.2f}  R={r[i]:.2f}  F1={f[i]:.2f}  n={int(sup[i])}")
    return {"acc": acc, "macro_f1": macro_f1}


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

    # Class distribution
    dist = Counter(go_instance_ekman(i) for i in data if go_instance_ekman(i))
    print("Ekman class distribution:")
    for lbl in EKMAN_6:
        print(f"  {lbl:<10s}: {dist.get(lbl, 0)}")

    import random
    rng = random.Random(42)
    def random_fn(text: str) -> str:
        return rng.choice(list(EA_TO_EKMAN_REV := ["FEAR","RAGE","SEEKING","SADNESS",
                                                   "DISGUST","CARE","PANIC_GRIEF",
                                                   "PLAY","LUST","SURPRISE"]))

    print("\n--- random ---")
    p, t = run_baseline("random", random_fn, data)
    r_res = report("random", p, t)

    print("\n--- keyword ---")
    p, t = run_baseline("keyword", keyword_baseline, data)
    k_res = report("keyword", p, t)

    print("\n--- model_rates ---")
    reset_model_cache()
    p, t = run_baseline("model_rates", model_rates_baseline, data)
    m_res = report("model_rates", p, t)

    print("\n" + "=" * 60)
    print("  Ekman-6 Summary")
    print("=" * 60)
    print(f"  random:       acc={r_res['acc']:.3f}  F1={r_res['macro_f1']:.3f}")
    print(f"  keyword:      acc={k_res['acc']:.3f}  F1={k_res['macro_f1']:.3f}")
    print(f"  model_rates:  acc={m_res['acc']:.3f}  F1={m_res['macro_f1']:.3f}")
    print(f"\n  Δ keyword - model = {k_res['acc'] - m_res['acc']:+.3f}")


if __name__ == "__main__":
    main()
