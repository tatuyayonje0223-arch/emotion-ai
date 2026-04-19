"""Valence × arousal regression evaluation.

Usage:
    PYTHONPATH=. python phase9/run_va_eval.py --full --limit 500
"""
from __future__ import annotations

import argparse
import time

from phase9.baselines import reset_model_cache
from phase9.dataset import load_sample, load_goemotions_full, split_single_label
from phase9.dimensional import (
    VA_BASELINES, ground_truth_va, pearson_r, mae, bivariate_r2,
)


def collect_predictions(name, fn, instances):
    gt_v, gt_a = [], []
    pred_v, pred_a = [], []
    t0 = time.time()
    for inst in instances:
        true_label = inst.primary_ea
        if true_label is None:
            continue
        gtv, gta = ground_truth_va(true_label)
        gt_v.append(gtv); gt_a.append(gta)
        p = fn(inst.text)
        pred_v.append(p.valence); pred_a.append(p.arousal)
    print(f"  [{name}] {len(pred_v)} preds in {time.time()-t0:.1f}s")
    return gt_v, gt_a, pred_v, pred_a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    if args.full:
        print(f"Loading GoEmotions {args.split} limit={args.limit}...")
        data = load_goemotions_full(split=args.split)
        data = split_single_label(data)
    else:
        print("Loading embedded sample...")
        data = load_sample()
        data = split_single_label(data)

    if args.limit:
        data = data[:args.limit]
    print(f"Instances: {len(data)}\n")

    results = {}
    for name, fn in VA_BASELINES.items():
        print(f"--- {name} ---")
        if name == "model_va":
            reset_model_cache()
        results[name] = collect_predictions(name, fn, data)

    # Summary
    print("\n" + "=" * 70)
    print("  Dimensional affect regression — V and A separately + joint R^2")
    print("=" * 70)
    print(f"  {'baseline':<15s}  {'V Pearson':>10s}  {'V MAE':>8s}  "
          f"{'A Pearson':>10s}  {'A MAE':>8s}  {'Joint R^2':>10s}")
    for name, (gv, ga, pv, pa) in results.items():
        vr = pearson_r(pv, gv)
        vm = mae(pv, gv)
        ar = pearson_r(pa, ga)
        am = mae(pa, ga)
        r2 = bivariate_r2(pv, pa, gv, ga)
        print(f"  {name:<15s}  {vr:>+10.3f}  {vm:>8.3f}  "
              f"{ar:>+10.3f}  {am:>8.3f}  {r2:>+10.3f}")

    # Primary comparison: model vs keyword on valence
    print("\n" + "=" * 70)
    print("  Primary comparison: model_va vs keyword_va")
    print("=" * 70)
    _, _, mpv, mpa = results["model_va"]
    _, _, kpv, kpa = results["keyword_va"]
    gv, ga, _, _ = results["model_va"]   # ground truth (same across baselines)
    mv_r = pearson_r(mpv, gv); mv_m = mae(mpv, gv)
    kv_r = pearson_r(kpv, gv); kv_m = mae(kpv, gv)
    ma_r = pearson_r(mpa, ga); ma_m = mae(mpa, ga)
    ka_r = pearson_r(kpa, ga); ka_m = mae(kpa, ga)
    print(f"  Valence Pearson: model {mv_r:+.3f}  vs keyword {kv_r:+.3f}  delta={mv_r-kv_r:+.3f}")
    print(f"  Valence MAE:    model {mv_m:.3f}  vs keyword {kv_m:.3f}  delta={mv_m-kv_m:+.3f}")
    print(f"  Arousal Pearson: model {ma_r:+.3f}  vs keyword {ka_r:+.3f}  delta={ma_r-ka_r:+.3f}")
    print(f"  Arousal MAE:    model {ma_m:.3f}  vs keyword {ka_m:.3f}  delta={ma_m-ka_m:+.3f}")

    # Interpretation
    if mv_r > kv_r and ma_r > ka_r:
        print("\n  RESULT: model beats keyword on BOTH valence and arousal correlation")
    elif mv_r > kv_r:
        print("\n  RESULT: model beats keyword on valence only")
    elif ma_r > ka_r:
        print("\n  RESULT: model beats keyword on arousal only")
    else:
        print("\n  RESULT: keyword >= model on both dimensions — null holds dimensionally too")


if __name__ == "__main__":
    main()
