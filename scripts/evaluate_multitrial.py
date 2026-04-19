"""Multi-trial Monte Carlo evaluation — tests whether "36/36 PASS" is seed-stable.

The default evaluator (optimize_adex.py / evaluate_adex_direct.py) uses trial_num=0
only. With 300ms simulations and small populations, single-trial rates carry
~10-35% noise (empirically measured). This script runs each scenario N times and
reports mean/std + per-target pass stability (k out of N trials passed).

Usage:
    python scripts/evaluate_multitrial.py --n-trials 5 [--adex | --izh]
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.optimize_adex import SCENARIOS


def run_multitrial(use_adex: bool, n_trials: int) -> dict:
    from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
    from src.brian2_circuits.shared_core_network import SharedCoreConfig

    cfg = SharedCoreConfig(use_adex=use_adex)
    results = {}  # key: (emotion, cond, pop) -> list of rates

    for emotion, scenarios in SCENARIOS.items():
        for cond_name, inputs, targets in scenarios:
            for t in range(n_trials):
                # Fresh brain per trial gives independent noise realization
                # (trial_num drives noise_rng seed inside run_trial)
                brain = EmotionBrainV2(config=cfg)
                brain._step_count = t  # shift seed per trial
                result = brain.process(**inputs)
                rates = result.all_rates

                for pop, lo, hi, typical in targets:
                    key = (emotion, cond_name, pop, lo, hi)
                    results.setdefault(key, []).append(rates.get(pop, 0.0))

    return results


def report(label: str, results: dict, n_trials: int) -> tuple[int, int, int]:
    print(f"\n=== {label} ({n_trials}-trial MC) ===")
    print(f"  {'scenario/pop':<45s} {'mean':>6s} {'std':>5s} {'pass_k/n':>10s} {'range':>12s}")
    stable_pass = 0
    unstable = 0
    stable_fail = 0
    total = len(results)
    for (emotion, cond, pop, lo, hi), rates in results.items():
        rates = np.array(rates)
        mean = rates.mean()
        std = rates.std()
        n_pass = int(((rates >= lo) & (rates <= hi)).sum())
        key = f"{emotion}/{cond}/{pop}"
        rng = f"[{rates.min():.1f}-{rates.max():.1f}]"
        if n_pass == n_trials:
            tag = "STABLE_PASS"
            stable_pass += 1
        elif n_pass == 0:
            tag = "STABLE_FAIL"
            stable_fail += 1
        else:
            tag = "UNSTABLE"
            unstable += 1
        print(f"  {key:<45s} {mean:6.2f} {std:5.2f} {n_pass:>3d}/{n_trials:>3d} {rng:>12s}  {tag}")

    print(f"\n  Stable PASS: {stable_pass}/{total}")
    print(f"  Unstable:    {unstable}/{total}")
    print(f"  Stable FAIL: {stable_fail}/{total}")
    return stable_pass, unstable, stable_fail


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--model", choices=["adex", "izh", "both"], default="both")
    args = parser.parse_args()

    t0 = time.time()
    if args.model in ("adex", "both"):
        adex_results = run_multitrial(use_adex=True, n_trials=args.n_trials)
        report("AdEx", adex_results, args.n_trials)
    if args.model in ("izh", "both"):
        izh_results = run_multitrial(use_adex=False, n_trials=args.n_trials)
        report("Izhikevich", izh_results, args.n_trials)

    print(f"\n  Elapsed: {time.time() - t0:.1f}s")
