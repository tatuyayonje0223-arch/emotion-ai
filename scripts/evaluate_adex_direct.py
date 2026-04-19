"""Evaluate AdEx directly from shared_core_network.py tonic_drives (no overrides).

Unlike optimize_adex.py which passes tonic_overrides, this uses the values
from shared_core_network.py as-is. Use this to measure the effect of
hand-tuning tonic drives in the source file.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.optimize_adex import SCENARIOS, TOTAL_TARGETS


def evaluate_direct(verbose: bool = True) -> tuple[int, int, float]:
    from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
    from src.brian2_circuits.shared_core_network import SharedCoreConfig

    cfg = SharedCoreConfig(use_adex=True)

    passed = 0
    total = 0
    penalty = 0.0

    for emotion, scenarios in SCENARIOS.items():
        for cond_name, inputs, targets in scenarios:
            brain = EmotionBrainV2(config=cfg)  # NO tonic_overrides
            result = brain.process(**inputs)
            rates = result.all_rates

            for pop, lo, hi, typical in targets:
                rate = rates.get(pop, 0.0)
                total += 1
                in_range = lo <= rate <= hi
                if in_range:
                    passed += 1
                    status = "PASS"
                else:
                    status = "FAIL"
                    if rate < lo:
                        penalty += (lo - rate) / max(typical, 1.0)
                    else:
                        penalty += (rate - hi) / max(typical, 1.0)

                if verbose:
                    print(f"  {status}  {emotion}/{cond_name}/{pop}: "
                          f"{rate:.1f} Hz  [{lo}-{hi}]")

    return passed, total, penalty


if __name__ == "__main__":
    t0 = time.time()
    print("Direct AdEx evaluation (no tonic_overrides)...")
    passed, total, penalty = evaluate_direct(verbose=True)
    elapsed = time.time() - t0
    print(f"\n  Total: {passed}/{total} PASS, penalty={penalty:.2f}")
    print(f"  Elapsed: {elapsed:.1f}s")
