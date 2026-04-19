"""Sweep tonic values for failing AdEx populations to find tunable ones.

For each failure, test the target rate at several tonic values.
Uses _tonic_overrides so we don't need to edit source files.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
from src.brian2_circuits.shared_core_network import SharedCoreConfig


SWEEPS = [
    # (population, scenario_name, inputs, target_range, pop_to_report, tonic_values)
    ("il",       "extinction_recall",  {"threat": 0.1},                (7, 13),  "il",      [3.5, 4.5, 5.5, 6.5]),
    ("dr",       "sadness_suppressed", {"loss": 0.8},                  (2, 4),   "dr",      [3.0, 3.5, 4.0, 4.5]),
    ("vta_da_lat", "pause",            {"loss": 0.5},                  (0, 1),   "vta_da_lat", [2.5, 2.0, 1.5, 1.0]),
    ("lc",       "novelty_burst",      {"novelty": 0.9},               (8, 16),  "lc",      [2.8, 4.0, 5.5, 7.0]),
    ("lust_mpoa", "sexual_arousal",    {"social": 0.7, "reward": 0.4}, (7, 15),  "vta_da_lat", [3.5, 3.0, 2.5, 2.0]),
    ("habenula", "reward_omission",    {"loss": 0.5},                  (10, 20), "habenula", [2.5, 2.0, 1.5, 1.0]),
    ("putamen",  "disgust_recognition", {"contamination": 0.8},        (7, 13),  "putamen", [4.5, 6.0, 7.5, 9.0]),
]


def run_sweep():
    cfg = SharedCoreConfig(use_adex=True)

    for (pop, scenario, inputs, (lo, hi), report_pop, tonic_vals) in SWEEPS:
        print(f"\n-- {pop} ({scenario}) - target {report_pop} [{lo}-{hi}] --")
        for tval in tonic_vals:
            brain = EmotionBrainV2(config=cfg, tonic_overrides={pop: tval})
            r = brain.process(**inputs)
            rate = r.all_rates.get(report_pop, 0.0)
            in_range = lo <= rate <= hi
            status = "PASS" if in_range else "FAIL"
            print(f"  tonic({pop})={tval:.1f} -> {report_pop}={rate:.2f} Hz  [{status}]")


if __name__ == "__main__":
    run_sweep()
