"""V2 Emotion Circuits -- Quantitative Validation Against Literature Targets.

EmotionBrainV2.process()を直接使用し、発火率をターゲットと比較する。
各シナリオは独立したEmotionBrainV2インスタンスで実行（STDP蓄積を防ぐ）。

Usage:
    PYTHONPATH=. PYTHONIOENCODING=utf-8 python scripts/run_v2_validation.py
"""

from __future__ import annotations

import sys
import time

from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
from src.calibration.quantitative_targets_v2 import (
    ALL_TARGETS, FiringRateTarget, RatioTarget,
)


def run_scenario(name: str, **kwargs) -> dict[str, float]:
    """独立したBrainインスタンスで1シナリオを実行。"""
    brain = EmotionBrainV2()
    state = brain.process(**kwargs)
    return dict(state.all_rates)


def main():
    t0 = time.time()

    print("=" * 90)
    print("  V2 Emotion Circuits -- Quantitative Validation")
    print("  Using EmotionBrainV2.process() directly (independent instances per scenario)")
    print("=" * 90)

    # ── Scenarios ──
    scenarios: dict[str, dict[str, dict]] = {
        "fear": {
            "baseline": {"threat": 0.0},
            "cs_evoked": {"threat": 0.8},
            "extinction_recall": {"threat": 0.1},
        },
        "rage": {
            "baseline": {"frustration": 0.0},
            "social_encounter": {"frustration": 0.3},
            "investigation": {"frustration": 0.5},
            "attack": {"frustration": 0.8},
        },
        "seeking": {
            "tonic": {"reward": 0.0},
            "phasic_burst": {"reward": 0.8},
            "pause": {"loss": 0.5},
        },
        "sadness": {
            "baseline": {"loss": 0.0},
            "depression": {"loss": 0.8},
        },
        "disgust": {
            "baseline": {"contamination": 0.0},
            "stimulus": {"contamination": 0.8},
        },
        "care": {
            "social_bonding": {"social": 0.8, "attachment_need": 0.6},
        },
        "panic_grief": {
            "separation": {"loss": 0.8, "attachment_need": 0.8},
        },
        "play": {
            "social_play": {"social": 0.7, "reward": 0.5, "novelty": 0.3},
        },
        "lust": {
            "sexual_arousal": {"social": 0.7, "reward": 0.4},
        },
        "surprise": {
            "novelty_burst": {"novelty": 0.9},
            "novelty": {"novelty": 0.9},
        },
    }

    all_rates: dict[str, dict[str, dict[str, float]]] = {}
    trial = 0
    for emo, conditions in scenarios.items():
        print(f"\n  {emo.upper()}:")
        all_rates[emo] = {}
        for cond, kwargs in conditions.items():
            print(f"    {cond} ({kwargs}) ...", end=" ", flush=True)
            rates = run_scenario(cond, **kwargs)
            all_rates[emo][cond] = rates
            trial += 1
            top = sorted(rates.items(), key=lambda x: x[1], reverse=True)[:5]
            top_str = ", ".join(f"{k}={v:.1f}" for k, v in top)
            print(f"done [{top_str}]")

    # ── Validate ──
    results = []
    for emo_name, (fr_targets, ratio_targets) in ALL_TARGETS.items():
        emo_data = all_rates.get(emo_name, {})

        for t in fr_targets:
            cond_data = emo_data.get(t.condition, {})
            actual = cond_data.get(t.region, 0.0)
            passed = t.min_hz <= actual <= t.max_hz
            results.append({
                "emotion": emo_name, "region": t.region, "condition": t.condition,
                "target": f"[{t.min_hz}-{t.max_hz}]", "actual": actual,
                "passed": passed, "source": t.source,
            })

        for rt in ratio_targets:
            if rt.name.startswith("FEAR: SOM"):
                cs_data = emo_data.get("cs_evoked", {})
                num = cs_data.get("cel_som", 0)
                den = cs_data.get("cel_pkcd", 0)
            elif rt.name.startswith("FEAR: Conditioned"):
                num = emo_data.get("cs_evoked", {}).get("la_exc", 0)
                den = emo_data.get("baseline", {}).get("la_exc", 0)
            elif rt.name.startswith("RAGE: VMH"):
                num = emo_data.get("attack", {}).get("vmh", 0)
                den = emo_data.get("baseline", {}).get("vmh", 0)
            elif rt.name.startswith("SEEKING: DA"):
                num = emo_data.get("phasic_burst", {}).get("vta_da_lat", 0)
                den = emo_data.get("tonic", {}).get("vta_da_lat", 0)
            elif rt.name.startswith("SADNESS: sgACC"):
                num = emo_data.get("depression", {}).get("sgacc", 0)
                den = emo_data.get("baseline", {}).get("sgacc", 0)
            else:
                continue

            ratio = num / den if den > 0 else float('inf')
            passed = rt.min_ratio <= ratio <= rt.max_ratio
            results.append({
                "emotion": emo_name, "region": rt.name, "condition": "ratio",
                "target": f"[{rt.min_ratio}-{rt.max_ratio}]", "actual": ratio,
                "passed": passed, "source": rt.source,
            })

    # ── Report ──
    elapsed = time.time() - t0
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)

    print(f"\n{'=' * 90}")
    print(f"  VALIDATION REPORT  |  {n_pass}/{n_total} PASS  |  {n_pass/n_total*100:.1f}%  |  {elapsed:.1f}s")
    print(f"{'=' * 90}")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        emo = r["emotion"].upper()
        rgn = f"{r['region']}/{r['condition']}" if r["condition"] != "ratio" else r["region"]
        actual_str = f"{r['actual']:.1f}" if r["actual"] != float('inf') else "INF"
        note = ""
        if not r["passed"]:
            if r["actual"] == float('inf'):
                note = "  << division by zero (baseline=0)"
            elif isinstance(r["actual"], float) and r["actual"] < 0.01:
                note = "  << ZERO"
            else:
                note = f"  << {'HIGH' if r['actual'] > 0 else 'LOW'}"
        print(f"  {status:4s}  {emo:10s} {rgn:40s} {r['target']:15s} {actual_str:>10s}  {r['source'][:40]}{note}")

    print(f"\n  Per-emotion:")
    for emo in ALL_TARGETS:
        emo_results = [r for r in results if r["emotion"] == emo]
        emo_pass = sum(1 for r in emo_results if r["passed"])
        print(f"    {emo.upper():12s} {emo_pass}/{len(emo_results)}")

    print(f"\n  Overall: {n_pass}/{n_total} ({n_pass/n_total*100:.1f}%)")


if __name__ == "__main__":
    main()
