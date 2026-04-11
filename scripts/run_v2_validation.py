"""V2 Emotion Circuits — Quantitative Validation Against Literature Targets.

Runs each emotion scenario through EmotionBrainV2 and compares firing rates
against the 232-paper-derived targets in quantitative_targets_v2.py.

Usage:
    PYTHONPATH=. PYTHONIOENCODING=utf-8 python scripts/run_v2_validation.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
from src.brian2_circuits.shared_core_network import SharedCoreConfig
from src.calibration.quantitative_targets_v2 import (
    ALL_TARGETS,
    FiringRateTarget,
    RatioTarget,
    FEAR_RATIOS,
    RAGE_RATIOS,
    SEEKING_RATIOS,
    SADNESS_RATIOS,
)


# ════════════════════════════════════════════════════════════════
# Result container
# ════════════════════════════════════════════════════════════════

@dataclass
class TargetResult:
    target_name: str
    emotion: str
    condition: str
    min_val: float
    max_val: float
    actual: float
    passed: bool
    source: str
    note: str = ""


# ════════════════════════════════════════════════════════════════
# Scenario runner
# ════════════════════════════════════════════════════════════════

def run_scenarios(brain: EmotionBrainV2) -> dict[str, dict[str, dict[str, float]]]:
    """Run all emotion scenarios and collect per-region firing rates.

    Returns:
        {emotion: {condition_label: {region: firing_rate}}}
    """
    results: dict[str, dict[str, dict[str, float]]] = {}

    # ── FEAR scenarios ──
    print("  Running FEAR baseline...", flush=True)
    state_base = brain.process(threat=0.0)
    fear_baseline = dict(state_base.all_rates)

    print("  Running FEAR cs_evoked (threat=0.8)...", flush=True)
    state_fear = brain.process(threat=0.8)
    fear_evoked = dict(state_fear.all_rates)

    print("  Running FEAR extinction (threat=0.1)...", flush=True)
    state_ext = brain.process(threat=0.1)
    fear_extinction = dict(state_ext.all_rates)

    results["fear"] = {
        "baseline": fear_baseline,
        "cs_evoked": fear_evoked,
        "fear_expression": fear_evoked,  # same trial for CeM
        "fear_burst": fear_evoked,       # PL burst during fear
        "extinction_recall": fear_extinction,
        "freezing": fear_evoked,         # vlPAG during fear
    }

    # ── RAGE scenarios ──
    print("  Running RAGE baseline...", flush=True)
    state_rage_base = brain.process(frustration=0.0)
    rage_baseline = dict(state_rage_base.all_rates)

    print("  Running RAGE social_encounter (frustration=0.3)...", flush=True)
    state_rage_social = brain.process(frustration=0.3)
    rage_social = dict(state_rage_social.all_rates)

    print("  Running RAGE investigation (frustration=0.5)...", flush=True)
    state_rage_inv = brain.process(frustration=0.5)
    rage_inv = dict(state_rage_inv.all_rates)

    print("  Running RAGE attack (frustration=0.8)...", flush=True)
    state_rage_attack = brain.process(frustration=0.8)
    rage_attack = dict(state_rage_attack.all_rates)

    results["rage"] = {
        "baseline": rage_baseline,
        "social_encounter": rage_social,
        "investigation": rage_inv,
        "attack": rage_attack,
    }

    # ── SEEKING scenarios ──
    print("  Running SEEKING tonic (reward=0.0)...", flush=True)
    state_seek_tonic = brain.process(reward=0.0)
    seek_tonic = dict(state_seek_tonic.all_rates)

    print("  Running SEEKING phasic_burst (reward=0.8)...", flush=True)
    state_seek_burst = brain.process(reward=0.8)
    seek_burst = dict(state_seek_burst.all_rates)

    print("  Running SEEKING pause (reward=-0.1, loss=0.5)...", flush=True)
    # Negative RPE: loss activates LHb which inhibits VTA
    state_seek_pause = brain.process(reward=0.0, loss=0.5)
    seek_pause = dict(state_seek_pause.all_rates)

    results["seeking"] = {
        "tonic": seek_tonic,
        "phasic_burst": seek_burst,
        "pause": seek_pause,
        "reward": seek_burst,  # NAc shell during reward
    }

    # ── SADNESS scenarios ──
    print("  Running SADNESS baseline (loss=0.0)...", flush=True)
    state_sad_base = brain.process(loss=0.0)
    sad_baseline = dict(state_sad_base.all_rates)

    print("  Running SADNESS depression (loss=0.8)...", flush=True)
    state_sad = brain.process(loss=0.8)
    sad_depressed = dict(state_sad.all_rates)

    results["sadness"] = {
        "baseline": sad_baseline,
        "depression": sad_depressed,
        "reward_omission": sad_depressed,  # habenula in loss
        "sadness_suppressed": sad_depressed,  # DR suppressed in depression
    }

    # ── DISGUST scenarios ──
    print("  Running DISGUST baseline (contamination=0.0)...", flush=True)
    state_disg_base = brain.process(contamination=0.0)
    disg_baseline = dict(state_disg_base.all_rates)

    print("  Running DISGUST stimulus (contamination=0.8)...", flush=True)
    state_disg = brain.process(contamination=0.8)
    disg_stim = dict(state_disg.all_rates)

    results["disgust"] = {
        "baseline": disg_baseline,
        "disgust_stimulus": disg_stim,
        "contamination": disg_stim,
        "disgust_recognition": disg_stim,
    }

    return results


# ════════════════════════════════════════════════════════════════
# Validation logic
# ════════════════════════════════════════════════════════════════

def validate_firing_rates(
    scenarios: dict[str, dict[str, dict[str, float]]],
) -> list[TargetResult]:
    """Compare scenario firing rates against literature targets."""
    results: list[TargetResult] = []

    for emotion, (fr_targets, _ratio_targets) in ALL_TARGETS.items():
        emotion_scenarios = scenarios.get(emotion, {})

        for t in fr_targets:
            condition_data = emotion_scenarios.get(t.condition, {})
            actual = condition_data.get(t.region, -1.0)

            if actual < 0:
                results.append(TargetResult(
                    target_name=f"{t.region}/{t.condition}",
                    emotion=emotion,
                    condition=t.condition,
                    min_val=t.min_hz,
                    max_val=t.max_hz,
                    actual=actual,
                    passed=False,
                    source=t.source,
                    note="REGION NOT FOUND in scenario rates",
                ))
                continue

            passed = t.min_hz <= actual <= t.max_hz
            note = ""
            if not passed:
                if actual < t.min_hz:
                    note = f"TOO LOW by {t.min_hz - actual:.1f} Hz"
                else:
                    note = f"TOO HIGH by {actual - t.max_hz:.1f} Hz"

            results.append(TargetResult(
                target_name=f"{t.region}/{t.condition}",
                emotion=emotion,
                condition=t.condition,
                min_val=t.min_hz,
                max_val=t.max_hz,
                actual=round(actual, 2),
                passed=passed,
                source=t.source,
            note=note,
            ))

    return results


def validate_ratios(
    scenarios: dict[str, dict[str, dict[str, float]]],
) -> list[TargetResult]:
    """Validate firing rate ratios."""
    results: list[TargetResult] = []

    # FEAR ratios
    fear_sc = scenarios.get("fear", {})

    # SOM+/PKCd+ ratio (during cs_evoked)
    cs_data = fear_sc.get("cs_evoked", {})
    som_rate = cs_data.get("cel_som", 0)
    pkcd_rate = cs_data.get("cel_pkcd", 0)
    if pkcd_rate > 0:
        ratio = som_rate / pkcd_rate
    else:
        ratio = float("inf") if som_rate > 0 else 0
    rt = FEAR_RATIOS[0]
    results.append(TargetResult(
        target_name="FEAR: SOM+/PKCd+ ratio",
        emotion="fear",
        condition="cs_evoked",
        min_val=rt.min_ratio,
        max_val=rt.max_ratio,
        actual=round(ratio, 2),
        passed=rt.min_ratio <= ratio <= rt.max_ratio,
        source=rt.source,
        note="" if rt.min_ratio <= ratio <= rt.max_ratio else
              f"ratio={ratio:.2f}, target=[{rt.min_ratio}, {rt.max_ratio}]",
    ))

    # Conditioned/Baseline LA
    la_base = fear_sc.get("baseline", {}).get("la_exc", 0)
    la_cs = cs_data.get("la_exc", 0)
    if la_base > 0:
        ratio_la = la_cs / la_base
    else:
        ratio_la = float("inf") if la_cs > 0 else 0
    rt2 = FEAR_RATIOS[1]
    results.append(TargetResult(
        target_name="FEAR: Conditioned/Baseline LA",
        emotion="fear",
        condition="cs_evoked vs baseline",
        min_val=rt2.min_ratio,
        max_val=rt2.max_ratio,
        actual=round(ratio_la, 2),
        passed=rt2.min_ratio <= ratio_la <= rt2.max_ratio,
        source=rt2.source,
        note="" if rt2.min_ratio <= ratio_la <= rt2.max_ratio else
              f"ratio={ratio_la:.2f}, target=[{rt2.min_ratio}, {rt2.max_ratio}]",
    ))

    # RAGE ratio: VMH attack/baseline
    rage_sc = scenarios.get("rage", {})
    vmh_base = rage_sc.get("baseline", {}).get("vmh", 0)
    vmh_attack = rage_sc.get("attack", {}).get("vmh", 0)
    if vmh_base > 0:
        ratio_vmh = vmh_attack / vmh_base
    else:
        ratio_vmh = float("inf") if vmh_attack > 0 else 0
    rt3 = RAGE_RATIOS[0]
    results.append(TargetResult(
        target_name="RAGE: VMH attack/baseline",
        emotion="rage",
        condition="attack vs baseline",
        min_val=rt3.min_ratio,
        max_val=rt3.max_ratio,
        actual=round(ratio_vmh, 2),
        passed=rt3.min_ratio <= ratio_vmh <= rt3.max_ratio,
        source=rt3.source,
        note="" if rt3.min_ratio <= ratio_vmh <= rt3.max_ratio else
              f"ratio={ratio_vmh:.2f}, target=[{rt3.min_ratio}, {rt3.max_ratio}]",
    ))

    # SEEKING ratio: DA burst/tonic
    seek_sc = scenarios.get("seeking", {})
    vta_tonic = seek_sc.get("tonic", {}).get("vta_da_lat", 0)
    vta_burst = seek_sc.get("phasic_burst", {}).get("vta_da_lat", 0)
    if vta_tonic > 0:
        ratio_vta = vta_burst / vta_tonic
    else:
        ratio_vta = float("inf") if vta_burst > 0 else 0
    rt4 = SEEKING_RATIOS[0]
    results.append(TargetResult(
        target_name="SEEKING: DA burst/tonic",
        emotion="seeking",
        condition="phasic_burst vs tonic",
        min_val=rt4.min_ratio,
        max_val=rt4.max_ratio,
        actual=round(ratio_vta, 2),
        passed=rt4.min_ratio <= ratio_vta <= rt4.max_ratio,
        source=rt4.source,
        note="" if rt4.min_ratio <= ratio_vta <= rt4.max_ratio else
              f"ratio={ratio_vta:.2f}, target=[{rt4.min_ratio}, {rt4.max_ratio}]",
    ))

    # SADNESS ratio: sgACC depression/healthy
    sad_sc = scenarios.get("sadness", {})
    sgacc_base = sad_sc.get("baseline", {}).get("sgacc", 0)
    sgacc_dep = sad_sc.get("depression", {}).get("sgacc", 0)
    if sgacc_base > 0:
        ratio_sg = sgacc_dep / sgacc_base
    else:
        ratio_sg = float("inf") if sgacc_dep > 0 else 0
    rt5 = SADNESS_RATIOS[0]
    results.append(TargetResult(
        target_name="SADNESS: sgACC depression/healthy",
        emotion="sadness",
        condition="depression vs baseline",
        min_val=rt5.min_ratio,
        max_val=rt5.max_ratio,
        actual=round(ratio_sg, 2),
        passed=rt5.min_ratio <= ratio_sg <= rt5.max_ratio,
        source=rt5.source,
        note="" if rt5.min_ratio <= ratio_sg <= rt5.max_ratio else
              f"ratio={ratio_sg:.2f}, target=[{rt5.min_ratio}, {rt5.max_ratio}]",
    ))

    return results


# ════════════════════════════════════════════════════════════════
# Report printer
# ════════════════════════════════════════════════════════════════

def print_report(
    fr_results: list[TargetResult],
    ratio_results: list[TargetResult],
    elapsed: float,
) -> None:
    """Print a formatted validation report."""
    all_results = fr_results + ratio_results
    n_total = len(all_results)
    n_pass = sum(1 for r in all_results if r.passed)
    n_fail = n_total - n_pass
    score = n_pass / n_total if n_total > 0 else 0

    sep = "=" * 80
    thin_sep = "-" * 80

    print()
    print(sep)
    print("  V2 EMOTION CIRCUITS -- QUANTITATIVE VALIDATION REPORT")
    print(f"  232 verified papers | {n_total} targets | elapsed {elapsed:.1f}s")
    print(sep)
    print()

    # ── Firing Rate Targets ──
    print("FIRING RATE TARGETS")
    print(thin_sep)
    print(f"{'Emotion':<10} {'Region/Condition':<30} {'Target (Hz)':<16} {'Actual (Hz)':<13} {'Result':<6} {'Source'}")
    print(thin_sep)

    current_emotion = ""
    for r in fr_results:
        em_label = r.emotion.upper() if r.emotion != current_emotion else ""
        current_emotion = r.emotion
        status = "PASS" if r.passed else "FAIL"
        target_str = f"[{r.min_val:.0f}-{r.max_val:.0f}]"
        actual_str = f"{r.actual:.1f}" if r.actual >= 0 else "N/A"
        note_str = f"  << {r.note}" if r.note else ""
        print(f"{em_label:<10} {r.target_name:<30} {target_str:<16} {actual_str:<13} {status:<6} {r.source}{note_str}")

    print()

    # ── Ratio Targets ──
    print("RATIO TARGETS")
    print(thin_sep)
    print(f"{'Name':<35} {'Target':<16} {'Actual':<13} {'Result':<6} {'Source'}")
    print(thin_sep)

    for r in ratio_results:
        status = "PASS" if r.passed else "FAIL"
        target_str = f"[{r.min_val:.1f}-{r.max_val:.1f}]"
        actual_str = f"{r.actual:.2f}" if r.actual != float("inf") else "INF"
        note_str = f"  << {r.note}" if r.note else ""
        print(f"{r.target_name:<35} {target_str:<16} {actual_str:<13} {status:<6} {r.source}{note_str}")

    print()

    # ── Summary ──
    print(sep)
    print("  SUMMARY")
    print(sep)
    print(f"  Total targets:  {n_total}")
    print(f"  PASS:           {n_pass}")
    print(f"  FAIL:           {n_fail}")
    print(f"  Overall score:  {score:.1%} ({n_pass}/{n_total})")
    print()

    # ── Per-emotion breakdown ──
    print("  Per-emotion breakdown:")
    for emotion in ["fear", "rage", "seeking", "sadness", "disgust"]:
        em_results = [r for r in all_results if r.emotion == emotion]
        em_pass = sum(1 for r in em_results if r.passed)
        em_total = len(em_results)
        em_score = em_pass / em_total if em_total > 0 else 0
        print(f"    {emotion.upper():<12} {em_pass}/{em_total} ({em_score:.0%})")

    print()

    # ── Failed targets with suggestions ──
    failed = [r for r in all_results if not r.passed]
    if failed:
        print("  FAILED TARGETS -- Adjustment Suggestions:")
        print(thin_sep)
        for r in failed:
            print(f"  [{r.emotion.upper()}] {r.target_name}")
            print(f"    Actual: {r.actual:.2f}  |  Target: [{r.min_val:.1f}, {r.max_val:.1f}]")
            if r.actual >= 0 and r.actual < r.min_val:
                deficit_pct = (r.min_val - r.actual) / max(r.min_val, 0.1) * 100
                print(f"    -> Increase drive/weight to {r.target_name.split('/')[0]} "
                      f"(deficit ~{deficit_pct:.0f}%)")
            elif r.actual > r.max_val:
                excess_pct = (r.actual - r.max_val) / max(r.max_val, 0.1) * 100
                print(f"    -> Reduce drive/weight to {r.target_name.split('/')[0]} "
                      f"(excess ~{excess_pct:.0f}%)")
            elif r.actual < 0:
                print(f"    -> Region not found in scenario output")
            print()
    else:
        print("  All targets PASSED!")
        print()

    print(sep)
    print(f"  Validation complete. Score: {score:.1%}")
    print(sep)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 80)
    print("  V2 Emotion Circuits -- Quantitative Validation")
    print("  Comparing against 232-paper literature targets")
    print("=" * 80)
    print()

    t0 = time.time()

    # Create EmotionBrainV2
    print("[1/3] Building EmotionBrainV2...", flush=True)
    brain = EmotionBrainV2()
    print(f"       Built: {brain.total_neurons} spiking neurons, "
          f"{len(brain.population_names)} populations")
    print()

    # Run scenarios
    print("[2/3] Running emotion scenarios...", flush=True)
    scenarios = run_scenarios(brain)
    print()

    # Print raw rates for reference
    print("[DEBUG] Raw firing rates per scenario:")
    for emotion, conditions in scenarios.items():
        print(f"  {emotion.upper()}:")
        for cond, rates in conditions.items():
            # Only print regions relevant to this emotion
            relevant = {k: v for k, v in rates.items() if v > 0.01}
            if relevant:
                sorted_rates = sorted(relevant.items(), key=lambda x: -x[1])[:10]
                rate_strs = [f"{k}={v:.1f}" for k, v in sorted_rates]
                print(f"    {cond}: {', '.join(rate_strs)}")
    print()

    # Validate
    print("[3/3] Validating against literature targets...", flush=True)
    fr_results = validate_firing_rates(scenarios)
    ratio_results = validate_ratios(scenarios)

    elapsed = time.time() - t0

    # Report
    print_report(fr_results, ratio_results, elapsed)

    # Exit code
    all_results = fr_results + ratio_results
    n_pass = sum(1 for r in all_results if r.passed)
    n_total = len(all_results)
    score = n_pass / n_total if n_total > 0 else 0

    if score >= 0.8:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
