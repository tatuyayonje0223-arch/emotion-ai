"""V2 Emotion Circuits -- Quantitative Validation Against Literature Targets.

Runs each emotion scenario through the SharedCoreNetwork (with all 10 registered
spiking circuits) and compares firing rates against the 232-paper-derived targets
in quantitative_targets_v2.py.

NOTE: Bypasses EmotionBrainV2.process() because its __init__ registers all 10
circuits as spiking but process() still references removed mean-field objects.
Instead, we build the SharedCoreNetwork directly with the same registration
functions and drive the spiking network via run_trial(drive_overrides=...).

Usage:
    PYTHONPATH=. PYTHONIOENCODING=utf-8 python scripts/run_v2_validation.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

from src.brian2_circuits.shared_core_network import SharedCoreNetwork, SharedCoreConfig
from src.brian2_circuits.emotion_circuits_v2 import (
    register_fear_circuit,
    register_rage_circuit,
    register_seeking_circuit,
    register_sadness_circuit,
    register_disgust_circuit,
    register_care_circuit,
    register_panic_grief_circuit,
    register_play_circuit,
    register_lust_circuit,
    register_surprise_circuit,
)
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
# Build the network
# ════════════════════════════════════════════════════════════════

def build_core() -> SharedCoreNetwork:
    """Build a SharedCoreNetwork with all 10 emotion circuits registered."""
    cfg = SharedCoreConfig()
    core = SharedCoreNetwork(cfg)
    register_fear_circuit(core)
    register_rage_circuit(core)
    register_seeking_circuit(core)
    register_sadness_circuit(core)
    register_disgust_circuit(core)
    register_care_circuit(core)
    register_panic_grief_circuit(core)
    register_play_circuit(core)
    register_lust_circuit(core)
    register_surprise_circuit(core)
    core.build()
    return core


# ════════════════════════════════════════════════════════════════
# Drive construction helpers (mirrored from EmotionBrainV2.process)
# ════════════════════════════════════════════════════════════════

def make_overrides(
    cfg: SharedCoreConfig,
    threat: float = 0.0,
    reward: float = 0.0,
    frustration: float = 0.0,
    loss: float = 0.0,
    contamination: float = 0.0,
    novelty: float = 0.0,
    pain: float = 0.0,
) -> dict[str, np.ndarray]:
    """Build drive_overrides dict exactly as EmotionBrainV2.process() does."""
    n_steps = int(cfg.duration_ms / cfg.dt_ms)
    overrides: dict[str, np.ndarray] = {}

    # FEAR drive
    if threat > 0.1 or pain > 0.1:
        la_drive = np.zeros((n_steps, 40))
        cs_start, cs_end = int(50 / cfg.dt_ms), int(250 / cfg.dt_ms)
        la_drive[cs_start:cs_end, :15] = 15.0 * max(0.5, threat * 2)
        if pain > 0.1:
            la_drive[cs_start:cs_end, :] += 10.0 * pain
        overrides["la_exc"] = la_drive

        pl_drive = np.zeros((n_steps, 15))
        pl_drive[cs_start:cs_end, :5] = 4.0 * threat
        overrides["pl"] = pl_drive

        il_drive = np.zeros((n_steps, 15))
        il_drive[cs_start:cs_end, :5] = 4.0 * (1 - threat)
        overrides["il"] = il_drive

    # RAGE drive
    if frustration > 0.1:
        mea_drive = np.zeros((n_steps, 20))
        mea_drive[50:, :] = 8.0 * frustration
        overrides["mea"] = mea_drive

        vmh_drive = np.zeros((n_steps, 25))
        vmh_drive[50:, :] = 10.0 * frustration + 3.0 * threat
        overrides["vmh"] = vmh_drive

    # SEEKING drive
    if reward > 0.1:
        vta_drive = np.zeros((n_steps, 30))
        burst_start = int(100 / cfg.dt_ms)
        burst_end = int(200 / cfg.dt_ms)
        vta_drive[burst_start:burst_end, :] = 5.0 * reward
        overrides["vta_da_lat"] = vta_drive

        ofc_drive = np.zeros((n_steps, 15))
        ofc_drive[50:, :] = 5.0 * reward
        overrides["ofc_reward"] = ofc_drive

    # SADNESS drive
    if loss > 0.1:
        sg_drive = np.zeros((n_steps, 20))
        sg_drive[:, :] = 8.0 * loss
        overrides["sgacc"] = sg_drive

        hab_drive = np.zeros((n_steps, 15))
        hab_drive[:, :] = 5.0 * loss
        overrides["habenula"] = hab_drive

    # DISGUST drive
    if contamination > 0.1:
        nts_drive = np.zeros((n_steps, 10))
        nts_drive[50:, :] = 12.0 * contamination
        overrides["nts_disgust"] = nts_drive

        aic_drive = np.zeros((n_steps, 20))
        aic_drive[50:, :] = 6.0 * contamination
        overrides["aic"] = aic_drive

    # SURPRISE drive
    if novelty > 0.3:
        lc_drive = np.zeros((n_steps, 15))
        lc_drive[50:150, :] = 15.0 * novelty
        overrides["lc"] = lc_drive

    return overrides


# ════════════════════════════════════════════════════════════════
# Scenario runner
# ════════════════════════════════════════════════════════════════

def run_trial(core: SharedCoreNetwork, trial_num: int, **kwargs) -> dict[str, float]:
    """Run one trial with given emotion inputs, return rates dict."""
    cfg = core.cfg
    overrides = make_overrides(cfg, **kwargs)
    result = core.run_trial(drive_overrides=overrides, trial_num=trial_num)
    return dict(result.rates)


def run_scenarios(core: SharedCoreNetwork) -> dict[str, dict[str, dict[str, float]]]:
    """Run all emotion scenarios and collect per-region firing rates.

    Returns:
        {emotion: {condition_label: {region: firing_rate}}}
    """
    results: dict[str, dict[str, dict[str, float]]] = {}
    trial = 0

    def _run(label: str, **kwargs) -> dict[str, float]:
        nonlocal trial
        trial += 1
        print(f"    {label} ...", flush=True)
        return run_trial(core, trial, **kwargs)

    # ── FEAR scenarios ──
    print("  FEAR:", flush=True)
    fear_baseline = _run("baseline (threat=0)")
    fear_evoked = _run("cs_evoked (threat=0.8)", threat=0.8)
    fear_extinction = _run("extinction_recall (threat=0.1)", threat=0.1)

    results["fear"] = {
        "baseline": fear_baseline,
        "cs_evoked": fear_evoked,
        "fear_expression": fear_evoked,
        "fear_burst": fear_evoked,
        "extinction_recall": fear_extinction,
        "freezing": fear_evoked,
    }

    # ── RAGE scenarios ──
    print("  RAGE:", flush=True)
    rage_baseline = _run("baseline (frustration=0)")
    rage_social = _run("social_encounter (frustration=0.3)", frustration=0.3)
    rage_inv = _run("investigation (frustration=0.5)", frustration=0.5)
    rage_attack = _run("attack (frustration=0.8)", frustration=0.8)

    results["rage"] = {
        "baseline": rage_baseline,
        "social_encounter": rage_social,
        "investigation": rage_inv,
        "attack": rage_attack,
    }

    # ── SEEKING scenarios ──
    print("  SEEKING:", flush=True)
    seek_tonic = _run("tonic (reward=0)")
    seek_burst = _run("phasic_burst (reward=0.8)", reward=0.8)
    seek_pause = _run("pause (loss=0.5)", loss=0.5)

    results["seeking"] = {
        "tonic": seek_tonic,
        "phasic_burst": seek_burst,
        "pause": seek_pause,
        "reward": seek_burst,
    }

    # ── SADNESS scenarios ──
    print("  SADNESS:", flush=True)
    sad_baseline = _run("baseline (loss=0)")
    sad_depressed = _run("depression (loss=0.8)", loss=0.8)

    results["sadness"] = {
        "baseline": sad_baseline,
        "depression": sad_depressed,
        "reward_omission": sad_depressed,
        "sadness_suppressed": sad_depressed,
    }

    # ── DISGUST scenarios ──
    print("  DISGUST:", flush=True)
    disg_baseline = _run("baseline (contamination=0)")
    disg_stim = _run("stimulus (contamination=0.8)", contamination=0.8)

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

    def _ratio(num: float, den: float) -> float:
        if den > 0:
            return num / den
        return float("inf") if num > 0 else 0.0

    def _check(name: str, emotion: str, condition: str,
               ratio: float, rt: RatioTarget | None = None,
               min_r: float = 0, max_r: float = 0,
               source: str = "") -> None:
        if rt:
            min_r, max_r, source = rt.min_ratio, rt.max_ratio, rt.source
        passed = min_r <= ratio <= max_r
        note = "" if passed else f"ratio={ratio:.2f}, target=[{min_r}, {max_r}]"
        results.append(TargetResult(
            target_name=name,
            emotion=emotion,
            condition=condition,
            min_val=min_r,
            max_val=max_r,
            actual=round(ratio, 2),
            passed=passed,
            source=source,
            note=note,
        ))

    # FEAR: SOM+/PKCd+ ratio
    fear_sc = scenarios.get("fear", {})
    cs_data = fear_sc.get("cs_evoked", {})
    _check("FEAR: SOM+/PKCd+ ratio", "fear", "cs_evoked",
           _ratio(cs_data.get("cel_som", 0), cs_data.get("cel_pkcd", 0)),
           rt=FEAR_RATIOS[0])

    # FEAR: Conditioned/Baseline LA
    la_base = fear_sc.get("baseline", {}).get("la_exc", 0)
    la_cs = cs_data.get("la_exc", 0)
    _check("FEAR: Conditioned/Baseline LA", "fear", "cs_evoked vs baseline",
           _ratio(la_cs, la_base), rt=FEAR_RATIOS[1])

    # RAGE: VMH attack/baseline
    rage_sc = scenarios.get("rage", {})
    vmh_base = rage_sc.get("baseline", {}).get("vmh", 0)
    vmh_attack = rage_sc.get("attack", {}).get("vmh", 0)
    _check("RAGE: VMH attack/baseline", "rage", "attack vs baseline",
           _ratio(vmh_attack, vmh_base), rt=RAGE_RATIOS[0])

    # SEEKING: DA burst/tonic
    seek_sc = scenarios.get("seeking", {})
    vta_tonic = seek_sc.get("tonic", {}).get("vta_da_lat", 0)
    vta_burst = seek_sc.get("phasic_burst", {}).get("vta_da_lat", 0)
    _check("SEEKING: DA burst/tonic", "seeking", "phasic_burst vs tonic",
           _ratio(vta_burst, vta_tonic), rt=SEEKING_RATIOS[0])

    # SADNESS: sgACC depression/healthy
    sad_sc = scenarios.get("sadness", {})
    sgacc_base = sad_sc.get("baseline", {}).get("sgacc", 0)
    sgacc_dep = sad_sc.get("depression", {}).get("sgacc", 0)
    _check("SADNESS: sgACC depression/healthy", "sadness", "depression vs baseline",
           _ratio(sgacc_dep, sgacc_base), rt=SADNESS_RATIOS[0])

    return results


# ════════════════════════════════════════════════════════════════
# Report printer
# ════════════════════════════════════════════════════════════════

def print_report(
    fr_results: list[TargetResult],
    ratio_results: list[TargetResult],
    elapsed: float,
    n_neurons: int,
    n_pops: int,
) -> None:
    """Print a formatted validation report."""
    all_results = fr_results + ratio_results
    n_total = len(all_results)
    n_pass = sum(1 for r in all_results if r.passed)
    n_fail = n_total - n_pass
    score = n_pass / n_total if n_total > 0 else 0

    sep = "=" * 90
    thin_sep = "-" * 90

    print()
    print(sep)
    print("  V2 EMOTION CIRCUITS -- QUANTITATIVE VALIDATION REPORT")
    print(f"  232 verified papers | {n_total} targets | {n_neurons} neurons | {n_pops} populations")
    print(f"  elapsed {elapsed:.1f}s")
    print(sep)
    print()

    # -- Firing Rate Targets --
    print("FIRING RATE TARGETS")
    print(thin_sep)
    print(f"{'Emotion':<10} {'Region/Condition':<32} {'Target (Hz)':<16} {'Actual (Hz)':<14} {'Result':<6} Source")
    print(thin_sep)

    current_emotion = ""
    for r in fr_results:
        em_label = r.emotion.upper() if r.emotion != current_emotion else ""
        current_emotion = r.emotion
        status = "PASS" if r.passed else "FAIL"
        target_str = f"[{r.min_val:.0f}-{r.max_val:.0f}]"
        actual_str = f"{r.actual:.1f}" if r.actual >= 0 else "N/A"
        note_str = f"  << {r.note}" if r.note else ""
        print(f"{em_label:<10} {r.target_name:<32} {target_str:<16} {actual_str:<14} {status:<6} {r.source}{note_str}")

    print()

    # -- Ratio Targets --
    print("RATIO TARGETS")
    print(thin_sep)
    print(f"{'Name':<38} {'Target':<16} {'Actual':<14} {'Result':<6} Source")
    print(thin_sep)

    for r in ratio_results:
        status = "PASS" if r.passed else "FAIL"
        target_str = f"[{r.min_val:.1f}-{r.max_val:.1f}]"
        if r.actual == float("inf"):
            actual_str = "INF"
        else:
            actual_str = f"{r.actual:.2f}"
        note_str = f"  << {r.note}" if r.note else ""
        print(f"{r.target_name:<38} {target_str:<16} {actual_str:<14} {status:<6} {r.source}{note_str}")

    print()

    # -- Summary --
    print(sep)
    print("  SUMMARY")
    print(sep)
    print(f"  Total targets:  {n_total}")
    print(f"  PASS:           {n_pass}")
    print(f"  FAIL:           {n_fail}")
    print(f"  Overall score:  {score:.1%} ({n_pass}/{n_total})")
    print()

    # Per-emotion breakdown
    print("  Per-emotion breakdown:")
    for emotion in ["fear", "rage", "seeking", "sadness", "disgust"]:
        em_results = [r for r in all_results if r.emotion == emotion]
        em_pass = sum(1 for r in em_results if r.passed)
        em_total = len(em_results)
        em_score = em_pass / em_total if em_total > 0 else 0
        bar = "#" * em_pass + "." * (em_total - em_pass)
        print(f"    {emotion.upper():<12} {em_pass}/{em_total}  ({em_score:>4.0%})  [{bar}]")

    print()

    # -- Failed targets with suggestions --
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
    print("=" * 90)
    print("  V2 Emotion Circuits -- Quantitative Validation")
    print("  Comparing against 232-paper literature targets")
    print("=" * 90)
    print()

    t0 = time.time()

    # Build network
    print("[1/3] Building SharedCoreNetwork with all 10 emotion circuits...", flush=True)
    core = build_core()
    n_neurons = core.total_neurons
    n_pops = len(core.population_names)
    print(f"       Built: {n_neurons} spiking neurons, {n_pops} populations")
    print(f"       Populations: {', '.join(core.population_names)}")
    print()

    # Run scenarios
    print("[2/3] Running emotion scenarios...", flush=True)
    scenarios = run_scenarios(core)
    print()

    # Print raw rates for reference
    print("[DEBUG] Key firing rates per scenario:")
    for emotion, conditions in scenarios.items():
        print(f"  {emotion.upper()}:")
        for cond, rates in conditions.items():
            # Skip duplicate alias conditions
            if cond in ("fear_expression", "fear_burst", "freezing", "reward",
                        "reward_omission", "sadness_suppressed", "contamination",
                        "disgust_recognition"):
                continue
            relevant = {k: v for k, v in sorted(rates.items(), key=lambda x: -x[1])
                        if v > 0.5}
            if relevant:
                rate_strs = [f"{k}={v:.1f}" for k, v in list(relevant.items())[:12]]
                print(f"    {cond}: {', '.join(rate_strs)}")
    print()

    # Validate
    print("[3/3] Validating against literature targets...", flush=True)
    fr_results = validate_firing_rates(scenarios)
    ratio_results = validate_ratios(scenarios)

    elapsed = time.time() - t0

    # Report
    print_report(fr_results, ratio_results, elapsed, n_neurons, n_pops)

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
