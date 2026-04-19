"""Baseline rate validation — probes resting-state firing rates without scenario drive.

Exposes a validation gap: scripts/optimize_adex.py + quantitative_targets_v2.py test only
SCENARIO-EVOKED rates. Resting-state baseline rates are not checked, so populations can
pass "STRICT 100%" while firing 2-100x their physiological baseline.

This script runs both AdEx and Izhikevich with no inputs and compares baseline rates
to published physiological tonic ranges.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Published tonic baseline ranges (no stimulus, awake/resting state).
# Format: population -> (lo_hz, hi_hz, citation)
BASELINE_TARGETS = {
    # Monoaminergic cells — low tonic firing in awake resting state
    "lc":         (1.0, 3.0,  "Sara & Bouret 2012 tonic LC 1-3 Hz"),
    "dr":         (1.0, 3.0,  "Aghajanian 1977 / Allers 2003 DR 5-HT tonic"),
    "vta_da_lat": (3.0, 8.0,  "Grace 2007 VTA DA tonic 3-8 Hz"),
    # Cortical/limbic baseline
    "il":         (1.0, 5.0,  "Quirk 2002 IL baseline rare spontaneous"),
    "pl":         (1.0, 5.0,  "Courtin 2014 PL baseline"),
    "aic":        (1.0, 5.0,  "Craig 2009 insula baseline"),
    "sgacc":      (1.0, 8.0,  "Mayberg 1999 sgACC baseline ~healthy"),
    "dacc":       (1.0, 5.0,  "Eisenberger 2003 dACC baseline"),
    # Limbic/subcortical
    "la_exc":     (1.0, 5.0,  "Quirk 2002 LA baseline 1-5 Hz"),
    "ba_exc":     (3.0, 8.0,  "Duvarci & Pare 2014 BA 3-8 Hz"),
    "mea":        (3.0, 8.0,  "Hong 2014 MeA 3-8 Hz"),
    "vmh":        (2.0, 5.0,  "Lee 2014 VMH baseline 2-5 Hz"),
    "bnst":       (3.0, 5.0,  "Davis 2010 BNST baseline 3-5 Hz"),
    # Striatum — near silent at rest
    "putamen":        (0.0, 1.0, "Humphries 2005 MSN <1 Hz at rest"),
    "nac_shell_d1":   (0.0, 1.0, "MSN D1 near silent at rest"),
    "nac_shell_d2":   (0.0, 1.0, "MSN D2 near silent at rest"),
    # Habenula / other
    "habenula":   (1.0, 10.0, "Matsumoto 2007 LHb baseline (broad)"),
    "mpoa":       (1.0, 5.0,  "Kohl 2018 MPOA baseline"),
    "pvn_crh":    (1.0, 5.0,  "PVN CRH baseline"),
    "pvn_oxt":    (0.5, 3.0,  "Bhatt 2019 OXT pulsatile baseline"),
}


def probe_baseline(use_adex: bool) -> dict[str, float]:
    from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
    from src.brian2_circuits.shared_core_network import SharedCoreConfig

    cfg = SharedCoreConfig(use_adex=use_adex)
    brain = EmotionBrainV2(config=cfg)
    result = brain.process()  # no inputs -> resting state
    return dict(result.all_rates)


def report(label: str, rates: dict[str, float]) -> tuple[int, int]:
    print(f"\n=== {label} baseline rates ===")
    passed = 0
    total = 0
    for pop, (lo, hi, note) in BASELINE_TARGETS.items():
        rate = rates.get(pop, 0.0)
        total += 1
        in_range = lo <= rate <= hi
        if in_range:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {status}  {pop}: {rate:.2f} Hz  [{lo}-{hi}]  ({note})")
    print(f"\n  {label}: {passed}/{total} baseline PASS")
    return passed, total


if __name__ == "__main__":
    t0 = time.time()
    print("Probing baseline firing rates (no scenario drive)...")

    adex_rates = probe_baseline(use_adex=True)
    izh_rates  = probe_baseline(use_adex=False)

    a_pass, a_total = report("AdEx", adex_rates)
    i_pass, i_total = report("Izhikevich", izh_rates)

    elapsed = time.time() - t0
    print(f"\n  Summary: AdEx {a_pass}/{a_total}  |  Izh {i_pass}/{i_total}")
    print(f"  Elapsed: {elapsed:.1f}s")
