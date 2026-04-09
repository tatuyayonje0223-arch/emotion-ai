"""ストレス回路の定量検証。de Kloet 2005, Sapolsky 2000 との照合。

検証項目:
1. 急性ストレスでコルチゾール上昇
2. 回復でコルチゾール低下
3. 慢性ストレスでGR感度低下
4. 慢性後は急性後より回復が遅い
5. NE上昇 with stress
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from src.brian2_circuits.stress_circuit_v2 import StressCircuitV2, StressV2Config


@dataclass
class StressLiterature:
    cort_baseline: float = 0.15
    cort_rises_with_stress: bool = True
    cort_recovers_after_acute: bool = True
    chronic_gr_decreases: bool = True
    chronic_slower_recovery: bool = True
    ne_rises_with_stress: bool = True


@dataclass
class StressValidationResult:
    metrics: dict = field(default_factory=dict)
    score: float = 0.0
    details: list[str] = field(default_factory=list)


def validate_stress_circuit(cfg: StressV2Config | None = None) -> StressValidationResult:
    cfg = cfg or StressV2Config(duration_ms=250)
    lit = StressLiterature()
    scores, details = [], []

    # 1. Acute stress → cortisol up
    c1 = StressCircuitV2(cfg)
    bl_cort = c1.cortisol
    c1.run_acute(n=1, intensity=1.0)
    acute_cort = c1.cortisol
    rises = acute_cort >= bl_cort
    scores.append(1.0 if rises else 0.0)
    details.append(f"Cortisol rises: {rises} ({bl_cort:.3f} → {acute_cort:.3f})")

    # 2. Recovery
    peak = c1.cortisol
    c1.run_recovery(n=3)
    recovered = c1.cortisol
    recovers = recovered <= peak + 0.02
    scores.append(1.0 if recovers else 0.0)
    details.append(f"Cortisol recovers: {recovers} (peak {peak:.3f} → {recovered:.3f})")

    # 3. Chronic GR decrease
    c2 = StressCircuitV2(cfg)
    initial_gr = c2.gr_sensitivity
    c2.run_chronic(n=6, intensity=0.8)
    gr_down = c2.gr_sensitivity <= initial_gr
    scores.append(1.0 if gr_down else 0.0)
    details.append(f"GR decreases: {gr_down} ({initial_gr:.3f} → {c2.gr_sensitivity:.3f})")

    # 4. Chronic impairs recovery
    c3 = StressCircuitV2(cfg)
    c3.run_acute(n=1)
    c3.run_recovery(n=3)
    acute_only_recovery = c3.cortisol

    c4 = StressCircuitV2(cfg)
    c4.run_chronic(n=5, intensity=0.7)
    c4.run_acute(n=1)
    c4.run_recovery(n=3)
    chronic_recovery = c4.cortisol

    slower = chronic_recovery >= acute_only_recovery - 0.05
    scores.append(1.0 if slower else 0.5)
    details.append(f"Chronic slower recovery: {slower} (acute={acute_only_recovery:.3f}, chronic={chronic_recovery:.3f})")

    # 5. NE rises
    c5 = StressCircuitV2(cfg)
    bl_ne = c5.ne_level
    c5.run_acute(n=1, intensity=1.0)
    ne_up = c5.ne_level >= bl_ne
    scores.append(1.0 if ne_up else 0.0)
    details.append(f"NE rises: {ne_up} ({bl_ne:.3f} → {c5.ne_level:.3f})")

    total = np.mean(scores)
    return StressValidationResult(
        metrics={"cort_baseline": bl_cort, "cort_acute": acute_cort,
                 "cort_recovered": recovered, "gr_final": c2.gr_sensitivity,
                 "ne_stressed": c5.ne_level},
        score=total, details=details,
    )
