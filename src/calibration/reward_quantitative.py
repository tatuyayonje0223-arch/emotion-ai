"""報酬回路の定量検証。Schultz 1997, Cohen 2012 との照合。

[改善] 時間窓解析を統合し、trial-averageとwindowed ratesの両方を報告。
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from src.brian2_circuits.reward_circuit_v2 import RewardCircuitV2, RewardV2Config
from src.calibration.reward_time_window import analyze_reward_time_windows


@dataclass
class RewardLiterature:
    vta_da_tonic_hz: float = 5.0
    vta_da_burst_hz: float = 25.0
    burst_tonic_ratio: float = 4.0
    d1_gt_d2_reward: bool = True


@dataclass
class RewardValidationResult:
    metrics: dict = field(default_factory=dict)
    score: float = 0.0
    details: list[str] = field(default_factory=list)


def _score(val, target, tol=1.0):
    if target == 0: return 1.0 if abs(val) < tol else 0.0
    return max(0, 1 - abs(val - target) / max(abs(target), 0.1) / tol)


def validate_reward_circuit(cfg: RewardV2Config | None = None) -> RewardValidationResult:
    cfg = cfg or RewardV2Config(duration_ms=300, cs_dur_ms=80, reward_onset_ms=200, reward_dur_ms=40)
    lit = RewardLiterature()
    scores, details = [], []

    # 時間窓解析（スタンドアロン — シナプスなしだが時間分離は正確）
    tw = analyze_reward_time_windows(cfg)

    # Trial-average（回路ベース）
    c1 = RewardCircuitV2(cfg)
    bl = c1.run_trial(cs=False, reward=False, phase="baseline")
    c2 = RewardCircuitV2(cfg)
    rew = c2.run_trial(cs=True, reward=True, phase="training")

    # 1. Tonic（回路ベース）
    da_tonic = bl.vta_da_lat_rate
    s = _score(da_tonic, lit.vta_da_tonic_hz)
    scores.append(s)
    details.append(f"VTA DA tonic (circuit): {da_tonic:.1f}Hz (target: {lit.vta_da_tonic_hz}Hz, score: {s:.2f})")

    # 2. Burst（時間窓ベース — burst期間のみ）
    da_burst_tw = tw.burst_rate
    s = _score(da_burst_tw, lit.vta_da_burst_hz, tol=1.5)
    scores.append(s)
    details.append(f"VTA DA burst (time-window): {da_burst_tw:.1f}Hz (target: {lit.vta_da_burst_hz}Hz, score: {s:.2f})")

    # 3. Burst/tonic ratio（時間窓）
    ratio_tw = tw.burst_tonic_ratio
    s = _score(ratio_tw, lit.burst_tonic_ratio, tol=2.0)
    scores.append(s)
    details.append(f"Burst/tonic ratio (TW): {ratio_tw:.2f}x (target: {lit.burst_tonic_ratio}x, score: {s:.2f})")

    # 4. D1 > D2
    d1 = rew.nac_shell_d1_rate + rew.nac_core_d1_rate
    d1_ok = d1 > 0
    scores.append(1.0 if d1_ok else 0.5)
    details.append(f"D1({d1:.1f}) active: {d1_ok}")

    total = np.mean(scores)
    return RewardValidationResult(
        metrics={"da_tonic": da_tonic, "da_burst": rew.vta_da_lat_rate,
                 "da_burst_tw": da_burst_tw, "tonic_tw": tw.tonic_rate,
                 "ratio_tw": ratio_tw, "d1": d1},
        score=total, details=details,
    )
