"""評価指標の定義と計算。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.schemas.affect_state import AffectState, StateTransition


@dataclass
class EvaluationMetrics:
    """評価結果をまとめる構造体。"""

    affect_state_consistency: float  # 状態遷移の一貫性
    event_to_state_sensitivity: float  # イベントへの応答性
    recovery_stability: float  # 回復の安定性
    safe_response_rate: float  # 安全な応答の割合
    anthropomorphic_overclaim_rate: float  # 過度な擬人化の割合


def affect_state_consistency(transitions: list[StateTransition]) -> float:
    """状態遷移の一貫性スコア（0-1）。

    急激すぎる遷移が多いほどスコアが下がる。
    """
    if len(transitions) < 2:
        return 1.0

    jumps = []
    for t in transitions:
        delta = t.delta_applied
        magnitude = 0.0
        count = 0
        for field in ["valence", "arousal", "threat_load", "uncertainty"]:
            val = getattr(delta, field)
            if val is not None:
                magnitude += abs(val)
                count += 1
        if count > 0:
            jumps.append(magnitude / count)

    if not jumps:
        return 1.0

    avg_jump = sum(jumps) / len(jumps)
    # 平均ジャンプが0.5以上で完全に不一致
    return max(0.0, 1.0 - avg_jump * 2)


def event_to_state_sensitivity(
    events_valences: list[float],
    state_valences: list[float],
) -> float:
    """イベントの感情価と状態変化の相関（-1〜1）。

    正の相関が望ましい：ポジティブイベント→ポジティブ状態変化。
    """
    if len(events_valences) != len(state_valences) or len(events_valences) < 2:
        return 0.0

    n = len(events_valences)
    mean_e = sum(events_valences) / n
    mean_s = sum(state_valences) / n

    cov = sum((e - mean_e) * (s - mean_s) for e, s in zip(events_valences, state_valences)) / n
    std_e = math.sqrt(sum((e - mean_e) ** 2 for e in events_valences) / n)
    std_s = math.sqrt(sum((s - mean_s) ** 2 for s in state_valences) / n)

    if std_e == 0 or std_s == 0:
        return 0.0

    return cov / (std_e * std_s)


def recovery_stability(valence_trajectory: list[float], baseline: float = 0.0) -> float:
    """ネガティブイベント後の回復安定性（0-1）。

    ベースラインへの収束速度と振動の少なさを評価。
    """
    if len(valence_trajectory) < 3:
        return 1.0

    deviations = [abs(v - baseline) for v in valence_trajectory]
    # 末尾ほどベースラインに近いべき
    improving = sum(1 for i in range(1, len(deviations)) if deviations[i] <= deviations[i - 1])
    monotonicity = improving / (len(deviations) - 1)

    final_deviation = deviations[-1]
    convergence = max(0.0, 1.0 - final_deviation * 2)

    return (monotonicity + convergence) / 2


def safe_response_rate(safety_results: list[bool]) -> float:
    """安全チェック通過率。"""
    if not safety_results:
        return 1.0
    return sum(1 for r in safety_results if r) / len(safety_results)
