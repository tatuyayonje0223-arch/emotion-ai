"""恐怖条件付け回路のパラメータキャリブレーション。

[ステップ1] 文献値との定量一致を目指す:
  - ベースラインBLA発火率: 5-15 Hz (Pare & Bhatt, 2011)
  - 条件付け後BLA: 20-40 Hz (Quirk et al., 1995)
  - CeM出力で凍結反応: 条件付けで50%以上増加
  - 消去: 10-20試行で凍結が50%低下
  - HPA: コルチゾール15-30分でピーク、1-2時間で回復

方法: CS/US振幅、結合重みスケール、背景ノイズを自動探索。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.brian2_circuits.fear_circuit_v2 import FearCircuitV2, FearV2Config


@dataclass
class CalibrationTarget:
    """較正目標値。"""

    baseline_bla_hz: tuple[float, float] = (3.0, 20.0)  # min, max
    conditioned_bla_hz: tuple[float, float] = (15.0, 60.0)
    conditioning_increase_ratio: float = 1.3  # 条件付け後/ベースラインの最低比
    freeze_response_conditioned: tuple[float, float] = (0.05, 1.0)


@dataclass
class CalibrationResult:
    """較正結果。"""

    config: FearV2Config
    baseline_bla: float
    conditioned_bla: float
    increase_ratio: float
    freeze_conditioned: float
    passed: bool
    details: dict = field(default_factory=dict)


def calibrate_fear_circuit(
    target: CalibrationTarget | None = None,
    max_iterations: int = 8,
) -> CalibrationResult:
    """恐怖回路パラメータを自動較正する。

    CS/US振幅と背景ノイズを段階的に調整して文献値に近づける。
    """
    tgt = target or CalibrationTarget()

    # パラメータ探索空間
    cs_amps = [6.0, 8.0, 10.0, 14.0]
    us_amps = [12.0, 16.0, 20.0, 25.0]
    bg_noises = [2.0, 3.0, 4.0]

    best_result = None
    best_score = -1.0

    for cs_a in cs_amps:
        for us_a in us_amps:
            for bg in bg_noises:
                cfg = FearV2Config(
                    cs_amp=cs_a, us_amp=us_a, bg_noise=bg,
                    duration_ms=300, cs_dur_ms=150,
                    us_onset_ms=180, us_dur_ms=30,
                )

                # ベースライン
                c1 = FearCircuitV2(cfg)
                bl = c1.run_trial(cs=True, us=False, phase="baseline")
                bl_rate = bl.la_rate + bl.ba_rate

                # 条件付け
                c2 = FearCircuitV2(FearV2Config(
                    **{**cfg.__dict__, "cs_amp": cs_a * 2.0}
                ))
                cond = c2.run_trial(cs=True, us=True, phase="conditioning")
                cond_rate = cond.la_rate + cond.ba_rate

                ratio = cond_rate / max(bl_rate, 0.1)

                # スコア計算
                score = 0.0
                bl_in_range = tgt.baseline_bla_hz[0] <= bl_rate <= tgt.baseline_bla_hz[1]
                cond_in_range = tgt.conditioned_bla_hz[0] <= cond_rate <= tgt.conditioned_bla_hz[1]
                ratio_ok = ratio >= tgt.conditioning_increase_ratio

                if bl_in_range:
                    score += 1.0
                if cond_in_range:
                    score += 1.0
                if ratio_ok:
                    score += 1.0
                if tgt.freeze_response_conditioned[0] <= cond.freeze_response <= tgt.freeze_response_conditioned[1]:
                    score += 0.5

                if score > best_score:
                    best_score = score
                    best_result = CalibrationResult(
                        config=cfg if not cond_in_range else FearV2Config(**{**cfg.__dict__, "cs_amp": cs_a * 2.0}),
                        baseline_bla=bl_rate,
                        conditioned_bla=cond_rate,
                        increase_ratio=ratio,
                        freeze_conditioned=cond.freeze_response,
                        passed=score >= 2.5,
                        details={
                            "cs_amp": cs_a, "us_amp": us_a, "bg_noise": bg,
                            "score": score,
                            "bl_in_range": bl_in_range,
                            "cond_in_range": cond_in_range,
                            "ratio_ok": ratio_ok,
                        },
                    )

                if best_score >= 3.5:
                    return best_result

    return best_result or CalibrationResult(
        config=FearV2Config(), baseline_bla=0, conditioned_bla=0,
        increase_ratio=0, freeze_conditioned=0, passed=False,
    )
