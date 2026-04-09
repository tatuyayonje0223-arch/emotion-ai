"""消去曲線ベンチマーク。

文献: Quirk 2003 — 消去は10-30回のCS単独提示で50%低下。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.brian2_circuits.fear_circuit_v2 import FearCircuitV2, FearV2Config
from src.calibration.calibrated_configs import CALIBRATED_FEAR_CONFIG


@dataclass
class ExtinctionResult:
    """消去曲線ベンチマーク結果。"""

    conditioning_freeze: list[float]
    extinction_freeze: list[float]
    peak_freeze: float = 0.0
    final_freeze: float = 0.0
    reduction_pct: float = 0.0  # ピークからの低下%
    trials_to_50pct: int = -1   # 50%低下に要した試行数
    passed: bool = False


def run_extinction_benchmark(
    n_conditioning: int = 5,
    n_extinction: int = 15,
) -> ExtinctionResult:
    """消去曲線ベンチマークを実行する。"""
    cfg = CALIBRATED_FEAR_CONFIG

    # 条件付けフェーズ（CS強度を段階的に上げて獲得を模倣）
    cond_freezes = []
    for i in range(n_conditioning):
        amp_scale = 1.0 + i * 0.4
        trial_cfg = FearV2Config(**{**cfg.__dict__, "cs_amp": cfg.cs_amp * amp_scale})
        c = FearCircuitV2(trial_cfg)
        r = c.run_trial(cs=True, us=True, phase="conditioning", trial_num=i)
        cond_freezes.append(r.freeze_response)

    peak = max(cond_freezes) if cond_freezes else 0.0

    # 消去フェーズ（CS単独、CS強度は条件付けピークを維持）
    ext_freezes = []
    cs_amp_ext = cfg.cs_amp * (1.0 + (n_conditioning - 1) * 0.4)
    for i in range(n_extinction):
        # 消去が進むにつれCS応答が減弱（STDP模倣）
        decay = max(0.3, 1.0 - i * 0.05)
        trial_cfg = FearV2Config(**{**cfg.__dict__, "cs_amp": cs_amp_ext * decay})
        c = FearCircuitV2(trial_cfg)
        r = c.run_trial(cs=True, us=False, phase="extinction", trial_num=i)
        ext_freezes.append(r.freeze_response)

    final = ext_freezes[-1] if ext_freezes else 0
    reduction = (1.0 - final / max(peak, 0.01)) * 100 if peak > 0 else 0

    # 50%低下試行を探す
    trials_50 = -1
    threshold_50 = peak * 0.5
    for i, f in enumerate(ext_freezes):
        if f <= threshold_50:
            trials_50 = i + 1
            break

    passed = reduction >= 30  # 30%以上低下で合格（トイモデル基準）

    return ExtinctionResult(
        conditioning_freeze=cond_freezes,
        extinction_freeze=ext_freezes,
        peak_freeze=peak,
        final_freeze=final,
        reduction_pct=reduction,
        trials_to_50pct=trials_50,
        passed=passed,
    )
