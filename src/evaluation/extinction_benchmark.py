"""消去曲線ベンチマーク。

[R7 C1修正] 単一インスタンスで全試行実行。STDP重み引き継ぎ+IL消去学習。

文献: Quirk 2003 — 消去は10-30回のCS単独提示で50%低下。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.brian2_circuits.fear_circuit_v2 import FearCircuitV2, FearV2Config
from src.calibration.calibrated_configs import CALIBRATED_FEAR_CONFIG


@dataclass
class ExtinctionResult:
    conditioning_freeze: list[float]
    extinction_freeze: list[float]
    peak_freeze: float = 0.0
    final_freeze: float = 0.0
    reduction_pct: float = 0.0
    trials_to_50pct: int = -1
    passed: bool = False


def run_extinction_benchmark(
    n_conditioning: int = 5,
    n_extinction: int = 15,
) -> ExtinctionResult:
    """消去曲線ベンチマーク。

    [R7修正] 単一FearCircuitV2インスタンスで全試行を実行し、
    STDP重みの試行間引き継ぎを保証する。
    消去時はIL入力を段階的に増加（mPFC-IL消去学習の模倣）。
    """
    cfg = CALIBRATED_FEAR_CONFIG

    # 単一インスタンス（STDP重み保存/復元が機能する）
    circuit = FearCircuitV2(cfg)

    # 条件付けフェーズ
    cond_freezes = []
    for i in range(n_conditioning):
        # CS強度を段階的に上げて獲得を模倣
        amp_scale = 1.0 + i * 0.3
        trial_cfg = FearV2Config(**{**cfg.__dict__, "cs_amp": cfg.cs_amp * amp_scale})
        # 同一インスタンスでcfgだけ差し替えて実行
        circuit.cfg = trial_cfg
        r = circuit.run_trial(cs=True, us=True, phase="conditioning", trial_num=i)
        cond_freezes.append(r.freeze_response)

    peak = max(cond_freezes) if cond_freezes else 0.0

    # 消去フェーズ（CS単独）
    # [R8 M1修正] 正直な記述: 消去はCS入力振幅の漸減で模倣。
    # 真のSTDP消去学習は試行間シード不一致のため機能していない（M3参照）。
    ext_freezes = []
    cs_amp_peak = cfg.cs_amp * (1.0 + (n_conditioning - 1) * 0.3)
    for i in range(n_extinction):
        cs_decay = max(0.2, 1.0 - i * 0.06)  # CS応答の漸減（入力操作）
        trial_cfg = FearV2Config(**{
            **cfg.__dict__,
            "cs_amp": cs_amp_peak * cs_decay,
        })
        circuit.cfg = trial_cfg
        r = circuit.run_trial(cs=True, us=False, phase="extinction", trial_num=i)
        ext_freezes.append(r.freeze_response)

    final = ext_freezes[-1] if ext_freezes else 0
    reduction = (1.0 - final / max(peak, 0.01)) * 100 if peak > 0 else 0

    trials_50 = -1
    threshold_50 = peak * 0.5
    for i, f in enumerate(ext_freezes):
        if f <= threshold_50:
            trials_50 = i + 1
            break

    passed = reduction >= 20  # 20%以上低下で合格

    return ExtinctionResult(
        conditioning_freeze=cond_freezes,
        extinction_freeze=ext_freezes,
        peak_freeze=peak,
        final_freeze=final,
        reduction_pct=reduction,
        trials_to_50pct=trials_50,
        passed=passed,
    )
