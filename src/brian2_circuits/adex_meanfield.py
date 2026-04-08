"""AdEx mean-field モデル。背景脳領域の高速近似。

Destexhe lab (2025, Nature Computational Science) に基づく:
  単一ニューロン(AdEx) → スパイキングネットワーク → mean-field
  の4スケール統一手法を簡略実装。

Wilson-Cowanとの違い:
  - AdExニューロンの集団発火率を閉形式で近似
  - 興奮性/抑制性集団のE-Iバランスを解析的に追跡
  - スパイキング回路とのインターフェースが自然

背景領域に使用:
  島皮質、ACC、海馬、dlPFC
  （これらは恐怖/報酬/ストレスの中核回路ではないが、
  全脳ダイナミクスに寄与する）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdExMFParams:
    """AdEx mean-field パラメータ。"""

    # 集団サイズ
    n_exc: int = 8000       # 興奮性ニューロン数（仮想）
    n_inh: int = 2000       # 抑制性ニューロン数（仮想）

    # AdExニューロンパラメータ（mean-field化済み）
    tau_exc: float = 20.0   # 興奮性時定数 (ms)
    tau_inh: float = 10.0   # 抑制性時定数 (ms)

    # 結合重み（集団レベル）
    w_ee: float = 1.5       # E→E
    w_ei: float = 1.0       # E→I
    w_ie: float = 3.0       # I→E (抑制)
    w_ii: float = 0.5       # I→I

    # 非線形伝達関数パラメータ（AdExのF-I曲線近似）
    gain: float = 0.04      # 利得
    threshold: float = 5.0  # 閾値電流 (nA概念的)

    # 適応
    tau_adaptation: float = 500.0  # 適応時定数 (ms)
    b_adaptation: float = 0.02     # 適応強度

    # 外部入力
    baseline_drive: float = 3.0    # 基底外部入力


@dataclass
class MeanFieldState:
    """1つのmean-field領域の状態。"""

    name: str
    rate_exc: float = 2.0   # 興奮性集団の発火率 (Hz)
    rate_inh: float = 5.0   # 抑制性集団の発火率 (Hz)
    adaptation: float = 0.0  # 適応変数
    output: float = 0.0     # 下流への出力（rate_excから導出）


def _transfer_function(current: float, gain: float, threshold: float) -> float:
    """AdExのF-I曲線の近似（ソフトプラス）。"""
    x = gain * (current - threshold)
    if x > 20:
        return x  # 線形領域
    if x < -20:
        return 0.0
    return np.log1p(np.exp(x))


def step_meanfield(
    state: MeanFieldState,
    ext_exc: float = 0.0,
    ext_inh: float = 0.0,
    neuromod: dict[str, float] | None = None,
    params: AdExMFParams | None = None,
    dt: float = 1.0,
) -> MeanFieldState:
    """mean-field領域を1ステップ(dt ms)更新する。

    dν_e/dt = (-ν_e + F_e(I_e)) / τ_e
    dν_i/dt = (-ν_i + F_i(I_i)) / τ_i
    dw/dt = (-w + b * ν_e) / τ_w

    I_e = w_ee * ν_e - w_ie * ν_i + ext_exc + baseline - w_adapt
    I_i = w_ei * ν_e - w_ii * ν_i + ext_inh
    """
    p = params or AdExMFParams()
    nm = neuromod or {}

    # 神経修飾による利得変調
    ne_mod = nm.get("norepinephrine", 0.0)  # NE: 利得↑
    da_mod = nm.get("dopamine", 0.0)        # DA: 基底入力↑
    sht_mod = nm.get("serotonin", 0.0)      # 5-HT: 抑制↑
    gaba_mod = nm.get("gaba", 0.0)          # GABA: 直接抑制

    effective_gain = p.gain * (1.0 + ne_mod * 0.5)
    effective_baseline = p.baseline_drive + da_mod * 2.0

    # 興奮性集団への入力
    I_exc = (
        p.w_ee * state.rate_exc
        - p.w_ie * state.rate_inh
        + ext_exc
        + effective_baseline
        - state.adaptation
        - gaba_mod * 3.0
    )

    # 抑制性集団への入力
    I_inh = (
        p.w_ei * state.rate_exc
        - p.w_ii * state.rate_inh
        + ext_inh
        + sht_mod * 2.0  # 5-HTは抑制性介在ニューロンを活性化
    )

    # 伝達関数
    target_exc = _transfer_function(I_exc, effective_gain, p.threshold)
    target_inh = _transfer_function(I_inh, p.gain, p.threshold * 0.8)

    # 時間更新
    d_exc = (-state.rate_exc + target_exc) / p.tau_exc * dt
    d_inh = (-state.rate_inh + target_inh) / p.tau_inh * dt
    d_adapt = (-state.adaptation + p.b_adaptation * state.rate_exc) / p.tau_adaptation * dt

    new = MeanFieldState(
        name=state.name,
        rate_exc=max(0.0, state.rate_exc + d_exc),
        rate_inh=max(0.0, state.rate_inh + d_inh),
        adaptation=max(0.0, state.adaptation + d_adapt),
    )
    new.output = new.rate_exc  # 出力 = 興奮性集団の発火率
    return new


class MeanFieldRegion:
    """mean-field領域のラッパー。状態と履歴を管理する。"""

    def __init__(self, name: str, params: AdExMFParams | None = None):
        self.name = name
        self.params = params or AdExMFParams()
        self.state = MeanFieldState(name=name)
        self._history: list[float] = []

    def step(self, ext_exc: float = 0.0, ext_inh: float = 0.0,
             neuromod: dict[str, float] | None = None, dt: float = 1.0) -> MeanFieldState:
        self.state = step_meanfield(self.state, ext_exc, ext_inh, neuromod, self.params, dt)
        self._history.append(self.state.rate_exc)
        return self.state

    @property
    def output(self) -> float:
        return self.state.output

    @property
    def rate_history(self) -> list[float]:
        return list(self._history)

    def reset(self) -> None:
        self.state = MeanFieldState(name=self.name)
        self._history.clear()
