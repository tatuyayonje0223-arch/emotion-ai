"""脳領域モデル。各領域をニューラルマス（興奮性/抑制性集団）として実装する。

各領域は:
- excitatory: 興奮性集団の活性（0-1）
- inhibitory: 抑制性集団の活性（0-1）
- neuromodulatory_sensitivity: 各神経伝達物質への感度
- 入力を受け取り、内部ダイナミクスを更新し、出力を生成する

Wilson-Cowan型の集団ダイナミクスを基礎とし、
神経修飾による利得変調（gain modulation）を加える。
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field


def _sigmoid(x: float, gain: float = 1.0, threshold: float = 0.0) -> float:
    """シグモイド活性化関数。神経集団の入出力特性。"""
    return 1.0 / (1.0 + math.exp(-gain * (x - threshold)))


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class RegionState(BaseModel):
    """1脳領域の状態。"""

    name: str
    excitatory: float = Field(0.1, ge=0.0, le=1.0, description="興奮性集団活性")
    inhibitory: float = Field(0.1, ge=0.0, le=1.0, description="抑制性集団活性")
    output: float = Field(0.0, ge=0.0, le=1.0, description="下流への出力信号")

    # 内部パラメータ
    tau_e: float = Field(0.05, description="興奮性時定数")
    tau_i: float = Field(0.08, description="抑制性時定数")
    w_ee: float = Field(1.5, description="E→E結合重み")
    w_ei: float = Field(1.0, description="E→I結合重み")
    w_ie: float = Field(1.5, description="I→E結合重み（抑制）")
    w_ii: float = Field(0.5, description="I→I結合重み")
    gain_e: float = Field(4.0, description="興奮性シグモイド利得")
    gain_i: float = Field(4.0, description="抑制性シグモイド利得")
    threshold_e: float = Field(0.3, description="興奮性閾値")
    threshold_i: float = Field(0.3, description="抑制性閾値")
    baseline_input: float = Field(0.0, description="基底入力（自発活動）")


class AmygdalaState(RegionState):
    """扁桃体。脅威検出・情動学習・条件付け。"""
    name: str = "amygdala"
    w_ee: float = 2.0  # 強い自己興奮（脅威持続）
    threshold_e: float = 0.25  # 低閾値（敏感）
    fear_conditioning_strength: float = Field(0.0, ge=0.0, le=1.0)
    ne_sensitivity: float = Field(0.8, description="NE感度（高→過敏）")
    gaba_sensitivity: float = Field(0.7, description="GABA感度（高→抑制されやすい）")


class PFCState(BaseModel):
    """前頭前皮質。3サブ領域を持つ。"""
    name: str = "pfc"
    vmPFC: RegionState = Field(default_factory=lambda: RegionState(
        name="vmPFC", w_ee=1.2, threshold_e=0.35, baseline_input=0.1,
    ))
    dlPFC: RegionState = Field(default_factory=lambda: RegionState(
        name="dlPFC", w_ee=1.0, threshold_e=0.4, baseline_input=0.15,
    ))
    OFC: RegionState = Field(default_factory=lambda: RegionState(
        name="OFC", w_ee=1.3, threshold_e=0.3, baseline_input=0.1,
    ))
    # PFC全体としての扁桃体抑制力
    amygdala_inhibition_strength: float = Field(0.5, ge=0.0, le=1.0)


class InsulaState(RegionState):
    """島皮質。内受容統合・主観的感情の基盤。"""
    name: str = "insula"
    w_ee: float = 1.3
    threshold_e: float = 0.3
    interoceptive_input: float = Field(0.3, ge=0.0, le=1.0, description="内受容信号強度")


class ACCState(RegionState):
    """前帯状皮質。コンフリクト監視・エラー検出。"""
    name: str = "acc"
    w_ee: float = 1.2
    threshold_e: float = 0.35
    conflict_signal: float = Field(0.0, ge=0.0, le=1.0)


class HippocampusState(RegionState):
    """海馬。文脈記憶・情動記憶。"""
    name: str = "hippocampus"
    w_ee: float = 1.4
    threshold_e: float = 0.3
    memory_encoding_strength: float = Field(0.5, ge=0.0, le=1.0)
    cortisol_suppression: float = Field(0.0, ge=0.0, le=1.0, description="コルチゾールによる抑制")


class HypothalamusState(RegionState):
    """視床下部。HPA軸起点・自律神経制御。"""
    name: str = "hypothalamus"
    w_ee: float = 1.0
    threshold_e: float = 0.3
    crh_output: float = Field(0.0, ge=0.0, le=1.0, description="CRH分泌量")
    sympathetic_tone: float = Field(0.3, ge=0.0, le=1.0, description="交感神経緊張度")
    parasympathetic_tone: float = Field(0.5, ge=0.0, le=1.0, description="副交感神経緊張度")


class PAGState(RegionState):
    """中脳水道周囲灰白質。基本的情動反応。"""
    name: str = "pag"
    w_ee: float = 1.8  # 強い反応性
    threshold_e: float = 0.4  # 高閾値（強い入力で発動）
    freeze_output: float = Field(0.0, ge=0.0, le=1.0)
    flight_output: float = Field(0.0, ge=0.0, le=1.0)
    fight_output: float = Field(0.0, ge=0.0, le=1.0)
    endorphin_release: float = Field(0.0, ge=0.0, le=1.0)


class VentralStriatumState(RegionState):
    """腹側線条体/側坐核。報酬処理・動機づけ。"""
    name: str = "ventral_striatum"
    w_ee: float = 1.5
    threshold_e: float = 0.3
    reward_prediction_error: float = Field(0.0, ge=-1.0, le=1.0)
    da_sensitivity: float = Field(0.9, description="ドーパミン感度")


class BrainstemState(BaseModel):
    """脳幹核群。神経伝達物質の産生源。"""
    name: str = "brainstem"
    locus_coeruleus: RegionState = Field(default_factory=lambda: RegionState(
        name="locus_coeruleus", w_ee=1.0, threshold_e=0.3, baseline_input=0.2,
    ))
    raphe_nuclei: RegionState = Field(default_factory=lambda: RegionState(
        name="raphe_nuclei", w_ee=1.0, threshold_e=0.3, baseline_input=0.3,
    ))
    vta: RegionState = Field(default_factory=lambda: RegionState(
        name="vta", w_ee=1.2, threshold_e=0.3, baseline_input=0.2,
    ))


def update_region(
    state: RegionState,
    external_excitatory: float = 0.0,
    external_inhibitory: float = 0.0,
    neuromod_gain: float = 1.0,
    dt: float = 0.01,
) -> RegionState:
    """Wilson-Cowan型ダイナミクスで領域を1ステップ更新する。

    dE/dt = (-E + S_e(w_ee*E - w_ie*I + external_e + baseline)) / tau_e
    dI/dt = (-I + S_i(w_ei*E - w_ii*I + external_i)) / tau_i
    output = E * neuromod_gain
    """
    updated = state.model_copy(deep=True)

    # 興奮性集団への入力
    input_e = (
        state.w_ee * state.excitatory
        - state.w_ie * state.inhibitory
        + external_excitatory
        + state.baseline_input
    )
    target_e = _sigmoid(input_e, gain=state.gain_e * neuromod_gain, threshold=state.threshold_e)
    de = (-state.excitatory + target_e) / state.tau_e * dt
    updated.excitatory = _clamp(state.excitatory + de)

    # 抑制性集団への入力
    input_i = (
        state.w_ei * state.excitatory
        - state.w_ii * state.inhibitory
        + external_inhibitory
    )
    target_i = _sigmoid(input_i, gain=state.gain_i, threshold=state.threshold_i)
    di = (-state.inhibitory + target_i) / state.tau_i * dt
    updated.inhibitory = _clamp(state.inhibitory + di)

    # 出力 = 興奮性活性 × 神経修飾利得
    updated.output = _clamp(updated.excitatory * neuromod_gain)

    return updated
