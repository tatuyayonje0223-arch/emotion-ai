"""ホメオスタシスと身体性モデル。HPA軸・自律神経系・内受容。

HPA軸カスケード:
  視床下部(CRH) → 下垂体(ACTH) → 副腎(コルチゾール) → 海馬/PFC(負のフィードバック)

自律神経系:
  交感神経（闘争逃走）↔ 副交感神経（休息消化）のバランス

内受容:
  心拍・呼吸・筋緊張 → 島皮質 → 情動体験への影響
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HPAAxisState(BaseModel):
    """HPA軸（視床下部-下垂体-副腎系）の状態。"""

    crh: float = Field(0.1, ge=0.0, le=1.0, description="CRH（副腎皮質刺激ホルモン放出因子）")
    acth: float = Field(0.1, ge=0.0, le=1.0, description="ACTH（副腎皮質刺激ホルモン）")
    cortisol: float = Field(0.2, ge=0.0, le=1.0, description="コルチゾール")
    cortisol_baseline: float = Field(0.2, description="基底コルチゾールレベル")
    negative_feedback_gain: float = Field(0.6, description="負のフィードバック強度")


class AutonomicState(BaseModel):
    """自律神経系の状態。"""

    sympathetic: float = Field(0.3, ge=0.0, le=1.0, description="交感神経活性（闘争逃走）")
    parasympathetic: float = Field(0.5, ge=0.0, le=1.0, description="副交感神経活性（休息消化）")

    @property
    def balance(self) -> float:
        """自律神経バランス。正=交感優位、負=副交感優位。"""
        return self.sympathetic - self.parasympathetic

    @property
    def total_arousal(self) -> float:
        """自律神経性覚醒度。"""
        return min(1.0, self.sympathetic * 0.7 + (1.0 - self.parasympathetic) * 0.3)


class InteroceptiveState(BaseModel):
    """内受容状態。身体内部からの信号。"""

    heart_rate_signal: float = Field(0.3, ge=0.0, le=1.0, description="心拍関連信号")
    respiratory_signal: float = Field(0.3, ge=0.0, le=1.0, description="呼吸関連信号")
    muscle_tension: float = Field(0.2, ge=0.0, le=1.0, description="筋緊張")
    gut_signal: float = Field(0.2, ge=0.0, le=1.0, description="消化管信号")
    energy_level: float = Field(0.7, ge=0.0, le=1.0, description="エネルギーレベル")
    pain_signal: float = Field(0.0, ge=0.0, le=1.0, description="疼痛信号")

    @property
    def aggregate_distress(self) -> float:
        """身体的苦痛の総合指標。"""
        return min(1.0, (
            self.heart_rate_signal * 0.2
            + self.muscle_tension * 0.2
            + self.pain_signal * 0.3
            + (1.0 - self.energy_level) * 0.15
            + self.gut_signal * 0.15
        ))

    @property
    def insula_input(self) -> float:
        """島皮質への統合内受容入力。"""
        return min(1.0, (
            self.heart_rate_signal * 0.25
            + self.respiratory_signal * 0.15
            + self.muscle_tension * 0.15
            + self.gut_signal * 0.15
            + self.pain_signal * 0.2
            + (1.0 - self.energy_level) * 0.1
        ))


class BodyState(BaseModel):
    """身体状態の統合。"""

    hpa: HPAAxisState = Field(default_factory=HPAAxisState)
    autonomic: AutonomicState = Field(default_factory=AutonomicState)
    interoception: InteroceptiveState = Field(default_factory=InteroceptiveState)


def update_hpa_axis(hpa: HPAAxisState, amygdala_output: float, dt: float = 0.01) -> HPAAxisState:
    """HPA軸を1ステップ更新する。

    扁桃体出力 → CRH↑ → ACTH↑ → コルチゾール↑ → (海馬/PFC経由で)CRH↓（負のフィードバック）
    """
    updated = hpa.model_copy(deep=True)

    # CRH: 扁桃体出力で増加、コルチゾールで抑制
    crh_drive = amygdala_output * 0.5
    crh_inhibition = hpa.cortisol * hpa.negative_feedback_gain
    d_crh = (crh_drive - crh_inhibition - hpa.crh * 0.1) * dt * 5
    updated.crh = max(0.0, min(1.0, hpa.crh + d_crh))

    # ACTH: CRHに追従（遅延あり）
    d_acth = (updated.crh * 0.8 - hpa.acth * 0.15) * dt * 3
    updated.acth = max(0.0, min(1.0, hpa.acth + d_acth))

    # コルチゾール: ACTHに追従（さらに遅延）+ 基底レベルへの強い回帰
    d_cort = (updated.acth * 0.4 - (hpa.cortisol - hpa.cortisol_baseline) * 0.2) * dt * 3
    updated.cortisol = max(0.0, min(1.0, hpa.cortisol + d_cort))

    return updated


def update_autonomic(
    autonomic: AutonomicState,
    amygdala_output: float,
    pag_output: float,
    pfc_inhibition: float,
    dt: float = 0.01,
) -> AutonomicState:
    """自律神経系を更新する。

    扁桃体/PAG → 交感神経↑
    PFC抑制 → 副交感神経↑（リラクゼーション）
    """
    updated = autonomic.model_copy(deep=True)

    # 交感神経: 扁桃体+PAGで活性化、PFC抑制で減衰
    sym_drive = (amygdala_output * 0.5 + pag_output * 0.3) * (1.0 - pfc_inhibition * 0.4)
    d_sym = (sym_drive - autonomic.sympathetic * 0.15) * dt * 3
    updated.sympathetic = max(0.0, min(1.0, autonomic.sympathetic + d_sym))

    # 副交感神経: 交感神経と相反、PFC抑制で回復
    para_drive = 0.5 + pfc_inhibition * 0.3 - amygdala_output * 0.3
    d_para = (para_drive - autonomic.parasympathetic) * dt * 2
    updated.parasympathetic = max(0.0, min(1.0, autonomic.parasympathetic + d_para))

    return updated


def update_interoception(
    intero: InteroceptiveState,
    autonomic: AutonomicState,
    cortisol: float,
    endorphin: float,
    dt: float = 0.01,
) -> InteroceptiveState:
    """内受容状態を自律神経系と神経化学から更新する。"""
    updated = intero.model_copy(deep=True)

    # 心拍: 交感神経で上昇
    updated.heart_rate_signal = max(0.0, min(1.0,
        intero.heart_rate_signal + (autonomic.sympathetic * 0.4 - intero.heart_rate_signal * 0.2) * dt * 3
    ))

    # 呼吸: 覚醒と連動
    updated.respiratory_signal = max(0.0, min(1.0,
        intero.respiratory_signal + (autonomic.total_arousal * 0.3 - intero.respiratory_signal * 0.15) * dt * 3
    ))

    # 筋緊張: 交感神経+コルチゾール
    updated.muscle_tension = max(0.0, min(1.0,
        intero.muscle_tension + ((autonomic.sympathetic + cortisol) * 0.25 - intero.muscle_tension * 0.2) * dt * 2
    ))

    # エネルギー: コルチゾール高値で消耗、副交感優位で回復
    energy_drain = cortisol * 0.1 + autonomic.sympathetic * 0.05
    energy_recovery = autonomic.parasympathetic * 0.08
    updated.energy_level = max(0.0, min(1.0,
        intero.energy_level + (energy_recovery - energy_drain) * dt * 2
    ))

    # 疼痛: エンドルフィンで抑制
    updated.pain_signal = max(0.0, min(1.0,
        intero.pain_signal * (1.0 - endorphin * 0.5)
    ))

    return updated
