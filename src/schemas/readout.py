"""共通readoutスキーマ。全バックエンドが使う唯一の情動出力型。

[NC2修正] 5つのreadout計算パスを1つの正規パスに統一する。
SensoryInputとEmotionReadoutをbrain.pyから分離し、全系統の共通型とする。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class SensoryInput(BaseModel):
    """外部感覚入力。全バックエンドの共通入力型。"""

    threat_signal: float = Field(0.0, ge=0.0, le=1.0)
    reward_signal: float = Field(0.0, ge=0.0, le=1.0)
    social_signal: float = Field(0.0, ge=0.0, le=1.0)
    novelty_signal: float = Field(0.0, ge=0.0, le=1.0)
    pain_input: float = Field(0.0, ge=0.0, le=1.0)
    context_input: float = Field(0.0, ge=0.0, le=1.0)


class EmotionReadout(BaseModel):
    """情動readout。全バックエンドの共通出力型。"""

    valence: float = Field(0.0, description="快不快")
    arousal: float = Field(0.0, description="覚醒")
    threat_load: float = Field(0.0, description="脅威")
    reward_drive: float = Field(0.0, description="報酬動機")
    social_warmth: float = Field(0.0, description="社会的温かさ")
    cognitive_control: float = Field(0.0, description="認知制御")
    body_distress: float = Field(0.0, description="身体的苦痛")
    energy: float = Field(0.0, description="エネルギー")
    memory_encoding_boost: float = Field(0.0, description="記憶強化")


def readout_to_affect_state(readout: EmotionReadout, step: int = 0):
    """EmotionReadout → AffectState変換（唯一の正規変換）。"""
    from src.schemas.affect_state import AffectState
    return AffectState(
        valence=max(-1, min(1, readout.valence)),
        arousal=max(0, min(1, readout.arousal)),
        motivational_salience=max(0, min(1, readout.reward_drive)),
        perceived_control=max(0, min(1, readout.cognitive_control)),
        uncertainty=max(0, min(1, 1.0 - readout.cognitive_control)),
        trust=max(0, min(1, readout.social_warmth)),
        threat_load=max(0, min(1, readout.threat_load)),
        fatigue=max(0, min(1, 1.0 - readout.energy)),
        step_count=step,
    )
