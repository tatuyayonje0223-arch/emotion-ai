"""情動内部状態のスキーマ定義。全状態変数はここで一元管理する。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AffectState(BaseModel):
    """内部情動状態。連続値として保持し、時間経過で更新される。"""

    state_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # 主要状態変数（全て -1.0〜1.0 または 0.0〜1.0）
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="快・不快")
    arousal: float = Field(0.3, ge=0.0, le=1.0, description="覚醒度")
    motivational_salience: float = Field(0.2, ge=0.0, le=1.0, description="動機づけ的重要度")
    perceived_control: float = Field(0.5, ge=0.0, le=1.0, description="主観的制御感")
    uncertainty: float = Field(0.3, ge=0.0, le=1.0, description="不確実性")
    trust: float = Field(0.5, ge=0.0, le=1.0, description="信頼")
    threat_load: float = Field(0.0, ge=0.0, le=1.0, description="脅威負荷")
    fatigue: float = Field(0.0, ge=0.0, le=1.0, description="認知負荷・疲労")

    # 制御モード
    regulation_mode: Literal[
        "reappraisal", "suppression", "acceptance", "adaptive"
    ] = "adaptive"

    # メタ情報
    step_count: int = 0
    last_event_id: str | None = None

    def to_vector(self) -> list[float]:
        """状態変数を数値ベクトルとして返す。"""
        return [
            self.valence,
            self.arousal,
            self.motivational_salience,
            self.perceived_control,
            self.uncertainty,
            self.trust,
            self.threat_load,
            self.fatigue,
        ]

    @staticmethod
    def variable_names() -> list[str]:
        return [
            "valence", "arousal", "motivational_salience", "perceived_control",
            "uncertainty", "trust", "threat_load", "fatigue",
        ]


class AffectDelta(BaseModel):
    """状態変数への加算的変化量。None は変更なしを意味する。"""

    valence: float | None = None
    arousal: float | None = None
    motivational_salience: float | None = None
    perceived_control: float | None = None
    uncertainty: float | None = None
    trust: float | None = None
    threat_load: float | None = None
    fatigue: float | None = None
    regulation_mode: Literal[
        "reappraisal", "suppression", "acceptance", "adaptive"
    ] | None = None


class StateTransition(BaseModel):
    """状態遷移の監査ログエントリ。"""

    transition_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    previous_state_id: UUID
    new_state_id: UUID
    event_id: str
    trigger: str  # "appraisal" | "regulation" | "decay" | "external"
    delta_applied: AffectDelta
    reason: str
