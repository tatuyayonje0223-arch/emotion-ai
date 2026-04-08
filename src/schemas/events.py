"""イベント（外界入力）のスキーマ定義。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PerceptionSignal(BaseModel):
    """知覚層から抽出された信号。不確実性を必ず含む。"""

    modality: Literal["text", "audio", "facial", "behavioral", "physiological", "context"]
    sentiment_score: float = Field(0.0, ge=-1.0, le=1.0)
    arousal_estimate: float = Field(0.0, ge=0.0, le=1.0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    features: dict[str, Any] = Field(default_factory=dict)


class AppraisalResult(BaseModel):
    """評価層の出力。イベントを目標・記憶・文脈に照らして評価した結果。"""

    goal_relevance: float = Field(0.0, ge=-1.0, le=1.0, description="正=促進, 負=阻害")
    novelty: float = Field(0.0, ge=0.0, le=1.0)
    controllability: float = Field(0.5, ge=0.0, le=1.0)
    uncertainty_change: float = Field(0.0, ge=-1.0, le=1.0)
    social_significance: float = Field(0.0, ge=0.0, le=1.0)
    reward_threat_balance: float = Field(0.0, ge=-1.0, le=1.0, description="正=報酬, 負=脅威")
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class EmotionEvent(BaseModel):
    """システムに入力されるイベント。知覚信号と文脈を含む。"""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: Literal[
        "user_message", "system_event", "environmental",
        "social_feedback", "internal_tick",
    ]
    source: str = "unknown"
    raw_content: str = ""
    perception_signals: list[PerceptionSignal] = Field(default_factory=list)
    appraisal: AppraisalResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
