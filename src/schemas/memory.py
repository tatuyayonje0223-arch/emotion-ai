"""記憶エントリのスキーマ定義。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """エピソード記憶の1エントリ。情動的重要度を必ず持つ。"""

    memory_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0

    memory_type: Literal["episodic", "semantic", "procedural"] = "episodic"

    # 内容
    event_id: str
    summary: str
    raw_content: str = ""
    tags: list[str] = Field(default_factory=list)

    # 情動情報
    emotional_salience: float = Field(0.0, ge=0.0, le=1.0, description="情動的重要度")
    valence_at_encoding: float = Field(0.0, ge=-1.0, le=1.0)
    arousal_at_encoding: float = Field(0.0, ge=0.0, le=1.0)

    # 信頼度・出所
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    provenance: str = "system"

    # 減衰
    current_strength: float = Field(1.0, ge=0.0, le=1.0, description="時間減衰後の強度")

    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    """記憶検索クエリ。現在の感情状態によるバイアスを含む。"""

    query_text: str = ""
    query_tags: list[str] = Field(default_factory=list)
    current_valence: float = 0.0
    current_arousal: float = 0.3
    affect_bias_weight: float = Field(0.3, ge=0.0, le=1.0)
    max_results: int = 10
    min_strength: float = 0.1


class RetrievalResult(BaseModel):
    """記憶検索結果。"""

    memory: MemoryEntry
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    affect_match_score: float = Field(0.0, ge=0.0, le=1.0)
    combined_score: float = Field(0.0, ge=0.0, le=1.0)
