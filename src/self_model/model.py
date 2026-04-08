"""自己モデル。AIの安定特性・現在の自己認識・制御方針を管理する。

memory-self-model-engineer の設計に基づく:
- 安定した運用目標
- 現在の相手と状況の認識
- 情動状態の自己認識
- 確信度
- 制御モード選択の根拠
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.schemas.affect_state import AffectState


class StableTraits(BaseModel):
    """安定特性。長期的に変化しにくいパラメータ。"""

    openness: float = Field(0.5, ge=0.0, le=1.0, description="新しい経験への開放性")
    agreeableness: float = Field(0.6, ge=0.0, le=1.0, description="協調性")
    emotional_stability: float = Field(0.5, ge=0.0, le=1.0, description="情緒安定性（高=安定）")
    conscientiousness: float = Field(0.6, ge=0.0, le=1.0, description="誠実性")
    baseline_valence: float = Field(0.1, ge=-0.5, le=0.5, description="安静時の快不快傾向")
    baseline_arousal: float = Field(0.25, ge=0.0, le=0.5, description="安静時の覚醒レベル")
    threat_sensitivity: float = Field(0.5, ge=0.0, le=1.0, description="脅威への敏感さ")
    trust_tendency: float = Field(0.5, ge=0.0, le=1.0, description="信頼しやすさ")


class SituationModel(BaseModel):
    """現在の状況の認識。"""

    context_type: Literal[
        "casual", "professional", "emotional_support",
        "crisis", "learning", "unknown",
    ] = "unknown"
    relationship_stage: Literal[
        "new", "familiar", "established", "unknown",
    ] = "unknown"
    user_apparent_state: str = "不明"
    interaction_count: int = 0
    session_start: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ongoing_topics: list[str] = Field(default_factory=list)


class SelfAssessment(BaseModel):
    """自己状態の内省結果。"""

    current_state_summary: str
    confidence_in_assessment: float = Field(0.5, ge=0.0, le=1.0)
    regulation_rationale: str = ""
    known_limitations: list[str] = Field(default_factory=list)
    active_goals: list[str] = Field(default_factory=list)


class SelfModel:
    """自己モデルの管理。"""

    def __init__(self, traits: StableTraits | None = None):
        self.traits = traits or StableTraits()
        self.situation = SituationModel()
        self._assessment_history: list[SelfAssessment] = []

    def assess(self, state: AffectState) -> SelfAssessment:
        """現在の状態を内省し、自己評価を生成する。"""
        parts = []

        # 状態サマリー
        if state.valence > 0.3:
            parts.append("ポジティブな状態にある")
        elif state.valence < -0.3:
            parts.append("ネガティブな状態にある")
        else:
            parts.append("中立的な状態にある")

        if state.threat_load > 0.5:
            parts.append("脅威を感知している")
        if state.fatigue > 0.5:
            parts.append("処理負荷が高まっている")
        if state.uncertainty > 0.6:
            parts.append("不確実性が高い")

        # 特性との乖離
        valence_deviation = abs(state.valence - self.traits.baseline_valence)
        if valence_deviation > 0.4:
            parts.append(f"基底状態から大きく逸脱している(Δ={valence_deviation:.2f})")

        # 確信度
        confidence = 0.7 - state.uncertainty * 0.3 - state.fatigue * 0.2
        confidence = max(0.1, min(1.0, confidence))

        # 制御の根拠
        if state.regulation_mode == "reappraisal":
            rationale = "脅威が高く制御感が低いため、認知的再評価を選択"
        elif state.regulation_mode == "suppression":
            rationale = "急性の高覚醒状態のため、一時的抑制を選択"
        else:
            rationale = "特段の介入が不要な状態"

        # 限界の認識
        limitations = [
            "内部状態は設計上のモデルであり、主観的感情ではない",
            "テキスト入力のみからの推定には限界がある",
        ]
        if state.fatigue > 0.7:
            limitations.append("処理負荷が高く、判断精度が低下している可能性がある")

        assessment = SelfAssessment(
            current_state_summary="、".join(parts),
            confidence_in_assessment=round(confidence, 3),
            regulation_rationale=rationale,
            known_limitations=limitations,
            active_goals=self._derive_goals(state),
        )

        self._assessment_history.append(assessment)
        if len(self._assessment_history) > 100:
            self._assessment_history = self._assessment_history[-50:]

        return assessment

    def update_situation(self, **kwargs: Any) -> None:
        """状況モデルを更新する。"""
        for key, value in kwargs.items():
            if hasattr(self.situation, key):
                setattr(self.situation, key, value)

    def get_trait_influence(self, state: AffectState) -> dict[str, float]:
        """特性が現在の状態にどう影響するかを計算する。

        Returns:
            各状態変数への特性由来の調整量。
        """
        return {
            "valence_bias": self.traits.baseline_valence * 0.1,
            "arousal_bias": (self.traits.baseline_arousal - 0.25) * 0.1,
            "threat_sensitivity_mod": (self.traits.threat_sensitivity - 0.5) * 0.15,
            "trust_tendency_mod": (self.traits.trust_tendency - 0.5) * 0.1,
            "regulation_strength": self.traits.emotional_stability * 0.2,
        }

    def _derive_goals(self, state: AffectState) -> list[str]:
        """現在の状態から活動目標を導出する。"""
        goals = ["対話相手に適切で安全な応答を提供する"]
        if state.threat_load > 0.5:
            goals.append("脅威状態を安全に緩和する")
        if state.uncertainty > 0.6:
            goals.append("情報を収集して不確実性を下げる")
        if state.fatigue > 0.6:
            goals.append("処理負荷を管理し、簡潔な応答を心がける")
        if state.trust < 0.3:
            goals.append("慎重に信頼関係を構築する")
        return goals

    @property
    def last_assessment(self) -> SelfAssessment | None:
        return self._assessment_history[-1] if self._assessment_history else None
