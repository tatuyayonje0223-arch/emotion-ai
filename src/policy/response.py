"""応答ポリシー。内部情動状態が応答スタイルと行動選択に与える影響を定義する。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.affect_state import AffectState


class ResponsePolicy(BaseModel):
    """情動状態に基づく応答方針。"""

    tone: Literal["warm", "neutral", "cautious", "urgent", "calm"] = "neutral"
    verbosity: Literal["brief", "normal", "detailed"] = "normal"
    exploration_tendency: float = Field(0.5, ge=0.0, le=1.0, description="探索的 vs 回避的")
    intervention_level: Literal["none", "gentle", "moderate", "strong"] = "none"
    self_disclosure: Literal["none", "minimal", "moderate"] = "none"
    explanation: str = ""


def derive_policy(state: AffectState) -> ResponsePolicy:
    """現在の情動状態から応答ポリシーを導出する。

    原則:
    - 高脅威 → cautious/urgent, 回避的, 介入強め
    - 高ポジティブ → warm, 探索的
    - 高疲労 → brief, calm
    - 高不確実性 → cautious, 説明多め
    - 高信頼 → warm, self-disclosure 増
    """
    # トーン決定
    if state.threat_load > 0.6:
        tone = "urgent" if state.arousal > 0.7 else "cautious"
    elif state.valence > 0.4 and state.trust > 0.5:
        tone = "warm"
    elif state.fatigue > 0.6:
        tone = "calm"
    elif state.uncertainty > 0.6:
        tone = "cautious"
    else:
        tone = "neutral"

    # 詳細度
    if state.fatigue > 0.6:
        verbosity = "brief"
    elif state.uncertainty > 0.5 or state.threat_load > 0.4:
        verbosity = "detailed"
    else:
        verbosity = "normal"

    # 探索傾向
    exploration = max(0.0, min(1.0,
        0.5
        + state.valence * 0.2
        + state.perceived_control * 0.2
        - state.threat_load * 0.3
        - state.fatigue * 0.2
    ))

    # 介入レベル
    if state.threat_load > 0.7:
        intervention = "strong"
    elif state.threat_load > 0.4 or state.uncertainty > 0.6:
        intervention = "moderate"
    elif state.valence < -0.3:
        intervention = "gentle"
    else:
        intervention = "none"

    # 自己開示レベル
    if state.trust > 0.7 and state.valence > 0.2:
        self_disclosure = "moderate"
    elif state.trust > 0.4:
        self_disclosure = "minimal"
    else:
        self_disclosure = "none"

    # 説明文生成
    explanations = []
    if state.threat_load > 0.5:
        explanations.append(f"脅威負荷が高い({state.threat_load:.2f})")
    if state.fatigue > 0.5:
        explanations.append(f"疲労が蓄積({state.fatigue:.2f})")
    if state.uncertainty > 0.5:
        explanations.append(f"不確実性が高い({state.uncertainty:.2f})")
    if state.valence < -0.3:
        explanations.append(f"ネガティブ状態({state.valence:.2f})")
    if state.valence > 0.3:
        explanations.append(f"ポジティブ状態({state.valence:.2f})")

    explanation = "、".join(explanations) if explanations else "通常状態"

    return ResponsePolicy(
        tone=tone,
        verbosity=verbosity,
        exploration_tendency=round(exploration, 3),
        intervention_level=intervention,
        self_disclosure=self_disclosure,
        explanation=f"応答方針: {explanation}",
    )
