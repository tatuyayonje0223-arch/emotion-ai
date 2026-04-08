"""評価エンジン。イベントを目標・記憶・文脈に照らして評価し、AppraisalResult を返す。"""

from __future__ import annotations

from src.schemas.affect_state import AffectState
from src.schemas.events import AppraisalResult, EmotionEvent, PerceptionSignal


def _aggregate_perception(signals: list[PerceptionSignal]) -> tuple[float, float, float]:
    """知覚信号を信頼度加重平均で統合する。"""
    if not signals:
        return 0.0, 0.0, 0.2

    total_weight = sum(s.confidence for s in signals)
    if total_weight == 0:
        return 0.0, 0.0, 0.1

    sentiment = sum(s.sentiment_score * s.confidence for s in signals) / total_weight
    arousal = sum(s.arousal_estimate * s.confidence for s in signals) / total_weight
    avg_confidence = total_weight / len(signals)

    return sentiment, arousal, avg_confidence


def appraise(
    event: EmotionEvent,
    current_state: AffectState,
    memory_context: list[dict] | None = None,
) -> AppraisalResult:
    """イベントを評価する。

    評価次元:
    - goal_relevance: イベントが現在の目標を促進(+)か阻害(-)か
    - novelty: 予想外度合い
    - controllability: 対処可能性
    - uncertainty_change: 不確実性の増減
    - social_significance: 社会的重要度
    - reward_threat_balance: 報酬と脅威のバランス

    MVP実装: 知覚信号 + 現在の内部状態から推定。
    """
    sentiment, arousal, signal_confidence = _aggregate_perception(event.perception_signals)

    # goal_relevance: sentimentをベースに、現在の制御感で調整
    goal_relevance = sentiment * 0.7 + (current_state.perceived_control - 0.5) * 0.3
    goal_relevance = max(-1.0, min(1.0, goal_relevance))

    # novelty: arousalが高いほど新規性が高い、加えて現在のuncertaintyが低いほど驚き大
    novelty = arousal * 0.6 + (1.0 - current_state.uncertainty) * arousal * 0.4
    novelty = min(1.0, novelty)

    # controllability: 現在の制御感と脅威レベルから推定
    controllability = current_state.perceived_control * 0.7 + (1.0 - current_state.threat_load) * 0.3
    controllability = max(0.0, min(1.0, controllability))

    # uncertainty_change: ネガティブ+高覚醒→不確実性↑、ポジティブ→不確実性↓
    if sentiment < -0.3 and arousal > 0.5:
        uncertainty_change = min(1.0, arousal * 0.5)
    elif sentiment > 0.3:
        uncertainty_change = max(-1.0, -sentiment * 0.3)
    else:
        uncertainty_change = 0.0

    # social_significance: テキストの社会的手がかりから推定（MVP: 簡易）
    social_cues = _count_social_cues(event.raw_content)
    social_significance = min(1.0, social_cues * 0.2 + abs(sentiment) * 0.3)

    # reward_threat_balance
    threat_features = sum(
        s.features.get("threat_hits", 0) for s in event.perception_signals
    )
    if threat_features > 0:
        reward_threat = max(-1.0, sentiment - threat_features * 0.2)
    else:
        reward_threat = sentiment * 0.8
    reward_threat = max(-1.0, min(1.0, reward_threat))

    # 信頼度: 信号の信頼度と情報量
    confidence = signal_confidence * 0.7 + (0.3 if memory_context else 0.1)
    confidence = min(1.0, confidence)

    return AppraisalResult(
        goal_relevance=round(goal_relevance, 4),
        novelty=round(novelty, 4),
        controllability=round(controllability, 4),
        uncertainty_change=round(uncertainty_change, 4),
        social_significance=round(social_significance, 4),
        reward_threat_balance=round(reward_threat, 4),
        confidence=round(confidence, 4),
    )


def _count_social_cues(text: str) -> int:
    """テキスト中の社会的手がかり語数を返す（MVP: 簡易カウント）。"""
    social_words = {
        "みんな", "チーム", "家族", "友達", "上司", "同僚", "社会",
        "people", "team", "family", "friend", "boss", "everyone",
    }
    count = 0
    text_lower = text.lower()
    for w in social_words:
        if w in text_lower:
            count += 1
    return count
