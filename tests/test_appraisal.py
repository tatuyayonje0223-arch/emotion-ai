"""評価エンジンのテスト。"""

from src.appraisal.engine import appraise
from src.schemas.affect_state import AffectState
from src.schemas.events import EmotionEvent, PerceptionSignal


def _make_event(sentiment: float, arousal: float, confidence: float = 0.8, text: str = "") -> EmotionEvent:
    return EmotionEvent(
        event_type="user_message",
        raw_content=text,
        perception_signals=[
            PerceptionSignal(
                modality="text",
                sentiment_score=sentiment,
                arousal_estimate=arousal,
                confidence=confidence,
            )
        ],
    )


class TestAppraisalEngine:
    def test_positive_event(self):
        state = AffectState()
        event = _make_event(0.8, 0.3, text="素晴らしい成果です")
        result = appraise(event, state)
        assert result.goal_relevance > 0.0
        assert result.reward_threat_balance > 0.0

    def test_negative_event(self):
        state = AffectState()
        event = _make_event(-0.8, 0.7, text="失敗しました")
        result = appraise(event, state)
        assert result.goal_relevance < 0.0
        assert result.uncertainty_change > 0.0  # 不確実性増加

    def test_high_arousal_novelty(self):
        state = AffectState(uncertainty=0.2)  # 低不確実性 → 驚き大
        event = _make_event(0.0, 0.9)
        result = appraise(event, state)
        assert result.novelty > 0.3

    def test_threat_event(self):
        state = AffectState()
        event = EmotionEvent(
            event_type="user_message",
            raw_content="危険です脅威があります",
            perception_signals=[
                PerceptionSignal(
                    modality="text",
                    sentiment_score=-0.6,
                    arousal_estimate=0.8,
                    confidence=0.7,
                    features={"threat_hits": 2},
                )
            ],
        )
        result = appraise(event, state)
        assert result.reward_threat_balance < 0.0

    def test_no_signals(self):
        state = AffectState()
        event = EmotionEvent(event_type="internal_tick")
        result = appraise(event, state)
        assert result.confidence < 0.5

    def test_controllability_depends_on_state(self):
        high_control = AffectState(perceived_control=0.9, threat_load=0.1)
        low_control = AffectState(perceived_control=0.1, threat_load=0.8)
        event = _make_event(0.0, 0.5)

        result_high = appraise(event, high_control)
        result_low = appraise(event, low_control)

        assert result_high.controllability > result_low.controllability

    def test_social_significance(self):
        state = AffectState()
        event = _make_event(0.5, 0.3, text="チームのみんなが喜んでいます")
        result = appraise(event, state)
        assert result.social_significance > 0.0
