"""スキーマのバリデーションテスト。"""

import pytest

from src.schemas.affect_state import AffectDelta, AffectState, StateTransition
from src.schemas.events import AppraisalResult, EmotionEvent, PerceptionSignal
from src.schemas.memory import MemoryEntry, RetrievalQuery


class TestAffectState:
    def test_default_creation(self):
        state = AffectState()
        assert state.valence == 0.0
        assert state.arousal == 0.3
        assert state.regulation_mode == "adaptive"
        assert state.step_count == 0

    def test_to_vector(self):
        state = AffectState()
        vec = state.to_vector()
        assert len(vec) == 8
        assert all(isinstance(v, float) for v in vec)

    def test_variable_names(self):
        names = AffectState.variable_names()
        assert len(names) == 8
        assert "valence" in names
        assert "arousal" in names

    def test_range_validation(self):
        with pytest.raises(Exception):
            AffectState(valence=2.0)  # > 1.0
        with pytest.raises(Exception):
            AffectState(arousal=-0.1)  # < 0.0

    def test_state_id_unique(self):
        s1 = AffectState()
        s2 = AffectState()
        assert s1.state_id != s2.state_id


class TestAffectDelta:
    def test_none_means_no_change(self):
        delta = AffectDelta()
        assert delta.valence is None
        assert delta.arousal is None

    def test_partial_update(self):
        delta = AffectDelta(valence=0.1, arousal=0.2)
        assert delta.valence == 0.1
        assert delta.threat_load is None


class TestPerceptionSignal:
    def test_creation(self):
        signal = PerceptionSignal(
            modality="text",
            sentiment_score=0.5,
            arousal_estimate=0.3,
            confidence=0.8,
        )
        assert signal.modality == "text"
        assert signal.confidence == 0.8


class TestEmotionEvent:
    def test_default_creation(self):
        event = EmotionEvent(event_type="user_message", raw_content="hello")
        assert event.event_type == "user_message"
        assert event.raw_content == "hello"
        assert len(event.perception_signals) == 0


class TestAppraisalResult:
    def test_default_values(self):
        result = AppraisalResult()
        assert result.goal_relevance == 0.0
        assert result.confidence == 0.5


class TestMemoryEntry:
    def test_creation(self):
        entry = MemoryEntry(
            event_id="test-001",
            summary="テスト記憶",
            emotional_salience=0.7,
            valence_at_encoding=0.5,
        )
        assert entry.emotional_salience == 0.7
        assert entry.current_strength == 1.0

    def test_salience_range(self):
        with pytest.raises(Exception):
            MemoryEntry(event_id="x", summary="x", emotional_salience=1.5)
