"""セッション管理のテスト。"""

from src.api.session import ConversationSession, SessionManager
from src.config.settings import ExperimentConfig
from src.self_model.model import StableTraits


class TestConversationSession:
    def test_create_session(self):
        session = ConversationSession()
        assert session.session_id is not None
        assert session.info.interaction_count == 0

    def test_process_text(self):
        session = ConversationSession()
        result = session.process(text="嬉しいニュースです！")
        assert "state" in result
        assert "safety" in result
        assert "self_assessment" in result
        assert result["step"] > 0

    def test_multiple_interactions(self):
        session = ConversationSession()
        for text in ["こんにちは", "嬉しい", "不安です"]:
            session.process(text)
        assert session.info.interaction_count == 3

    def test_trajectory_recording(self):
        session = ConversationSession()
        session.process("テスト")
        session.process("テスト2")
        assert session.trajectory.size == 2

    def test_get_stats(self):
        session = ConversationSession()
        session.process("テスト")
        stats = session.get_stats()
        assert "total_steps" in stats
        assert "interaction_count" in stats

    def test_self_model_assessment(self):
        session = ConversationSession()
        result = session.process("脅威があります！危険！")
        assert "self_assessment" in result
        assert result["self_assessment"]["confidence"] > 0

    def test_custom_traits(self):
        traits = StableTraits(threat_sensitivity=0.9, emotional_stability=0.3)
        session = ConversationSession(traits=traits)
        assert session.self_model.traits.threat_sensitivity == 0.9

    def test_reset(self):
        session = ConversationSession()
        session.process("テスト")
        session.reset()
        assert session.info.interaction_count == 0

    def test_trajectory_chart(self):
        session = ConversationSession()
        for i in range(10):
            session.process(f"メッセージ{i}")
        chart = session.get_trajectory_chart("valence")
        assert "valence" in chart


class TestSessionManager:
    def test_create_and_list(self):
        manager = SessionManager()
        s1 = manager.create_session()
        s2 = manager.create_session()
        sessions = manager.list_sessions()
        assert len(sessions) == 2

    def test_get_session(self):
        manager = SessionManager()
        s = manager.create_session()
        retrieved = manager.get_session(s.session_id)
        assert retrieved is not None
        assert retrieved.session_id == s.session_id

    def test_remove_session(self):
        manager = SessionManager()
        s = manager.create_session()
        assert manager.remove_session(s.session_id) is True
        assert manager.get_session(s.session_id) is None

    def test_independent_sessions(self):
        manager = SessionManager()
        s1 = manager.create_session()
        s2 = manager.create_session()
        s1.process("嬉しい")
        assert s1.info.interaction_count == 1
        assert s2.info.interaction_count == 0
