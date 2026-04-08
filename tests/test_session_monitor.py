"""セッション安全監視のテスト。"""

from src.safety.session_monitor import SessionSafetyMonitor
from src.schemas.affect_state import AffectState


class TestSessionSafetyMonitor:
    def test_no_alerts_normal(self):
        monitor = SessionSafetyMonitor()
        state = AffectState()
        alerts = monitor.record_state(state)
        assert len(alerts) == 0

    def test_rapid_trust_escalation(self):
        monitor = SessionSafetyMonitor()
        for i in range(15):
            trust = 0.3 + i * 0.05
            state = AffectState(trust=min(1.0, trust))
            alerts = monitor.record_state(state)
        # 10ステップでtrust 0.3→0.8 = +0.5 → アラート
        has_attachment = any(a.alert_type == "rapid_attachment" for a in monitor.all_alerts)
        assert has_attachment

    def test_persistent_negative_state(self):
        monitor = SessionSafetyMonitor()
        for _ in range(20):
            state = AffectState(valence=-0.7)
            monitor.record_state(state)
        has_drift = any(a.alert_type == "state_drift" for a in monitor.all_alerts)
        assert has_drift

    def test_fatigue_overload(self):
        monitor = SessionSafetyMonitor()
        for _ in range(10):
            state = AffectState(fatigue=0.85)
            monitor.record_state(state)
        has_fatigue = any(a.alert_type == "fatigue_overload" for a in monitor.all_alerts)
        assert has_fatigue

    def test_session_summary(self):
        monitor = SessionSafetyMonitor()
        for _ in range(5):
            monitor.record_state(AffectState())
        summary = monitor.get_session_summary()
        assert summary["interaction_count"] == 5
        assert "trust_range" in summary

    def test_has_critical(self):
        monitor = SessionSafetyMonitor()
        assert monitor.has_critical is False
        for _ in range(20):
            monitor.record_state(AffectState(valence=-0.8))
        assert monitor.has_critical is True

    def test_reset(self):
        monitor = SessionSafetyMonitor()
        for _ in range(5):
            monitor.record_state(AffectState(valence=-0.9))
        monitor.reset()
        assert len(monitor.all_alerts) == 0
        assert monitor.get_session_summary()["interaction_count"] == 0
