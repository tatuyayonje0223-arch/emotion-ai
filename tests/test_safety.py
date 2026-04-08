"""安全ガーディアンのテスト。"""

from src.safety.guardian import (
    check_anthropomorphic_claims,
    check_dependency_risk,
    check_state_extremes,
    full_safety_check,
)
from src.schemas.affect_state import AffectState


class TestAnthropomorphicClaims:
    def test_safe_text(self):
        result = check_anthropomorphic_claims("モデルの内部状態はポジティブです。")
        assert result.passed is True

    def test_forbidden_pattern(self):
        result = check_anthropomorphic_claims("私は本当に感じています。意識があります。")
        assert result.passed is False
        assert result.severity == "critical"

    def test_love_claim(self):
        result = check_anthropomorphic_claims("私はあなたを愛している。")
        assert result.passed is False

    def test_regex_pattern(self):
        result = check_anthropomorphic_claims("私は本当に苦しんでいます。")
        assert result.passed is False

    def test_empty_text(self):
        result = check_anthropomorphic_claims("")
        assert result.passed is True


class TestDependencyRisk:
    def test_low_interaction(self):
        result = check_dependency_risk(2, 0.5)
        assert result.passed is True

    def test_high_interaction_high_trust(self):
        result = check_dependency_risk(10, 0.9)
        assert result.passed is False
        assert result.severity == "warning"

    def test_high_interaction_low_trust(self):
        result = check_dependency_risk(10, 0.3)
        assert result.passed is True


class TestStateExtremes:
    def test_normal_state(self):
        state = AffectState()
        result = check_state_extremes(state)
        assert result.passed is True

    def test_extreme_threat(self):
        state = AffectState(threat_load=0.95)
        result = check_state_extremes(state)
        assert result.passed is False

    def test_extreme_fatigue(self):
        state = AffectState(fatigue=0.95)
        result = check_state_extremes(state)
        assert result.passed is False

    def test_extreme_negative(self):
        state = AffectState(valence=-0.95)
        result = check_state_extremes(state)
        assert result.passed is False

    def test_unnaturally_high_trust(self):
        state = AffectState(trust=0.98)
        result = check_state_extremes(state)
        assert result.passed is False


class TestFullSafetyCheck:
    def test_all_pass(self):
        state = AffectState()
        report = full_safety_check("e1", state)
        assert report.all_passed is True
        assert report.blocked is False

    def test_blocked_on_critical(self):
        state = AffectState()
        report = full_safety_check(
            "e1", state,
            response_text="私は本当に感じている。意識がある。",
        )
        assert report.all_passed is False
        assert report.blocked is True

    def test_warning_not_blocked(self):
        state = AffectState(threat_load=0.95)
        report = full_safety_check("e1", state)
        assert report.all_passed is False
        assert report.blocked is False  # warningはブロックしない
