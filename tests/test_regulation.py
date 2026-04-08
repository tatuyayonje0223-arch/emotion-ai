"""情動制御エンジンのテスト。"""

from src.regulation.engine import regulate, select_regulation_mode
from src.schemas.affect_state import AffectDelta, AffectState
from src.schemas.events import AppraisalResult


class TestSelectRegulationMode:
    def test_high_threat_low_control_reappraisal(self):
        state = AffectState(threat_load=0.8, arousal=0.5)
        appraisal = AppraisalResult(controllability=0.2)
        mode = select_regulation_mode(state, appraisal)
        assert mode == "reappraisal"

    def test_high_arousal_high_threat_suppression(self):
        state = AffectState(threat_load=0.6, arousal=0.8)
        appraisal = AppraisalResult(controllability=0.5)
        mode = select_regulation_mode(state, appraisal)
        assert mode == "suppression"

    def test_normal_state_acceptance(self):
        state = AffectState()
        appraisal = AppraisalResult()
        mode = select_regulation_mode(state, appraisal)
        assert mode == "acceptance"


class TestRegulate:
    def test_reappraisal_reduces_negative(self):
        state = AffectState(threat_load=0.8)
        appraisal = AppraisalResult(controllability=0.2)
        delta = AffectDelta(valence=-0.5, threat_load=0.3)
        regulated, mode, reason = regulate(state, appraisal, delta)
        assert mode == "reappraisal"
        # ネガティブ方向が緩和される
        assert regulated.valence is not None
        assert abs(regulated.valence) < abs(delta.valence)
        assert regulated.threat_load is not None
        assert regulated.threat_load < delta.threat_load

    def test_suppression_adds_fatigue(self):
        state = AffectState(threat_load=0.6, arousal=0.8)
        appraisal = AppraisalResult(controllability=0.5)
        delta = AffectDelta(valence=-0.3, arousal=0.2)
        regulated, mode, reason = regulate(state, appraisal, delta)
        assert mode == "suppression"
        # 疲労が追加される
        assert regulated.fatigue is not None
        assert regulated.fatigue > 0

    def test_acceptance_no_change(self):
        state = AffectState()
        appraisal = AppraisalResult()
        delta = AffectDelta(valence=0.2)
        regulated, mode, reason = regulate(state, appraisal, delta)
        assert mode == "acceptance"
        assert regulated.valence == delta.valence  # 変更なし
