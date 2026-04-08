"""情動状態ストアと動態のテスト。"""

from src.affect_state.dynamics import compute_decay, compute_hysteresis
from src.affect_state.store import AffectStateStore, apply_delta
from src.schemas.affect_state import AffectDelta, AffectState


class TestApplyDelta:
    def test_basic_update(self):
        state = AffectState(valence=0.0)
        delta = AffectDelta(valence=0.3)
        new_state, transition = apply_delta(state, delta, "e1", "appraisal", "test")
        assert new_state.valence == 0.3
        assert new_state.step_count == 1
        assert transition.event_id == "e1"

    def test_clamp_upper(self):
        state = AffectState(valence=0.9)
        delta = AffectDelta(valence=0.5)
        new_state, _ = apply_delta(state, delta, "e1", "test", "test")
        assert new_state.valence == 1.0

    def test_clamp_lower(self):
        state = AffectState(valence=-0.9)
        delta = AffectDelta(valence=-0.5)
        new_state, _ = apply_delta(state, delta, "e1", "test", "test")
        assert new_state.valence == -1.0

    def test_none_fields_unchanged(self):
        state = AffectState(valence=0.5, arousal=0.7)
        delta = AffectDelta(valence=-0.2)  # arousal は None
        new_state, _ = apply_delta(state, delta, "e1", "test", "test")
        assert new_state.valence == 0.3
        assert new_state.arousal == 0.7  # 変化なし

    def test_regulation_mode_change(self):
        state = AffectState()
        delta = AffectDelta(regulation_mode="suppression")
        new_state, _ = apply_delta(state, delta, "e1", "test", "test")
        assert new_state.regulation_mode == "suppression"


class TestAffectStateStore:
    def test_initial_state(self):
        store = AffectStateStore()
        assert store.current.step_count == 0

    def test_update(self):
        store = AffectStateStore()
        delta = AffectDelta(valence=0.2)
        transition = store.update(delta, "e1", "test", "test reason")
        assert store.current.valence == 0.2
        assert len(store.history) == 1
        assert transition.reason == "test reason"

    def test_multiple_updates(self):
        store = AffectStateStore()
        store.update(AffectDelta(valence=0.1), "e1", "test", "")
        store.update(AffectDelta(valence=0.1), "e2", "test", "")
        assert abs(store.current.valence - 0.2) < 1e-6
        assert store.current.step_count == 2
        assert len(store.history) == 2

    def test_reset(self):
        store = AffectStateStore()
        store.update(AffectDelta(valence=0.5), "e1", "test", "")
        store.reset()
        assert store.current.valence == 0.0
        assert len(store.history) == 0


class TestDecay:
    def test_valence_decays_toward_zero(self):
        state = AffectState(valence=0.5)
        delta = compute_decay(state)
        assert delta.valence is not None
        assert delta.valence < 0  # 正の valence は減少

    def test_negative_valence_decays(self):
        state = AffectState(valence=-0.5)
        delta = compute_decay(state)
        assert delta.valence is not None
        assert delta.valence > 0  # 負の valence は増加（0へ向かう）

    def test_arousal_decays_to_baseline(self):
        high_arousal = AffectState(arousal=0.8)
        delta = compute_decay(high_arousal)
        assert delta.arousal is not None
        assert delta.arousal < 0  # ベースライン(0.2)に向かって減少

    def test_zero_state_minimal_decay(self):
        state = AffectState(valence=0.0, arousal=0.2, threat_load=0.0, fatigue=0.0)
        delta = compute_decay(state)
        assert abs(delta.valence or 0) < 0.01
        assert abs(delta.arousal or 0) < 0.01


class TestHysteresis:
    def test_high_threat_dampens_reduction(self):
        state = AffectState(threat_load=0.8)
        delta = AffectDelta(threat_load=-0.4)
        dampened = compute_hysteresis(state, delta)
        assert dampened.threat_load is not None
        assert abs(dampened.threat_load) < abs(delta.threat_load)  # 減衰された

    def test_low_threat_no_dampening(self):
        state = AffectState(threat_load=0.3)
        delta = AffectDelta(threat_load=-0.2)
        dampened = compute_hysteresis(state, delta)
        assert dampened.threat_load == delta.threat_load  # 変化なし

    def test_low_trust_dampens_increase(self):
        state = AffectState(trust=0.1)
        delta = AffectDelta(trust=0.3)
        dampened = compute_hysteresis(state, delta)
        assert dampened.trust is not None
        assert dampened.trust < delta.trust  # 信頼増加が抑制
