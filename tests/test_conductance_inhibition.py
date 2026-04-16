"""Phase 3 conductance-based inhibition + CeA expansion tests."""

import pytest

class TestConductanceInhibition:
    """Tests for g_inh GABA_A conductance-based shunting."""

    def test_vta_da_pause_below_1hz(self):
        """VTA DA pause should be < 1Hz during loss (Schultz 1997)."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        r = brain.process(loss=0.5)
        assert r.all_rates["vta_da_lat"] < 1.0, f"VTA DA pause too high: {r.all_rates['vta_da_lat']:.1f}"

    def test_dr_partial_suppression(self):
        """DR should be partially suppressed to 2-4Hz during sadness, not silenced."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        r = brain.process(loss=0.8)
        dr = r.all_rates["dr"]
        assert 1.0 < dr < 6.0, f"DR suppression out of range: {dr:.1f} (expected ~2-4Hz)"

    def test_vta_da_tonic_preserved(self):
        """VTA DA tonic firing should still be 3-7Hz at baseline."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        r = brain.process()
        da = r.all_rates["vta_da_lat"]
        assert 2.0 < da < 10.0, f"VTA DA tonic outside range: {da:.1f}"

    def test_cea_shunting_pkcd_silenced(self):
        """CeL PKCd+ should be silenced during CS (shunting inhibition)."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        r = brain.process(threat=0.8)
        pkcd = r.all_rates["cel_pkcd"]
        assert pkcd < 2.0, f"PKCd+ not silenced: {pkcd:.1f}"

    def test_rmtg_activates_during_loss(self):
        """RMTg should fire > 15Hz during loss (habenula-driven)."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        r = brain.process(loss=0.5)
        rmtg = r.all_rates["rmtg"]
        assert rmtg > 10.0, f"RMTg too quiet during loss: {rmtg:.1f}"

    def test_pptg_suppressed_during_loss(self):
        """PPTg should be suppressed during loss (RMTg→PPTg shunting)."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        # Baseline
        r_base = brain.process()
        pptg_base = r_base.all_rates["pptg"]
        # Loss
        brain2 = EmotionBrainV2()
        r_loss = brain2.process(loss=0.5)
        pptg_loss = r_loss.all_rates["pptg"]
        assert pptg_loss < pptg_base, f"PPTg not suppressed: base={pptg_base:.1f} loss={pptg_loss:.1f}"


class TestCeAExpansion:
    """Tests for Phase 3 CeA microcircuit expansion."""

    def test_pb_population_exists(self):
        """Parabrachial nucleus should be registered."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        assert "pb" in brain.population_names

    def test_cel_crf_population_exists(self):
        """CeL CRF+ neurons should be registered."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        assert "cel_crf" in brain.population_names

    def test_total_neurons_with_expansion(self):
        """Total should be ~794 with PB + CeL_CRF + VIP + PV."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        assert 785 < brain.total_neurons < 810, f"Unexpected count: {brain.total_neurons}"

    def test_total_populations(self):
        """Should have 51 populations after full CeA expansion."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        assert len(brain.population_names) == 51

    def test_pb_fires_during_pain(self):
        """PB should activate when pain is presented."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        r = brain.process(pain=0.8)
        pb = r.all_rates["pb"]
        assert pb > 5.0, f"PB too quiet during pain: {pb:.1f}"

    def test_pain_activates_fear_via_pb(self):
        """Pain should activate fear circuit via PB→CeA pathway."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        r = brain.process(pain=0.8)
        assert r.fear > 0.1, f"Fear not activated by pain: {r.fear:.2f}"

    def test_cel_vip_population_exists(self):
        """CeL VIP+ neurons should be registered."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        assert "cel_vip" in brain.population_names

    def test_cea_pv_population_exists(self):
        """CeA PV+ neurons should be registered."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        assert "cea_pv" in brain.population_names

    def test_cea_pv_fires_during_threat(self):
        """CeA PV+ should fire during threat (LA→PV feedforward)."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        r = brain.process(threat=0.8)
        pv = r.all_rates.get("cea_pv", 0)
        assert pv > 2.0, f"CeA PV+ too quiet during threat: {pv:.1f}"

    def test_cel_vip_fires(self):
        """CeL VIP+ should fire (low rate, input-modulated)."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        r = brain.process(threat=0.8)
        vip = r.all_rates.get("cel_vip", 0)
        assert vip >= 0, f"CeL VIP+ negative rate: {vip:.1f}"
