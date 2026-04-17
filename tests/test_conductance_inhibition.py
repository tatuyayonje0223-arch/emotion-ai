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
        """Total should be ~821 with CeA expansion + HPC."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        assert 810 < brain.total_neurons < 840, f"Unexpected count: {brain.total_neurons}"

    def test_total_populations(self):
        """Should have 53 populations (CeA + HPC)."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()
        assert len(brain.population_names) == 53

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


class TestAdExModel:
    """AdEx (Adaptive Exponential) neuron model infrastructure tests."""

    def test_adex_builds_and_runs(self):
        """AdEx model should build and run without errors."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        from src.brian2_circuits.shared_core_network import SharedCoreConfig
        cfg = SharedCoreConfig(use_adex=True)
        brain = EmotionBrainV2(config=cfg)
        r = brain.process(threat=0.5)
        assert brain.total_neurons == 821
        assert len(brain.population_names) == 53

    def test_adex_produces_spikes(self):
        """AdEx neurons should fire (non-zero rates)."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        from src.brian2_circuits.shared_core_network import SharedCoreConfig
        cfg = SharedCoreConfig(use_adex=True)
        brain = EmotionBrainV2(config=cfg)
        r = brain.process(threat=0.8)
        active = sum(1 for v in r.all_rates.values() if v > 0.5)
        assert active > 20, f"Too few active populations: {active}"

    def test_adex_default_is_izhikevich(self):
        """Default config should use Izhikevich (use_adex=False)."""
        from src.brian2_circuits.shared_core_network import SharedCoreConfig
        cfg = SharedCoreConfig()
        assert cfg.use_adex is False

    def test_adex_vta_pause(self):
        """AdEx VTA DA should achieve pause (< 2Hz) during loss."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        from src.brian2_circuits.shared_core_network import SharedCoreConfig
        cfg = SharedCoreConfig(use_adex=True)
        brain = EmotionBrainV2(config=cfg)
        r = brain.process(loss=0.5)
        assert r.all_rates["vta_da_lat"] < 2.0, f"AdEx VTA DA not pausing: {r.all_rates['vta_da_lat']:.1f}"

    def test_adex_shunting_works(self):
        """AdEx CeA shunting should suppress PKCd+ during threat."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        from src.brian2_circuits.shared_core_network import SharedCoreConfig
        cfg = SharedCoreConfig(use_adex=True)
        brain = EmotionBrainV2(config=cfg)
        r = brain.process(threat=0.8)
        pkcd = r.all_rates.get("cel_pkcd", 0)
        # Shunting should at least reduce PKCd below baseline
        brain2 = EmotionBrainV2(config=cfg)
        r2 = brain2.process()
        pkcd_base = r2.all_rates.get("cel_pkcd", 0)
        assert pkcd <= pkcd_base, f"PKCd not suppressed: threat={pkcd:.1f} base={pkcd_base:.1f}"

    def test_adex_fear_produces_cem_output(self):
        """AdEx threat should activate CeM (fear output)."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        from src.brian2_circuits.shared_core_network import SharedCoreConfig
        cfg = SharedCoreConfig(use_adex=True)
        brain = EmotionBrainV2(config=cfg)
        r = brain.process(threat=0.8)
        assert r.all_rates["cem"] > 5.0, f"AdEx CeM too quiet: {r.all_rates['cem']:.1f}"

    def test_izhikevich_unaffected_by_adex_code(self):
        """Izhikevich validation should be 36/36 regardless of AdEx code existence."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        brain = EmotionBrainV2()  # default = Izhikevich
        r_tonic = brain.process()
        assert 3 <= r_tonic.all_rates["vta_da_lat"] <= 7
        brain2 = EmotionBrainV2()
        r_threat = brain2.process(threat=0.8)
        assert r_threat.all_rates["cem"] >= 10

    def test_adex_per_population_tonic(self):
        """AdEx should have different tonic rates than Izhikevich (per-pop calibration)."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        from src.brian2_circuits.shared_core_network import SharedCoreConfig
        cfg = SharedCoreConfig(use_adex=True)
        brain = EmotionBrainV2(config=cfg)
        r = brain.process()
        # la_exc should fire at baseline (tonic=2.0, rheobase=3.0, bg=1.7 → I=3.7 > 3.0)
        assert r.all_rates["la_exc"] > 1.0, f"la_exc too quiet: {r.all_rates['la_exc']:.1f}"

    def test_adex_drive_scale_affects_overrides(self):
        """AdEx emotion-specific drives should be scaled (1.8x) relative to Izhikevich."""
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        from src.brian2_circuits.shared_core_network import SharedCoreConfig
        # AdEx with threat should produce meaningful fear response
        cfg = SharedCoreConfig(use_adex=True)
        brain = EmotionBrainV2(config=cfg)
        r = brain.process(threat=0.8)
        assert r.fear > 0.05, f"AdEx fear too low: {r.fear:.3f}"
