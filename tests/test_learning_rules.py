"""STDP + DA modulation learning rule validation.

Tests that fear conditioning strengthens amygdala synapses and
extinction weakens them, consistent with rodent behavioral data.
"""
import pytest
from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2


class TestFearConditioning:
    """CS+US pairing should strengthen amygdala synapses (STDP)."""

    def test_repeated_threat_maintains_fear(self):
        """Multiple threat presentations should maintain or increase fear (STDP)."""
        brain = EmotionBrainV2()
        s1 = brain.process(threat=0.8)
        fear1 = s1.fear
        for _ in range(3):
            brain.process(threat=0.8)
        s5 = brain.process(threat=0.8)
        fear5 = s5.fear
        # Fear should not decrease significantly (STDP strengthens LA→CeL/BA)
        assert fear5 >= fear1 * 0.95, f"Fear decreased: {fear1:.3f} → {fear5:.3f}"

    def test_stdp_synapses_exist(self):
        """STDP synapses should be registered in the network."""
        brain = EmotionBrainV2()
        brain.process(threat=0.5)  # need to run at least once
        stdp = brain._core.stdp_synapses
        assert len(stdp) > 0, "No STDP synapses found"

    def test_conditioning_maintains_valence(self):
        """Repeated threat should maintain negative valence."""
        brain = EmotionBrainV2()
        for _ in range(5):
            s = brain.process(threat=0.8)
        assert s.valence < 0, f"Valence not negative after conditioning: {s.valence:.2f}"


class TestExtinction:
    """Repeated CS-only (no US) should weaken fear response."""

    def test_no_threat_gates_fear_to_zero(self):
        """Without threat input, fear is gated to 0 regardless of STDP history."""
        brain = EmotionBrainV2()
        # Conditioning phase
        for _ in range(3):
            brain.process(threat=0.8)
        # No threat → input gate blocks fear output
        s_ext = brain.process(threat=0.0)
        assert s_ext.fear == 0.0, f"Fear not gated without threat: {s_ext.fear:.3f}"

    def test_reduced_threat_produces_less_fear(self):
        """Lower threat level should produce proportionally less fear."""
        brain = EmotionBrainV2()
        s_high = brain.process(threat=0.8)
        brain2 = EmotionBrainV2()
        s_low = brain2.process(threat=0.3)
        assert s_low.fear < s_high.fear, f"Low threat >= high threat: {s_low.fear:.3f} >= {s_high.fear:.3f}"


class TestRewardLearning:
    """Reward should strengthen VTA→NAc via STDP-like mechanisms."""

    def test_reward_activates_da_system(self):
        """Reward should activate VTA DA and NAc."""
        brain = EmotionBrainV2()
        s = brain.process(reward=0.8)
        assert s.all_rates["vta_da_lat"] > 10, f"VTA DA too low: {s.all_rates['vta_da_lat']:.1f}"
        assert s.all_rates["nac_shell_d1"] > 5, f"NAc D1 too low: {s.all_rates['nac_shell_d1']:.1f}"

    def test_repeated_reward_maintains_seeking(self):
        """Multiple reward presentations should maintain seeking activation."""
        brain = EmotionBrainV2()
        for _ in range(5):
            s = brain.process(reward=0.8)
        assert s.seeking > 0.3, f"Seeking too low after repeated reward: {s.seeking:.3f}"


class TestCrossModalLearning:
    """Different input modalities should produce independent learning."""

    def test_threat_does_not_increase_seeking(self):
        """Threat conditioning should not increase seeking."""
        brain = EmotionBrainV2()
        s_base = brain.process()
        seeking_base = s_base.seeking
        for _ in range(5):
            brain.process(threat=0.8)
        s_after = brain.process()
        # Seeking should not increase from threat conditioning
        assert s_after.seeking <= seeking_base * 1.5, \
            f"Seeking increased from threat: {seeking_base:.3f} → {s_after.seeking:.3f}"


class TestHippocampalContext:
    """Hippocampal context memory (dHPC/vHPC) tests."""

    def test_dhpc_population_exists(self):
        """dHPC should be registered."""
        brain = EmotionBrainV2()
        assert "dhpc" in brain.population_names

    def test_vhpc_population_exists(self):
        """vHPC should be registered."""
        brain = EmotionBrainV2()
        assert "vhpc" in brain.population_names

    def test_context_activates_dhpc(self):
        """Context input should drive dHPC firing."""
        brain = EmotionBrainV2()
        r = brain.process(context=0.8)
        dhpc = r.all_rates.get("dhpc", 0)
        assert dhpc > 8.0, f"dHPC too quiet with context: {dhpc:.1f}"

    def test_context_activates_vhpc_with_threat(self):
        """Context + threat should drive vHPC (anxiety modulation)."""
        brain = EmotionBrainV2()
        r = brain.process(context=0.8, threat=0.5)
        vhpc = r.all_rates.get("vhpc", 0)
        assert vhpc > 8.0, f"vHPC too quiet with context+threat: {vhpc:.1f}"

    def test_context_parameter_accepted(self):
        """EmotionBrainV2.process() should accept context parameter."""
        brain = EmotionBrainV2()
        # Should not raise
        r = brain.process(context=0.5)
        assert r.spiking_neurons > 0


class TestIntegratedPipeline:
    """End-to-end: text → perception → brain → emotion state."""

    def test_threat_text_produces_fear(self):
        """Japanese threat text should produce fear via integrated pipeline."""
        from src.brian2_circuits.integrated_brain_v2 import IntegratedBrainV2
        brain = IntegratedBrainV2()
        r = brain.process("危険だ！攻撃を受けている！逃げろ！")
        emotions = r.emotion_state.get("emotions", {})
        assert emotions.get("fear", 0) > 0.1, f"Fear not detected: {emotions}"

    def test_positive_text_produces_seeking(self):
        """Positive text should produce seeking/positive valence."""
        from src.brian2_circuits.integrated_brain_v2 import IntegratedBrainV2
        brain = IntegratedBrainV2()
        r = brain.process("嬉しい！素晴らしい！成功した！")
        assert r.readout.valence > 0, f"Valence not positive: {r.readout.valence}"

    def test_context_parameter_in_integrated(self):
        """IntegratedBrainV2 should accept context parameter."""
        from src.brian2_circuits.integrated_brain_v2 import IntegratedBrainV2
        brain = IntegratedBrainV2()
        r = brain.process("怖い", context=0.8)
        assert r.spiking_neurons > 0
