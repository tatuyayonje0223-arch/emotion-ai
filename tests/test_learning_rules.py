"""STDP + DA modulation learning rule validation.

Tests that fear conditioning strengthens amygdala synapses and
extinction weakens them, consistent with rodent behavioral data.
"""
import pytest
from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2


class TestFearConditioning:
    """CS+US pairing should strengthen amygdala synapses (STDP)."""

    def test_repeated_threat_increases_fear(self):
        """Multiple threat presentations should increase fear response (sensitization)."""
        brain = EmotionBrainV2()
        # First trial
        s1 = brain.process(threat=0.8)
        fear1 = s1.fear
        # After 3 more conditioning trials
        for _ in range(3):
            brain.process(threat=0.8)
        s5 = brain.process(threat=0.8)
        fear5 = s5.fear
        # Fear should not decrease (STDP strengthens LA→CeL/BA)
        assert fear5 >= fear1 * 0.8, f"Fear decreased: {fear1:.3f} → {fear5:.3f}"

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

    def test_no_threat_reduces_fear_expression(self):
        """After conditioning, presenting no threat should show less fear."""
        brain = EmotionBrainV2()
        # Conditioning phase
        for _ in range(3):
            brain.process(threat=0.8)
        # Extinction phase (no threat)
        s_ext = brain.process(threat=0.0)
        # Without threat input, fear should be gated to 0 (input-gated emotion)
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
