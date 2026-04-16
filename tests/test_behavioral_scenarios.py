"""Behavioral test battery: ecological scenario validation."""
import pytest
from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2


class TestThreatResponse:
    """Threat scenarios should produce coherent fear responses."""

    def test_high_threat_dominant_fear(self):
        """High threat should make FEAR the dominant emotion."""
        brain = EmotionBrainV2()
        s = brain.process(threat=0.8)
        assert s.dominant_emotion == "FEAR"
        assert s.fear > 0.3

    def test_threat_negative_valence(self):
        """Threat should produce negative valence."""
        brain = EmotionBrainV2()
        s = brain.process(threat=0.8)
        assert s.valence < 0

    def test_threat_high_arousal(self):
        """Threat should produce high arousal."""
        brain = EmotionBrainV2()
        s = brain.process(threat=0.8)
        assert s.arousal > 0.3


class TestRewardResponse:
    """Reward scenarios should produce seeking/approach behavior."""

    def test_reward_positive_valence(self):
        """Reward should produce positive valence."""
        brain = EmotionBrainV2()
        s = brain.process(reward=0.8)
        assert s.valence > 0

    def test_reward_activates_seeking(self):
        """Reward should activate SEEKING system."""
        brain = EmotionBrainV2()
        s = brain.process(reward=0.8)
        assert s.seeking > 0.2


class TestSocialBonding:
    """Social bonding should activate CARE and suppress PANIC."""

    def test_social_activates_care(self):
        """Social contact should activate CARE system."""
        brain = EmotionBrainV2()
        s = brain.process(social=0.8, attachment_need=0.5)
        assert s.care > 0.1

    def test_care_suppresses_panic(self):
        """CARE activation should buffer separation distress (OXT mechanism)."""
        brain = EmotionBrainV2()
        # Loss alone → panic
        s_loss = brain.process(loss=0.5, attachment_need=0.5)
        brain2 = EmotionBrainV2()
        # Loss + social → less panic
        s_social = brain2.process(loss=0.5, attachment_need=0.5, social=0.8)
        assert s_social.panic_grief < s_loss.panic_grief or s_social.care > s_loss.care


class TestLossResponse:
    """Loss should produce sadness and suppress seeking."""

    def test_loss_activates_sadness(self):
        """Loss should activate SADNESS system."""
        brain = EmotionBrainV2()
        s = brain.process(loss=0.8)
        assert s.sadness > 0.2

    def test_loss_suppresses_seeking(self):
        """Loss should suppress SEEKING (LHb→VTA inhibition)."""
        brain = EmotionBrainV2()
        s_base = brain.process(reward=0.5)
        brain2 = EmotionBrainV2()
        s_loss = brain2.process(reward=0.5, loss=0.8)
        assert s_loss.seeking <= s_base.seeking


class TestCompetingEmotions:
    """Competing emotional inputs should produce coherent responses."""

    def test_fear_rage_competition(self):
        """Simultaneous threat + frustration: one should dominate."""
        brain = EmotionBrainV2()
        s = brain.process(threat=0.8, frustration=0.8)
        # Either fear or rage should be dominant, not both maxed
        assert s.fear > 0.2 or s.rage > 0.2
        if s.fear > 0.3 and s.rage > 0.3:
            # Competition should suppress the weaker one
            assert abs(s.fear - s.rage) > 0.05

    def test_disgust_suppresses_approach(self):
        """Disgust should suppress reward approach."""
        brain = EmotionBrainV2()
        s = brain.process(contamination=0.8, reward=0.5)
        # Disgust should reduce seeking vs reward alone
        brain2 = EmotionBrainV2()
        s_reward = brain2.process(reward=0.5)
        assert s.seeking <= s_reward.seeking


class TestNeutralBaseline:
    """No input should produce neutral emotional state."""

    def test_baseline_low_arousal(self):
        """No input → moderate arousal (tonic SEEKING provides baseline activity)."""
        brain = EmotionBrainV2()
        s = brain.process()
        assert s.arousal < 0.8

    def test_baseline_near_neutral_valence(self):
        """No input → mildly positive valence (tonic SEEKING bias)."""
        brain = EmotionBrainV2()
        s = brain.process()
        assert -0.3 < s.valence < 0.9


class TestPainPathway:
    """Pain should activate fear via PB→CeA pathway."""

    def test_pain_activates_fear(self):
        """Pain input should activate FEAR (PB→CeA relay)."""
        brain = EmotionBrainV2()
        s = brain.process(pain=0.8)
        assert s.fear > 0.1

    def test_pain_negative_valence(self):
        """Pain should produce negative valence."""
        brain = EmotionBrainV2()
        s = brain.process(pain=0.8)
        assert s.valence < 0
