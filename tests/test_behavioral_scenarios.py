"""Behavioral test battery: ecological scenario validation.

Tests cross-emotion interactions and ecological validity of the 10-emotion model.
Tonic SEEKING (VTA DA baseline ~6Hz) produces mild positive valence/arousal at rest.
"""
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
        assert s.valence < -0.1

    def test_threat_high_arousal(self):
        """Threat should produce high arousal."""
        brain = EmotionBrainV2()
        s = brain.process(threat=0.8)
        assert s.arousal > 0.4


class TestRewardResponse:
    """Reward scenarios should produce seeking/approach behavior."""

    def test_reward_increases_valence(self):
        """Reward should increase valence above baseline."""
        brain_base = EmotionBrainV2()
        s_base = brain_base.process()
        brain = EmotionBrainV2()
        s = brain.process(reward=0.8)
        assert s.valence >= s_base.valence

    def test_reward_activates_seeking(self):
        """Reward should strongly activate SEEKING system."""
        brain = EmotionBrainV2()
        s = brain.process(reward=0.8)
        assert s.seeking > 0.3


class TestSocialBonding:
    """Social bonding should activate CARE and buffer distress."""

    def test_social_activates_care(self):
        """Social contact should activate CARE system."""
        brain = EmotionBrainV2()
        s = brain.process(social=0.8, attachment_need=0.5)
        assert s.care > 0.1

    def test_care_buffers_panic(self):
        """Social contact during loss should increase CARE (OXT mechanism)."""
        brain = EmotionBrainV2()
        s_loss = brain.process(loss=0.5, attachment_need=0.5)
        brain2 = EmotionBrainV2()
        s_social = brain2.process(loss=0.5, attachment_need=0.5, social=0.8)
        # Social contact must increase care
        assert s_social.care > s_loss.care


class TestLossResponse:
    """Loss should produce sadness and suppress seeking."""

    def test_loss_activates_sadness(self):
        """Loss should activate SADNESS system."""
        brain = EmotionBrainV2()
        s = brain.process(loss=0.8)
        assert s.sadness > 0.2
        assert s.valence < 0

    def test_loss_suppresses_seeking(self):
        """Loss should reduce tonic SEEKING via LHb→VTA→DA pause."""
        brain = EmotionBrainV2()
        s_base = brain.process()
        brain2 = EmotionBrainV2()
        s_loss = brain2.process(loss=0.8)
        # Sadness cross-interaction suppresses seeking
        assert s_loss.seeking < s_base.seeking


class TestCompetingEmotions:
    """Competing emotional inputs should produce coherent responses."""

    def test_fear_rage_competition(self):
        """Simultaneous threat + frustration: stronger wins, weaker suppressed."""
        brain = EmotionBrainV2()
        s = brain.process(threat=0.8, frustration=0.8)
        assert s.fear > 0.2 and s.rage > 0.2
        # vlPAG/dlPAG competition: one should be suppressed by 30%
        assert abs(s.fear - s.rage) > 0.03

    def test_disgust_suppresses_approach(self):
        """Disgust should suppress reward approach."""
        brain = EmotionBrainV2()
        s = brain.process(contamination=0.8, reward=0.5)
        brain2 = EmotionBrainV2()
        s_reward = brain2.process(reward=0.5)
        assert s.seeking <= s_reward.seeking


class TestNeutralBaseline:
    """No input: tonic SEEKING produces mild positive valence/arousal."""

    def test_baseline_gated_emotions_zero(self):
        """Gated emotions (FEAR/RAGE/SADNESS/DISGUST) should be 0 at baseline."""
        brain = EmotionBrainV2()
        s = brain.process()
        assert s.fear == 0.0
        assert s.rage == 0.0
        assert s.sadness == 0.0
        assert s.disgust == 0.0

    def test_baseline_tonic_seeking(self):
        """Tonic VTA DA should produce mild SEEKING at baseline."""
        brain = EmotionBrainV2()
        s = brain.process()
        assert s.seeking > 0.1
        # Valence and arousal driven by tonic SEEKING
        assert 0.3 < s.valence < 0.9
        assert 0.3 < s.arousal < 0.8


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


class TestPlayResponse:
    """Social play should activate PLAY system."""

    def test_play_activates_with_social_and_novelty(self):
        """PLAY requires social + novelty/reward."""
        brain = EmotionBrainV2()
        s = brain.process(social=0.7, novelty=0.5, reward=0.3)
        assert s.play > 0.05

    def test_play_positive_valence(self):
        """Play should produce positive valence."""
        brain = EmotionBrainV2()
        s = brain.process(social=0.7, novelty=0.5, reward=0.3)
        assert s.valence > 0


class TestSurpriseResponse:
    """Novelty should activate SURPRISE system."""

    def test_novelty_activates_surprise(self):
        """High novelty should activate SURPRISE (LC NE burst)."""
        brain = EmotionBrainV2()
        s = brain.process(novelty=0.9)
        assert s.surprise > 0.1

    def test_surprise_high_arousal(self):
        """Surprise should produce high arousal."""
        brain = EmotionBrainV2()
        s = brain.process(novelty=0.9)
        assert s.arousal > 0.4


class TestLustResponse:
    """Social context should activate LUST system."""

    def test_social_activates_lust(self):
        """Social input should activate LUST."""
        brain = EmotionBrainV2()
        s = brain.process(social=0.7)
        assert s.lust > 0.05
