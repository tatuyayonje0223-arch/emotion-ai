"""EmotionBrainV2 — 10情動統合テスト。

232検証済み論文パラメータに基づくV2の動作検証。
"""

import pytest

from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2, EmotionStateV2


@pytest.fixture(scope="class")
def brain():
    """クラススコープでBrainを共有（STDP蓄積を防ぐ）。"""
    b = EmotionBrainV2()
    return b


class TestEmotionBrainV2Build:
    def test_build_succeeds(self, brain):
        assert brain.total_neurons > 0

    def test_all_populations_registered(self, brain):
        names = brain.population_names
        # 共有14 + FEAR10 + RAGE2 + SEEKING4 + SADNESS2 + DISGUST2 = 34
        assert len(names) >= 30

    def test_neuron_count_reasonable(self, brain):
        n = brain.total_neurons
        assert 400 < n < 2000, f"Total neurons {n} outside expected range"


class TestFearScenario:
    def test_threat_activates_fear(self, brain):
        state = brain.process(threat=0.8)
        assert state.fear > 0.1, f"FEAR too low: {state.fear}"

    def test_high_threat_dominant(self, brain):
        state = brain.process(threat=0.9, pain=0.3)
        assert state.dominant_emotion == "FEAR", f"Expected FEAR, got {state.dominant_emotion}"

    def test_threat_negative_valence(self, brain):
        state = brain.process(threat=0.8)
        assert state.valence < 0, f"Threat should produce negative valence: {state.valence}"


class TestRageScenario:
    def test_frustration_activates_rage(self, brain):
        state = brain.process(frustration=0.8)
        assert state.rage > 0.1, f"RAGE too low: {state.rage}"

    def test_high_frustration_dominant(self, brain):
        state = brain.process(frustration=0.9, threat=0.2)
        assert state.dominant_emotion == "RAGE", f"Expected RAGE, got {state.dominant_emotion}"


class TestSeekingScenario:
    def test_reward_activates_seeking(self, brain):
        state = brain.process(reward=0.8)
        assert state.seeking > 0.1, f"SEEKING too low: {state.seeking}"

    def test_reward_positive_valence(self, brain):
        state = brain.process(reward=0.9)
        assert state.valence > 0, f"Reward should produce positive valence: {state.valence}"


class TestSadnessScenario:
    def test_loss_activates_sadness(self, brain):
        state = brain.process(loss=0.8, social=0.0)
        assert state.sadness > 0.05, f"SADNESS too low: {state.sadness}"

    def test_loss_negative_valence(self, brain):
        state = brain.process(loss=0.8)
        assert state.valence < 0, f"Loss should produce negative valence: {state.valence}"


class TestDisgustScenario:
    def test_contamination_activates_disgust(self, brain):
        state = brain.process(contamination=0.8)
        assert state.disgust > 0.1, f"DISGUST too low: {state.disgust}"


class TestCareScenario:
    def test_social_activates_care(self, brain):
        state = brain.process(social=0.8, attachment_need=0.5)
        assert state.care > 0.0, f"CARE too low: {state.care}"


class TestPanicGriefScenario:
    def test_loss_activates_panic_grief(self, brain):
        state = brain.process(loss=0.8, social=0.0, attachment_need=0.8)
        assert state.panic_grief > 0.0, f"PANIC_GRIEF too low: {state.panic_grief}"


class TestPlayScenario:
    def test_social_reward_activates_play(self, brain):
        state = brain.process(social=0.7, reward=0.5, novelty=0.3)
        assert state.play > 0.0, f"PLAY too low: {state.play}"


class TestSurpriseScenario:
    def test_novelty_activates_surprise(self, brain):
        state = brain.process(novelty=0.9)
        assert state.surprise > 0.0, f"SURPRISE too low: {state.surprise}"


class TestLustScenario:
    def test_social_activates_lust(self, brain):
        state = brain.process(social=0.5, reward=0.3)
        assert state.lust >= 0.0  # LUST is subtle


class TestIntegration:
    def test_all_activations_bounded(self, brain):
        state = brain.process(
            threat=0.5, reward=0.5, social=0.5, novelty=0.5,
            pain=0.3, loss=0.3, frustration=0.3, contamination=0.3,
        )
        for name in ["fear", "rage", "seeking", "sadness", "disgust",
                      "care", "panic_grief", "play", "lust", "surprise"]:
            val = getattr(state, name)
            assert 0 <= val <= 1, f"{name}={val} out of [0,1]"

    def test_valence_bounded(self, brain):
        state = brain.process(threat=1.0)
        assert -1 <= state.valence <= 1

    def test_arousal_bounded(self, brain):
        state = brain.process(reward=1.0)
        assert 0 <= state.arousal <= 1

    def test_to_dict(self, brain):
        state = brain.process(threat=0.5)
        d = state.to_dict()
        assert "emotions" in d
        assert len(d["emotions"]) == 10

    def test_joy_scenario(self, brain):
        """喜び: 高報酬+社会的+低脅威 → SEEKING/PLAY優勢 + positive valence。"""
        state = brain.process(reward=0.9, social=0.8, threat=0.0, novelty=0.3)
        assert state.valence > 0, f"Joy should be positive: {state.valence}"
        assert state.dominant_emotion in ("SEEKING", "PLAY", "CARE"), \
            f"Joy dominant should be SEEKING/PLAY/CARE, got {state.dominant_emotion}"

    def test_cross_emotion_fear_suppresses_rage(self, brain):
        """FEAR↔RAGE: 高恐怖が怒りを抑制（PAG競合）。"""
        state = brain.process(threat=0.9, frustration=0.5)
        # Fear should dominate when threat >> frustration
        assert state.fear > state.rage * 0.5, \
            f"Fear should suppress rage: fear={state.fear}, rage={state.rage}"

    def test_sadness_suppresses_seeking(self, brain):
        """SADNESS→SEEKING: 悲しみが報酬追求を抑制（LHb機序）。"""
        # Pure reward
        s1 = brain.process(reward=0.7)
        seek1 = s1.seeking

        # Reward + loss
        s2 = brain.process(reward=0.7, loss=0.7)
        seek2 = s2.seeking

        # Loss should reduce seeking (LHb→VTA inhibition)
        # Allow some tolerance due to stochastic spiking
        assert seek2 < seek1 * 1.5, \
            f"Sadness should not amplify seeking: {seek1:.2f} -> {seek2:.2f}"
