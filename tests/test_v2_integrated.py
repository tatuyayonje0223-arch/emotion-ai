"""IntegratedBrainV2 E2Eテスト。テキスト→10情動→readout→ポリシー。"""

import pytest

from src.brian2_circuits.integrated_brain_v2 import IntegratedBrainV2


class TestIntegratedBrainV2:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.brain = IntegratedBrainV2()

    def test_process_returns_result(self):
        r = self.brain.process("Hello, how are you?")
        assert r.step == 1
        assert not r.blocked

    def test_threat_text(self):
        r = self.brain.process("I'm scared, there's danger everywhere")
        assert r.readout.threat_load > 0
        assert r.emotion_state["emotions"]["fear"] > 0

    def test_reward_text(self):
        r = self.brain.process("I got a wonderful surprise gift!")
        assert r.readout.reward_drive >= 0

    def test_social_text(self):
        r = self.brain.process("I love spending time with my family")
        assert r.readout.social_warmth >= 0

    def test_readout_bounded(self):
        r = self.brain.process("Everything is terrible and wonderful at once")
        assert -1 <= r.readout.valence <= 1
        assert 0 <= r.readout.arousal <= 1
        assert 0 <= r.readout.threat_load <= 1

    def test_emotion_state_has_10_emotions(self):
        r = self.brain.process("Test input")
        emotions = r.emotion_state.get("emotions", {})
        assert len(emotions) == 10

    def test_neuromodulation_present(self):
        r = self.brain.process("A stressful event occurred")
        assert "ecb_2ag" in r.neuromodulation
        assert "ach_nbm" in r.neuromodulation
        assert "theta_coherence" in r.neuromodulation

    def test_memory_encoding(self):
        self.brain.process("A very threatening event")
        self.brain.process("A wonderful reward")
        assert self.brain.memory_count >= 0

    def test_sleep_cycle(self):
        self.brain.process("Something memorable happened")
        results = self.brain.sleep(n_cycles=1)
        assert len(results) == 1

    def test_neuron_count(self):
        assert self.brain.total_neurons > 400

    def test_multiple_steps(self):
        for text in ["Hello", "I'm scared", "That's great", "I'm sad"]:
            r = self.brain.process(text)
            assert r.step > 0

    def test_policy_generated(self):
        r = self.brain.process("I need help")
        assert r.policy is not None
