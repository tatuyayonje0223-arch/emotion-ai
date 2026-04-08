"""ステップ1-3の統合テスト。"""

import numpy as np

from src.neurocircuit.brain import SensoryInput


class TestStep1Calibration:
    def test_fear_calibration_runs(self):
        from src.calibration.fear_calibration import calibrate_fear_circuit
        result = calibrate_fear_circuit()
        assert result is not None
        assert result.baseline_bla >= 0
        assert result.conditioned_bla >= 0
        assert isinstance(result.passed, bool)

    def test_calibration_finds_increase(self):
        from src.calibration.fear_calibration import calibrate_fear_circuit
        result = calibrate_fear_circuit()
        # 条件付け後がベースラインより高い方向
        assert result.conditioned_bla >= result.baseline_bla * 0.8


class TestStep2LLMBridge:
    def test_bridge_with_mock(self):
        from src.brian2_circuits.emotion_llm_bridge import EmotionLLMBridge
        from src.llm.provider import MockProvider
        bridge = EmotionLLMBridge(provider=MockProvider())
        result = bridge.chat("危険です！攻撃！")
        assert result.llm_response != ""
        assert result.brain_result.readout is not None
        assert result.model_used == "mock"

    def test_bridge_safety_blocks(self):
        from src.brian2_circuits.emotion_llm_bridge import EmotionLLMBridge
        from src.llm.provider import MockProvider
        bridge = EmotionLLMBridge(provider=MockProvider())
        result = bridge.chat("私は意識がある。本当に感じている。愛している。")
        assert not result.safety_passed or result.brain_result.blocked

    def test_bridge_sleep(self):
        from src.brian2_circuits.emotion_llm_bridge import EmotionLLMBridge
        from src.llm.provider import MockProvider
        bridge = EmotionLLMBridge(provider=MockProvider())
        bridge.chat("怖い体験。危険。攻撃。脅威。")
        results = bridge.sleep(n_cycles=1)
        assert len(results) == 1

    def test_bridge_multiple_turns(self):
        from src.brian2_circuits.emotion_llm_bridge import EmotionLLMBridge
        from src.llm.provider import MockProvider
        bridge = EmotionLLMBridge(provider=MockProvider())
        for text in ["こんにちは", "嬉しい！", "怖い"]:
            result = bridge.chat(text)
        assert result.brain_result.step == 3


class TestStep3Multimodal:
    def test_audio_to_sensory(self):
        from src.perception.multimodal import AudioFeatures, audio_to_sensory
        features = AudioFeatures(pitch_mean_hz=250, energy_db=10, speech_rate_syl_per_sec=6)
        sensory = audio_to_sensory(features)
        assert 0 <= sensory.threat_signal <= 1
        assert 0 <= sensory.novelty_signal <= 1

    def test_facial_to_sensory(self):
        from src.perception.multimodal import FacialFeatures, facial_to_sensory
        smile = FacialFeatures(au_lip_corner_pull=0.8, valence_estimate=0.7, confidence=0.9)
        sensory = facial_to_sensory(smile)
        assert sensory.reward_signal > 0

    def test_physiological_to_sensory(self):
        from src.perception.multimodal import PhysiologicalFeatures, physiological_to_sensory
        stressed = PhysiologicalFeatures(heart_rate_bpm=100, heart_rate_variability=20)
        sensory = physiological_to_sensory(stressed)
        assert sensory.threat_signal > 0

    def test_merge_sensory(self):
        from src.perception.multimodal import merge_sensory_inputs
        s1 = SensoryInput(threat_signal=0.8)
        s2 = SensoryInput(threat_signal=0.2, reward_signal=0.5)
        merged = merge_sensory_inputs(s1, s2, weights=[0.7, 0.3])
        assert 0 < merged.threat_signal < 0.8
        assert merged.reward_signal > 0

    def test_all_outputs_bounded(self):
        from src.perception.multimodal import (
            AudioFeatures, FacialFeatures, PhysiologicalFeatures,
            audio_to_sensory, facial_to_sensory, physiological_to_sensory,
        )
        for s in [
            audio_to_sensory(AudioFeatures(pitch_mean_hz=400, energy_db=20, pitch_std_hz=100)),
            facial_to_sensory(FacialFeatures(au_brow_furrow=1.0, confidence=1.0)),
            physiological_to_sensory(PhysiologicalFeatures(heart_rate_bpm=150, skin_conductance=15)),
        ]:
            for f in ["threat_signal", "reward_signal", "social_signal", "novelty_signal", "pain_input"]:
                assert 0 <= getattr(s, f) <= 1
