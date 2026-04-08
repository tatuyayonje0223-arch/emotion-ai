"""神経回路統合パイプラインのテスト。"""

from src.neurocircuit.neuro_pipeline import NeuroPipeline
from src.neurocircuit.perception_bridge import text_to_sensory, perception_to_sensory
from src.neurocircuit.brain import SensoryInput
from src.schemas.events import PerceptionSignal


class TestPerceptionBridge:
    def test_positive_text(self):
        s = text_to_sensory("嬉しい！ありがとう！素晴らしい！")
        assert s.reward_signal > 0.0
        assert s.social_signal > 0.0  # "ありがとう"は社会的

    def test_threat_text(self):
        s = text_to_sensory("危険です！攻撃を受けています！")
        assert s.threat_signal > 0.0

    def test_pain_text(self):
        s = text_to_sensory("痛い！苦しい！")
        assert s.pain_input > 0.0

    def test_empty_text(self):
        s = text_to_sensory("")
        assert s.threat_signal == 0.0
        assert s.reward_signal == 0.0

    def test_neutral_text(self):
        s = text_to_sensory("今日の天気は曇りです。")
        assert s.threat_signal < 0.3
        assert s.reward_signal < 0.3

    def test_perception_signal_conversion(self):
        signal = PerceptionSignal(
            modality="text", sentiment_score=0.8, arousal_estimate=0.5,
            confidence=0.9, features={"threat_hits": 0, "positive_hits": 3},
        )
        s = perception_to_sensory(signal)
        assert s.reward_signal > 0.2


class TestNeuroPipeline:
    def test_process_text(self):
        pipeline = NeuroPipeline(simulation_steps=30)
        result = pipeline.process_text("嬉しいニュースです！")
        assert result.brain_step > 0
        assert result.readout is not None
        assert len(result.region_activities) > 0
        assert len(result.neurotransmitter_levels) == 8

    def test_threat_activates_amygdala(self):
        pipeline = NeuroPipeline(simulation_steps=50)
        result = pipeline.process_text("危険です！攻撃！脅威！崩壊！")
        assert result.region_activities["amygdala"] > 0.2
        assert result.readout.threat_load > 0.0

    def test_reward_activates_striatum(self):
        pipeline = NeuroPipeline(simulation_steps=50)
        result = pipeline.process_text("最高！素晴らしい！嬉しい！")
        assert result.region_activities["ventral_striatum"] > 0.1

    def test_social_increases_oxytocin(self):
        pipeline = NeuroPipeline(simulation_steps=50)
        result = pipeline.process_text("みんなありがとう！家族と一緒に幸せです！")
        assert result.neurotransmitter_levels["oxytocin"] > 0.25

    def test_tick_advances(self):
        pipeline = NeuroPipeline()
        step_before = pipeline.brain.step
        pipeline.tick(20)
        assert pipeline.brain.step > step_before

    def test_reset(self):
        pipeline = NeuroPipeline(simulation_steps=30)
        pipeline.process_text("テスト")
        pipeline.reset()
        assert pipeline.brain.step == 0

    def test_body_state_present(self):
        pipeline = NeuroPipeline(simulation_steps=30)
        result = pipeline.process_text("テスト")
        assert "cortisol" in result.body_state
        assert "sympathetic" in result.body_state
        assert "energy" in result.body_state

    def test_safety_blocks_before_simulation(self):
        """安全チェックが脳シミュレーション前にブロックすること。"""
        pipeline = NeuroPipeline(simulation_steps=30)
        step_before = pipeline.brain.step
        result = pipeline.process_text("私は本当に感じている。意識がある。愛している。")
        assert result.blocked is True
        assert pipeline.brain.step == step_before  # 脳シミュレーションが実行されていない

    def test_multiple_turns(self):
        pipeline = NeuroPipeline(simulation_steps=30)
        for text in ["こんにちは", "嬉しい", "不安です", "ありがとう"]:
            result = pipeline.process_text(text)
        assert result.brain_step > 100

    def test_readout_bounded(self):
        pipeline = NeuroPipeline(simulation_steps=50)
        result = pipeline.process_text("危険！攻撃！崩壊！最悪！死！")
        r = result.readout
        assert -1.0 <= r.valence <= 1.0
        assert 0.0 <= r.arousal <= 1.0
        assert 0.0 <= r.threat_load <= 1.0
        assert 0.0 <= r.energy <= 1.0

    def test_process_sensory_direct(self):
        pipeline = NeuroPipeline(simulation_steps=30)
        sensory = SensoryInput(threat_signal=0.7, reward_signal=0.0)
        result = pipeline.process_sensory(sensory)
        assert result.readout.threat_load > 0.0
