"""Pankseppの7情動システム + 拡張のテスト。"""

from src.brian2_circuits.emotion_circuits import PankseppEmotionEngine


class TestPankseppEngine:
    def test_all_systems_exist(self):
        e = PankseppEmotionEngine()
        assert len(e.systems) == 10  # 7 Panksepp + 3 extended

    def test_fear_activates_with_threat(self):
        e = PankseppEmotionEngine()
        e.process(threat=0.8)
        assert e.systems["FEAR"].activation > 0.5

    def test_seeking_activates_with_reward(self):
        e = PankseppEmotionEngine()
        e.process(reward=0.8)
        assert e.systems["SEEKING"].activation > 0.3

    def test_rage_activates_with_frustration(self):
        e = PankseppEmotionEngine()
        e.process(frustration=0.8)
        assert e.systems["RAGE"].activation > 0.4

    def test_care_activates_with_social(self):
        e = PankseppEmotionEngine()
        e.process(social=0.8)
        assert e.systems["CARE"].activation > 0.4

    def test_panic_grief_activates_with_loss(self):
        e = PankseppEmotionEngine()
        e.process(loss=0.8)
        assert e.systems["PANIC_GRIEF"].activation > 0.3

    def test_play_activates_with_social_and_reward(self):
        e = PankseppEmotionEngine()
        e.process(social=0.7, reward=0.5)
        assert e.systems["PLAY"].activation > 0.3

    def test_disgust_activates_with_contamination(self):
        e = PankseppEmotionEngine()
        e.process(contamination=0.8)
        assert e.systems["DISGUST"].activation > 0.5

    def test_sadness_activates_with_loss(self):
        e = PankseppEmotionEngine()
        e.process(loss=0.7, social=0.0)
        assert e.systems["SADNESS"].activation > 0.3

    def test_surprise_activates_with_novelty(self):
        e = PankseppEmotionEngine()
        e.process(novelty=0.9)
        assert e.systems["SURPRISE"].activation > 0.5

    def test_dominant_emotion(self):
        e = PankseppEmotionEngine()
        e.process(threat=0.9)
        name, conf = e.get_dominant_emotion()
        assert name == "FEAR"
        assert conf > 0.5

    def test_compound_emotions(self):
        e = PankseppEmotionEngine()
        e.process(social=0.8, reward=0.6)
        compounds = e.get_compound_emotions()
        assert "LOVE" in compounds
        assert "OPTIMISM" in compounds
        assert compounds["LOVE"] > 0  # PLAY + CARE

    def test_integrated_readout(self):
        e = PankseppEmotionEngine()
        e.process(threat=0.5, reward=0.3, social=0.2)
        readout = e.get_integrated_readout()
        assert -1 <= readout["valence"] <= 1
        assert 0 <= readout["arousal"] <= 1
        assert "dominant_system" in readout
        assert "all_activations" in readout

    def test_joy_scenario(self):
        """喜び: 高報酬+社会的+低脅威 → SEEKING+PLAY優勢。"""
        e = PankseppEmotionEngine()
        e.process(reward=0.9, social=0.8, threat=0.0, novelty=0.3)
        readout = e.get_integrated_readout()
        assert readout["valence"] > 0  # ポジティブ
        assert readout["dominant_system"] in ("SEEKING", "PLAY", "CARE")

    def test_anger_scenario(self):
        """怒り: 高フラストレーション → RAGE優勢。"""
        e = PankseppEmotionEngine()
        e.process(frustration=0.9, threat=0.3)
        name, _ = e.get_dominant_emotion()
        assert name == "RAGE"

    def test_sadness_scenario(self):
        """悲しみ: 喪失+孤立 → PANIC_GRIEF/SADNESS。"""
        e = PankseppEmotionEngine()
        e.process(loss=0.8, social=0.0, reward=0.0)
        readout = e.get_integrated_readout()
        assert readout["valence"] < 0  # ネガティブ

    def test_all_activations_bounded(self):
        e = PankseppEmotionEngine()
        e.process(threat=1, reward=1, social=1, novelty=1, pain=1,
                  loss=1, frustration=1, contamination=1, attachment_need=1)
        for s in e.systems.values():
            assert 0 <= s.activation <= 1
