"""最終統合テスト。知覚ブリッジ強化+data_driven重み+偽陽性排除。"""

from src.perception.text_analyzer import analyze_text
from src.data_driven.run_data_driven_circuit import get_data_driven_weights


class TestJapanesePerception:
    """日本語知覚ブリッジの強化テスト。"""

    def test_japanese_threat(self):
        r = analyze_text("危険だ！攻撃を受けている！逃げろ！")
        assert r.features["threat_hits"] >= 2
        assert r.sentiment_score < 0

    def test_japanese_positive(self):
        r = analyze_text("嬉しい！素晴らしい！感動した！ありがとう！")
        assert r.features["positive_hits"] >= 3
        assert r.sentiment_score > 0

    def test_japanese_high_arousal(self):
        r = analyze_text("大変だ！緊急事態！パニック！")
        assert r.features["high_arousal_hits"] >= 2
        assert r.arousal_estimate > 0.2

    def test_japanese_pain(self):
        r = analyze_text("痛い！苦しい！死にそう！")
        assert r.features["negative_hits"] >= 2
        assert r.features["threat_hits"] >= 1

    def test_english_threat(self):
        r = analyze_text("Danger! Attack! Death! Destroy!")
        assert r.features["threat_hits"] >= 3

    def test_mixed_language(self):
        r = analyze_text("危険 danger 攻撃 attack")
        assert r.features["threat_hits"] >= 2


class TestDataDrivenWeights:
    """data_driven重み取得のテスト。"""

    def test_weights_returned(self):
        weights = get_data_driven_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_la_ba_weight_positive(self):
        weights = get_data_driven_weights()
        if "la_ba_weight" in weights:
            assert weights["la_ba_weight"] > 0

    def test_bla_cea_weight_positive(self):
        weights = get_data_driven_weights()
        if "bla_cea_weight" in weights:
            assert weights["bla_cea_weight"] > 0
