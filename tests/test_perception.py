"""テキスト知覚モジュールのテスト。"""

from src.perception.text_analyzer import analyze_text


class TestTextAnalyzer:
    def test_empty_input(self):
        result = analyze_text("")
        assert result.confidence < 0.2
        assert result.sentiment_score == 0.0

    def test_positive_text(self):
        result = analyze_text("嬉しい！素晴らしい結果です。ありがとう。")
        assert result.sentiment_score > 0.0
        assert result.confidence > 0.3
        assert result.modality == "text"

    def test_negative_text(self):
        result = analyze_text("悲しい。最悪の結果です。辛い。")
        assert result.sentiment_score < 0.0
        assert result.confidence > 0.3

    def test_neutral_text(self):
        result = analyze_text("今日の天気は曇りです。")
        assert abs(result.sentiment_score) < 0.3
        assert result.confidence < 0.5

    def test_high_arousal(self):
        result = analyze_text("びっくりした！衝撃的なニュースだ！")
        assert result.arousal_estimate > 0.2
        assert result.features["high_arousal_hits"] > 0

    def test_threat_detection(self):
        result = analyze_text("危険です。攻撃を受けています。")
        assert result.features["threat_hits"] > 0

    def test_mixed_sentiment(self):
        result = analyze_text("嬉しいけど不安もある。")
        assert result.features["positive_hits"] > 0
        assert result.features["negative_hits"] > 0

    def test_english_words(self):
        result = analyze_text("I'm happy and excited about this great news!")
        assert result.sentiment_score > 0.0

    def test_short_text_low_confidence(self):
        result = analyze_text("ok")
        assert result.confidence <= 0.3
