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


class TestPerceptionV3:
    """v3知覚ブリッジの強化テスト。"""

    def test_no_false_positive_yokohama(self):
        """横浜 should not trigger SEEKING (ハマ→ハマる/ハマっに修正済み)."""
        r = analyze_text("横浜に住んでいます")
        assert r.features["seeking_hits"] == 0

    def test_compound_proximity(self):
        """胸が苦しい → compound hit (proximity within 10 chars)."""
        r = analyze_text("胸が苦しい")
        assert r.features.get("compound_hits", 0) >= 1
        assert r.confidence > 0.4

    def test_compound_no_false_proximity(self):
        """胸がドキドキして頭が痛い → 胸が+痛い should NOT match (too far)."""
        r = analyze_text("胸がドキドキして楽しくて最高で頭が痛い")
        # "痛い" is >10 chars from "胸が", so no compound match for 胸が+痛い
        # But "頭が" + "痛い" could match if within proximity
        compound = r.features.get("compound_hits", 0)
        assert compound <= 1  # at most 頭が+痛い

    def test_v3_new_keywords_care(self):
        """かわいい should trigger CARE."""
        r = analyze_text("赤ちゃんがかわいい")
        assert r.features["care_hits"] >= 2  # 赤ちゃん + かわいい

    def test_v3_new_keywords_surprise(self):
        """ありえない should trigger SURPRISE."""
        r = analyze_text("ありえない！仰天した！")
        assert r.features["surprise_hits"] >= 2

    def test_dedup_no_double_count(self):
        """楽しい should count as PLAY only (not SEEKING after dedup)."""
        r = analyze_text("楽しい")
        assert r.features["play_hits"] >= 1
        assert r.features["seeking_hits"] == 0


class TestPerceptionBenchmark:
    """10情動×日英ベンチマーク（v3認識率100%検証）。"""

    CASES = [
        ("怖い！助けて！逃げろ！", "fear_hits"),
        ("I am terrified and scared", "fear_hits"),
        ("むかつく！許せない！ブチ切れ", "rage_hits"),
        ("I am furious and angry", "rage_hits"),
        ("嬉しい！やった！成功した！", "seeking_hits"),
        ("I am so happy and excited", "seeking_hits"),
        ("悲しい...辛い...切ない", "sadness_hits"),
        ("I feel so sad and lonely", "sadness_hits"),
        ("気持ち悪い！吐き気がする", "disgust_hits"),
        ("That is disgusting and gross", "disgust_hits"),
        ("大好き！ありがとう！癒される", "care_hits"),
        ("I love you, you are so kind", "care_hits"),
        ("寂しい...会いたい...帰ってきて", "panic_grief_hits"),
        ("I miss you, please come back", "panic_grief_hits"),
        ("楽しい！面白い！笑える！ウケる", "play_hits"),
        ("That was hilarious and fun", "play_hits"),
        ("セクシー...魅力的...ドキドキ", "lust_hits"),
        ("So attractive and seductive", "lust_hits"),
        ("びっくり！ありえない！まさか！", "surprise_hits"),
        ("I am shocked and amazed, wow", "surprise_hits"),
    ]

    def test_all_20_cases_detected(self):
        """All 20 benchmark cases should have >= 2 hits."""
        failed = []
        for text, key in self.CASES:
            r = analyze_text(text)
            hits = r.features.get(key, 0)
            if hits < 2:
                failed.append(f"{key}: {hits} hits for '{text[:30]}'")
        assert not failed, f"Failed: {failed}"


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
