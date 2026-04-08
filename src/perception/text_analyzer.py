"""テキスト入力からの情動信号抽出。MVPではキーワード+ヒューリスティクス。"""

from __future__ import annotations

import re

from src.schemas.events import PerceptionSignal

# 感情関連キーワード辞書（MVP: 拡張可能な最小構成）
_POSITIVE_WORDS = {
    "嬉しい", "楽しい", "ありがとう", "素晴らしい", "最高", "好き", "幸せ",
    "感謝", "良い", "素敵", "安心", "期待", "希望", "成功", "達成",
    "happy", "great", "thanks", "love", "wonderful", "excellent", "joy",
}

_NEGATIVE_WORDS = {
    "悲しい", "辛い", "怒り", "不安", "心配", "嫌い", "最悪", "失敗",
    "恐怖", "怖い", "痛い", "苦しい", "困った", "ストレス", "疲れた",
    "sad", "angry", "fear", "hate", "terrible", "awful", "worried", "stressed",
}

_HIGH_AROUSAL_WORDS = {
    "驚き", "びっくり", "緊急", "危険", "衝撃", "興奮", "パニック",
    "shocked", "urgent", "danger", "excited", "panic", "amazing",
}

_THREAT_WORDS = {
    "脅威", "危険", "攻撃", "批判", "失う", "死", "終わり", "崩壊",
    "threat", "danger", "attack", "lose", "death", "collapse", "destroy",
}


def analyze_text(text: str) -> PerceptionSignal:
    """テキストから情動知覚信号を抽出する。

    Returns:
        PerceptionSignal with sentiment, arousal, and confidence.
        confidence は信号強度に比例し、テキストが短い/曖昧なら低くなる。
    """
    if not text.strip():
        return PerceptionSignal(
            modality="text",
            sentiment_score=0.0,
            arousal_estimate=0.0,
            confidence=0.1,
            features={"word_count": 0, "method": "empty_input"},
        )

    text_lower = text.lower()
    word_count = max(1, len(re.findall(r"\w+", text_lower)))

    # 日本語はスペース区切りでないため、部分文字列マッチで検出する
    pos_count = sum(1 for w in _POSITIVE_WORDS if w in text_lower)
    neg_count = sum(1 for w in _NEGATIVE_WORDS if w in text_lower)
    high_arousal_count = sum(1 for w in _HIGH_AROUSAL_WORDS if w in text_lower)
    threat_count = sum(1 for w in _THREAT_WORDS if w in text_lower)

    total_emotional = pos_count + neg_count
    if total_emotional == 0:
        sentiment = 0.0
    else:
        sentiment = (pos_count - neg_count) / total_emotional
    sentiment = max(-1.0, min(1.0, sentiment))

    # arousal: 感情語の密度 + 高覚醒語
    base_arousal = min(1.0, total_emotional / max(word_count, 1) * 3.0)
    arousal_boost = min(0.4, high_arousal_count * 0.15)
    arousal = min(1.0, base_arousal + arousal_boost)

    # 信頼度: テキスト長と感情語の存在に依存
    if word_count < 3:
        confidence = 0.2
    elif total_emotional == 0:
        confidence = 0.3
    else:
        confidence = min(0.9, 0.4 + total_emotional * 0.1)

    return PerceptionSignal(
        modality="text",
        sentiment_score=sentiment,
        arousal_estimate=arousal,
        confidence=confidence,
        features={
            "word_count": word_count,
            "positive_hits": pos_count,
            "negative_hits": neg_count,
            "high_arousal_hits": high_arousal_count,
            "threat_hits": threat_count,
            "method": "keyword_heuristic_v1",
        },
    )
