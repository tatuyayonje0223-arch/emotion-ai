"""テキスト入力からの情動信号抽出。10情動キーワード辞書（日英両対応）。"""

from __future__ import annotations

import re

from src.schemas.events import PerceptionSignal

# ════════════════════════════════════════════════════════════════
# 10情動キーワード辞書（日本語 + 英語）
# 各辞書はその情動に特徴的な単語・表現を含む。
# 部分文字列マッチ（日本語対応）で検出する。
# ════════════════════════════════════════════════════════════════

_FEAR_WORDS = {
    # 日本語
    "怖い", "怖", "恐怖", "恐ろしい", "恐れ", "不安", "心配", "脅威", "危険",
    "パニック", "震え", "助けて", "助け", "逃げ", "逃げろ", "やばい",
    "恐怖症", "怯え", "ビクビク", "おびえ", "戦慄", "身震い",
    # 英語
    "scared", "afraid", "fear", "terrified", "frightened", "anxious", "panic",
    "dread", "horror", "alarmed", "phobia", "nervous", "worried", "threat",
    "danger", "help me", "run", "escape",
}

_RAGE_WORDS = {
    # 日本語
    "怒り", "怒", "腹立", "むかつく", "ムカつく", "イライラ", "いらいら",
    "激怒", "憤り", "キレ", "ふざけるな", "許せない", "ぶっ殺", "殴",
    "暴力", "攻撃", "喧嘩", "うざい", "ウザい", "頭にくる", "憎い", "憎",
    # 英語
    "angry", "furious", "rage", "enraged", "irritated", "annoyed", "frustrated",
    "outraged", "mad", "pissed", "hate", "violent", "punch", "fight",
    "revenge", "hostile", "aggressive",
}

_SEEKING_WORDS = {
    # 日本語
    "嬉しい", "楽しい", "素晴らしい", "最高", "幸せ", "喜び", "期待",
    "ワクワク", "わくわく", "希望", "成功", "達成", "報酬", "ご褒美",
    "やった", "面白い", "好奇心", "探索", "発見", "興味", "挑戦",
    # 英語
    "happy", "excited", "wonderful", "great", "joy", "delighted", "reward",
    "curious", "explore", "discover", "achieve", "success", "interesting",
    "fun", "awesome", "fantastic", "thrilled", "eager", "motivated",
}

_SADNESS_WORDS = {
    # 日本語
    "悲しい", "悲し", "辛い", "つらい", "寂しい", "さみしい", "孤独",
    "絶望", "落ち込", "憂鬱", "うつ", "鬱", "泣", "涙", "無理",
    "もうだめ", "やる気ない", "失意", "虚しい", "空虚", "惨め", "後悔",
    "喪失", "失った", "別れ", "死別",
    # 英語
    "sad", "depressed", "hopeless", "miserable", "grief", "sorrow", "crying",
    "tears", "lonely", "heartbroken", "devastated", "gloomy", "melancholy",
    "despair", "lost", "mourning", "bereaved", "empty",
}

_DISGUST_WORDS = {
    # 日本語
    "気持ち悪い", "きもい", "キモい", "吐き気", "嫌悪", "不潔", "汚い",
    "汚", "臭い", "くさい", "ゲロ", "嘔吐", "ムカムカ", "不快",
    "ゾッと", "ぞっと", "おぞましい", "最悪", "ドン引き", "嫌",
    # 英語
    "disgusting", "gross", "nausea", "repulsive", "revolting", "sick",
    "nasty", "filthy", "vomit", "stink", "creepy", "yuck", "eww",
    "contaminated", "rotten", "foul", "putrid", "abhorrent",
}

_CARE_WORDS = {
    # 日本語
    "大好き", "愛して", "愛", "ありがとう", "感謝", "優しい", "温かい",
    "守り", "世話", "育て", "抱きしめ", "慈愛", "思いやり", "絆",
    "家族", "赤ちゃん", "子供", "母", "親", "仲間",
    # 英語
    "love", "care", "nurture", "compassion", "kindness", "tender",
    "protect", "cherish", "adore", "affection", "bonding", "family",
    "baby", "child", "mother", "warm", "gentle", "hug", "grateful", "thanks",
}

_PANIC_GRIEF_WORDS = {
    # 日本語
    "別れ", "離れ", "失った", "喪失", "死別", "孤独", "見捨て",
    "置いていか", "一人ぼっち", "ひとりぼっち", "泣き叫", "号泣",
    "帰ってきて", "会いたい", "寂しくて", "耐えられ",
    # 英語
    "separation", "abandoned", "alone", "lost someone", "grief", "bereaved",
    "miss you", "come back", "left behind", "isolated", "forsaken",
    "heartbreak", "broken heart", "longing", "yearning",
}

_PLAY_WORDS = {
    # 日本語
    "遊び", "遊ぼ", "楽しい", "面白い", "笑", "ゲーム", "冗談",
    "ふざけ", "はしゃ", "じゃれ", "ワイワイ", "わいわい",
    "友達", "仲間", "みんな", "パーティ",
    # 英語
    "play", "fun", "game", "laugh", "joke", "silly", "playful",
    "party", "hang out", "together", "friends", "enjoy", "humor",
    "tickle", "wrestle", "chase",
}

_LUST_WORDS = {
    # 日本語
    "魅力", "セクシー", "色気", "欲望", "情熱", "恋", "ドキドキ",
    "惹かれ", "キス", "抱き", "触れ",
    # 英語
    "sexy", "attractive", "desire", "passion", "lust", "aroused",
    "seductive", "kiss", "touch", "intimate", "sensual",
}

_SURPRISE_WORDS = {
    # 日本語
    "びっくり", "驚き", "驚い", "まさか", "えっ", "うそ", "信じられない",
    "突然", "予想外", "意外", "衝撃", "唖然", "呆然",
    # 英語
    "surprised", "shocked", "unexpected", "wow", "amazing", "astonished",
    "unbelievable", "sudden", "startled", "jaw-dropping", "stunning",
}

# Legacy compatibility
_POSITIVE_WORDS = _SEEKING_WORDS | _CARE_WORDS | _PLAY_WORDS
_NEGATIVE_WORDS = _FEAR_WORDS | _RAGE_WORDS | _SADNESS_WORDS | _DISGUST_WORDS | _PANIC_GRIEF_WORDS

_HIGH_AROUSAL_WORDS = {
    "驚き", "びっくり", "緊急", "危険", "衝撃", "興奮", "パニック",
    "大変", "激しい", "突然", "急", "慌", "叫", "震え",
    "shocked", "urgent", "danger", "excited", "panic", "amazing",
    "alarm", "emergency", "sudden", "intense", "scream",
}

_THREAT_WORDS = _FEAR_WORDS  # threat = fear


def analyze_text(text: str) -> PerceptionSignal:
    """テキストから情動知覚信号を抽出する。10情動キーワードマッチング。"""
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

    # 10情動キーワードカウント
    counts = {
        "fear": sum(1 for w in _FEAR_WORDS if w in text_lower),
        "rage": sum(1 for w in _RAGE_WORDS if w in text_lower),
        "seeking": sum(1 for w in _SEEKING_WORDS if w in text_lower),
        "sadness": sum(1 for w in _SADNESS_WORDS if w in text_lower),
        "disgust": sum(1 for w in _DISGUST_WORDS if w in text_lower),
        "care": sum(1 for w in _CARE_WORDS if w in text_lower),
        "panic_grief": sum(1 for w in _PANIC_GRIEF_WORDS if w in text_lower),
        "play": sum(1 for w in _PLAY_WORDS if w in text_lower),
        "lust": sum(1 for w in _LUST_WORDS if w in text_lower),
        "surprise": sum(1 for w in _SURPRISE_WORDS if w in text_lower),
    }

    # ポジティブ/ネガティブ集計
    pos_count = counts["seeking"] + counts["care"] + counts["play"]
    neg_count = counts["fear"] + counts["rage"] + counts["sadness"] + counts["disgust"]
    total_emotional = pos_count + neg_count + counts["surprise"] + counts["panic_grief"] + counts["lust"]

    if total_emotional == 0:
        sentiment = 0.0
    else:
        sentiment = (pos_count - neg_count) / max(1, pos_count + neg_count)
    sentiment = max(-1.0, min(1.0, sentiment))

    # Arousal
    high_arousal_count = sum(1 for w in _HIGH_AROUSAL_WORDS if w in text_lower)
    base_arousal = min(1.0, total_emotional / max(word_count, 1) * 3.0)
    arousal_boost = min(0.4, high_arousal_count * 0.15)
    arousal = min(1.0, base_arousal + arousal_boost)

    # Confidence
    if total_emotional == 0:
        confidence = 0.2 if word_count < 3 else 0.3
    elif word_count < 3:
        confidence = min(0.7, 0.3 + total_emotional * 0.1)
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
            "threat_hits": counts["fear"],
            # 10情動カウント
            "fear_hits": counts["fear"],
            "rage_hits": counts["rage"],
            "seeking_hits": counts["seeking"],
            "sadness_hits": counts["sadness"],
            "disgust_hits": counts["disgust"],
            "care_hits": counts["care"],
            "panic_grief_hits": counts["panic_grief"],
            "play_hits": counts["play"],
            "lust_hits": counts["lust"],
            "surprise_hits": counts["surprise"],
            "method": "keyword_heuristic_v2",
        },
    )
