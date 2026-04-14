"""知覚ブリッジ。テキスト入力を神経回路のSensoryInputに変換する。

既存の知覚モジュール(text_analyzer/llm_analyzer)の出力を、
脳領域が受け取れるSensoryInput形式に変換する。
"""

from __future__ import annotations

from src.neurocircuit.brain import SensoryInput
from src.perception.text_analyzer import analyze_text
from src.schemas.events import PerceptionSignal


def text_to_sensory(text: str) -> SensoryInput:
    """テキストからSensoryInputを生成する。

    知覚信号の各次元を脳回路の入力チャンネルにマッピング:
    - threat_signal ← negative sentiment + threat keywords
    - reward_signal ← positive sentiment
    - social_signal ← social cues
    - novelty_signal ← arousal estimate (高覚醒≒新規性)
    - pain_input ← 強いネガティブ + 脅威
    - context_input ← テキスト長に比例（文脈情報量の簡易推定）
    """
    if not text.strip():
        return SensoryInput()

    signal = analyze_text(text)
    return perception_to_sensory(signal, text)


def perception_to_sensory(signal: PerceptionSignal, raw_text: str = "") -> SensoryInput:
    """PerceptionSignalからSensoryInputに変換する。"""
    sentiment = signal.sentiment_score
    arousal = signal.arousal_estimate
    confidence = signal.confidence
    threat_hits = signal.features.get("threat_hits", 0)
    pos_hits = signal.features.get("positive_hits", 0)
    neg_hits = signal.features.get("negative_hits", 0)

    # threat: ネガティブ感情 + 脅威語 + 高覚醒
    threat = 0.0
    if sentiment < -0.2:
        threat += abs(sentiment) * 0.5
    if threat_hits > 0:
        threat += min(0.4, threat_hits * 0.15)
    if arousal > 0.5 and sentiment < 0:
        threat += arousal * 0.2
    threat = min(1.0, threat * confidence)

    # reward: ポジティブ感情
    reward = max(0.0, sentiment) * 0.7 * confidence
    if pos_hits > 0:
        reward += min(0.3, pos_hits * 0.1)
    reward = min(1.0, reward)

    # social: 社会的手がかり
    social_words = {"みんな", "チーム", "家族", "友達", "ありがとう", "一緒", "仲間",
                    "people", "team", "family", "friend", "together", "thanks"}
    social_count = sum(1 for w in social_words if w in raw_text.lower())
    social = min(1.0, social_count * 0.2 + abs(sentiment) * 0.1)

    # novelty: 覚醒度ベース（脅威時は低下: Pessoa 2009 Nat Rev Neurosci — threat narrows attention）
    novelty = min(1.0, arousal * 0.8 * confidence)
    if threat > 0.3:
        novelty *= (1.0 - threat * 0.5)  # threat suppresses novelty processing

    # pain: 強いネガティブ + 脅威
    pain = 0.0
    pain_words = {"痛い", "苦しい", "死", "痛み", "pain", "hurt", "agony"}
    if any(w in raw_text.lower() for w in pain_words):
        pain = 0.5 + abs(min(0, sentiment)) * 0.3
    pain = min(1.0, pain)

    # context: テキスト情報量（文字数の対数）
    import math
    context = min(1.0, math.log1p(len(raw_text)) / 6.0) if raw_text else 0.0

    return SensoryInput(
        threat_signal=round(threat, 4),
        reward_signal=round(reward, 4),
        social_signal=round(social, 4),
        novelty_signal=round(novelty, 4),
        pain_input=round(pain, 4),
        context_input=round(context, 4),
    )
