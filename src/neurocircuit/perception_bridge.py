"""知覚ブリッジ。テキスト入力を神経回路のSensoryInputに変換する。

10情動キーワード辞書（v2）に基づき、各情動チャンネルに直接マッピング。
"""

from __future__ import annotations

import math

from src.neurocircuit.brain import SensoryInput
from src.perception.text_analyzer import analyze_text
from src.schemas.events import PerceptionSignal


def text_to_sensory(text: str) -> SensoryInput:
    """テキストからSensoryInputを生成する。

    10情動キーワードカウントから直接SensoryInputチャンネルにマッピング:
    - threat_signal ← fear_hits
    - reward_signal ← seeking_hits
    - social_signal ← care_hits + play_hits
    - novelty_signal ← surprise_hits (threat時は抑制: Pessoa 2009)
    - pain_input ← rage_hits + pain keywords
    - loss ← sadness_hits + panic_grief_hits (IntegratedBrainV2で使用)
    - frustration ← rage_hits (IntegratedBrainV2で使用)
    - contamination ← disgust_hits (IntegratedBrainV2で使用)
    """
    if not text.strip():
        return SensoryInput()

    signal = analyze_text(text)
    return perception_to_sensory(signal, text)


def perception_to_sensory(signal: PerceptionSignal, raw_text: str = "") -> SensoryInput:
    """PerceptionSignalからSensoryInputに変換する。"""
    features = signal.features
    confidence = signal.confidence

    # 10情動カウント取得
    fear = features.get("fear_hits", features.get("threat_hits", 0))
    rage = features.get("rage_hits", 0)
    seeking = features.get("seeking_hits", features.get("positive_hits", 0))
    sadness = features.get("sadness_hits", 0)
    disgust = features.get("disgust_hits", 0)
    care = features.get("care_hits", 0)
    panic_grief = features.get("panic_grief_hits", 0)
    play = features.get("play_hits", 0)
    lust = features.get("lust_hits", 0)
    surprise = features.get("surprise_hits", 0)

    # Keyword count → signal intensity (0-1)
    def _intensity(count: int, scale: float = 0.2) -> float:
        return min(1.0, count * scale * confidence)

    # threat_signal ← fear keywords
    threat = _intensity(fear, 0.15)

    # reward_signal ← seeking + care (positive emotions)
    reward = _intensity(seeking + care, 0.12)

    # social_signal ← care + play + lust
    social_words = {"みんな", "チーム", "家族", "友達", "一緒", "仲間",
                    "people", "team", "family", "friend", "together"}
    social_count = sum(1 for w in social_words if w in raw_text.lower())
    social = min(1.0, _intensity(care + play + lust, 0.12) + social_count * 0.15)

    # novelty_signal ← surprise (threat時は抑制: Pessoa 2009)
    novelty = _intensity(surprise, 0.15)
    if threat > 0.3:
        novelty *= (1.0 - threat * 0.5)

    # pain_input ← rage + pain words
    pain_words = {"痛い", "苦しい", "痛み", "pain", "hurt", "agony", "suffering"}
    pain_count = sum(1 for w in pain_words if w in raw_text.lower())
    pain = min(1.0, _intensity(rage, 0.1) + pain_count * 0.2 * confidence)

    # context: テキスト情報量
    context = min(1.0, math.log1p(len(raw_text)) / 6.0) if raw_text else 0.0

    return SensoryInput(
        threat_signal=round(threat, 4),
        reward_signal=round(reward, 4),
        social_signal=round(social, 4),
        novelty_signal=round(novelty, 4),
        pain_input=round(pain, 4),
        context_input=round(context, 4),
    )
