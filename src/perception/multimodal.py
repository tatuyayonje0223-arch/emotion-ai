"""マルチモーダル入力インターフェース。

[ステップ3] テキスト以外の入力ソースからSensoryInputを生成する基盤。
実際のハードウェア/外部APIは未接続だが、インターフェースを定義する。

サポート予定:
  - 音声: ピッチ/速度/エネルギー → arousal推定
  - 表情: AU(Action Unit) → valence/arousal推定
  - 心拍: BPM → 内受容信号（島皮質mean-fieldに直接注入）
  - テキスト: 既存の text_to_sensory()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.neurocircuit.brain import SensoryInput


@dataclass
class AudioFeatures:
    """音声特徴量。librosa等で抽出する想定。"""

    pitch_mean_hz: float = 0.0       # 基本周波数の平均
    pitch_std_hz: float = 0.0        # ピッチの変動
    energy_db: float = 0.0           # エネルギー(dB)
    speech_rate_syl_per_sec: float = 0.0  # 発話速度
    pause_ratio: float = 0.0         # ポーズの割合


@dataclass
class FacialFeatures:
    """表情特徴量。mediapipe等で抽出する想定。"""

    valence_estimate: float = 0.0    # 快不快推定(-1〜1)
    arousal_estimate: float = 0.0    # 覚醒推定(0〜1)
    au_brow_raise: float = 0.0      # 眉上げ(AU1+AU2)
    au_brow_furrow: float = 0.0     # 眉しかめ(AU4)
    au_lip_corner_pull: float = 0.0  # 口角上げ(AU12, 笑顔)
    au_lip_corner_depress: float = 0.0  # 口角下げ(AU15)
    confidence: float = 0.0


@dataclass
class PhysiologicalFeatures:
    """生理信号特徴量。ウェアラブル等から取得する想定。"""

    heart_rate_bpm: float = 70.0     # 心拍数
    heart_rate_variability: float = 50.0  # HRV (ms)
    skin_conductance: float = 0.0    # 皮膚電気伝導(μS)
    respiration_rate: float = 15.0   # 呼吸数(/min)


def audio_to_sensory(features: AudioFeatures) -> SensoryInput:
    """音声特徴量からSensoryInputに変換する。

    高ピッチ+速い発話→高arousal→novelty↑
    低エネルギー+遅い発話→低arousal
    ピッチ変動大→情動的発話→threat or reward
    """
    # arousal推定: ピッチ+速度+エネルギー
    pitch_norm = min(1.0, max(0.0, (features.pitch_mean_hz - 100) / 200))  # 100-300Hz正規化
    speed_norm = min(1.0, max(0.0, (features.speech_rate_syl_per_sec - 2) / 5))
    energy_norm = min(1.0, max(0.0, (features.energy_db + 40) / 60))  # -40〜20dB正規化

    arousal = (pitch_norm * 0.4 + speed_norm * 0.3 + energy_norm * 0.3)

    # 脅威推定: 高ピッチ変動 + 高エネルギー
    threat = 0.0
    if features.pitch_std_hz > 50 and features.energy_db > 0:
        threat = min(1.0, features.pitch_std_hz / 100 * 0.5)

    return SensoryInput(
        threat_signal=min(1.0, threat),
        novelty_signal=min(1.0, arousal * 0.8),
        context_input=min(1.0, 0.3),  # 音声が存在する=文脈情報あり
    )


def facial_to_sensory(features: FacialFeatures) -> SensoryInput:
    """表情特徴量からSensoryInputに変換する。

    笑顔(AU12)→reward↑
    眉しかめ(AU4)→threat↑
    """
    reward = max(0.0, features.au_lip_corner_pull * 0.7 + features.valence_estimate * 0.3)
    threat = max(0.0, features.au_brow_furrow * 0.5 - features.au_lip_corner_pull * 0.3)
    social = max(0.0, features.confidence * 0.5)

    return SensoryInput(
        threat_signal=min(1.0, threat * features.confidence),
        reward_signal=min(1.0, reward * features.confidence),
        social_signal=min(1.0, social),
        novelty_signal=min(1.0, features.arousal_estimate * 0.5),
    )


def physiological_to_sensory(features: PhysiologicalFeatures) -> SensoryInput:
    """生理信号からSensoryInputに変換する。

    高心拍→arousal↑→threat_signalとして扱う
    低HRV→ストレス→threat↑
    高皮膚電気伝導→覚醒↑
    """
    # 心拍の正規化（60-120BPM → 0-1）
    hr_norm = min(1.0, max(0.0, (features.heart_rate_bpm - 60) / 60))

    # HRVの逆数（低HRV=高ストレス）
    hrv_stress = max(0.0, 1.0 - features.heart_rate_variability / 100)

    # 皮膚電気伝導
    eda_norm = min(1.0, features.skin_conductance / 10.0)

    threat = (hr_norm * 0.3 + hrv_stress * 0.4 + eda_norm * 0.3) * 0.5
    pain = hr_norm * 0.2  # 高心拍は身体的苦痛のシグナルにもなる

    return SensoryInput(
        threat_signal=min(1.0, threat),
        pain_input=min(1.0, pain),
        novelty_signal=min(1.0, eda_norm * 0.5),
    )


def merge_sensory_inputs(*inputs: SensoryInput, weights: list[float] | None = None) -> SensoryInput:
    """複数のSensoryInputを信頼度加重で統合する。

    テキスト+音声+表情+生理信号を1つのSensoryInputに統合。
    """
    if not inputs:
        return SensoryInput()

    if weights is None:
        weights = [1.0] * len(inputs)

    total_w = sum(weights)
    if total_w == 0:
        return SensoryInput()

    fields = ["threat_signal", "reward_signal", "social_signal",
              "novelty_signal", "pain_input", "context_input"]

    merged = {}
    for f in fields:
        weighted_sum = sum(getattr(inp, f) * w for inp, w in zip(inputs, weights))
        merged[f] = float(min(1.0, max(0.0, weighted_sum / total_w)))

    return SensoryInput(**merged)
