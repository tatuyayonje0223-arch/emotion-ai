"""Pankseppの7情動システムの統合実装。

現在実装済み:
  FEAR: BLA/CeA (fear_circuit_v2.py)
  SEEKING: VTA/NAc (reward_circuit_v2.py) — 報酬=SEEKING系
  PANIC/GRIEF部分: HPA/BNST (stress_circuit_v2.py) — ストレス応答

新規追加:
  RAGE: 怒り/激怒 — MeA→VMH→PAG
  CARE: 養育/慈愛 — OXT系（既存のOXTを拡張）
  PANIC/GRIEF完全版: 分離苦痛 — ACC→BNST→PAG（オピオイド低下）
  PLAY: 遊び/社会的喜び — 視床→皮質（オピオイド+eCB）
  LUST: 簡略実装 — 視床下部（テストステロン/エストロゲン）
  DISGUST: 嫌悪 — 前島皮質（Barrett/Ekman）
  SADNESS: 悲しみ — subgenual ACC + オピオイド低下
  SURPRISE: 驚き — LC burst + 扁桃体（NE急増）

全システムをSensoryInputチャンネル→EmotionReadoutにマッピング。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field


@dataclass
class EmotionSystemState:
    """1つの情動システムの活性状態。"""

    name: str
    activation: float = 0.0  # 0-1
    valence: float = 0.0     # -1 to 1
    arousal: float = 0.0     # 0-1
    key_neurotransmitter: str = ""
    behavioral_output: str = ""


class PankseppEmotionEngine:
    """Pankseppの7情動システム + 拡張（嫌悪/悲しみ/驚き）。

    各システムは入力信号から活性度を計算し、
    統合readoutとして全システムの加重和を返す。
    """

    def __init__(self):
        self._systems: dict[str, EmotionSystemState] = {
            "SEEKING": EmotionSystemState("SEEKING", valence=0.7, arousal=0.6,
                                           key_neurotransmitter="DA", behavioral_output="exploration"),
            "RAGE": EmotionSystemState("RAGE", valence=-0.8, arousal=0.9,
                                       key_neurotransmitter="SubstanceP+Glu", behavioral_output="attack"),
            "FEAR": EmotionSystemState("FEAR", valence=-0.9, arousal=0.9,
                                       key_neurotransmitter="CRH+NE", behavioral_output="freeze/flight"),
            "LUST": EmotionSystemState("LUST", valence=0.6, arousal=0.7,
                                       key_neurotransmitter="testosterone+OXT", behavioral_output="approach"),
            "CARE": EmotionSystemState("CARE", valence=0.8, arousal=0.3,
                                       key_neurotransmitter="OXT+prolactin", behavioral_output="nurture"),
            "PANIC_GRIEF": EmotionSystemState("PANIC_GRIEF", valence=-0.7, arousal=0.6,
                                               key_neurotransmitter="opioid_low+CRH", behavioral_output="cry/seek_attachment"),
            "PLAY": EmotionSystemState("PLAY", valence=0.9, arousal=0.8,
                                       key_neurotransmitter="opioid+eCB+DA", behavioral_output="social_play"),
            # 拡張
            "DISGUST": EmotionSystemState("DISGUST", valence=-0.6, arousal=0.5,
                                           key_neurotransmitter="5HT", behavioral_output="rejection/avoidance"),
            "SADNESS": EmotionSystemState("SADNESS", valence=-0.5, arousal=0.2,
                                           key_neurotransmitter="opioid_low+5HT_low", behavioral_output="withdrawal/cry"),
            "SURPRISE": EmotionSystemState("SURPRISE", valence=0.0, arousal=0.9,
                                            key_neurotransmitter="NE_burst", behavioral_output="attention_redirect"),
        }

        # Plutchik複合情動
        self._compounds: dict[str, tuple[str, str]] = {
            "LOVE": ("PLAY", "CARE"),           # Joy+Trust → Love
            "AWE": ("FEAR", "SURPRISE"),          # Fear+Surprise → Awe
            "REMORSE": ("SADNESS", "DISGUST"),    # Sadness+Disgust → Remorse
            "CONTEMPT": ("DISGUST", "RAGE"),      # Disgust+Anger → Contempt
            "OPTIMISM": ("SEEKING", "PLAY"),      # Anticipation+Joy → Optimism
            "SUBMISSION": ("FEAR", "CARE"),        # Trust+Fear → Submission
            "AGGRESSIVENESS": ("RAGE", "SEEKING"), # Anger+Anticipation
        }

    def process(
        self,
        threat: float = 0.0,
        reward: float = 0.0,
        social: float = 0.0,
        novelty: float = 0.0,
        pain: float = 0.0,
        loss: float = 0.0,
        frustration: float = 0.0,
        contamination: float = 0.0,
        attachment_need: float = 0.0,
    ) -> dict[str, EmotionSystemState]:
        """入力信号から全情動システムの活性度を計算する。"""
        s = self._systems

        # FEAR: 脅威+疼痛
        s["FEAR"].activation = min(1.0, threat * 0.7 + pain * 0.3)

        # SEEKING: 報酬+新規性（好奇心）
        s["SEEKING"].activation = min(1.0, reward * 0.5 + novelty * 0.3 + (1 - threat) * 0.2)

        # RAGE: フラストレーション+脅威（闘争モード）
        s["RAGE"].activation = min(1.0, frustration * 0.6 + threat * 0.2 + pain * 0.2)

        # CARE: 社会的信号+愛着
        s["CARE"].activation = min(1.0, social * 0.6 + attachment_need * 0.3 + (1 - threat) * 0.1)

        # PANIC/GRIEF: 喪失+愛着欲求（分離苦痛）
        s["PANIC_GRIEF"].activation = min(1.0, loss * 0.5 + attachment_need * 0.3 + (1 - social) * 0.2)

        # PLAY: 社会的+報酬+低脅威
        s["PLAY"].activation = min(1.0, social * 0.4 + reward * 0.3 + (1 - threat) * 0.2 + novelty * 0.1)

        # LUST: （簡略）社会的信号の一部
        s["LUST"].activation = min(1.0, social * 0.3 + reward * 0.2)

        # DISGUST: 汚染シグナル
        s["DISGUST"].activation = min(1.0, contamination * 0.7 + (1 - reward) * 0.1)

        # SADNESS: 喪失+低報酬
        s["SADNESS"].activation = min(1.0, loss * 0.5 + (1 - reward) * 0.2 + (1 - social) * 0.2)

        # SURPRISE: 新規性の急増
        s["SURPRISE"].activation = min(1.0, novelty * 0.8 + abs(threat - 0.5) * 0.2)

        return dict(s)

    def get_dominant_emotion(self) -> tuple[str, float]:
        """最も活性の高い情動を返す。"""
        best = max(self._systems.values(), key=lambda s: s.activation)
        return best.name, best.activation

    def get_compound_emotions(self) -> dict[str, float]:
        """Plutchik複合情動の活性度を返す。"""
        compounds = {}
        for name, (e1, e2) in self._compounds.items():
            a1 = self._systems[e1].activation
            a2 = self._systems[e2].activation
            compounds[name] = min(1.0, (a1 + a2) / 2 * 1.2)  # 重なりでわずかに増幅
        return compounds

    def get_integrated_readout(self) -> dict[str, float]:
        """全システムの加重統合readout。"""
        total_activation = sum(s.activation for s in self._systems.values())
        if total_activation == 0:
            return {"valence": 0.0, "arousal": 0.3, "dominant": "none"}

        # 活性度加重平均
        valence = sum(s.activation * s.valence for s in self._systems.values()) / total_activation
        arousal = sum(s.activation * s.arousal for s in self._systems.values()) / total_activation

        dominant, conf = self.get_dominant_emotion()
        compounds = self.get_compound_emotions()
        top_compound = max(compounds, key=compounds.get) if compounds else "none"

        return {
            "valence": max(-1, min(1, valence)),
            "arousal": max(0, min(1, arousal)),
            "dominant_system": dominant,
            "dominant_confidence": conf,
            "top_compound": top_compound,
            "compound_confidence": compounds.get(top_compound, 0),
            "all_activations": {s.name: round(s.activation, 3) for s in self._systems.values()},
        }

    @property
    def systems(self) -> dict[str, EmotionSystemState]:
        return dict(self._systems)
