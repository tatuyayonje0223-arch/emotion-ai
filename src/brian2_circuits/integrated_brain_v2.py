"""統合脳モデル V2。全10情動回路 + 神経修飾 + 睡眠リプレイ。

V1 (integrated_brain.py) との違い:
  - 10情動回路（V1は3回路: fear/reward/stress）
  - 232検証済み論文パラメータ（V1は手動設定）
  - SharedCoreNetwork（V1はHybridBrain）
  - 5 spiking + 5 mean-field（V1は3 spiking + 4 mean-field）
  - 文献準拠の相互作用（V1は手動線形結合）

V1は凍結して維持。V2がテスト全パス後に正式システムに昇格。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2, EmotionStateV2
from src.brian2_circuits.neuromodulation import (
    EndocannabinoidState, AcetylcholineState, ThetaOscillator,
    StructuralPlasticityState,
    update_endocannabinoid, update_acetylcholine,
    compute_theta_coherence, update_structural_plasticity,
)
from src.brian2_circuits.sleep_replay import SleepReplayEngine
from src.neurocircuit.brain import SensoryInput, EmotionReadout
from src.neurocircuit.perception_bridge import text_to_sensory
from src.policy.response import ResponsePolicy, derive_policy
from src.safety.guardian import full_safety_check
from src.schemas.affect_state import AffectState


class IntegratedResultV2(BaseModel):
    """V2統合脳の出力。"""

    text: str = ""
    emotion_state: dict[str, Any] = Field(default_factory=dict)
    readout: EmotionReadout = Field(default_factory=EmotionReadout)
    policy: ResponsePolicy = Field(default_factory=ResponsePolicy)
    neuromodulation: dict[str, float] = Field(default_factory=dict)
    theta_coherence: float = 0.0
    memory_stats: dict[str, Any] = Field(default_factory=dict)
    spiking_neurons: int = 0
    blocked: bool = False
    step: int = 0


class IntegratedBrainV2:
    """V2統合情動脳。10情動回路 + 神経修飾 + 睡眠リプレイ。

    テキスト → 知覚ブリッジ → 安全チェック → EmotionBrainV2(10情動)
    → 神経修飾更新 → 記憶エンコード → readout → 応答ポリシー
    """

    def __init__(self, seed: int = 42):
        import random as _random
        self._rng = _random.Random(seed)
        self._emotion_brain = EmotionBrainV2()

        # 神経修飾 (V1と同じモジュール)
        self._ecb = EndocannabinoidState()
        self._ach = AcetylcholineState()
        self._theta_bla = ThetaOscillator(frequency_hz=6.0)
        self._theta_hippo = ThetaOscillator(frequency_hz=6.0)
        self._structural = StructuralPlasticityState()
        self._sleep_engine = SleepReplayEngine()
        self._step = 0
        self._is_extinction_mode = False

    def process(self, text: str) -> IntegratedResultV2:
        """テキストを処理し、10情動の統合結果を返す。"""
        self._step += 1

        # 1. 知覚ブリッジ
        sensory = text_to_sensory(text)

        # 2. 安全チェック
        dummy_state = AffectState()
        safety = full_safety_check(f"v2-{self._step}", dummy_state, text, self._step)
        if safety.blocked:
            return IntegratedResultV2(
                text=text, readout=EmotionReadout(),
                policy=derive_policy(dummy_state),
                blocked=True, step=self._step,
            )

        # 3. 10情動回路処理
        emotion = self._emotion_brain.process(
            threat=sensory.threat_signal,
            reward=sensory.reward_signal,
            social=sensory.social_signal,
            novelty=sensory.novelty_signal,
            pain=sensory.pain_input,
            loss=max(0, sensory.threat_signal * 0.3 - sensory.reward_signal * 0.5),
            frustration=max(0, sensory.pain_input * 0.5 + sensory.threat_signal * 0.3 - sensory.reward_signal * 0.3),
            contamination=max(0, sensory.pain_input * 0.3 - sensory.reward_signal * 0.5),
            attachment_need=sensory.social_signal * 0.5,
        )

        # 4. 神経修飾更新 (V2の発火率を使用)
        bla_activity = emotion.all_rates.get("la_exc", 0) / 100.0
        cem_activity = emotion.all_rates.get("cem", 0) / 100.0

        self._ecb = update_endocannabinoid(
            self._ecb, bla_activity, 0.3, self._is_extinction_mode, dt_ms=200.0,
        )
        self._ach = update_acetylcholine(
            self._ach, sensory.threat_signal, sensory.reward_signal, bla_activity, dt_ms=200.0,
        )

        # シータ振動
        for _ in range(200):
            self._theta_bla.step(dt_ms=1.0)
            self._theta_hippo.step(dt_ms=1.0)
            if sensory.threat_signal < 0.2:
                self._theta_hippo.phase += self._rng.gauss(0, 0.05)
        theta_coh = compute_theta_coherence(self._theta_bla, self._theta_hippo)

        # 構造的可塑性
        ltp = sensory.threat_signal > 0.3 or sensory.reward_signal > 0.3
        ltd = self._is_extinction_mode
        self._structural = update_structural_plasticity(
            self._structural, ltp, ltd, bla_activity, dt_ms=200.0,
        )

        # 5. 記憶エンコード (新: 10情動のsalienceを使用)
        max_emotion = max(emotion.fear, emotion.rage, emotion.seeking,
                          emotion.sadness, emotion.disgust)
        salience = max(max_emotion, sensory.threat_signal, sensory.reward_signal)
        if salience > 0.2:
            strength = salience * (1.0 + self._ach.nbm_ach * 0.5)
            self._sleep_engine.add_memory(
                event_id=f"v2-{self._step}",
                content=text[:100],
                salience=min(1.0, salience),
                valence=emotion.valence,
                strength=min(1.0, strength),
            )

        # 6. V1互換readout (V2情動 → 9次元EmotionReadout)
        ecb_mod = self._ecb.extinction_signal * 0.3
        ach_mod = self._ach.nbm_ach * 0.15
        theta_mod = theta_coh * 0.1
        spine_mod = (self._structural.spine_density - 1.0) * 0.1

        readout = EmotionReadout(
            valence=max(-1, min(1, emotion.valence + ecb_mod)),
            arousal=max(0, min(1, emotion.arousal + theta_mod)),
            threat_load=max(0, min(1, emotion.fear * 0.6 + emotion.rage * 0.3 - ecb_mod + spine_mod)),
            reward_drive=max(0, min(1, emotion.seeking)),
            social_warmth=max(0, min(1, emotion.care * 0.5 + emotion.play * 0.3)),
            cognitive_control=0.5,
            body_distress=max(0, min(1, emotion.sadness * 0.3 + emotion.panic_grief * 0.3 + emotion.disgust * 0.2)),
            energy=max(0, min(1, 1.0 - emotion.sadness * 0.3 - emotion.panic_grief * 0.2)),
            memory_encoding_boost=min(1, max_emotion * 0.5 + ach_mod),
        )

        # 7. 応答ポリシー
        affect = AffectState(
            valence=readout.valence, arousal=readout.arousal,
            threat_load=readout.threat_load, trust=readout.social_warmth,
            fatigue=1.0 - readout.energy,
        )
        policy = derive_policy(affect)

        return IntegratedResultV2(
            text=text,
            emotion_state=emotion.to_dict(),
            readout=readout,
            policy=policy,
            neuromodulation={
                "ecb_2ag": self._ecb.two_ag,
                "ecb_extinction": self._ecb.extinction_signal,
                "ach_nbm": self._ach.nbm_ach,
                "theta_coherence": theta_coh,
                "spine_density": self._structural.spine_density,
                "bdnf": self._structural.bdnf_level,
            },
            theta_coherence=theta_coh,
            memory_stats=self._sleep_engine.get_memory_stats(),
            spiking_neurons=self._emotion_brain.total_neurons,
            step=self._step,
        )

    def sleep(self, n_cycles: int = 1) -> list[dict]:
        results = []
        for _ in range(n_cycles):
            results.append(self._sleep_engine.run_sleep_cycle())
        return results

    def set_extinction_mode(self, enabled: bool) -> None:
        self._is_extinction_mode = enabled

    @property
    def memory_count(self) -> int:
        return len(self._sleep_engine.memories)

    @property
    def total_neurons(self) -> int:
        return self._emotion_brain.total_neurons

    def reset(self) -> None:
        self._emotion_brain.reset()
        self._ecb = EndocannabinoidState()
        self._ach = AcetylcholineState()
        self._structural = StructuralPlasticityState()
        self._step = 0
