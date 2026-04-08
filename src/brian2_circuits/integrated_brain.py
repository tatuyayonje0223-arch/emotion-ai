"""統合脳モデル。全モジュールを結合した最終的なEmotionBrain。

ハイブリッド脳 + 神経修飾 + 睡眠リプレイ + 知覚ブリッジ を統合。
テキスト入力からE2Eで動作する完全な情動AI。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from src.brian2_circuits.hybrid_brain import HybridBrain, HybridResult
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


class IntegratedResult(BaseModel):
    """統合脳の出力。"""

    text: str = ""
    readout: EmotionReadout
    policy: ResponsePolicy
    region_activities: dict[str, float] = Field(default_factory=dict)
    neuromodulation: dict[str, float] = Field(default_factory=dict)
    theta_coherence: float = 0.0
    memory_stats: dict[str, Any] = Field(default_factory=dict)
    virtual_neurons: int = 0
    blocked: bool = False
    step: int = 0


class EmotionBrain:
    """統合情動脳。全モジュールを結合したE2Eシステム。

    テキスト → 知覚ブリッジ → 安全チェック → ハイブリッド脳
    → 神経修飾更新 → 記憶エンコード → readout → 応答ポリシー

    睡眠リプレイは明示的に呼び出す。
    """

    def __init__(self, seed: int = 42):
        import random as _random
        self._rng = _random.Random(seed)  # [R5修正] 再現性のためシード固定
        self._hybrid = HybridBrain()
        self._ecb = EndocannabinoidState()
        self._ach = AcetylcholineState()
        self._theta_bla = ThetaOscillator(frequency_hz=6.0)
        self._theta_hippo = ThetaOscillator(frequency_hz=6.0)
        self._structural = StructuralPlasticityState()
        self._sleep_engine = SleepReplayEngine()
        self._step = 0
        self._is_extinction_mode = False

    def process(self, text: str) -> IntegratedResult:
        """テキストを処理する。"""
        self._step += 1

        # 1. 知覚ブリッジ
        sensory = text_to_sensory(text)

        # 2. 安全チェック
        dummy_state = AffectState()
        safety = full_safety_check(f"ibrain-{self._step}", dummy_state, text, self._step)
        if safety.blocked:
            return IntegratedResult(
                text=text, readout=EmotionReadout(),
                policy=derive_policy(dummy_state),
                blocked=True, step=self._step,
            )

        # 3. ハイブリッド脳処理
        h_result = self._hybrid.process(sensory)

        # 4. 神経修飾更新
        bla_activity = h_result.spiking_result.region_activities.get("la_exc", 0) / 100.0
        cem_activity = h_result.spiking_result.region_activities.get("cem", 0) / 100.0

        self._ecb = update_endocannabinoid(
            self._ecb, bla_activity, 0.3, self._is_extinction_mode, dt_ms=200.0,
        )
        self._ach = update_acetylcholine(
            self._ach, sensory.threat_signal, sensory.reward_signal, bla_activity, dt_ms=200.0,
        )

        # シータ振動更新（脅威で同期↑、中立で位相ノイズ）
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

        # 5. 記憶エンコード
        salience = max(sensory.threat_signal, sensory.reward_signal, sensory.pain_input)
        if salience > 0.2:
            strength = salience * (1.0 + self._ach.nbm_ach * 0.5)  # ACh→記憶強化
            self._sleep_engine.add_memory(
                event_id=f"ev-{self._step}",
                content=text[:100],
                salience=min(1.0, salience),
                valence=h_result.readout.valence,
                strength=min(1.0, strength),
            )

        # 6. readout（神経修飾の影響を統合）[問題4修正: 影響を増幅]
        base = h_result.readout
        ecb_mod = self._ecb.extinction_signal * 0.3         # 消去→valence改善+脅威低下
        ach_mod = self._ach.nbm_ach * 0.15                  # ACh→記憶強化
        theta_mod = theta_coh * 0.1                          # シータ同期→覚醒変調
        spine_mod = (self._structural.spine_density - 1.0) * 0.1  # スパイン→脅威感度

        readout = EmotionReadout(
            valence=max(-1, min(1, base.valence + ecb_mod)),
            arousal=max(0, min(1, base.arousal + theta_mod)),
            threat_load=max(0, min(1, base.threat_load - ecb_mod + spine_mod)),
            reward_drive=base.reward_drive,
            social_warmth=base.social_warmth,
            cognitive_control=base.cognitive_control,
            body_distress=base.body_distress,
            energy=base.energy,
            memory_encoding_boost=min(1, base.memory_encoding_boost + ach_mod),
        )

        # 7. 応答ポリシー
        affect = AffectState(
            valence=readout.valence, arousal=readout.arousal,
            threat_load=readout.threat_load, trust=readout.social_warmth,
            fatigue=1.0 - readout.energy,
        )
        policy = derive_policy(affect)

        return IntegratedResult(
            text=text,
            readout=readout,
            policy=policy,
            region_activities={**h_result.spiking_result.region_activities, **h_result.mf_rates},
            neuromodulation={
                "ecb_2ag": self._ecb.two_ag,
                "ecb_aea": self._ecb.aea_tone,
                "ecb_extinction": self._ecb.extinction_signal,
                "ach_nbm": self._ach.nbm_ach,
                "ach_cin": self._ach.nac_cin,
                "theta_coherence": theta_coh,
                "spine_density": self._structural.spine_density,
                "pnn_maturity": self._structural.pnn_maturity,
                "bdnf": self._structural.bdnf_level,
            },
            theta_coherence=theta_coh,
            memory_stats=self._sleep_engine.get_memory_stats(),
            virtual_neurons=h_result.total_virtual_neurons,
            step=self._step,
        )

    def sleep(self, n_cycles: int = 1) -> list[dict]:
        """睡眠リプレイを実行する。"""
        results = []
        for _ in range(n_cycles):
            results.append(self._sleep_engine.run_sleep_cycle())
        return results

    def set_extinction_mode(self, enabled: bool) -> None:
        """消去モードの設定。"""
        self._is_extinction_mode = enabled

    @property
    def memory_count(self) -> int:
        return len(self._sleep_engine.memories)

    @property
    def consolidated_memories(self) -> int:
        return len(self._sleep_engine.get_consolidated_memories())

    def reset(self) -> None:
        self._hybrid.reset()
        self._ecb = EndocannabinoidState()
        self._ach = AcetylcholineState()
        self._structural = StructuralPlasticityState()
        self._step = 0
