"""ハイブリッド脳モデル。スパイキング(精密) + mean-field(高速) の統合。

アーキテクチャ:
  [AdEx mean-field 層 — 背景領域]
    島皮質 ── ACC ── dlPFC ── 海馬
        ↕           ↕           ↕
  [Brian2 スパイキング層 — 中核情動回路]
    扁桃体(BLA/CeA) ── vmPFC/IL/PL ── VTA/NAc ── PVN/BNST
    (恐怖回路)        (制御/消去)   (報酬回路)   (ストレス回路)

インターフェース:
  mean-field → spiking: 発火率 → 外部電流（TimedArray経由）
  spiking → mean-field: スパイキング発火率 → 外部入力
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.brian2_circuits.adex_meanfield import AdExMFParams, MeanFieldRegion
from src.brian2_circuits.brian2_backend import Brian2Backend, Brian2Result
from src.neurocircuit.brain import SensoryInput, EmotionReadout


@dataclass
class HybridConfig:
    """ハイブリッド脳の設定。"""

    # mean-field背景領域
    mf_regions: list[str] = field(default_factory=lambda: [
        "insula", "acc", "dlpfc", "hippocampus",
    ])

    # 時間ステップ
    mf_dt: float = 1.0  # mean-fieldのdt (ms)
    mf_steps_per_trial: int = 200  # mean-fieldの更新ステップ数

    # mean-field → スパイキング 変換ゲイン
    mf_to_spike_gain: float = 0.3
    # スパイキング → mean-field 変換ゲイン
    spike_to_mf_gain: float = 0.1


@dataclass
class HybridResult:
    """ハイブリッドモデルの出力。"""

    # スパイキング層
    spiking_result: Brian2Result
    # mean-field層
    mf_rates: dict[str, float]
    # 統合readout
    readout: EmotionReadout
    # 統合情報
    total_virtual_neurons: int = 0


class HybridBrain:
    """スパイキング + mean-field のハイブリッド脳モデル。"""

    def __init__(self, config: HybridConfig | None = None):
        self.cfg = config or HybridConfig()
        self._spiking = Brian2Backend()

        # mean-field背景領域を構築
        self._mf_regions: dict[str, MeanFieldRegion] = {}
        mf_params = {
            "insula": AdExMFParams(baseline_drive=3.5),      # 内受容で高めの基底活動
            "acc": AdExMFParams(baseline_drive=3.0),          # コンフリクト監視
            "dlpfc": AdExMFParams(baseline_drive=4.0),        # 認知制御（高基底活動）
            "hippocampus": AdExMFParams(baseline_drive=2.5),  # 文脈記憶
        }
        for name in self.cfg.mf_regions:
            params = mf_params.get(name, AdExMFParams())
            self._mf_regions[name] = MeanFieldRegion(name, params)

        self._interaction_count = 0

    def process(self, sensory: SensoryInput) -> HybridResult:
        """SensoryInputを処理する。

        1. mean-field背景領域を更新
        2. mean-field出力をスパイキング入力に変換
        3. スパイキング回路を実行
        4. スパイキング出力をmean-fieldにフィードバック
        5. 統合readoutを計算
        """
        self._interaction_count += 1

        # === 1. mean-field背景領域の更新 ===
        # 感覚入力のマッピング
        mf_inputs = {
            "insula": sensory.pain_input * 3.0 + sensory.threat_signal * 1.0,
            "acc": sensory.novelty_signal * 2.0 + sensory.threat_signal * 1.5,
            "dlpfc": sensory.context_input * 2.0,
            "hippocampus": sensory.context_input * 3.0 + sensory.novelty_signal * 1.0,
        }

        for _ in range(self.cfg.mf_steps_per_trial):
            for name, region in self._mf_regions.items():
                ext = mf_inputs.get(name, 0.0)
                region.step(ext_exc=ext, dt=self.cfg.mf_dt)

        # === 2. スパイキング回路を実行 ===
        # mean-fieldからの入力をSensoryInputに加算
        _c = lambda v: float(max(0.0, min(1.0, v)))
        enhanced_sensory = SensoryInput(
            threat_signal=_c(sensory.threat_signal + self._mf_regions["insula"].output * self.cfg.mf_to_spike_gain * 0.1),
            reward_signal=_c(sensory.reward_signal),
            social_signal=_c(sensory.social_signal),
            novelty_signal=_c(sensory.novelty_signal + self._mf_regions["acc"].output * self.cfg.mf_to_spike_gain * 0.05),
            pain_input=_c(sensory.pain_input),
            context_input=_c(sensory.context_input + self._mf_regions["hippocampus"].output * self.cfg.mf_to_spike_gain * 0.1),
        )

        spike_result = self._spiking.process(enhanced_sensory)

        # === 3. スパイキング→mean-fieldフィードバック ===
        # 扁桃体活性がACC/島皮質に影響
        amygdala_rate = spike_result.region_activities.get("la_exc", 0) + spike_result.region_activities.get("cem", 0)
        if "insula" in self._mf_regions:
            self._mf_regions["insula"].step(ext_exc=amygdala_rate * self.cfg.spike_to_mf_gain)
        if "acc" in self._mf_regions:
            self._mf_regions["acc"].step(ext_exc=amygdala_rate * self.cfg.spike_to_mf_gain * 0.5)

        # === 4. 統合readout ===
        mf_rates = {name: region.output for name, region in self._mf_regions.items()}

        # mean-field貢献をreadoutに統合
        base_readout = spike_result.readout
        dlpfc_control = self._mf_regions.get("dlpfc", MeanFieldRegion("x")).output
        hippo_context = self._mf_regions.get("hippocampus", MeanFieldRegion("x")).output
        insula_intero = self._mf_regions.get("insula", MeanFieldRegion("x")).output

        integrated_readout = EmotionReadout(
            valence=base_readout.valence,
            arousal=base_readout.arousal,
            threat_load=base_readout.threat_load,
            reward_drive=base_readout.reward_drive,
            social_warmth=base_readout.social_warmth,
            cognitive_control=min(1.0, base_readout.cognitive_control + dlpfc_control * 0.01),
            body_distress=min(1.0, base_readout.body_distress + insula_intero * 0.005),
            energy=base_readout.energy,
            memory_encoding_boost=min(1.0, base_readout.memory_encoding_boost + hippo_context * 0.005),
        )

        # ニューロン数の計算
        # [監査C3修正] 正直な表記: スパイキング(実際に個別発火) + mean-field(集団近似)
        spiking_neurons = sum(
            v for k, v in {
                "fear": 217, "reward": 190, "stress": 140,
            }.items() if k in spike_result.circuits_activated
        )
        mf_regions_count = len(self._mf_regions)
        # mean-fieldは「仮想ニューロン数」ではなく「近似対象の集団サイズ」
        # 実態は1領域あたり3状態変数(rate_exc, rate_inh, adaptation)

        return HybridResult(
            spiking_result=spike_result,
            mf_rates=mf_rates,
            readout=integrated_readout,
            total_virtual_neurons=spiking_neurons,  # スパイキングのみカウント
        )

    @property
    def mf_states(self) -> dict[str, float]:
        return {name: region.output for name, region in self._mf_regions.items()}

    def reset(self) -> None:
        for region in self._mf_regions.values():
            region.reset()
        self._interaction_count = 0
