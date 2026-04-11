"""Brian2バックエンド。NeuroPipelineからBrian2スパイキング回路を呼び出す。

[監査Fix5] NeuroPipelineにBrian2バックエンドを追加し、
テキスト入力→知覚ブリッジ→Brian2スパイキング→readout_v2→応答ポリシーの
フルパスを実現する。
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from src.brian2_circuits.fear_circuit_v2 import FearCircuitV2, FearV2Config
from src.brian2_circuits.reward_circuit_v2 import RewardCircuitV2, RewardV2Config
from src.brian2_circuits.stress_circuit_v2 import StressCircuitV2, StressV2Config
from src.neurocircuit.brain import SensoryInput, EmotionReadout


class Brian2Result(BaseModel):
    """Brian2バックエンドの出力。"""

    readout: EmotionReadout
    region_activities: dict[str, float] = Field(default_factory=dict)
    neurotransmitter_levels: dict[str, float] = Field(default_factory=dict)
    circuits_activated: list[str] = Field(default_factory=list)


class Brian2Backend:
    """Brian2スパイキング回路のバックエンド。

    SensoryInputの各チャンネルに応じて恐怖/報酬/ストレス回路を起動し、
    結果を統合したEmotionReadoutを返す。
    """

    def __init__(self):
        # [SBI較正済みconfig] ABC rejection (n=50, score=0.928) 由来
        from src.calibration.calibrated_configs import (
            CALIBRATED_FEAR_CONFIG, CALIBRATED_REWARD_CONFIG, CALIBRATED_STRESS_CONFIG,
        )
        self._fear_cfg = CALIBRATED_FEAR_CONFIG
        self._reward_cfg = CALIBRATED_REWARD_CONFIG
        self._stress_cfg = CALIBRATED_STRESS_CONFIG
        self._interaction_count = 0

        # PersistentFearCircuit（真のSTDP学習用、オプション）
        self._persistent_fear = None
        self._use_persistent_fear = False

        # [問題3修正] readout_v2 (PCA) を統合
        from src.brian2_circuits.readout_v2 import SpikingReadout
        self._readout_pca = SpikingReadout(n_components=3)
        self._rate_history: list[tuple[list[float], str]] = []

        # [R5修正] homeostatic plasticity を統合
        from src.brian2_circuits.homeostatic_plasticity import HomeostaticController
        self._homeostatic = HomeostaticController(n_neurons=547)
        self._fear_circuit_ref = None  # homeostatic→fear重みフィードバック用

    def get_pca_emotion_state(self) -> dict[str, object]:
        """PCAクラスタリングによる情動状態を返す。

        Returns:
            {
                "dominant_label": str,   # 最も近いクラスタのラベル（例: "threat", "reward", "neutral"）
                "confidence": float,     # 支配的クラスタの類似度スコア (0-1)
                "all_scores": dict,      # 全クラスタの類似度スコア
                "is_fitted": bool,       # PCAが適合済みか
            }
        """
        if not self._readout_pca.is_fitted:
            return {
                "dominant_label": "unknown",
                "confidence": 0.0,
                "all_scores": {},
                "is_fitted": False,
            }

        # 最新の活動ベクトルがなければunknown
        if not self._rate_history:
            return {
                "dominant_label": "unknown",
                "confidence": 0.0,
                "all_scores": {},
                "is_fitted": True,
            }

        import numpy as np
        latest_rates = np.array(self._rate_history[-1][0])
        classification = self._readout_pca.classify(latest_rates)
        dominant_label = max(classification, key=classification.get)
        confidence = classification[dominant_label]

        return {
            "dominant_label": dominant_label,
            "confidence": confidence,
            "all_scores": classification,
            "is_fitted": True,
        }

    def process(self, sensory: SensoryInput) -> Brian2Result:
        """SensoryInputを処理し、Brian2回路の結果を統合する。"""
        self._interaction_count += 1
        activities: dict[str, float] = {}
        circuits_used = []

        fear_freeze = 0.0
        fear_anxiety = 0.0
        reward_approach = 0.0
        stress_cortisol = 0.0

        # 恐怖回路: 脅威信号 > 0.1 で起動
        if sensory.threat_signal > 0.1 or sensory.pain_input > 0.1:
            cfg = FearV2Config(
                **{**self._fear_cfg.__dict__,
                   "cs_amp": self._fear_cfg.cs_amp * max(0.5, sensory.threat_signal * 2),
                   "us_amp": self._fear_cfg.us_amp * sensory.pain_input if sensory.pain_input > 0.1 else 0,
                   "sustained_threat_amp": self._fear_cfg.sustained_threat_amp * sensory.threat_signal},
            )
            circuit = FearCircuitV2(cfg)
            self._fear_circuit_ref = circuit  # homeostatic feedbackのため参照保持
            us = sensory.pain_input > 0.3
            sustained = sensory.threat_signal > 0.3 and sensory.pain_input < 0.2
            result = circuit.run_trial(cs=True, us=us, sustained_threat=sustained, phase="process")
            fear_freeze = result.freeze_response
            fear_anxiety = result.anxiety_level
            activities.update({
                "la_exc": result.la_rate, "ba_exc": result.ba_rate,
                "cel_som": result.cel_som_rate, "cem": result.cem_rate,
                "bnst": result.bnst_rate,
            })
            circuits_used.append("fear")

        # 報酬回路: 報酬信号 > 0.1 で起動
        if sensory.reward_signal > 0.1:
            circuit = RewardCircuitV2(self._reward_cfg)
            result = circuit.run_trial(cs=True, reward=True, phase="process")
            reward_approach = result.approach_tendency
            activities.update({
                "vta_da_lat": result.vta_da_lat_rate,
                "nac_shell_d1": result.nac_shell_d1_rate,
            })
            circuits_used.append("reward")

        # ストレス回路: 脅威+疼痛が高い場合
        if sensory.threat_signal > 0.3 or sensory.pain_input > 0.2:
            circuit = StressCircuitV2(self._stress_cfg)
            intensity = max(sensory.threat_signal, sensory.pain_input)
            result = circuit.run_acute(n=1, intensity=intensity)[0]
            stress_cortisol = result.cortisol
            activities.update({
                "pvn": result.pvn_rate, "lc": result.lc_rate,
            })
            circuits_used.append("stress")

        # [H6修正] 回路間相互影響
        # 恐怖→ストレス: CeA活性がHPA軸を駆動
        if fear_freeze > 0.2 and "stress" not in circuits_used:
            stress_cortisol = fear_freeze * 0.3  # CeA→PVN経路の簡易近似

        # 報酬→恐怖: DA高値が扁桃体を抑制（安心信号）
        if reward_approach > 0.3 and fear_freeze > 0:
            fear_freeze *= (1.0 - reward_approach * 0.3)  # DA→BLA抑制

        # ストレス→報酬: コルチゾール高値がDA系を抑制
        if stress_cortisol > 0.3 and reward_approach > 0:
            reward_approach *= (1.0 - stress_cortisol * 0.2)

        # [問題3修正] readout_v2 (PCA) 学習データ蓄積
        import numpy as np
        rate_vector = np.array([
            activities.get("la_exc", 0), activities.get("ba_exc", 0),
            activities.get("cel_som", 0), activities.get("cem", 0),
            activities.get("bnst", 0), activities.get("vta_da_lat", 0),
            activities.get("nac_shell_d1", 0), activities.get("pvn", 0),
        ])

        # PCA学習データを蓄積
        label = "threat" if sensory.threat_signal > 0.3 else ("reward" if sensory.reward_signal > 0.3 else "neutral")
        self._rate_history.append((rate_vector.tolist(), label))

        # 10サンプル以上溜まったらPCAを適合（未適合時のみ）
        if len(self._rate_history) >= 10 and not self._readout_pca.is_fitted:
            from src.brian2_circuits.readout_v2 import ReadoutTrainingData
            rates_matrix = np.array([r for r, _ in self._rate_history])
            labels = [l for _, l in self._rate_history]
            data = ReadoutTrainingData(
                rates_matrix=rates_matrix, labels=labels,
                population_names=["la", "ba", "cs", "cm", "bn", "vt", "na", "pv"],
            )
            self._readout_pca.fit(data)

        # 蓄積が増えたらPCAを定期的に再適合（50サンプルごと）
        if self._readout_pca.is_fitted and len(self._rate_history) % 50 == 0:
            from src.brian2_circuits.readout_v2 import ReadoutTrainingData
            rates_matrix = np.array([r for r, _ in self._rate_history])
            labels = [l for _, l in self._rate_history]
            data = ReadoutTrainingData(
                rates_matrix=rates_matrix, labels=labels,
                population_names=["la", "ba", "cs", "cm", "bn", "vt", "na", "pv"],
            )
            self._readout_pca.fit(data)

        # --- readout統合: PCA適合後はPCAが主（70%）、手動が従（30%） ---
        # 手動線形結合（フォールバック / 従readout）
        linear_valence = reward_approach * 0.5 - fear_freeze * 0.3 - stress_cortisol * 0.2
        linear_arousal = max(fear_freeze, reward_approach, stress_cortisol * 2) * 0.7 + 0.2
        linear_threat = fear_freeze * 0.6 + fear_anxiety * 0.3 + stress_cortisol * 0.1

        if self._readout_pca.is_fitted:
            # PCAが適合済み → PCA主（70%）+ 手動従（30%）
            pca_result = self._readout_pca.to_emotion_readout(rate_vector)
            pca_valence = pca_result.get("valence", 0.0)
            pca_arousal = pca_result.get("arousal", 0.5)

            valence = pca_valence * 0.7 + linear_valence * 0.3
            arousal = pca_arousal * 0.7 + linear_arousal * 0.3
            threat = linear_threat  # 脅威はPCAクラスタでは捉えにくいため手動維持
        else:
            # PCA未適合 → 手動線形結合100%（後方互換フォールバック）
            valence = linear_valence
            arousal = linear_arousal
            threat = linear_threat

        readout = EmotionReadout(
            valence=max(-1, min(1, valence)),
            arousal=max(0, min(1, arousal)),
            threat_load=max(0, min(1, threat)),
            reward_drive=max(0, min(1, reward_approach)),
            social_warmth=max(0, min(1, sensory.social_signal * 0.5)),
            cognitive_control=0.5,
            body_distress=max(0, min(1, stress_cortisol)),
            energy=max(0, min(1, 1.0 - stress_cortisol * 0.5)),
            memory_encoding_boost=max(0, min(1, fear_freeze * 0.5)),
        )

        # [R6修正] homeostatic plasticity: 発火率追跡+スケーリング→次試行の重みに反映
        total_spikes = sum(v for v in activities.values() if isinstance(v, (int, float)))
        self._homeostatic.update_rates(int(total_spikes), 200.0)
        self._homeostatic.update_bcm(dt_ms=200.0)

        # スケーリング係数を恐怖回路の試行間重みに反映
        scaling = float(self._homeostatic.get_scaling_factors().mean())
        if hasattr(self, '_fear_circuit_ref') and self._fear_circuit_ref is not None:
            for key, weights in self._fear_circuit_ref._saved_weights.items():
                self._fear_circuit_ref._saved_weights[key] = np.clip(weights * scaling, 0, 15)

        return Brian2Result(
            readout=readout,
            region_activities=activities,
            circuits_activated=circuits_used,
        )
