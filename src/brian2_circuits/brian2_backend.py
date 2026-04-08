"""Brian2バックエンド。NeuroPipelineからBrian2スパイキング回路を呼び出す。

[監査Fix5] NeuroPipelineにBrian2バックエンドを追加し、
テキスト入力→知覚ブリッジ→Brian2スパイキング→readout_v2→応答ポリシーの
フルパスを実現する。
"""

from __future__ import annotations

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
        self._fear_cfg = FearV2Config(duration_ms=200, cs_dur_ms=100, us_onset_ms=130, us_dur_ms=20)
        self._reward_cfg = RewardV2Config(duration_ms=200, cs_dur_ms=50, reward_dur_ms=25)
        self._stress_cfg = StressV2Config(duration_ms=200)
        self._interaction_count = 0

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

        # readout統合
        valence = (reward_approach * 0.5 - fear_freeze * 0.3 - stress_cortisol * 0.2)
        arousal = max(fear_freeze, reward_approach, stress_cortisol * 2) * 0.7 + 0.2
        threat = fear_freeze * 0.6 + fear_anxiety * 0.3 + stress_cortisol * 0.1

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

        return Brian2Result(
            readout=readout,
            region_activities=activities,
            circuits_activated=circuits_used,
        )
