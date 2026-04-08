"""神経回路統合パイプライン。テキスト→知覚ブリッジ→脳モデル→readout→応答ポリシー→安全。

既存のEmotionPipelineの抽象レイヤーを、神経回路ダイナミクスで置き換える。
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from src.audit.logger import AuditLogger
from src.neurocircuit.brain import BrainState, EmotionReadout, SensoryInput, compute_readout, step_brain
from src.neurocircuit.perception_bridge import text_to_sensory
from src.policy.response import ResponsePolicy, derive_policy
from src.safety.guardian import SafetyReport, full_safety_check
from src.schemas.affect_state import AffectState


class NeuroResult(BaseModel):
    """神経回路パイプライン1ステップの出力。"""

    event_id: str = ""
    input_text: str = ""
    sensory_input: SensoryInput
    readout: EmotionReadout
    response_policy: ResponsePolicy
    safety_report: SafetyReport
    brain_step: int
    region_activities: dict[str, float]
    neurotransmitter_levels: dict[str, float]
    body_state: dict[str, float]
    blocked: bool = False


def _readout_to_affect_state(readout: EmotionReadout, step: int) -> AffectState:
    """EmotionReadoutを既存のAffectStateに変換する（安全チェック/ポリシー互換）。"""
    return AffectState(
        valence=readout.valence,
        arousal=readout.arousal,
        motivational_salience=readout.reward_drive,
        perceived_control=readout.cognitive_control,
        uncertainty=max(0.0, 1.0 - readout.cognitive_control),
        trust=readout.social_warmth,
        threat_load=readout.threat_load,
        fatigue=max(0.0, 1.0 - readout.energy),
        step_count=step,
    )


class NeuroPipeline:
    """神経回路ベースの感情処理パイプライン。

    [監査Fix5] backend="brian2" でBrian2スパイキング回路を使用可能。
    """

    def __init__(self, simulation_steps: int = 50, dt: float = 0.02,
                 backend: str = "wilson_cowan"):
        self._backend_name = backend
        self._brain = BrainState()
        self._sim_steps = simulation_steps
        self._dt = dt
        self._audit = AuditLogger("logs/neuro_audit.jsonl")
        self._interaction_count = 0

        # Brian2/ハイブリッドバックエンド（オプション）
        self._brian2_backend = None
        self._hybrid_brain = None
        if backend == "brian2":
            from src.brian2_circuits.brian2_backend import Brian2Backend
            self._brian2_backend = Brian2Backend()
        elif backend == "hybrid":
            from src.brian2_circuits.hybrid_brain import HybridBrain
            self._hybrid_brain = HybridBrain()

    @property
    def brain(self) -> BrainState:
        return self._brain

    @property
    def readout(self) -> EmotionReadout:
        return compute_readout(self._brain)

    @property
    def affect_state(self) -> AffectState:
        return _readout_to_affect_state(self.readout, self._brain.step)

    def process_text(self, text: str) -> NeuroResult:
        """テキストを処理する。知覚→脳シミュレーション→readout→ポリシー→安全。"""
        self._interaction_count += 1

        # 1. 安全チェック（状態変更前）
        current_affect = self.affect_state
        safety = full_safety_check(
            event_id=f"neuro-{self._brain.step}",
            state=current_affect,
            response_text=text,
            interaction_count=self._interaction_count,
        )
        if safety.blocked:
            return NeuroResult(
                event_id=f"neuro-{self._brain.step}",
                input_text=text,
                sensory_input=SensoryInput(),
                readout=self.readout,
                response_policy=derive_policy(current_affect),
                safety_report=safety,
                brain_step=self._brain.step,
                region_activities=self._get_region_activities(),
                neurotransmitter_levels=self._brain.neurotransmitters.effective_levels(),
                body_state=self._get_body_state(),
                blocked=True,
            )

        # 2. テキスト→SensoryInput変換
        sensory = text_to_sensory(text)

        # 3. 脳シミュレーション（バックエンド分岐）
        if self._brian2_backend is not None:
            # [Fix5] Brian2スパイキング回路バックエンド
            b2_result = self._brian2_backend.process(sensory)
            readout = b2_result.readout
            affect = _readout_to_affect_state(readout, self._interaction_count)
            policy = derive_policy(affect)
            return NeuroResult(
                event_id=f"neuro-b2-{self._interaction_count}",
                input_text=text,
                sensory_input=sensory,
                readout=readout,
                response_policy=policy,
                safety_report=safety,
                brain_step=self._interaction_count,
                region_activities=b2_result.region_activities,
                neurotransmitter_levels=b2_result.neurotransmitter_levels,
                body_state={},
            )

        # [Phase3] ハイブリッドバックエンド
        if self._hybrid_brain is not None:
            h_result = self._hybrid_brain.process(sensory)
            readout = h_result.readout
            affect = _readout_to_affect_state(readout, self._interaction_count)
            policy = derive_policy(affect)
            return NeuroResult(
                event_id=f"neuro-hybrid-{self._interaction_count}",
                input_text=text,
                sensory_input=sensory,
                readout=readout,
                response_policy=policy,
                safety_report=safety,
                brain_step=self._interaction_count,
                region_activities={**h_result.spiking_result.region_activities, **h_result.mf_rates},
                neurotransmitter_levels=h_result.spiking_result.neurotransmitter_levels,
                body_state={"total_virtual_neurons": h_result.total_virtual_neurons},
            )

        # Wilson-Cowan バックエンド（デフォルト）
        for _ in range(self._sim_steps):
            self._brain = step_brain(self._brain, sensory, self._dt)

        # 4. Readout計算
        readout = compute_readout(self._brain)
        affect = _readout_to_affect_state(readout, self._brain.step)

        # 5. 応答ポリシー
        policy = derive_policy(affect)

        return NeuroResult(
            event_id=f"neuro-{self._brain.step}",
            input_text=text,
            sensory_input=sensory,
            readout=readout,
            response_policy=policy,
            safety_report=safety,
            brain_step=self._brain.step,
            region_activities=self._get_region_activities(),
            neurotransmitter_levels=self._brain.neurotransmitters.effective_levels(),
            body_state=self._get_body_state(),
        )

    def process_sensory(self, sensory: SensoryInput) -> NeuroResult:
        """SensoryInputを直接処理する。"""
        for _ in range(self._sim_steps):
            self._brain = step_brain(self._brain, sensory, self._dt)

        readout = compute_readout(self._brain)
        affect = _readout_to_affect_state(readout, self._brain.step)

        return NeuroResult(
            sensory_input=sensory,
            readout=readout,
            response_policy=derive_policy(affect),
            safety_report=full_safety_check(f"neuro-{self._brain.step}", affect),
            brain_step=self._brain.step,
            region_activities=self._get_region_activities(),
            neurotransmitter_levels=self._brain.neurotransmitters.effective_levels(),
            body_state=self._get_body_state(),
        )

    def tick(self, steps: int = 10) -> None:
        """時間経過（入力なし）。"""
        empty = SensoryInput()
        for _ in range(steps):
            self._brain = step_brain(self._brain, empty, self._dt)

    def reset(self) -> None:
        self._brain = BrainState()
        self._interaction_count = 0

    def _get_region_activities(self) -> dict[str, float]:
        b = self._brain
        return {
            "amygdala": b.amygdala.output,
            "vmPFC": b.pfc.vmPFC.output,
            "dlPFC": b.pfc.dlPFC.output,
            "OFC": b.pfc.OFC.output,
            "insula": b.insula.output,
            "acc": b.acc.output,
            "hippocampus": b.hippocampus.output,
            "hypothalamus": b.hypothalamus.output,
            "pag": b.pag.output,
            "ventral_striatum": b.ventral_striatum.output,
            "locus_coeruleus": b.brainstem.locus_coeruleus.output,
            "raphe_nuclei": b.brainstem.raphe_nuclei.output,
            "vta": b.brainstem.vta.output,
        }

    def _get_body_state(self) -> dict[str, float]:
        body = self._brain.body
        return {
            "cortisol": body.hpa.cortisol,
            "crh": body.hpa.crh,
            "sympathetic": body.autonomic.sympathetic,
            "parasympathetic": body.autonomic.parasympathetic,
            "heart_rate": body.interoception.heart_rate_signal,
            "energy": body.interoception.energy_level,
            "muscle_tension": body.interoception.muscle_tension,
            "pain": body.interoception.pain_signal,
        }
