"""介入シミュレーション。病変研究・薬理操作を再現し、回路機能を検証する。

WBE進捗再評価(2025)の評価フレームワークに基づく:
  回路機能レベルの検証 = 介入(病変/薬理/パラメータ操作)に対する予測が合うか

実装する介入タイプ:
  1. 病変 (lesion): 特定領域の出力を0にする
  2. 薬理操作 (pharmacological): 特定伝達物質のレベルを操作
  3. 刺激 (stimulation): 特定領域に直接入力を加える
  4. 結合操作 (connection): 特定結合の重みを変更
"""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.neurocircuit.brain import BrainState, SensoryInput, step_brain, compute_readout, EmotionReadout


class InterventionType(str, Enum):
    LESION = "lesion"
    PHARMACOLOGICAL = "pharmacological"
    STIMULATION = "stimulation"
    CONNECTION = "connection"


class Intervention(BaseModel):
    """1つの介入操作。"""

    intervention_type: InterventionType
    target: str  # 領域名 or 伝達物質名
    parameter: str = ""  # 操作対象パラメータ
    value: float = 0.0  # 設定値 or 変化量
    description: str = ""


class InterventionResult(BaseModel):
    """介入実験の結果。"""

    intervention: Intervention
    baseline_readout: EmotionReadout
    post_intervention_readout: EmotionReadout
    prediction: str = ""
    actual_match: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


def apply_lesion(brain: BrainState, region_name: str) -> BrainState:
    """病変: 指定領域の出力を0にクランプする。

    神経科学の病変研究を再現:
    - 扁桃体病変 → 恐怖反応の消失
    - vmPFC病変 → 情動制御の障害
    - 海馬病変 → 文脈依存記憶の障害
    """
    modified = brain.model_copy(deep=True)

    region_map = {
        "amygdala": modified.amygdala,
        "vmPFC": modified.pfc.vmPFC,
        "dlPFC": modified.pfc.dlPFC,
        "OFC": modified.pfc.OFC,
        "insula": modified.insula,
        "acc": modified.acc,
        "hippocampus": modified.hippocampus,
        "hypothalamus": modified.hypothalamus,
        "pag": modified.pag,
        "ventral_striatum": modified.ventral_striatum,
        "locus_coeruleus": modified.brainstem.locus_coeruleus,
        "raphe_nuclei": modified.brainstem.raphe_nuclei,
        "vta": modified.brainstem.vta,
    }

    if region_name in region_map:
        region = region_map[region_name]
        region.excitatory = 0.0
        region.inhibitory = 0.0
        region.output = 0.0

    return modified


def apply_pharmacological(brain: BrainState, neurotransmitter: str, tonic_change: float = 0.0, block_reuptake: bool = False) -> BrainState:
    """薬理操作: 特定伝達物質のレベルを操作する。

    臨床・実験で使われる介入を再現:
    - SSRI → セロトニン再取り込み阻害 → tonic↑
    - ベンゾジアゼピン → GABA作動 → tonic↑
    - ドーパミン拮抗薬 → DA tonic↓
    - β遮断薬 → NE tonic↓
    """
    modified = brain.model_copy(deep=True)
    nt = modified.neurotransmitters

    if not hasattr(nt, neurotransmitter):
        return modified

    target = getattr(nt, neurotransmitter)
    target.tonic = max(0.0, min(1.0, target.tonic + tonic_change))

    if block_reuptake:
        target.reuptake_rate *= 0.3  # 再取り込み阻害

    return modified


def apply_stimulation(brain: BrainState, region_name: str, intensity: float) -> BrainState:
    """直接刺激: 特定領域に外部入力を加える。

    DBS(深部脳刺激)やTMSを模倣。
    """
    modified = brain.model_copy(deep=True)

    region_map = {
        "amygdala": modified.amygdala,
        "vmPFC": modified.pfc.vmPFC,
        "dlPFC": modified.pfc.dlPFC,
        "acc": modified.acc,
        "ventral_striatum": modified.ventral_striatum,
    }

    if region_name in region_map:
        region = region_map[region_name]
        region.excitatory = max(0.0, min(1.0, region.excitatory + intensity))
        region.output = max(0.0, min(1.0, region.output + intensity * 0.8))

    return modified


def run_lesion_experiment(
    region_name: str,
    sensory: SensoryInput,
    steps: int = 80,
    prediction: str = "",
) -> InterventionResult:
    """病変実験を実行する。

    1. ベースライン（正常脳）で刺激を処理
    2. 病変後に同じ刺激を処理
    3. readoutの差分を計測
    """
    # ベースライン
    baseline_brain = BrainState()
    for _ in range(steps):
        baseline_brain = step_brain(baseline_brain, sensory, dt=0.02)
    baseline_readout = compute_readout(baseline_brain)

    # 病変後
    lesioned_brain = BrainState()
    lesioned_brain = apply_lesion(lesioned_brain, region_name)
    for _ in range(steps):
        lesioned_brain = step_brain(lesioned_brain, sensory, dt=0.02)
        lesioned_brain = apply_lesion(lesioned_brain, region_name)  # 毎ステップ病変を維持
    lesioned_readout = compute_readout(lesioned_brain)

    return InterventionResult(
        intervention=Intervention(
            intervention_type=InterventionType.LESION,
            target=region_name,
            description=f"{region_name}の完全病変",
        ),
        baseline_readout=baseline_readout,
        post_intervention_readout=lesioned_readout,
        prediction=prediction,
        details={
            "delta_threat": lesioned_readout.threat_load - baseline_readout.threat_load,
            "delta_valence": lesioned_readout.valence - baseline_readout.valence,
            "delta_arousal": lesioned_readout.arousal - baseline_readout.arousal,
            "delta_reward": lesioned_readout.reward_drive - baseline_readout.reward_drive,
            "delta_social": lesioned_readout.social_warmth - baseline_readout.social_warmth,
        },
    )


def run_pharmacological_experiment(
    neurotransmitter: str,
    tonic_change: float,
    sensory: SensoryInput,
    steps: int = 80,
    prediction: str = "",
    block_reuptake: bool = False,
) -> InterventionResult:
    """薬理実験を実行する。"""
    # ベースライン
    baseline_brain = BrainState()
    for _ in range(steps):
        baseline_brain = step_brain(baseline_brain, sensory, dt=0.02)
    baseline_readout = compute_readout(baseline_brain)

    # 薬理操作後
    drug_brain = BrainState()
    drug_brain = apply_pharmacological(drug_brain, neurotransmitter, tonic_change, block_reuptake)
    for _ in range(steps):
        drug_brain = step_brain(drug_brain, sensory, dt=0.02)
        # 薬理効果を持続させる
        nt = getattr(drug_brain.neurotransmitters, neurotransmitter, None)
        if nt and tonic_change != 0:
            nt.tonic = max(0.0, min(1.0, nt.tonic + tonic_change * 0.01))
    drug_readout = compute_readout(drug_brain)

    return InterventionResult(
        intervention=Intervention(
            intervention_type=InterventionType.PHARMACOLOGICAL,
            target=neurotransmitter,
            value=tonic_change,
            description=f"{neurotransmitter} tonic {tonic_change:+.2f}" + (" + 再取り込み阻害" if block_reuptake else ""),
        ),
        baseline_readout=baseline_readout,
        post_intervention_readout=drug_readout,
        prediction=prediction,
    )


# === 標準的な介入実験セット（神経科学の知見に基づく予測付き） ===

STANDARD_LESION_EXPERIMENTS = [
    {
        "region": "amygdala",
        "sensory": SensoryInput(threat_signal=0.8),
        "prediction": "扁桃体病変 → 脅威への反応が低下（恐怖条件付けの障害）",
        "expected_direction": {"threat_load": "decrease"},
    },
    {
        "region": "vmPFC",
        "sensory": SensoryInput(threat_signal=0.6),
        "prediction": "vmPFC病変 → 扁桃体のトップダウン抑制が失われ、脅威反応が増大",
        "expected_direction": {"threat_load": "increase_or_same"},
    },
    {
        "region": "vta",
        "sensory": SensoryInput(reward_signal=0.8),
        "prediction": "VTA病変 → ドーパミン産生低下 → 報酬反応の減弱",
        "expected_direction": {"reward_drive": "decrease"},
    },
    {
        "region": "hippocampus",
        "sensory": SensoryInput(threat_signal=0.5, context_input=0.7),
        "prediction": "海馬病変 → 文脈依存の情動記憶が障害される",
        "expected_direction": {"memory_encoding_boost": "decrease"},
    },
]

STANDARD_PHARMA_EXPERIMENTS = [
    {
        "neurotransmitter": "serotonin",
        "tonic_change": 0.3,
        "block_reuptake": True,
        "sensory": SensoryInput(threat_signal=0.5),
        "prediction": "SSRI様（5-HT↑） → 脅威反応と不安の軽減",
        "expected_direction": {"threat_load": "decrease"},
    },
    {
        "neurotransmitter": "gaba",
        "tonic_change": 0.3,
        "sensory": SensoryInput(threat_signal=0.6),
        "prediction": "ベンゾジアゼピン様（GABA↑） → 扁桃体抑制 → 不安軽減",
        "expected_direction": {"threat_load": "decrease"},
    },
    {
        "neurotransmitter": "norepinephrine",
        "tonic_change": -0.2,
        "sensory": SensoryInput(threat_signal=0.6),
        "prediction": "β遮断薬様（NE↓） → 覚醒・脅威反応の低下",
        "expected_direction": {"arousal": "decrease"},
    },
]
