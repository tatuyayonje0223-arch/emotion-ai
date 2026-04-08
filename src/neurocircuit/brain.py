"""統合脳モデル。全領域 + 神経伝達物質 + ホメオスタシスを結合して時間発展させる。

1ステップ:
  1. 外部入力（感覚信号）を受け取る
  2. 各脳幹核の更新 → 神経伝達物質レベルの更新
  3. 各領域のWilson-Cowanダイナミクスを解剖学的結合に基づいて更新
  4. HPA軸・自律神経系・内受容の更新
  5. 神経伝達物質の再取り込み
  6. 可塑性の適用
  7. 観測量（valence, arousal等）のreadout
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from src.neurocircuit.connectivity import ANATOMICAL_CONNECTIONS, get_connections_to
from src.neurocircuit.homeostasis import (
    BodyState, update_autonomic, update_hpa_axis, update_interoception,
)
from src.neurocircuit.neurotransmitters import NeurotransmitterSystem, apply_reuptake
from src.neurocircuit.plasticity import PlasticityParams, emotional_memory_tag
from src.neurocircuit.regions import (
    AmygdalaState, ACCState, BrainstemState, HippocampusState,
    HypothalamusState, InsulaState, PAGState, PFCState,
    RegionState, VentralStriatumState, update_region,
)


class SensoryInput(BaseModel):
    """外部感覚入力。"""

    threat_signal: float = Field(0.0, ge=0.0, le=1.0, description="脅威的入力")
    reward_signal: float = Field(0.0, ge=0.0, le=1.0, description="報酬的入力")
    social_signal: float = Field(0.0, ge=0.0, le=1.0, description="社会的入力")
    novelty_signal: float = Field(0.0, ge=0.0, le=1.0, description="新規性入力")
    pain_input: float = Field(0.0, ge=0.0, le=1.0, description="疼痛入力")
    context_input: float = Field(0.0, ge=0.0, le=1.0, description="文脈入力")


class EmotionReadout(BaseModel):
    """神経回路ダイナミクスから導出される観測量。

    旧来の抽象変数は、回路活動の関数として再定義される。
    """

    valence: float = Field(0.0, description="快不快 ≈ DA(NAc) + END - CORT - 扁桃体脅威")
    arousal: float = Field(0.0, description="覚醒 ≈ NE(LC) + Glu + 交感神経")
    threat_load: float = Field(0.0, description="脅威 ≈ 扁桃体出力 + CORT + NE")
    reward_drive: float = Field(0.0, description="報酬動機 ≈ DA(NAc) + NAc活性")
    social_warmth: float = Field(0.0, description="社会的温かさ ≈ OXT + PFC-扁桃体抑制")
    cognitive_control: float = Field(0.0, description="認知制御 ≈ dlPFC活性 + 5-HT")
    body_distress: float = Field(0.0, description="身体的苦痛 ≈ 内受容苦痛指標")
    energy: float = Field(0.0, description="エネルギー ≈ 内受容エネルギー")
    memory_encoding_boost: float = Field(0.0, description="記憶強化 ≈ 情動タグ強度")


class BrainState(BaseModel):
    """脳全体の統合状態。"""

    # 脳領域
    amygdala: AmygdalaState = Field(default_factory=AmygdalaState)
    pfc: PFCState = Field(default_factory=PFCState)
    insula: InsulaState = Field(default_factory=InsulaState)
    acc: ACCState = Field(default_factory=ACCState)
    hippocampus: HippocampusState = Field(default_factory=HippocampusState)
    hypothalamus: HypothalamusState = Field(default_factory=HypothalamusState)
    pag: PAGState = Field(default_factory=PAGState)
    ventral_striatum: VentralStriatumState = Field(default_factory=VentralStriatumState)
    brainstem: BrainstemState = Field(default_factory=BrainstemState)

    # 神経伝達物質
    neurotransmitters: NeurotransmitterSystem = Field(default_factory=NeurotransmitterSystem)

    # 身体
    body: BodyState = Field(default_factory=BodyState)

    # 可塑性
    plasticity: PlasticityParams = Field(default_factory=PlasticityParams)

    # メタ
    step: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _get_region_output(brain: BrainState, name: str) -> float:
    """名前から領域出力を取得する。"""
    mapping = {
        "amygdala": brain.amygdala.output,
        "vmPFC": brain.pfc.vmPFC.output,
        "dlPFC": brain.pfc.dlPFC.output,
        "OFC": brain.pfc.OFC.output,
        "insula": brain.insula.output,
        "acc": brain.acc.output,
        "hippocampus": brain.hippocampus.output,
        "hypothalamus": brain.hypothalamus.output,
        "pag": brain.pag.output,
        "ventral_striatum": brain.ventral_striatum.output,
        "locus_coeruleus": brain.brainstem.locus_coeruleus.output,
        "raphe_nuclei": brain.brainstem.raphe_nuclei.output,
        "vta": brain.brainstem.vta.output,
    }
    return mapping.get(name, 0.0)


def _compute_region_input(brain: BrainState, target: str) -> tuple[float, float]:
    """解剖学的結合から興奮性/抑制性入力を集計する。"""
    conns = get_connections_to(target)
    exc_total = 0.0
    inh_total = 0.0
    for c in conns:
        src_output = _get_region_output(brain, c.source)
        if c.conn_type == "excitatory":
            exc_total += src_output * c.weight
        else:
            inh_total += src_output * c.weight
    return exc_total, inh_total


def step_brain(brain: BrainState, sensory: SensoryInput, dt: float = 0.01) -> BrainState:
    """脳全体を1ステップ更新する。"""
    new = brain.model_copy(deep=True)
    nt = new.neurotransmitters

    # === 1. 脳幹核の更新（神経伝達物質の産生） ===
    # 青斑核(LC): 脅威+扁桃体で活性化 → NE産生
    lc_exc = sensory.threat_signal * 0.5 + brain.amygdala.output * 0.4
    new.brainstem.locus_coeruleus = update_region(
        brain.brainstem.locus_coeruleus, external_excitatory=lc_exc, dt=dt,
    )
    nt.norepinephrine.phasic += new.brainstem.locus_coeruleus.output * 0.15 * dt

    # 縫線核(Raphe): 基底活性 + PFC入力 → 5-HT産生
    raphe_exc = brain.pfc.vmPFC.output * 0.3
    new.brainstem.raphe_nuclei = update_region(
        brain.brainstem.raphe_nuclei, external_excitatory=raphe_exc, dt=dt,
    )
    nt.serotonin.phasic += new.brainstem.raphe_nuclei.output * 0.1 * dt

    # VTA: 報酬信号 → DA産生
    vta_exc = sensory.reward_signal * 0.6 + brain.ventral_striatum.reward_prediction_error * 0.3
    new.brainstem.vta = update_region(
        brain.brainstem.vta, external_excitatory=max(0, vta_exc), dt=dt,
    )
    nt.dopamine.phasic += new.brainstem.vta.output * 0.2 * dt

    # === 2. 主要領域の更新 ===

    # 扁桃体: 脅威入力 + 解剖学的入力、NE利得変調、GABA/5-HT抑制
    amy_exc, amy_inh = _compute_region_input(brain, "amygdala")
    amy_exc += sensory.threat_signal * 0.6 + sensory.novelty_signal * 0.2
    ne_gain = 1.0 + nt.norepinephrine.effective_level * brain.amygdala.ne_sensitivity * 0.5
    gaba_inh = nt.gaba.effective_level * brain.amygdala.gaba_sensitivity * 0.3
    sht_inh = nt.serotonin.effective_level * 0.2
    new.amygdala = update_region(
        brain.amygdala, amy_exc, amy_inh + gaba_inh + sht_inh, neuromod_gain=ne_gain, dt=dt,
    )
    new.amygdala.name = "amygdala"
    new.amygdala.ne_sensitivity = brain.amygdala.ne_sensitivity
    new.amygdala.gaba_sensitivity = brain.amygdala.gaba_sensitivity

    # PFC各サブ領域
    for sub_name in ["vmPFC", "dlPFC", "OFC"]:
        sub = getattr(brain.pfc, sub_name)
        exc, inh = _compute_region_input(brain, sub_name)
        if sub_name == "vmPFC":
            exc += sensory.social_signal * 0.2
        elif sub_name == "OFC":
            exc += sensory.reward_signal * 0.3
        sht_boost = nt.serotonin.effective_level * 0.2
        updated_sub = update_region(sub, exc, inh, neuromod_gain=1.0 + sht_boost, dt=dt)
        updated_sub.name = sub_name
        setattr(new.pfc, sub_name, updated_sub)

    # 島皮質: 内受容入力
    ins_exc, ins_inh = _compute_region_input(brain, "insula")
    ins_exc += brain.body.interoception.insula_input * 0.6
    new.insula = update_region(brain.insula, ins_exc, ins_inh, dt=dt)
    new.insula.name = "insula"
    new.insula.interoceptive_input = brain.body.interoception.insula_input

    # ACC
    acc_exc, acc_inh = _compute_region_input(brain, "acc")
    new.acc = update_region(brain.acc, acc_exc, acc_inh, dt=dt)
    new.acc.name = "acc"
    new.acc.conflict_signal = min(1.0, abs(new.amygdala.output - new.pfc.vmPFC.output))

    # 海馬: コルチゾール高値で抑制
    hip_exc, hip_inh = _compute_region_input(brain, "hippocampus")
    hip_exc += sensory.context_input * 0.3
    cort_suppression = max(0, brain.body.hpa.cortisol - 0.6) * 2.0
    new.hippocampus = update_region(brain.hippocampus, hip_exc, hip_inh + cort_suppression, dt=dt)
    new.hippocampus.name = "hippocampus"
    new.hippocampus.cortisol_suppression = min(1.0, cort_suppression)

    # 視床下部
    hyp_exc, hyp_inh = _compute_region_input(brain, "hypothalamus")
    hyp_exc += new.amygdala.output * 0.5
    new.hypothalamus = update_region(brain.hypothalamus, hyp_exc, hyp_inh, dt=dt)
    new.hypothalamus.name = "hypothalamus"
    new.hypothalamus.crh_output = new.hypothalamus.output * 0.7

    # PAG: 強い脅威で発動
    pag_exc, pag_inh = _compute_region_input(brain, "pag")
    pag_exc += sensory.pain_input * 0.4
    new.pag = update_region(brain.pag, pag_exc, pag_inh, dt=dt)
    new.pag.name = "pag"
    # 防御反応の分類
    if new.pag.output > 0.7:
        new.pag.fight_output = new.pag.output * 0.3
        new.pag.flight_output = new.pag.output * 0.4
        new.pag.freeze_output = new.pag.output * 0.3
    elif new.pag.output > 0.4:
        new.pag.freeze_output = new.pag.output * 0.6
        new.pag.flight_output = new.pag.output * 0.3
        new.pag.fight_output = new.pag.output * 0.1
    else:
        new.pag.freeze_output = 0.0
        new.pag.flight_output = 0.0
        new.pag.fight_output = 0.0
    new.pag.endorphin_release = max(0, new.pag.output - 0.5) * 0.4
    nt.endorphin.phasic += new.pag.endorphin_release * dt

    # 腹側線条体/NAc
    vs_exc, vs_inh = _compute_region_input(brain, "ventral_striatum")
    vs_exc += sensory.reward_signal * 0.4
    da_gain = 1.0 + nt.dopamine.effective_level * brain.ventral_striatum.da_sensitivity * 0.5
    new.ventral_striatum = update_region(brain.ventral_striatum, vs_exc, vs_inh, neuromod_gain=da_gain, dt=dt)
    new.ventral_striatum.name = "ventral_striatum"
    new.ventral_striatum.da_sensitivity = brain.ventral_striatum.da_sensitivity
    new.ventral_striatum.reward_prediction_error = min(1.0, max(-1.0,
        sensory.reward_signal - brain.ventral_striatum.output,
    ))

    # === 3. オキシトシン: 社会的入力で放出 ===
    nt.oxytocin.phasic += sensory.social_signal * 0.15 * dt

    # === 4. コルチゾール: HPA軸更新 ===
    new.body.hpa = update_hpa_axis(brain.body.hpa, new.amygdala.output, dt)
    nt.cortisol.tonic = new.body.hpa.cortisol

    # === 5. 自律神経系更新 ===
    pfc_inhibition = new.pfc.vmPFC.output * new.pfc.amygdala_inhibition_strength
    new.body.autonomic = update_autonomic(
        brain.body.autonomic, new.amygdala.output, new.pag.output, pfc_inhibition, dt,
    )

    # === 6. 内受容更新 ===
    new.body.interoception = update_interoception(
        brain.body.interoception, new.body.autonomic,
        new.body.hpa.cortisol, nt.endorphin.effective_level, dt,
    )

    # === 7. 神経伝達物質の再取り込み ===
    new.neurotransmitters = apply_reuptake(nt)

    # === 8. メタ更新 ===
    new.step = brain.step + 1
    new.timestamp = datetime.now(timezone.utc)

    return new


def compute_readout(brain: BrainState) -> EmotionReadout:
    """神経回路状態から観測可能な情動変数を計算する。"""
    nt = brain.neurotransmitters

    valence = (
        nt.dopamine.effective_level * 0.3
        + nt.endorphin.effective_level * 0.2
        + brain.ventral_striatum.output * 0.2
        - nt.cortisol.effective_level * 0.3
        - brain.amygdala.output * 0.3
        + nt.serotonin.effective_level * 0.1
    )

    arousal = (
        nt.norepinephrine.effective_level * 0.4
        + nt.glutamate.effective_level * 0.2
        + brain.body.autonomic.sympathetic * 0.3
        + brain.brainstem.locus_coeruleus.output * 0.1
    )

    # 脅威: ベースラインレベルを差し引き、実際の脅威活性化のみを反映
    amygdala_threat = max(0.0, brain.amygdala.output - 0.3)
    cortisol_excess = max(0.0, nt.cortisol.effective_level - 0.25)
    ne_excess = max(0.0, nt.norepinephrine.effective_level - 0.35)
    threat_load = (
        amygdala_threat * 0.5
        + cortisol_excess * 0.2
        + ne_excess * 0.15
        + brain.pag.output * 0.15
    )

    reward_drive = (
        nt.dopamine.effective_level * 0.4
        + brain.ventral_striatum.output * 0.4
        + brain.pfc.OFC.output * 0.2
    )

    social_warmth = (
        nt.oxytocin.effective_level * 0.5
        + brain.pfc.vmPFC.output * brain.pfc.amygdala_inhibition_strength * 0.3
        + nt.serotonin.effective_level * 0.2
    )

    cognitive_control = (
        brain.pfc.dlPFC.output * 0.5
        + nt.serotonin.effective_level * 0.3
        + (1.0 - brain.amygdala.output) * 0.2
    )

    body_distress = brain.body.interoception.aggregate_distress
    energy = brain.body.interoception.energy_level

    memory_boost = emotional_memory_tag(
        0.5,
        brain.amygdala.output,
        nt.cortisol.effective_level,
        nt.norepinephrine.effective_level,
    )

    return EmotionReadout(
        valence=max(-1.0, min(1.0, valence)),
        arousal=max(0.0, min(1.0, arousal)),
        threat_load=max(0.0, min(1.0, threat_load)),
        reward_drive=max(0.0, min(1.0, reward_drive)),
        social_warmth=max(0.0, min(1.0, social_warmth)),
        cognitive_control=max(0.0, min(1.0, cognitive_control)),
        body_distress=max(0.0, min(1.0, body_distress)),
        energy=max(0.0, min(1.0, energy)),
        memory_encoding_boost=max(0.0, min(1.0, memory_boost)),
    )
