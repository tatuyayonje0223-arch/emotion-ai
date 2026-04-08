"""神経伝達物質ダイナミクス。

主要8系統をモデル化:
- Dopamine (DA): 報酬予測、動機づけ — VTA/NAc 経路
- Serotonin (5-HT): 気分調整、衝動抑制 — 縫線核→広域
- Norepinephrine (NE): 覚醒、注意、闘争逃走 — 青斑核→広域
- GABA: 抑制性、不安軽減 — 広域抑制
- Glutamate (Glu): 興奮性、学習 — 広域興奮
- Oxytocin (OXT): 社会的結合、信頼 — 視床下部→扁桃体
- Cortisol (CORT): ストレス応答 — HPA軸
- Endorphin (END): 疼痛調節、快感 — PAG/NAc

各伝達物質は:
- tonic level (基底レベル): ゆっくり変動する持続的活性
- phasic burst (位相的発火): イベント駆動の急速な変化
- reuptake/degradation: 自然減衰
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NeurotransmitterState(BaseModel):
    """1種の神経伝達物質の状態。"""

    tonic: float = Field(0.5, ge=0.0, le=1.0, description="基底レベル")
    phasic: float = Field(0.0, ge=-1.0, le=1.0, description="位相的変化（正=放出, 負=抑制）")
    reuptake_rate: float = Field(0.1, ge=0.0, le=1.0, description="再取り込み速度")

    @property
    def effective_level(self) -> float:
        """実効レベル = tonic + phasic、0-1にクランプ。"""
        return max(0.0, min(1.0, self.tonic + self.phasic))


class NeurotransmitterSystem(BaseModel):
    """全神経伝達物質系の統合状態。"""

    dopamine: NeurotransmitterState = Field(
        default_factory=lambda: NeurotransmitterState(tonic=0.5, reuptake_rate=0.15),
    )
    serotonin: NeurotransmitterState = Field(
        default_factory=lambda: NeurotransmitterState(tonic=0.5, reuptake_rate=0.08),
    )
    norepinephrine: NeurotransmitterState = Field(
        default_factory=lambda: NeurotransmitterState(tonic=0.3, reuptake_rate=0.12),
    )
    gaba: NeurotransmitterState = Field(
        default_factory=lambda: NeurotransmitterState(tonic=0.5, reuptake_rate=0.1),
    )
    glutamate: NeurotransmitterState = Field(
        default_factory=lambda: NeurotransmitterState(tonic=0.4, reuptake_rate=0.2),
    )
    oxytocin: NeurotransmitterState = Field(
        default_factory=lambda: NeurotransmitterState(tonic=0.3, reuptake_rate=0.05),
    )
    cortisol: NeurotransmitterState = Field(
        default_factory=lambda: NeurotransmitterState(tonic=0.2, reuptake_rate=0.03),
    )
    endorphin: NeurotransmitterState = Field(
        default_factory=lambda: NeurotransmitterState(tonic=0.3, reuptake_rate=0.06),
    )

    def all_names(self) -> list[str]:
        return ["dopamine", "serotonin", "norepinephrine", "gaba",
                "glutamate", "oxytocin", "cortisol", "endorphin"]

    def get(self, name: str) -> NeurotransmitterState:
        return getattr(self, name)

    def effective_levels(self) -> dict[str, float]:
        return {name: self.get(name).effective_level for name in self.all_names()}


def apply_reuptake(system: NeurotransmitterSystem) -> NeurotransmitterSystem:
    """全伝達物質の再取り込み（位相成分の減衰）を適用する。"""
    updated = system.model_copy(deep=True)
    for name in updated.all_names():
        nt = updated.get(name)
        # phasic → 0 へ指数減衰
        nt.phasic *= (1.0 - nt.reuptake_rate)
        if abs(nt.phasic) < 0.001:
            nt.phasic = 0.0
        # tonic のホメオスタシス（基底値への緩やかな回帰）
        baseline = _BASELINES.get(name, 0.5)
        nt.tonic += (baseline - nt.tonic) * 0.02
    return updated


# 各伝達物質のホメオスタシス基底値
_BASELINES: dict[str, float] = {
    "dopamine": 0.5,
    "serotonin": 0.5,
    "norepinephrine": 0.3,
    "gaba": 0.5,
    "glutamate": 0.4,
    "oxytocin": 0.3,
    "cortisol": 0.2,
    "endorphin": 0.3,
}
