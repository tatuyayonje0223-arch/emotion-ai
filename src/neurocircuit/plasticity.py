"""シナプス可塑性ルール。脳に準拠した学習メカニズム。

- ヘブ則: 同時活性による結合強化
- 報酬変調学習: ドーパミンRPEによる強化/減弱
- 情動タグ付け: 扁桃体活性による記憶固定化の優先
- 恐怖条件付け/消去: 扁桃体-PFC回路の可塑性
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlasticityParams(BaseModel):
    """可塑性パラメータ。"""

    hebbian_rate: float = Field(0.01, ge=0.0, le=0.1, description="ヘブ則学習率")
    reward_modulation_rate: float = Field(0.02, ge=0.0, le=0.1, description="報酬変調学習率")
    fear_conditioning_rate: float = Field(0.03, ge=0.0, le=0.1, description="恐怖条件付け速度")
    extinction_rate: float = Field(0.005, ge=0.0, le=0.1, description="消去速度")
    weight_decay: float = Field(0.001, ge=0.0, le=0.01, description="結合重み自然減衰")
    max_weight: float = Field(2.0, description="結合重み上限")
    min_weight: float = Field(0.0, description="結合重み下限")


def hebbian_update(
    weight: float,
    pre_activity: float,
    post_activity: float,
    params: PlasticityParams,
) -> float:
    """ヘブ則: pre と post が同時に活性なら結合を強化する。

    Δw = η * pre * post - decay * w
    """
    dw = params.hebbian_rate * pre_activity * post_activity - params.weight_decay * weight
    new_weight = weight + dw
    return max(params.min_weight, min(params.max_weight, new_weight))


def reward_modulated_update(
    weight: float,
    pre_activity: float,
    post_activity: float,
    reward_prediction_error: float,
    params: PlasticityParams,
) -> float:
    """報酬変調学習: RPE × ヘブ相関。

    ドーパミン報酬予測誤差が正なら強化、負なら減弱。
    三要素則: Δw = η * RPE * pre * post
    """
    dw = (
        params.reward_modulation_rate * reward_prediction_error * pre_activity * post_activity
        - params.weight_decay * weight
    )
    new_weight = weight + dw
    return max(params.min_weight, min(params.max_weight, new_weight))


def fear_conditioning(
    amygdala_cs_weight: float,
    cs_activity: float,
    us_activity: float,
    amygdala_activity: float,
    params: PlasticityParams,
) -> float:
    """恐怖条件付け: CS-US連合による扁桃体結合の強化。

    条件刺激(CS)と無条件刺激(US)が同時提示されると、
    CS→扁桃体の結合が強化される。
    """
    if us_activity > 0.3:  # USが存在する時のみ
        dw = params.fear_conditioning_rate * cs_activity * us_activity
    else:
        dw = 0.0
    new_weight = amygdala_cs_weight + dw
    return max(params.min_weight, min(params.max_weight, new_weight))


def extinction(
    amygdala_cs_weight: float,
    cs_activity: float,
    us_absence: bool,
    pfc_inhibition: float,
    params: PlasticityParams,
) -> float:
    """消去: CS単独提示 + PFC抑制により恐怖反応が減弱。

    USなしでCSが提示され、PFCが扁桃体を抑制している時、
    CS→扁桃体の結合が徐々に減弱する。
    """
    if us_absence and cs_activity > 0.2 and pfc_inhibition > 0.3:
        dw = -params.extinction_rate * cs_activity * pfc_inhibition
    else:
        dw = 0.0
    new_weight = amygdala_cs_weight + dw
    return max(params.min_weight, min(params.max_weight, new_weight))


def emotional_memory_tag(
    memory_strength: float,
    amygdala_activity: float,
    cortisol_level: float,
    norepinephrine_level: float,
) -> float:
    """情動タグ付け記憶: 扁桃体活性+NE+コルチゾールで記憶固定化を促進。

    ただしコルチゾールが極端に高い場合は海馬機能を抑制し逆効果。
    """
    # 適度なコルチゾール(0.3-0.6)は記憶を促進、高すぎると抑制
    if cortisol_level < 0.6:
        cortisol_boost = cortisol_level * 0.5
    else:
        cortisol_boost = 0.3 - (cortisol_level - 0.6) * 1.0  # 逆U字

    tag_strength = (
        amygdala_activity * 0.4
        + norepinephrine_level * 0.3
        + max(0.0, cortisol_boost) * 0.3
    )

    enhanced_strength = memory_strength + tag_strength * 0.2
    return max(0.0, min(1.0, enhanced_strength))
