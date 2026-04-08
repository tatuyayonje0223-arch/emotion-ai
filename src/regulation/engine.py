"""情動制御エンジン。再評価・抑制・受容の3戦略を実装する。"""

from __future__ import annotations

from typing import Literal

from src.config.settings import RegulationConfig, get_config
from src.schemas.affect_state import AffectDelta, AffectState
from src.schemas.events import AppraisalResult


RegulationMode = Literal["reappraisal", "suppression", "acceptance", "adaptive"]


def select_regulation_mode(state: AffectState, appraisal: AppraisalResult) -> RegulationMode:
    """現在の状態と評価に基づき、最適な制御モードを選択する。

    アダプティブモード:
    - 高脅威+低制御 → 再評価（脅威の意味を変える）
    - 高覚醒+高脅威 → 抑制（急性反応の緩和）
    - 軽度ネガティブ → 受容（過剰制御を避ける）
    - それ以外 → 受容
    """
    cfg = get_config().regulation
    if cfg.mode != "adaptive":
        return cfg.mode

    # 高脅威+低制御 → 再評価
    if state.threat_load > 0.6 and appraisal.controllability < 0.4:
        return "reappraisal"

    # 高覚醒+高脅威 → 抑制
    if state.arousal > 0.7 and state.threat_load > 0.5:
        return "suppression"

    # デフォルト → 受容
    return "acceptance"


def regulate(
    state: AffectState,
    appraisal: AppraisalResult,
    proposed_delta: AffectDelta,
    config: RegulationConfig | None = None,
) -> tuple[AffectDelta, RegulationMode, str]:
    """提案された状態変化を制御し、調整後のdeltaと制御モード・理由を返す。

    Returns:
        (adjusted_delta, mode_used, explanation)
    """
    cfg = config or get_config().regulation
    mode = select_regulation_mode(state, appraisal)

    if mode == "reappraisal":
        return _reappraise(proposed_delta, cfg), mode, "脅威の意味を再解釈し、ネガティブ影響を緩和"

    if mode == "suppression":
        return _suppress(proposed_delta, cfg), mode, "急性の情動反応を一時的に抑制"

    # acceptance: 変更なし
    return proposed_delta, mode, "軽度のため、情動状態をそのまま受容"


def _reappraise(delta: AffectDelta, cfg: RegulationConfig) -> AffectDelta:
    """再評価: ネガティブ方向の変化を減衰させ、制御感を少し回復させる。"""
    adjusted = delta.model_copy()
    strength = cfg.reappraisal_strength

    if adjusted.valence is not None and adjusted.valence < 0:
        adjusted.valence = adjusted.valence * (1.0 - strength)
    if adjusted.threat_load is not None and adjusted.threat_load > 0:
        adjusted.threat_load = adjusted.threat_load * (1.0 - strength)
    if adjusted.uncertainty is not None and adjusted.uncertainty > 0:
        adjusted.uncertainty = adjusted.uncertainty * (1.0 - strength * 0.5)

    # 制御感を少し回復
    control_recovery = strength * 0.1
    if adjusted.perceived_control is None:
        adjusted.perceived_control = control_recovery
    else:
        adjusted.perceived_control = adjusted.perceived_control + control_recovery

    return adjusted


def _suppress(delta: AffectDelta, cfg: RegulationConfig) -> AffectDelta:
    """抑制: 全ての変化を一律に減衰させる（代償として疲労増加）。"""
    adjusted = delta.model_copy()
    strength = cfg.suppression_strength

    for field in ["valence", "arousal", "threat_load", "uncertainty"]:
        val = getattr(adjusted, field)
        if val is not None:
            setattr(adjusted, field, val * (1.0 - strength))

    # 抑制の代償: 疲労増加
    fatigue_cost = strength * 0.05
    if adjusted.fatigue is None:
        adjusted.fatigue = fatigue_cost
    else:
        adjusted.fatigue = adjusted.fatigue + fatigue_cost

    return adjusted
