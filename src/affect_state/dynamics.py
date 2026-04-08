"""状態変数の自然減衰・回復・ヒステリシスのダイナミクス。"""

from __future__ import annotations

from src.config.settings import DecayConfig, get_config
from src.schemas.affect_state import AffectDelta, AffectState


def compute_decay(state: AffectState, config: DecayConfig | None = None) -> AffectDelta:
    """時間経過による自然減衰を計算する。

    原則:
    - valence は 0 に向かって減衰（快も不快も薄れる）
    - arousal は基底レベル（0.2）に向かって減衰
    - threat_load は 0 に向かって緩やかに減衰
    - fatigue は 0 に向かって自然回復
    - trust, perceived_control, uncertainty は自然減衰しない（イベント駆動のみ）
    """
    cfg = config or get_config().decay

    # valence → 0 へ
    if abs(state.valence) < 0.01:
        valence_delta = -state.valence  # 完全リセット
    else:
        valence_delta = -state.valence * cfg.valence_decay

    # arousal → 0.2 (基底) へ
    arousal_baseline = 0.2
    arousal_delta = (arousal_baseline - state.arousal) * cfg.arousal_decay

    # threat_load → 0 へ
    threat_delta = -state.threat_load * cfg.threat_decay

    # fatigue → 0 へ（回復）
    fatigue_delta = -state.fatigue * cfg.fatigue_recovery

    return AffectDelta(
        valence=round(valence_delta, 6),
        arousal=round(arousal_delta, 6),
        threat_load=round(threat_delta, 6),
        fatigue=round(fatigue_delta, 6),
    )


def compute_hysteresis(state: AffectState, proposed_delta: AffectDelta) -> AffectDelta:
    """ヒステリシス: 極端な状態からの急激な逆転を抑制する。

    高脅威状態からの急速な安心化や、高信頼からの急速な不信を減衰させる。
    これは「一度恐怖を感じると簡単には安心しない」のモデル化。
    """
    dampened = proposed_delta.model_copy()

    # 高脅威(>0.7)時、脅威減少を半減
    if state.threat_load > 0.7 and (dampened.threat_load or 0) < 0:
        dampened.threat_load = (dampened.threat_load or 0) * 0.5

    # 高覚醒(>0.8)時、覚醒減少を半減
    if state.arousal > 0.8 and (dampened.arousal or 0) < 0:
        dampened.arousal = (dampened.arousal or 0) * 0.5

    # 低信頼(<0.2)時、信頼増加を半減
    if state.trust < 0.2 and (dampened.trust or 0) > 0:
        dampened.trust = (dampened.trust or 0) * 0.5

    return dampened
