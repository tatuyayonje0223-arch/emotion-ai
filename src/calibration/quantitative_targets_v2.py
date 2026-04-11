"""V2回路の定量的バリデーションターゲット。

232検証済み論文から導出した発火率・応答比・時間窓ターゲット。
各ターゲットに論文出典を付与。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FiringRateTarget:
    """発火率ターゲット。"""
    region: str
    condition: str
    min_hz: float
    max_hz: float
    source: str


@dataclass
class RatioTarget:
    """応答比ターゲット。"""
    name: str
    numerator: str
    denominator: str
    min_ratio: float
    max_ratio: float
    source: str


# ═══════════════════════════════════════════════════════════
# FEAR (既存V1ターゲット + V2拡張)
# ═══════════════════════════════════════════════════════════

FEAR_TARGETS = [
    FiringRateTarget("la_exc", "baseline", 1, 8, "Quirk 2002; Repa 2001"),
    FiringRateTarget("la_exc", "cs_evoked", 8, 35, "Quirk 2002; Repa 2001"),
    FiringRateTarget("cel_som", "cs_evoked", 5, 25, "Ciocchi 2010 Nature"),
    FiringRateTarget("cel_pkcd", "cs_evoked", 0, 5, "Ciocchi 2010 Nature (decrease)"),
    FiringRateTarget("cem", "fear_expression", 8, 30, "Ciocchi 2010; Duvarci & Pare 2014"),
    FiringRateTarget("pl", "fear_burst", 10, 45, "Courtin 2014 Nature"),
    FiringRateTarget("il", "extinction_recall", 5, 20, "Quirk 2002 Nature"),
    FiringRateTarget("vlpag", "freezing", 5, 20, "Tovote 2015 Nat Rev Neurosci"),
]

FEAR_RATIOS = [
    RatioTarget("SOM+/PKCd+ ratio", "cel_som", "cel_pkcd", 2.0, 5.0, "Ciocchi 2010"),
    RatioTarget("Conditioned/Baseline LA", "la_cs", "la_base", 2.0, 8.0, "Repa 2001"),
]

# ═══════════════════════════════════════════════════════════
# RAGE (新規)
# ═══════════════════════════════════════════════════════════

RAGE_TARGETS = [
    FiringRateTarget("mea", "baseline", 3, 10, "Hong 2014"),
    FiringRateTarget("mea", "social_encounter", 8, 30, "Hong 2014"),
    FiringRateTarget("vmh", "baseline", 2, 8, "Lee 2014; Falkner 2016"),
    FiringRateTarget("vmh", "investigation", 5, 20, "Lee 2014"),
    FiringRateTarget("vmh", "attack", 15, 55, "Lee 2014; Falkner 2016"),
    FiringRateTarget("dlpag", "attack", 10, 45, "Falkner 2020"),
]

RAGE_RATIOS = [
    RatioTarget("VMH attack/baseline", "vmh_attack", "vmh_base", 4.0, 15.0,
                "Lee 2014; Falkner 2016"),
]

# ═══════════════════════════════════════════════════════════
# SEEKING (既存reward拡張)
# ═══════════════════════════════════════════════════════════

SEEKING_TARGETS = [
    FiringRateTarget("vta_da_lat", "tonic", 3, 10, "Grace 2007; Schultz 1997"),
    FiringRateTarget("vta_da_lat", "phasic_burst", 15, 35, "Schultz 1997; calibrated=25Hz"),
    FiringRateTarget("vta_da_lat", "pause", 0, 1, "Schultz 1997 (negative RPE)"),
    FiringRateTarget("nac_shell_d1", "reward", 5, 25, "Roitman 2005"),
]

SEEKING_RATIOS = [
    RatioTarget("DA burst/tonic", "vta_burst", "vta_tonic", 2.5, 8.0,
                "Schultz 1997; Grace 2007"),
]

# ═══════════════════════════════════════════════════════════
# SADNESS (新規)
# ═══════════════════════════════════════════════════════════

SADNESS_TARGETS = [
    FiringRateTarget("sgacc", "depression", 5, 30,
                     "Mayberg 1999 (+20-40% metabolic increase)"),
    FiringRateTarget("habenula", "reward_omission", 5, 25,
                     "Matsumoto & Hikosaka 2007 Nature"),
    FiringRateTarget("dr", "sadness_suppressed", 0, 3,
                     "Estimated 20-40% reduction; Caspi 2003"),
]

SADNESS_RATIOS = [
    RatioTarget("sgACC depression/healthy", "sgacc_dep", "sgacc_healthy", 1.2, 1.6,
                "Mayberg 1999; Drevets 1997"),
]

# ═══════════════════════════════════════════════════════════
# DISGUST (新規)
# ═══════════════════════════════════════════════════════════

DISGUST_TARGETS = [
    FiringRateTarget("aic", "disgust_stimulus", 5, 30,
                     "Phillips 1997 Nature (fMRI z=3-5)"),
    FiringRateTarget("nts_disgust", "contamination", 5, 25,
                     "Carter 2015 (PBN CGRP)"),
    FiringRateTarget("putamen", "disgust_recognition", 3, 20,
                     "Calder 2000 Nat Neurosci; Sprengelmeyer 1996"),
]

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════

ALL_TARGETS = {
    "fear": (FEAR_TARGETS, FEAR_RATIOS),
    "rage": (RAGE_TARGETS, RAGE_RATIOS),
    "seeking": (SEEKING_TARGETS, SEEKING_RATIOS),
    "sadness": (SADNESS_TARGETS, SADNESS_RATIOS),
    "disgust": (DISGUST_TARGETS, []),
}
