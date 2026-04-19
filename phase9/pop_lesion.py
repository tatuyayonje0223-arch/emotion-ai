"""Population-level lesion: silence a specific neuron group by setting its tonic
drive to a large negative value (hyperpolarizing it below rheobase).

Difference from phase9/lesioned.py (input-level):
  - Input lesion zeroes text→drive keyword translation for one emotion
  - Population lesion zeroes the neuron group's firing directly at the circuit level

Population-level is the neuroscience gold-standard control: even if inputs arrive
to the silenced region, the neurons don't fire, so the circuit readout collapses.

Populations selected per emotion (well-documented core populations):
  FEAR     -> la_exc (lateral amygdala pyramidal; LeDoux 2000)
  RAGE     -> vmh (ventromedial hypothalamus; Lee 2014)
  SADNESS  -> sgacc (subgenual anterior cingulate; Mayberg 1999)
  SEEKING  -> vta_da_lat (VTA dopamine; Schultz 1997)
  CARE     -> mpoa (medial preoptic area; Kohl 2018)
  PANIC    -> dacc (dorsal anterior cingulate; Eisenberger 2003)
  DISGUST  -> aic (anterior insular cortex; Small 2003)
  PLAY     -> pfa_thalamus (play facilitating area; Siviy 2011)
  LUST     -> lust_mpoa (sexual MPOA; Dominguez 2005)
  SURPRISE -> lc (locus coeruleus; Sara 2012)

Control: lesioning pvn_oxt (oxytocin) should affect CARE specifically but not
  FEAR/RAGE/SADNESS.
"""
from __future__ import annotations

from typing import Callable

from phase9.emotion_mapping import EMOTIONAI_LABELS


# Populations silenced per emotion lesion.
# Target the READOUT populations directly (these are what fear_act, rage_act, etc. read)
# to ensure the readout collapses even if upstream sources keep firing.
#
# From emotion_circuits_v2.py readout code:
#   fear_act = cem*0.6 + vlpag*0.2 + bnst*0.2
#   rage_act = vmh*0.5 + dlpag*0.3 + mea*0.2
#   seeking_act = vta_da_lat*0.4 + nac_shell_d1*0.3 + ofc_reward*0.3
#   sadness_act = sgacc*0.4 + habenula*0.3 + aic*0.15 + pvn_crh*0.15
#   disgust_act = aic*0.4 + nts_disgust*0.3 + putamen*0.3
#   care_act = mpoa*0.4 + pvn_oxt*0.3 + care_bnst*0.15 + vta_da_lat*0.15
#   panic_act = dacc*0.35 + bnst*0.25 + grief_pag*0.2 + aic*0.1 + pvn_crh*0.1
#   play_act = pfa_thalamus*0.4 + play_cortex*0.3 + nac_shell_d1*0.15 + dlpag*0.15
#   lust_act = lust_mpoa*0.4 + lust_hypo*0.3 + vta_da_lat*0.15 + pvn_oxt*0.15
#   surprise_act = lc*0.3 + surprise_amygdala*0.25 + surprise_pfc*0.25 + aic*0.2
#
# Lesion ALL readout contributors for each emotion so the readout formula
# collapses to ~0 (even small residual wins argmax when everything else is 0).
POP_LESION_TARGETS: dict[str, list[str]] = {
    # Extended FEAR lesion: includes upstream amygdala nuclei to overcome
    # multi-pathway redundancy. Phase 9.10 v1 showed readout-only lesion
    # (cem/vlpag/bnst) couldn't silence FEAR due to la_exc→ba_exc→cem cascade.
    "FEAR":        ["cem", "vlpag", "bnst",
                    "la_exc", "ba_exc", "cel_som"],             # extended amygdala
    "RAGE":        ["vmh", "dlpag", "mea"],                     # 0.5+0.3+0.2
    "SEEKING":     ["vta_da_lat", "nac_shell_d1", "ofc_reward"],# 0.4+0.3+0.3
    "SADNESS":     ["sgacc", "habenula", "aic", "pvn_crh"],     # 0.4+0.3+0.15+0.15
    "DISGUST":     ["aic", "nts_disgust", "putamen"],           # 0.4+0.3+0.3
    "CARE":        ["mpoa", "pvn_oxt", "care_bnst", "vta_da_lat"],# 0.4+0.3+0.15+0.15
    "PANIC_GRIEF": ["dacc", "bnst", "grief_pag", "aic", "pvn_crh"],# 0.35+0.25+0.20+0.10+0.10
    "PLAY":        ["pfa_thalamus", "play_cortex", "nac_shell_d1", "dlpag"],# 0.4+0.3+0.15+0.15
    "LUST":        ["lust_mpoa", "lust_hypo", "vta_da_lat", "pvn_oxt"],# 0.4+0.3+0.15+0.15
    "SURPRISE":    ["lc", "surprise_amygdala", "surprise_pfc", "aic"],# 0.3+0.25+0.25+0.2
}

# Effective "silence" value: tonic = -10 hyperpolarizes well below rheobase
# for all cell types (Izh: rheobase ~2-4, AdEx: rheobase ~1.8-5.4).
SILENCE_TONIC = -10.0


# Cache: one brain per lesion config (created on first call, reused across texts).
_BRAIN_CACHE: dict[tuple[str, bool], object] = {}


def make_pop_lesioned_baseline(lesion_emotion: str, use_adex: bool = False) -> Callable[[str], str]:
    """Create a baseline that silences lesion_emotion's primary population."""
    target_pops = POP_LESION_TARGETS.get(lesion_emotion, [])
    overrides = {pop: SILENCE_TONIC for pop in target_pops}
    cache_key = (lesion_emotion, use_adex)

    def predict(text: str) -> str:
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        from src.brian2_circuits.shared_core_network import SharedCoreConfig
        from src.perception.text_analyzer import analyze_text
        from phase9.baselines import _HIT_TO_DRIVE, _STATE_TO_EA

        if cache_key not in _BRAIN_CACHE:
            cfg = SharedCoreConfig(use_adex=use_adex)
            _BRAIN_CACHE[cache_key] = EmotionBrainV2(config=cfg, tonic_overrides=overrides)
        brain = _BRAIN_CACHE[cache_key]

        signal = analyze_text(text)
        feats = signal.features or {}
        drives = {
            "threat": 0.0, "reward": 0.0, "social": 0.0, "novelty": 0.0,
            "pain": 0.0, "loss": 0.0, "frustration": 0.0, "contamination": 0.0,
            "attachment_need": 0.0,
        }
        for hit_key, drive_key in _HIT_TO_DRIVE.items():
            c = int(feats.get(hit_key, 0))
            if c > 0:
                drives[drive_key] = min(1.0, drives.get(drive_key, 0.0) + 0.4 * c)

        result = brain.process(**drives)
        best_label, best_val = "SURPRISE", 0.0
        for state_attr, ea_label in _STATE_TO_EA.items():
            v = float(getattr(result, state_attr, 0.0))
            if v > best_val:
                best_val = v
                best_label = ea_label
        return best_label

    predict.__name__ = f"pop_lesion_{lesion_emotion.lower()}"
    return predict


if __name__ == "__main__":
    # Quick smoke test — lesion FEAR and verify fear prediction collapses
    fn_fear = make_pop_lesioned_baseline("FEAR")
    fn_rage = make_pop_lesioned_baseline("RAGE")
    print("Pop-level lesion smoke test:")
    for text, expected in [
        ("I'm so scared right now", "FEAR"),
        ("I'm absolutely furious", "RAGE"),
        ("I feel sad today", "SADNESS"),
    ]:
        f = fn_fear(text)
        r = fn_rage(text)
        print(f"  '{text[:30]}' (true={expected}):  lesion-FEAR={f}  lesion-RAGE={r}")
