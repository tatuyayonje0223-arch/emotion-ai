"""Lesioned baselines: silence one emotion's primary drive inputs.

Tests circuit specificity: if lesioning EMOTION X's inputs specifically reduces
accuracy on TRUE X instances (but not on other emotions), the circuit has
functional specificity even if overall accuracy is poor.

This is a WEAKER form of lesion than zeroing the circuit populations; it operates
at the input drive level. The overlap between inputs (e.g., social drives
care/play/lust) means lesioning is not perfectly isolated.

Primary drive mapping for each emotion (per emotion_circuits_v2.py process()):
  FEAR        -> threat, pain
  RAGE        -> frustration
  SEEKING     -> reward (when social < 0.5)
  SADNESS     -> loss
  DISGUST     -> contamination
  CARE        -> social + attachment_need (overlaps w/ play, lust)
  PANIC_GRIEF -> loss + attachment_need
  PLAY        -> social + reward + novelty (combination)
  LUST        -> social (secondary: reward)
  SURPRISE    -> novelty
"""
from __future__ import annotations

from typing import Callable
from functools import partial

from phase9.baselines import model_rates_baseline
from phase9.emotion_mapping import EMOTIONAI_LABELS


# Primary drive keys to zero out for each emotion lesion.
# Chose conservative set — only zero the MOST direct drive.
LESION_DRIVES: dict[str, list[str]] = {
    "FEAR":        ["threat", "pain"],
    "RAGE":        ["frustration"],
    "SEEKING":     ["reward"],
    "SADNESS":     ["loss"],
    "DISGUST":     ["contamination"],
    "CARE":        ["attachment_need"],       # social shared; only lesion attachment_need
    "PANIC_GRIEF": ["loss", "attachment_need"],   # loss overlaps w/ sadness
    "PLAY":        ["novelty"],                # social shared; only lesion novelty
    "LUST":        ["social"],                 # overlaps but lust's main channel
    "SURPRISE":    ["novelty"],
}


def make_lesioned_baseline(lesion_emotion: str) -> Callable[[str], str]:
    """Create a baseline that zeros the lesion_emotion's primary drives
    before running the model."""
    zero_keys = LESION_DRIVES.get(lesion_emotion, [])

    def predict(text: str) -> str:
        # Monkey-patch the drive aggregation. Simplest: call the model baseline
        # but override the drives mid-stream. We re-implement the relevant slice here.
        from src.perception.text_analyzer import analyze_text
        from phase9.baselines import _get_brain, _HIT_TO_DRIVE, _STATE_TO_EA

        signal = analyze_text(text)
        feats = signal.features or {}

        drives: dict[str, float] = {
            "threat": 0.0, "reward": 0.0, "social": 0.0, "novelty": 0.0,
            "pain": 0.0, "loss": 0.0, "frustration": 0.0, "contamination": 0.0,
            "attachment_need": 0.0,
        }
        for hit_key, drive_key in _HIT_TO_DRIVE.items():
            c = int(feats.get(hit_key, 0))
            if c > 0:
                drives[drive_key] = min(1.0, drives.get(drive_key, 0.0) + 0.4 * c)

        # Apply lesion — zero the specified drives
        for k in zero_keys:
            drives[k] = 0.0

        brain = _get_brain(use_adex=False)
        result = brain.process(**drives)
        best_label, best_val = "SURPRISE", -1.0
        for state_attr, ea_label in _STATE_TO_EA.items():
            v = float(getattr(result, state_attr, 0.0))
            if v > best_val:
                best_val = v
                best_label = ea_label
        return best_label

    predict.__name__ = f"lesioned_{lesion_emotion.lower()}"
    return predict


def all_lesion_baselines() -> dict[str, Callable[[str], str]]:
    """Return dict of {'lesioned_FEAR': fn, ...} for all 10 emotions."""
    return {
        f"lesioned_{emo}": make_lesioned_baseline(emo)
        for emo in EMOTIONAI_LABELS
    }


if __name__ == "__main__":
    # Quick sanity — lesion FEAR and see if "scared" prediction changes
    fn_normal = model_rates_baseline
    fn_lesion_fear = make_lesioned_baseline("FEAR")
    fn_lesion_seeking = make_lesioned_baseline("SEEKING")

    text = "I'm so scared right now"
    print(f"Text: {text}")
    print(f"  normal          -> {fn_normal(text)}")
    print(f"  lesioned FEAR   -> {fn_lesion_fear(text)}")
    print(f"  lesioned SEEKING -> {fn_lesion_seeking(text)}")

    text = "I won the lottery, so happy!"
    print(f"\nText: {text}")
    print(f"  normal          -> {fn_normal(text)}")
    print(f"  lesioned FEAR   -> {fn_lesion_fear(text)}")
    print(f"  lesioned SEEKING -> {fn_lesion_seeking(text)}")
