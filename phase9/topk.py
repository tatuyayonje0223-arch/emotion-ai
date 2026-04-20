"""Top-k accuracy evaluation on 10-way classification.

Argmax (top-1) is strict. Top-k checks if the true label is in the top-k
ranked predictions. If the model "comes close" even when wrong, top-k
reveals that partial signal.

Interpretation:
  top-1 = keyword 28%, model 22% → model strictly loses
  top-3 = ? → if model's top-3 captures true emotion better than keyword's
          top-3, model ranks emotions more informatively even without
          winning argmax
"""
from __future__ import annotations

from typing import Callable

from phase9.emotion_mapping import EMOTIONAI_LABELS


def ranked_prediction_keyword(text: str) -> list[str]:
    """Return 10 EmotionAI labels sorted by keyword hit count (descending)."""
    from src.perception.text_analyzer import analyze_text
    sig = analyze_text(text)
    feats = sig.features or {}

    hit_map = {
        "FEAR": feats.get("fear_hits", 0),
        "RAGE": feats.get("rage_hits", 0),
        "SEEKING": feats.get("seeking_hits", 0),
        "SADNESS": feats.get("sadness_hits", 0),
        "DISGUST": feats.get("disgust_hits", 0),
        "CARE": feats.get("care_hits", 0),
        "PANIC_GRIEF": feats.get("panic_grief_hits", 0),
        "PLAY": feats.get("play_hits", 0),
        "LUST": feats.get("lust_hits", 0),
        "SURPRISE": feats.get("surprise_hits", 0),
    }
    return sorted(hit_map.keys(), key=lambda k: -hit_map[k])


def ranked_prediction_model(text: str, use_adex: bool = False) -> list[str]:
    """Return 10 EmotionAI labels sorted by model activation (descending)."""
    from phase9.baselines import _get_brain, _HIT_TO_DRIVE, _STATE_TO_EA
    from src.perception.text_analyzer import analyze_text

    sig = analyze_text(text)
    feats = sig.features or {}
    drives = {
        "threat": 0.0, "reward": 0.0, "social": 0.0, "novelty": 0.0,
        "pain": 0.0, "loss": 0.0, "frustration": 0.0, "contamination": 0.0,
        "attachment_need": 0.0,
    }
    for hit_key, drive_key in _HIT_TO_DRIVE.items():
        c = int(feats.get(hit_key, 0))
        if c > 0:
            drives[drive_key] = min(1.0, drives.get(drive_key, 0.0) + 0.4 * c)

    brain = _get_brain(use_adex=use_adex)
    result = brain.process(**drives)

    activations = {}
    for state_attr, ea_label in _STATE_TO_EA.items():
        activations[ea_label] = float(getattr(result, state_attr, 0.0))
    # Sort by activation desc
    return sorted(activations.keys(), key=lambda k: -activations[k])


def topk_accuracy(ranked_preds: list[list[str]], trues: list[str], k: int) -> float:
    """Fraction of instances where true label is in top-k."""
    assert len(ranked_preds) == len(trues)
    hits = 0
    for ranked, true in zip(ranked_preds, trues):
        if true in ranked[:k]:
            hits += 1
    return hits / len(trues) if trues else 0.0


RANKERS: dict[str, Callable[[str], list[str]]] = {
    "keyword": ranked_prediction_keyword,
    "model":   ranked_prediction_model,
}
