"""Phase 9.8: Dimensional affect regression (valence × arousal).

Hypothesis: 10-way classification is hard, but the underlying valence and
arousal dimensions may be captured by the neural simulation more naturally
than by keyword matching. Rationale:
  - VTA DA, CARE circuits → positive valence
  - FEAR, RAGE, SADNESS circuits → negative valence
  - LC NE, RAGE, SURPRISE → high arousal
  - SADNESS, CARE → low arousal

Ground truth: each GoEmotions label has published affect norms. We construct
V/A values per label using Russell's circumplex + Warriner 2013 proxies.

Baselines compared:
  - random: uniform [-1, 1] × [0, 1]
  - keyword: text_analyzer's sentiment_score (valence proxy) + arousal_estimate
  - model: IntegratedBrainV2.process().valence / .arousal

Metrics: Pearson r, MAE (per dimension) + bivariate R^2
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# Russell circumplex / Warriner 2013 approximate V/A norms per EmotionAI label.
# Valence: -1 (unpleasant) to +1 (pleasant)
# Arousal: 0 (calm) to 1 (activated)
EA_TO_VA: dict[str, tuple[float, float]] = {
    "FEAR":        (-0.8, 0.8),   # negative, high arousal
    "RAGE":        (-0.9, 0.9),   # very negative, very high arousal
    "SEEKING":     ( 0.6, 0.6),   # positive, moderate arousal (anticipation)
    "SADNESS":     (-0.7, 0.2),   # negative, low arousal
    "DISGUST":     (-0.7, 0.5),   # negative, moderate arousal
    "CARE":        ( 0.7, 0.3),   # positive, low arousal (warm)
    "PANIC_GRIEF": (-0.7, 0.6),   # negative, moderate-high arousal
    "PLAY":        ( 0.8, 0.8),   # very positive, high arousal
    "LUST":        ( 0.6, 0.7),   # positive, high arousal
    "SURPRISE":    ( 0.0, 0.8),   # neutral valence, high arousal
}


@dataclass
class VAPrediction:
    valence: float
    arousal: float


def ground_truth_va(ea_label: str) -> tuple[float, float]:
    """Return (valence, arousal) ground truth for an EmotionAI label."""
    return EA_TO_VA.get(ea_label, (0.0, 0.5))


# ═══════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════

import random as _random


def make_random_va_baseline(seed: int = 42):
    rng = _random.Random(seed)
    def predict(text: str) -> VAPrediction:
        return VAPrediction(valence=rng.uniform(-1, 1), arousal=rng.uniform(0, 1))
    return predict


def keyword_va_baseline(text: str) -> VAPrediction:
    """Use text_analyzer's sentiment + arousal estimates directly."""
    from src.perception.text_analyzer import analyze_text
    sig = analyze_text(text)
    return VAPrediction(valence=sig.sentiment_score, arousal=sig.arousal_estimate)


def model_va_baseline(text: str, use_adex: bool = False) -> VAPrediction:
    """IntegratedBrainV2 emits valence and arousal in EmotionStateV2."""
    from phase9.baselines import _get_brain, _HIT_TO_DRIVE
    from src.perception.text_analyzer import analyze_text

    sig = analyze_text(text)
    feats = sig.features or {}

    drives: dict[str, float] = {
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
    return VAPrediction(valence=float(result.valence), arousal=float(result.arousal))


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

def pearson_r(x: list[float], y: list[float]) -> float:
    xa = np.array(x); ya = np.array(y)
    if xa.std() == 0 or ya.std() == 0:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def mae(x: list[float], y: list[float]) -> float:
    return float(np.mean(np.abs(np.array(x) - np.array(y))))


def bivariate_r2(x_v, x_a, y_v, y_a) -> float:
    """R^2 for bivariate prediction: 1 - SSE/(SS_tot for both dims)."""
    sse = np.sum((np.array(x_v) - np.array(y_v))**2) + np.sum((np.array(x_a) - np.array(y_a))**2)
    mean_v = np.mean(y_v); mean_a = np.mean(y_a)
    sst = np.sum((np.array(y_v) - mean_v)**2) + np.sum((np.array(y_a) - mean_a)**2)
    return float(1 - sse / sst) if sst > 0 else 0.0


# ═══════════════════════════════════════════════════════════
# Control baseline: hybrid (keyword hits + model's V/A weight table, no simulation)
# ═══════════════════════════════════════════════════════════

# Mirrors the weights in src/brian2_circuits/emotion_circuits_v2.py lines 907, 916.
# DO NOT modify independently — these must stay in sync with the model readout.
_VALENCE_WEIGHTS: dict[str, float] = {
    "FEAR": -0.9, "RAGE": -0.8, "SEEKING": 0.7, "SADNESS": -0.5,
    "DISGUST": -0.6, "CARE": 0.8, "PANIC_GRIEF": -0.7,
    "PLAY": 0.9, "LUST": 0.6, "SURPRISE": 0.0,
}
_AROUSAL_WEIGHTS: dict[str, float] = {
    "FEAR": 0.9, "RAGE": 0.9, "SEEKING": 0.6, "SADNESS": 0.2,
    "DISGUST": 0.5, "CARE": 0.3, "PANIC_GRIEF": 0.6,
    "PLAY": 0.8, "LUST": 0.7, "SURPRISE": 0.9,
}

_HIT_KEY_TO_EA: dict[str, str] = {
    "fear_hits": "FEAR",
    "rage_hits": "RAGE",
    "seeking_hits": "SEEKING",
    "sadness_hits": "SADNESS",
    "disgust_hits": "DISGUST",
    "care_hits": "CARE",
    "panic_grief_hits": "PANIC_GRIEF",
    "play_hits": "PLAY",
    "lust_hits": "LUST",
    "surprise_hits": "SURPRISE",
}


def hybrid_va_baseline(text: str) -> VAPrediction:
    """Control experiment: keyword emotion hits → same V/A weight table as model.

    Bypasses the neural simulation entirely. If hybrid_va ≈ model_va on the
    same dataset, the simulation contributes NO unique dimensional value —
    the advantage attributed to "model" in Phase 9.8 came from the hand-coded
    weight table, not from circuit dynamics.
    """
    from src.perception.text_analyzer import analyze_text
    sig = analyze_text(text)
    feats = sig.features or {}

    # Build emotion activation dict from keyword hit counts.
    # Normalize by total hits to match model's `/ total_act` division.
    emotion_act: dict[str, float] = {}
    for hit_key, ea_label in _HIT_KEY_TO_EA.items():
        emotion_act[ea_label] = float(feats.get(hit_key, 0))

    total_act = sum(emotion_act.values())
    if total_act < 1e-6:
        # No emotional keywords — fall back to neutral
        return VAPrediction(valence=0.0, arousal=0.5)

    valence = sum(act * _VALENCE_WEIGHTS[name] for name, act in emotion_act.items()) / total_act
    arousal = sum(act * _AROUSAL_WEIGHTS[name] for name, act in emotion_act.items()) / total_act

    # Clip to [-1, 1] / [0, 1] as model does
    valence = max(-1.0, min(1.0, valence))
    arousal = max(0.0, min(1.0, arousal))
    return VAPrediction(valence=valence, arousal=arousal)


VA_BASELINES = {
    "random_va":  make_random_va_baseline(seed=42),
    "keyword_va": keyword_va_baseline,
    "hybrid_va":  hybrid_va_baseline,  # control: keyword + weights, no simulation
    "model_va":   model_va_baseline,
}


if __name__ == "__main__":
    # Smoke test
    texts_labels = [
        ("I'm so scared right now", "FEAR"),
        ("I won the lottery, so happy!", "SEEKING"),
        ("I feel so sad today", "SADNESS"),
        ("Haha that was hilarious!", "PLAY"),
    ]
    for text, label in texts_labels:
        gt_v, gt_a = ground_truth_va(label)
        k = keyword_va_baseline(text)
        m = model_va_baseline(text)
        print(f"'{text[:30]}' ({label})")
        print(f"  GT      V={gt_v:+.2f}  A={gt_a:.2f}")
        print(f"  keyword V={k.valence:+.2f}  A={k.arousal:.2f}")
        print(f"  model   V={m.valence:+.2f}  A={m.arousal:.2f}")
