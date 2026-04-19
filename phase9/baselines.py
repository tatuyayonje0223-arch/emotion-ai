"""Baseline classifiers for Phase 9 behavioral validation.

All baselines implement the same interface:
    predict(text: str) -> str  (one of EMOTIONAI_LABELS)

- `random_baseline`: uniform random class per instance (stratified seed per call)
- `keyword_baseline`: uses existing src/perception/text_analyzer.py keyword hits
- `model_rates_baseline`: runs IntegratedBrainV2 and picks max-activation emotion
- `lesioned_model_baseline`: same as model but with one circuit's drive forced to 0

LLM baseline (GPT-4 zero-shot) scaffolded in phase9/llm_baseline.py separately.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from phase9.emotion_mapping import EMOTIONAI_LABELS


# ═══════════════════════════════════════════════════════════
# Random baseline
# ═══════════════════════════════════════════════════════════

def make_random_baseline(seed: int = 42) -> Callable[[str], str]:
    """Uniform random across 10 EmotionAI classes. Seed for reproducibility."""
    rng = random.Random(seed)
    def predict(text: str) -> str:
        return rng.choice(EMOTIONAI_LABELS)
    return predict


# ═══════════════════════════════════════════════════════════
# Keyword baseline (existing text_analyzer_v3)
# ═══════════════════════════════════════════════════════════

_KEYWORD_TO_EA: dict[str, str] = {
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


def keyword_baseline(text: str) -> str:
    """Argmax over the 10 keyword-hit counts from text_analyzer_v3.

    Tie-breaks go to higher-priority emotions by insertion order in _KEYWORD_TO_EA.
    Empty-input fallback: returns "SURPRISE" (as Sara & Bouret novelty default).
    """
    from src.perception.text_analyzer import analyze_text
    signal = analyze_text(text)
    feats = signal.features or {}

    best_label = "SURPRISE"
    best_count = 0
    for hit_key, ea_label in _KEYWORD_TO_EA.items():
        c = int(feats.get(hit_key, 0))
        if c > best_count:
            best_count = c
            best_label = ea_label
    return best_label


# ═══════════════════════════════════════════════════════════
# Model-rates baseline (IntegratedBrainV2 → argmax emotion)
# ═══════════════════════════════════════════════════════════

# Keyword-to-drive translation (simplified): map keyword hit counts to the
# 9 process() input signals. This is a crude "perception" layer standing in
# for a proper text→signal pipeline.
_HIT_TO_DRIVE: dict[str, str] = {
    "fear_hits":        "threat",
    "rage_hits":        "frustration",
    "seeking_hits":     "reward",
    "sadness_hits":     "loss",
    "disgust_hits":     "contamination",
    "care_hits":        "social",
    "panic_grief_hits": "attachment_need",
    "play_hits":        "social",       # play also uses social signal
    "lust_hits":        "social",       # lust uses social+reward
    "surprise_hits":    "novelty",
}


# Model output emotion names (lowercase in IntegratedBrainV2 state dict) → EMOTIONAI_LABELS
_STATE_TO_EA: dict[str, str] = {
    "fear":        "FEAR",
    "rage":        "RAGE",
    "seeking":     "SEEKING",
    "sadness":     "SADNESS",
    "disgust":     "DISGUST",
    "care":        "CARE",
    "panic_grief": "PANIC_GRIEF",
    "play":        "PLAY",
    "lust":        "LUST",
    "surprise":    "SURPRISE",
}


@dataclass
class _ModelBrainCache:
    brain: object | None = None

_cache = _ModelBrainCache()


def _get_brain(use_adex: bool = False):
    """Cached brain instance — avoid rebuilding per prediction."""
    if _cache.brain is None:
        from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
        from src.brian2_circuits.shared_core_network import SharedCoreConfig
        _cache.brain = EmotionBrainV2(config=SharedCoreConfig(use_adex=use_adex))
    return _cache.brain


def model_rates_baseline(text: str, use_adex: bool = False) -> str:
    """Run IntegratedBrainV2 with text→drive translation, return argmax emotion.

    Text → keyword hits → drive signals → 1 trial → state emotion activations → argmax.
    This tests: "does the circuit add value over simple keyword matching?"
    """
    from src.perception.text_analyzer import analyze_text
    signal = analyze_text(text)
    feats = signal.features or {}

    # Aggregate hits into drives (crude linear combination, scaled to [0, 1]).
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
    result = brain.process(**drives)  # Returns EmotionStateV2 (dataclass, not dict).

    # Argmax over emotion attributes of EmotionStateV2.
    # When all activations tie at 0 (no input drive triggered any emotion gate),
    # fall back to class priors by using lowest starting best_val=0.0 and
    # returning a dedicated "unknown" label. For now we keep SURPRISE as the
    # fallback; tie-breaking by iteration order was exposed as biased during
    # Phase 9 SEEKING-gate fix audit (pre-existing argmax pathology).
    best_label, best_val = "SURPRISE", 0.0
    for state_attr, ea_label in _STATE_TO_EA.items():
        v = float(getattr(result, state_attr, 0.0))
        if v > best_val:   # strict > to avoid iteration-order bias on ties
            best_val = v
            best_label = ea_label
    return best_label


def reset_model_cache() -> None:
    """Clear cached brain (e.g., to switch Izh ↔ AdEx)."""
    _cache.brain = None


# ═══════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════

BASELINES: dict[str, Callable[[str], str]] = {
    "random":         make_random_baseline(seed=42),
    "keyword":        keyword_baseline,
    "model_rates":    model_rates_baseline,
}


if __name__ == "__main__":
    test_texts = [
        "I'm so scared right now",
        "I'm absolutely furious",
        "I won the lottery, so happy!",
        "I feel so sad today",
        "This is disgusting",
        "Please take care of yourself",
        "I want that so badly",
        "Haha that was hilarious",
        "Wow, unexpected!",
    ]
    print(f"{'text':<35s} random keyword model_rates")
    for t in test_texts:
        r = BASELINES["random"](t)
        k = BASELINES["keyword"](t)
        m = BASELINES["model_rates"](t)
        print(f"  {t[:34]:<35s} {r:<7s} {k:<9s} {m}")
