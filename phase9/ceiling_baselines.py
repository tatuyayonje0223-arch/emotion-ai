"""Reference ceiling baselines: trained logistic regression + Gemini LLM.

Completes the Phase 9 comparison picture:
- Random ~10% (lower bound)
- Keyword argmax ~28% (current unsupervised baseline)
- LR on keyword features (supervised ML ceiling for the same features)
- Neural model ~22%
- Gemini zero-shot (semantic LLM ceiling, untrained)
- Oracle (ground truth = 100%, reference only)

Gemini via env var GEMINI_API_KEY. Free tier rate limit ~15 req/min.
"""
from __future__ import annotations

import os
import time
from typing import Callable

from phase9.emotion_mapping import EMOTIONAI_LABELS


# ═══════════════════════════════════════════════════════════
# Logistic regression on keyword hit features
# ═══════════════════════════════════════════════════════════

_LR_CACHE: dict = {}


def _extract_hit_features(text: str) -> list[float]:
    """Extract 10 keyword hit counts from text_analyzer."""
    from src.perception.text_analyzer import analyze_text
    feats = (analyze_text(text).features or {})
    hit_keys = [f"{e.lower()}_hits" for e in EMOTIONAI_LABELS]
    return [float(feats.get(k, 0)) for k in hit_keys]


def fit_lr_baseline(train_instances) -> Callable[[str], str]:
    """Fit sklearn LogisticRegression on keyword hit features."""
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X = []
    y = []
    for inst in train_instances:
        true = inst.primary_ea
        if true is None:
            continue
        X.append(_extract_hit_features(inst.text))
        y.append(true)
    X = np.array(X)
    y = np.array(y)

    print(f"  LR training: {len(X)} instances, {X.shape[1]} features")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    lr.fit(X, y)
    _LR_CACHE["model"] = lr
    _LR_CACHE["classes"] = lr.classes_

    def predict(text: str) -> str:
        feat = [_extract_hit_features(text)]
        return str(lr.predict(feat)[0])
    return predict


# ═══════════════════════════════════════════════════════════
# Gemini zero-shot
# ═══════════════════════════════════════════════════════════

_GEMINI_CACHE: dict = {}

_GEMINI_PROMPT = """You are classifying a short text into one of 10 emotion categories
based on Panksepp + Ekman taxonomy.

Categories (answer with exactly one):
- FEAR: threat, anxiety, nervousness, terror
- RAGE: anger, annoyance, fury, irritation
- SEEKING: joy, excitement, anticipation, gratitude, optimism, pride
- SADNESS: sorrow, disappointment, remorse
- DISGUST: revulsion, contempt
- CARE: love, caring, admiration, approval, bonding
- PANIC_GRIEF: grief, separation distress, embarrassment
- PLAY: amusement, humor, play
- LUST: desire, sexual arousal
- SURPRISE: shock, curiosity, confusion, realization

Text: {text}

Answer with only the category name in ALL CAPS. No explanation."""


def make_gemini_baseline(model_name: str = "gemini-2.5-flash",
                          rate_limit_sec: float = 4.0):
    """Gemini zero-shot emotion classifier.

    rate_limit_sec: sleep between calls to respect free-tier quota.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("pip install google-generativeai")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY env var")

    genai.configure(api_key=api_key)
    if "model" not in _GEMINI_CACHE:
        _GEMINI_CACHE["model"] = genai.GenerativeModel(model_name)
        _GEMINI_CACHE["last_call"] = 0.0

    def predict(text: str) -> str:
        # Rate limit
        since_last = time.time() - _GEMINI_CACHE["last_call"]
        if since_last < rate_limit_sec:
            time.sleep(rate_limit_sec - since_last)

        prompt = _GEMINI_PROMPT.format(text=text[:500])  # cap text
        try:
            resp = _GEMINI_CACHE["model"].generate_content(prompt)
            out = resp.text.strip().upper()
            _GEMINI_CACHE["last_call"] = time.time()
            # Extract valid label
            for label in EMOTIONAI_LABELS:
                if label in out:
                    return label
            return "SURPRISE"  # fallback if response unparseable
        except Exception as e:
            _GEMINI_CACHE["last_call"] = time.time()
            print(f"  Gemini error: {e}")
            return "SURPRISE"
    return predict


if __name__ == "__main__":
    import sys
    from phase9.dataset import load_sample

    data = load_sample()[:5]

    print("Gemini smoke test:")
    fn_g = make_gemini_baseline(rate_limit_sec=2.0)
    for inst in data:
        if inst.primary_ea is None:
            continue
        pred = fn_g(inst.text)
        mark = "✓" if pred == inst.primary_ea else "✗"
        print(f"  {mark} '{inst.text[:40]}' true={inst.primary_ea} -> Gemini={pred}")
