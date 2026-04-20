"""Phase 9.12 Coarse-grained Ekman 6-way emotion classification.

Ekman 1992 (the classical 6 basic emotions): anger, disgust, fear, joy, sadness, surprise.

Collapses:
- GoEmotions 27 → Ekman 6 (for ground truth)
- EmotionAI 10 → Ekman 6 (for model output)

Tests whether simpler granularity gives the neural model an advantage:
maybe 10-way was too fine and FEAR/RAGE/SADNESS etc. get conflated by LLM too.
"""
from __future__ import annotations


EKMAN_6: list[str] = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


# EmotionAI 10 → Ekman 6 collapse
EA_TO_EKMAN: dict[str, str] = {
    "FEAR":        "fear",
    "RAGE":        "anger",
    "SEEKING":     "joy",         # positive valence, reward-approach
    "SADNESS":     "sadness",
    "DISGUST":     "disgust",
    "CARE":        "joy",         # warm positive
    "PANIC_GRIEF": "sadness",     # grief-adjacent
    "PLAY":        "joy",
    "LUST":        "joy",         # positive, reward-adjacent
    "SURPRISE":    "surprise",
}


# GoEmotions 27 → Ekman 6
GO_TO_EKMAN: dict[str, str | None] = {
    "fear": "fear",
    "nervousness": "fear",
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "joy": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "optimism": "joy",
    "pride": "joy",
    "amusement": "joy",
    "love": "joy",
    "admiration": "joy",
    "approval": "joy",
    "caring": "joy",
    "desire": "joy",
    "relief": "joy",
    "sadness": "sadness",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "disgust": "disgust",
    "surprise": "surprise",
    "confusion": "surprise",
    "realization": "surprise",
    "curiosity": "surprise",
    "embarrassment": "sadness",   # contested; most map to self-directed negative
    "neutral": None,
}


def ea_to_ekman(ea_label: str) -> str | None:
    return EA_TO_EKMAN.get(ea_label)


def go_labels_to_ekman(go_labels: list[str]) -> list[str]:
    """Returns unique Ekman labels for multi-label GoEmotions instance."""
    out = set()
    for lbl in go_labels:
        ek = GO_TO_EKMAN.get(lbl)
        if ek:
            out.add(ek)
    return sorted(out)


if __name__ == "__main__":
    print(f"Ekman 6: {EKMAN_6}")
    print(f"\nEA → Ekman collapse:")
    from collections import Counter
    ea_dist = Counter(EA_TO_EKMAN.values())
    for ek in EKMAN_6:
        sources = [ea for ea, e in EA_TO_EKMAN.items() if e == ek]
        print(f"  {ek:<10s}: n_sources={ea_dist[ek]}  from: {sources}")

    print(f"\nGo → Ekman collapse:")
    go_dist = Counter(v for v in GO_TO_EKMAN.values() if v)
    for ek in EKMAN_6:
        sources = [g for g, e in GO_TO_EKMAN.items() if e == ek]
        print(f"  {ek:<10s}: n_sources={go_dist[ek]}  from: {sources}")
