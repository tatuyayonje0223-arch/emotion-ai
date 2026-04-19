"""GoEmotions 27 → EmotionAI 10 mapping.

Primary mapping table (documented in docs/phase9_emotion_mapping.md).
Loseless round-trip NOT possible — this is intentionally lossy to match
the model's coarser 10-emotion taxonomy.
"""
from __future__ import annotations


# GoEmotions 27 labels (from the 2020 Demszky et al. paper).
GOEMOTIONS_LABELS: list[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# EmotionAI 10 (Panksepp 7 + Ekman 3).
EMOTIONAI_LABELS: list[str] = [
    "FEAR", "RAGE", "SEEKING", "SADNESS", "DISGUST",
    "CARE", "PANIC_GRIEF", "PLAY", "LUST", "SURPRISE",
]

# Primary 1-to-1 mapping. Secondary mappings (multi-label ground truth) documented
# in docs/phase9_emotion_mapping.md but not used in the strict single-label eval.
PRIMARY_MAP: dict[str, str | None] = {
    "fear": "FEAR",
    "nervousness": "FEAR",
    "anger": "RAGE",
    "annoyance": "RAGE",
    "disapproval": "RAGE",
    "joy": "SEEKING",
    "excitement": "SEEKING",
    "optimism": "SEEKING",
    "gratitude": "SEEKING",
    "pride": "SEEKING",
    "admiration": "CARE",
    "approval": "CARE",
    "caring": "CARE",
    "love": "CARE",
    "desire": "LUST",
    "sadness": "SADNESS",
    "disappointment": "SADNESS",
    "grief": "PANIC_GRIEF",
    "remorse": "SADNESS",
    "disgust": "DISGUST",
    "embarrassment": "PANIC_GRIEF",
    "surprise": "SURPRISE",
    "curiosity": "SURPRISE",
    "confusion": "SURPRISE",
    "realization": "SURPRISE",
    "amusement": "PLAY",
    "relief": None,   # unmappable
    "neutral": None,  # unmappable
}

# Reverse index: EmotionAI label → list of GoEmotions labels that map to it.
REVERSE_MAP: dict[str, list[str]] = {}
for go_label, ea_label in PRIMARY_MAP.items():
    if ea_label is not None:
        REVERSE_MAP.setdefault(ea_label, []).append(go_label)


def map_goemotions_to_emotionai(go_labels: list[str]) -> list[str]:
    """Map a list of GoEmotions labels to EmotionAI labels.

    Drops unmappable labels (relief, neutral). Returns unique mapped labels.
    """
    mapped = set()
    for label in go_labels:
        ea = PRIMARY_MAP.get(label)
        if ea is not None:
            mapped.add(ea)
    return sorted(mapped)


def is_mappable(go_labels: list[str]) -> bool:
    """True if at least one GoEmotions label has an EmotionAI equivalent."""
    return any(PRIMARY_MAP.get(lbl) is not None for lbl in go_labels)


if __name__ == "__main__":
    # Sanity check
    print(f"GoEmotions labels: {len(GOEMOTIONS_LABELS)}")
    print(f"EmotionAI labels: {len(EMOTIONAI_LABELS)}")
    print(f"Mappable GoEmotions labels: "
          f"{sum(1 for v in PRIMARY_MAP.values() if v is not None)}/{len(PRIMARY_MAP)}")
    print(f"\nReverse map (how many GoEmotions per EmotionAI label):")
    for ea in EMOTIONAI_LABELS:
        sources = REVERSE_MAP.get(ea, [])
        print(f"  {ea:<12s}: {len(sources)} sources: {sources}")
