"""GoEmotions dataset loader with offline sample fallback.

Full dataset: https://huggingface.co/datasets/go_emotions (requires `pip install datasets`)
Sample subset embedded below allows pipeline testing without network.

Each instance: {"text": str, "go_labels": list[str], "ea_labels": list[str], "is_mappable": bool}
Filters out instances where no GoEmotions label maps to EmotionAI (relief/neutral only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from phase9.emotion_mapping import (
    GOEMOTIONS_LABELS, PRIMARY_MAP,
    map_goemotions_to_emotionai, is_mappable,
)


@dataclass
class LabeledInstance:
    text: str
    go_labels: list[str]
    ea_labels: list[str]

    @property
    def primary_ea(self) -> str | None:
        """Single EmotionAI label for strict single-label eval. None if unmappable or multi."""
        return self.ea_labels[0] if len(self.ea_labels) == 1 else None


# Embedded sample for offline pipeline testing (40 instances, hand-curated to be
# unambiguous single-label-mappable where possible). NOT a substitute for real
# GoEmotions evaluation.
SAMPLE_INSTANCES: list[tuple[str, list[str]]] = [
    ("I'm so scared right now, my heart is racing.", ["fear"]),
    ("That noise made me jump, I'm terrified.", ["fear"]),
    ("I'm really nervous about tomorrow's exam.", ["nervousness"]),
    ("I can't stop worrying about the test.", ["nervousness"]),
    ("I'm absolutely furious about this!", ["anger"]),
    ("Don't touch my stuff, that makes me so mad.", ["anger"]),
    ("That's really annoying, please stop.", ["annoyance"]),
    ("Ugh, traffic again, this is so irritating.", ["annoyance"]),
    ("I won the lottery, I'm so happy!", ["joy"]),
    ("What a beautiful day, I love this.", ["joy"]),
    ("I can't wait for the concert!", ["excitement"]),
    ("This is going to be awesome!", ["excitement"]),
    ("Thank you so much for helping me.", ["gratitude"]),
    ("I really appreciate your kindness.", ["gratitude"]),
    ("I love you so much, my friend.", ["love"]),
    ("You mean the world to me.", ["love"]),
    ("I care about your wellbeing.", ["caring"]),
    ("Take care of yourself.", ["caring"]),
    ("I want to eat that cake so badly.", ["desire"]),
    ("I've been craving this all week.", ["desire"]),
    ("I feel so sad today.", ["sadness"]),
    ("My dog passed away, I can't stop crying.", ["sadness"]),
    ("I'm really disappointed with the result.", ["disappointment"]),
    ("I expected better from you.", ["disappointment"]),
    ("This is disgusting, I can't eat it.", ["disgust"]),
    ("That smell makes me want to vomit.", ["disgust"]),
    ("Losing my grandmother was the worst grief.", ["grief"]),
    ("The funeral was heartbreaking.", ["grief"]),
    ("I'm so embarrassed right now.", ["embarrassment"]),
    ("Please don't look, I'm blushing.", ["embarrassment"]),
    ("Wow, I didn't see that coming!", ["surprise"]),
    ("That was completely unexpected!", ["surprise"]),
    ("I wonder how this machine works.", ["curiosity"]),
    ("Tell me more, I want to know everything.", ["curiosity"]),
    ("I'm confused by these instructions.", ["confusion"]),
    ("I don't understand what's happening.", ["confusion"]),
    ("Haha that joke was hilarious!", ["amusement"]),
    ("This comedy is really funny.", ["amusement"]),
    ("What a relief that's over.", ["relief"]),          # unmappable
    ("The weather is mild today.", ["neutral"]),          # unmappable
]


def load_sample() -> list[LabeledInstance]:
    """Load embedded 40-instance sample (offline, no dependencies)."""
    out = []
    for text, go_labels in SAMPLE_INSTANCES:
        ea = map_goemotions_to_emotionai(go_labels)
        out.append(LabeledInstance(text=text, go_labels=go_labels, ea_labels=ea))
    return out


def load_goemotions_full(split: str = "validation") -> list[LabeledInstance]:
    """Load full GoEmotions via HuggingFace datasets library.

    Requires: `pip install datasets`. Splits: train / validation / test.
    Returns mappable-only instances (drops pure relief/neutral).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "datasets library not installed. Run: pip install datasets\n"
            "Or use load_sample() for offline pipeline testing."
        ) from e

    ds = load_dataset("go_emotions", "simplified", split=split)

    label_names = ds.features["labels"].feature.names
    instances = []
    for row in ds:
        text = row["text"]
        go_labels = [label_names[i] for i in row["labels"]]
        if not is_mappable(go_labels):
            continue
        ea_labels = map_goemotions_to_emotionai(go_labels)
        instances.append(LabeledInstance(text=text, go_labels=go_labels, ea_labels=ea_labels))
    return instances


def split_single_label(instances: Iterable[LabeledInstance]) -> list[LabeledInstance]:
    """Filter to single-EA-label instances (strict eval variant)."""
    return [inst for inst in instances if len(inst.ea_labels) == 1]


if __name__ == "__main__":
    sample = load_sample()
    single = split_single_label(sample)
    print(f"Total sample instances: {len(sample)}")
    print(f"Single-label (mappable) instances: {len(single)}")
    print(f"Unmappable filtered: {len(sample) - len([i for i in sample if i.ea_labels])}")
    from collections import Counter
    dist = Counter(i.primary_ea for i in single)
    print(f"\nLabel distribution (single-label subset):")
    for label, count in dist.most_common():
        print(f"  {label}: {count}")
