"""EmotionBrainV2 デモ。10情動回路の動作を視覚的に確認する。

Usage:
    python scripts/emotion_brain_v2_demo.py
    python scripts/emotion_brain_v2_demo.py --text "I'm terrified"
"""

from __future__ import annotations

import argparse
import sys

from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2


def bar(value: float, width: int = 30, label: str = "") -> str:
    """0-1値をASCIIバーで表示。"""
    filled = int(value * width)
    return f"{label:15s} [{'█' * filled}{'░' * (width - filled)}] {value:.3f}"


def run_scenario(brain: EmotionBrainV2, name: str, **kwargs) -> None:
    """シナリオを実行して結果を表示。"""
    state = brain.process(**kwargs)
    print(f"\n{'=' * 55}")
    print(f"  Scenario: {name}")
    print(f"  Input: {', '.join(f'{k}={v}' for k, v in kwargs.items() if v > 0)}")
    print(f"{'=' * 55}")

    # 情動バー
    emotions = [
        ("FEAR", state.fear), ("RAGE", state.rage), ("SEEKING", state.seeking),
        ("SADNESS", state.sadness), ("DISGUST", state.disgust),
        ("CARE", state.care), ("PANIC_GRIEF", state.panic_grief),
        ("PLAY", state.play), ("LUST", state.lust), ("SURPRISE", state.surprise),
    ]
    emotions.sort(key=lambda x: x[1], reverse=True)

    for name, val in emotions:
        marker = " ◄" if val == max(e[1] for e in emotions) and val > 0 else ""
        print(f"  {bar(val, 30, name)}{marker}")

    print(f"\n  Valence:  {state.valence:+.3f}  {'(positive)' if state.valence > 0 else '(negative)' if state.valence < 0 else '(neutral)'}")
    print(f"  Arousal:  {state.arousal:.3f}")
    print(f"  Dominant: {state.dominant_emotion}")
    print(f"  Neurons:  {state.spiking_neurons} spiking")


def main():
    parser = argparse.ArgumentParser(description="EmotionBrainV2 Demo")
    parser.add_argument("--text", type=str, help="Process text input via IntegratedBrainV2")
    args = parser.parse_args()

    if args.text:
        from src.brian2_circuits.integrated_brain_v2 import IntegratedBrainV2
        brain = IntegratedBrainV2()
        result = brain.process(args.text)
        print(f"\nInput: \"{args.text}\"")
        emotions = result.emotion_state.get("emotions", {})
        for name, val in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {bar(val, 30, name.upper())}")
        print(f"\n  Valence: {result.readout.valence:+.3f}")
        print(f"  Arousal: {result.readout.arousal:.3f}")
        print(f"  Dominant: {result.emotion_state.get('dominant_emotion', 'unknown')}")
        return

    print("=" * 55)
    print("  EmotionBrainV2 — 10 Emotion Circuit Demo")
    print("  232 verified papers | 570 spiking neurons")
    print("=" * 55)

    brain = EmotionBrainV2()
    print(f"\n  Populations: {len(brain.population_names)}")
    print(f"  Spiking neurons: {brain.total_neurons}")

    # 10情動シナリオ
    scenarios = [
        ("Fear (threat)", {"threat": 0.8, "pain": 0.2}),
        ("Rage (frustration)", {"frustration": 0.8, "threat": 0.2}),
        ("Seeking (reward)", {"reward": 0.8, "novelty": 0.3}),
        ("Sadness (loss)", {"loss": 0.8, "social": 0.0}),
        ("Disgust (contamination)", {"contamination": 0.8}),
        ("Care (social bonding)", {"social": 0.8, "attachment_need": 0.6}),
        ("Panic/Grief (separation)", {"loss": 0.7, "attachment_need": 0.8, "social": 0.0}),
        ("Play (social joy)", {"social": 0.7, "reward": 0.5, "novelty": 0.4}),
        ("Surprise (novelty)", {"novelty": 0.9}),
        ("Joy (reward+social)", {"reward": 0.9, "social": 0.8, "novelty": 0.3}),
    ]

    for name, kwargs in scenarios:
        run_scenario(brain, name, **kwargs)

    print(f"\n{'=' * 55}")
    print("  Demo complete. All 10 emotions demonstrated.")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
