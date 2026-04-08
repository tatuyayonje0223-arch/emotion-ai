"""ハイブリッド脳モード対話CLI。

スパイキング(恐怖/報酬/ストレス) + mean-field(島皮質/ACC/dlPFC/海馬)
の統合脳モデルでリアルタイム対話。~40,000仮想ニューロン。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.neurocircuit.neuro_pipeline import NeuroPipeline


def _bar(v: float, w: int = 12) -> str:
    f = int(max(0, min(1, v)) * w)
    return "\u2588" * f + "\u2591" * (w - f)


def main():
    pipeline = NeuroPipeline(backend="hybrid")

    print("=" * 60)
    print("  Emotion AI - Hybrid Brain")
    print("  Spiking(fear/reward/stress) + MeanField(insula/ACC/dlPFC/hippo)")
    print("  ~40,000 virtual neurons")
    print("=" * 60)
    print("  /state /quit /reset")
    print("=" * 60)

    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            pipeline.reset()
            print("  Reset.")
            continue

        result = pipeline.process_text(text)

        if result.blocked:
            print(f"  [BLOCKED] {result.safety_report.block_reason}")
            continue

        r = result.readout
        print(f"\n  valence    {r.valence:+.3f} {_bar((r.valence+1)/2)}")
        print(f"  arousal     {r.arousal:.3f} {_bar(r.arousal)}")
        print(f"  threat      {r.threat_load:.3f} {_bar(r.threat_load)}")
        print(f"  reward      {r.reward_drive:.3f} {_bar(r.reward_drive)}")
        print(f"  control     {r.cognitive_control:.3f} {_bar(r.cognitive_control)}")

        if result.region_activities:
            active = {k: v for k, v in sorted(result.region_activities.items()) if v > 0.5}
            if active:
                parts = [f"{k}={v:.0f}" for k, v in active.items()]
                print(f"  active: {', '.join(parts)}")

        neurons = result.body_state.get("total_virtual_neurons", 0)
        if neurons:
            print(f"  neurons: {int(neurons):,}")

        print(f"  policy: {result.response_policy.tone}")


if __name__ == "__main__":
    main()
