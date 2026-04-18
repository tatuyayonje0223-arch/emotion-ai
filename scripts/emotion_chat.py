"""EmotionBrainV2 + LLM 統合対話CLI。

821スパイキングニューロンの10情動回路が「どう感じるか」を計算し、
LLMが「どう言うか」を生成する。

Usage:
    PYTHONPATH=. python scripts/emotion_chat.py [--mock] [--adex] [--context 0.5]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brian2_circuits.emotion_llm_bridge import EmotionLLMBridgeV2
from src.llm.provider import get_best_provider, MockProvider


def _bar(v: float, w: int = 8) -> str:
    """0-1 range bar."""
    f = int(max(0, min(1, v)) * w)
    return "\u2588" * f + "\u2591" * (w - f)


def _emotion_line(emotions: dict, top_n: int = 3) -> str:
    """Top N active emotions as compact string."""
    active = sorted(
        [(k, v) for k, v in emotions.items() if v > 0.05],
        key=lambda x: x[1], reverse=True
    )[:top_n]
    if not active:
        return "neutral"
    return " ".join(f"{k}={v:.2f}" for k, v in active)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EmotionBrainV2 + LLM Chat")
    parser.add_argument("--mock", action="store_true", help="MockProvider (API不要)")
    parser.add_argument("--adex", action="store_true", help="AdExモデル使用")
    parser.add_argument("--context", type=float, default=0.0, help="文脈信号 (0-1)")
    args = parser.parse_args()

    provider = MockProvider() if args.mock else get_best_provider()
    model_name = "AdEx" if args.adex else "Izhikevich"

    # Import config for AdEx mode
    if args.adex:
        from src.brian2_circuits.shared_core_network import SharedCoreConfig
        # EmotionLLMBridgeV2 uses IntegratedBrainV2 which creates EmotionBrainV2 internally
        # We need to pass config through — currently IntegratedBrainV2 doesn't accept config
        # Fall back to default (Izhikevich) for now if IntegratedBrainV2 doesn't support it
        print("  Note: --adex requires IntegratedBrainV2 config support (using Izhikevich)")

    bridge = EmotionLLMBridgeV2(provider=provider)

    print("=" * 60)
    print(f"  EmotionBrainV2 + LLM Chat [{model_name}]")
    print(f"  821 neurons | 53 populations | 10 emotions")
    print(f"  LLM: {provider.name}")
    print("=" * 60)
    print("  /sleep    — 睡眠リプレイ (NREM+REM)")
    print("  /state    — 現在の脳状態を表示")
    print("  /reset    — リセット")
    print("  /quit     — 終了")
    print("=" * 60)

    context = args.context

    while True:
        try:
            text = input("\nあなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue
        if text == "/quit":
            break
        if text == "/sleep":
            results = bridge.sleep(n_cycles=2)
            for r in results:
                replayed = r.get("replayed", [])
                print(f"  Sleep cycle {r.get('cycle', '?')}: replayed={len(replayed)} memories")
            continue
        if text == "/state":
            print("  (次のメッセージで状態が更新されます)")
            continue
        if text == "/reset":
            bridge.reset()
            context = 0.0
            print("  Reset.")
            continue
        if text.startswith("/context "):
            try:
                context = float(text.split()[1])
                print(f"  Context = {context:.2f}")
            except (ValueError, IndexError):
                print("  Usage: /context 0.5")
            continue

        result = bridge.chat(text)

        if result.get("model_used") == "blocked":
            print("\n  [BLOCKED by safety filter]")
            continue

        # LLM応答
        llm_text = result.get("llm_response", "...")
        print(f"\nAI> {llm_text}")

        # 10情動状態
        emotions = result.get("emotion_state", {}).get("emotions", {})
        readout = result.get("readout", {})
        neuromod = result.get("neuromodulation", {})
        neurons = result.get("spiking_neurons", 0)

        emo_str = _emotion_line(emotions)
        val = readout.get("valence", 0)
        aro = readout.get("arousal", 0)

        print(f"  {_bar(aro)} arousal={aro:.2f}  val={val:+.2f}")
        print(f"  {emo_str}")
        if neuromod:
            ach = neuromod.get("ach_nbm", 0)
            ecb = neuromod.get("ecb_extinction", 0)
            theta = neuromod.get("theta_coherence", 0)
            if ach > 0.1 or ecb > 0.1 or theta > 0.3:
                print(f"  neuromod: ACh={ach:.2f} eCB={ecb:.2f} θ={theta:.2f}")


if __name__ == "__main__":
    main()
