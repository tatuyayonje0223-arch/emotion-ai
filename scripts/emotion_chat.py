"""EmotionBrain + LLM 統合対話CLI。

脳モデルが「どう感じるか」を計算し、LLMが「どう言うか」を生成する。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brian2_circuits.emotion_llm_bridge import EmotionLLMBridge
from src.llm.provider import get_best_provider, MockProvider


def _bar(v: float, w: int = 10) -> str:
    f = int(max(0, min(1, (v + 1) / 2 if v < 0 else v)) * w)
    return "\u2588" * f + "\u2591" * (w - f)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="MockProvider (API不要)")
    args = parser.parse_args()

    provider = MockProvider() if args.mock else get_best_provider()
    bridge = EmotionLLMBridge(provider=provider)

    print("=" * 55)
    print("  EmotionBrain + LLM Chat")
    print(f"  LLM: {provider.name}")
    print("  ~40K neurons + sleep replay + neuromodulation")
    print("=" * 55)
    print("  /sleep  — 睡眠リプレイ")
    print("  /ext    — 消去モード切替")
    print("  /reset  — リセット")
    print("  /quit   — 終了")
    print("=" * 55)

    ext_mode = False

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
                print(f"  Sleep cycle {r['cycle']}: replayed={len(r['replayed'])}")
            continue
        if text == "/ext":
            ext_mode = not ext_mode
            bridge.set_extinction_mode(ext_mode)
            print(f"  Extinction mode: {'ON' if ext_mode else 'OFF'}")
            continue
        if text == "/reset":
            bridge.reset()
            print("  Reset.")
            continue

        result = bridge.chat(text)

        if result.brain_result.blocked:
            print("\n  [BLOCKED]")
            continue

        # LLM応答
        print(f"\nAI> {result.llm_response}")

        # 脳状態サマリー
        r = result.brain_result.readout
        nm = result.brain_result.neuromodulation
        print(f"  val={r.valence:+.2f} thr={r.threat_load:.2f} rew={r.reward_drive:.2f} "
              f"| ACh={nm.get('ach_nbm',0):.2f} eCB={nm.get('ecb_extinction',0):.2f} "
              f"theta={nm.get('theta_coherence',0):.2f}")


if __name__ == "__main__":
    main()
