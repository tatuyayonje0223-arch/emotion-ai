"""EmotionBrain E2Eデモ。恐怖条件付け→睡眠→消去→再テストの完全プロトコル。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brian2_circuits.integrated_brain import EmotionBrain


def _bar(v: float, w: int = 15) -> str:
    f = int(max(0, min(1, (v + 1) / 2 if v < 0 else v)) * w)
    return "\u2588" * f + "\u2591" * (w - f)


def _show(result, label: str = "") -> None:
    r = result.readout
    nm = result.neuromodulation
    print(f"  {'['+label+'] ' if label else ''}"
          f"val={r.valence:+.2f} aro={r.arousal:.2f} thr={r.threat_load:.2f} "
          f"rew={r.reward_drive:.2f} | "
          f"eCB={nm.get('ecb_extinction',0):.2f} ACh={nm.get('ach_nbm',0):.2f} "
          f"theta={nm.get('theta_coherence',0):.2f} spine={nm.get('spine_density',1):.2f}")


def main():
    brain = EmotionBrain()

    print("=" * 70)
    print("  EmotionBrain E2E Demo")
    print("  Hybrid(40K neurons) + eCB + ACh + Theta + Sleep Replay")
    print("=" * 70)

    # Phase 1: Baseline
    print("\n--- Phase 1: Baseline ---")
    for text in ["Hello.", "How are you today?"]:
        r = brain.process(text)
        _show(r, "baseline")

    # Phase 2: Fear Conditioning
    print("\n--- Phase 2: Fear Conditioning (5 trials) ---")
    for i in range(5):
        r = brain.process(f"Danger! Attack! Threat! Pain! (trial {i+1})")
        _show(r, f"CS+US {i+1}")

    print(f"  Memories encoded: {brain.memory_count}")

    # Phase 3: Sleep (memory consolidation)
    print("\n--- Phase 3: Sleep (3 cycles) ---")
    sleep_results = brain.sleep(n_cycles=3)
    for sr in sleep_results:
        print(f"  Cycle {sr['cycle']}: replayed={len(sr['replayed'])} "
              f"consolidated={len(sr['consolidated'])} scaled={sr['scaling_applied']}")
    stats = brain._sleep_engine.get_memory_stats()
    print(f"  Stats: {stats['consolidated']}/{stats['count']} consolidated, "
          f"spines={stats['spine_density']:.2f}")

    # Phase 4: Post-sleep test
    print("\n--- Phase 4: Post-sleep Fear Test ---")
    r = brain.process("Danger! Threat approaching!")
    _show(r, "post-sleep fear")

    # Phase 5: Extinction
    print("\n--- Phase 5: Extinction (5 trials) ---")
    brain.set_extinction_mode(True)
    for i in range(5):
        r = brain.process(f"The threat signal is present but nothing happens. (ext {i+1})")
        _show(r, f"extinction {i+1}")
    brain.set_extinction_mode(False)

    # Phase 6: Extinction test
    print("\n--- Phase 6: Post-extinction Test ---")
    r = brain.process("Danger signal again.")
    _show(r, "post-extinction")

    # Phase 7: Social/reward
    print("\n--- Phase 7: Social and Reward ---")
    r = brain.process("Thank you everyone! Great team! Happy together!")
    _show(r, "social+reward")

    print(f"\n{'=' * 70}")
    print(f"  Total steps: {brain._step}")
    print(f"  Memories: {brain.memory_count} (consolidated: {brain.consolidated_memories})")
    print(f"  Virtual neurons: ~40,000")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
