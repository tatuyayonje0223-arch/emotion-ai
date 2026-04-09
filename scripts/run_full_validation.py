"""全回路統合検証レポート。SBI + 定量検証 + 消去曲線。"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.fear_quantitative import validate_fear_circuit
from src.calibration.reward_quantitative import validate_reward_circuit
from src.calibration.stress_quantitative import validate_stress_circuit
from src.calibration.calibrated_configs import CALIBRATED_FEAR_CONFIG
from src.calibration.reward_time_window import analyze_reward_time_windows
from src.evaluation.extinction_benchmark import run_extinction_benchmark


def main():
    print("=" * 65)
    print("  EmotionAI Full Validation Report")
    print("=" * 65)

    # Fear
    print("\n--- Fear Circuit (SBI-calibrated) ---")
    fr = validate_fear_circuit(CALIBRATED_FEAR_CONFIG, n_conditioning=3, n_extinction=2)
    for d in fr.details[:6]: print(f"  {d}")
    print(f"  Score: {fr.score:.3f}")

    # Extinction
    print("\n--- Extinction Benchmark ---")
    er = run_extinction_benchmark(n_conditioning=4, n_extinction=8)
    print(f"  Peak freeze: {er.peak_freeze:.3f}")
    print(f"  Final freeze: {er.final_freeze:.3f}")
    print(f"  Reduction: {er.reduction_pct:.1f}%")
    print(f"  50% trial: {er.trials_to_50pct}")
    print(f"  Passed: {er.passed}")

    # Reward (time window)
    print("\n--- Reward Circuit (time-window) ---")
    rr = validate_reward_circuit()
    for d in rr.details: print(f"  {d}")
    print(f"  Score: {rr.score:.3f}")

    tw = analyze_reward_time_windows()
    print(f"  Tonic: {tw.tonic_rate:.1f}Hz  Burst: {tw.burst_rate:.1f}Hz  Ratio: {tw.burst_tonic_ratio:.2f}x")

    # Stress
    print("\n--- Stress Circuit ---")
    sr = validate_stress_circuit()
    for d in sr.details: print(f"  {d}")
    print(f"  Score: {sr.score:.3f}")

    # Summary
    avg = (fr.score + rr.score + sr.score) / 3
    print(f"\n{'=' * 65}")
    print(f"  Fear: {fr.score:.3f}  Reward: {rr.score:.3f}  Stress: {sr.score:.3f}")
    print(f"  Average: {avg:.3f}")
    print(f"  Extinction: {'PASS' if er.passed else 'FAIL'} ({er.reduction_pct:.0f}% reduction)")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
