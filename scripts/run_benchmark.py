"""ベンチマーク実行スクリプト。モード比較レポートを生成する。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark_runner import run_benchmark, compare_modes


def main():
    print("=" * 60)
    print("  Emotion AI — 自動ベンチマーク")
    print("=" * 60)

    # ヒューリスティクスモード
    print("\n[1/3] ヒューリスティクスモード実行中...")
    h_report = run_benchmark(mode_label="heuristic")
    print(f"  結果: {h_report.passed}/{h_report.total_scenarios} 通過 "
          f"({h_report.pass_rate:.0%})")
    print(f"  一貫性: {h_report.avg_consistency:.3f}")
    print(f"  安全違反: {h_report.total_safety_violations}")

    for r in h_report.scenario_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"    [{status}] {r.name} (consistency={r.consistency_score:.2f})")

    # モード比較
    print("\n[2/3] モード比較実行中...")
    comparison = compare_modes("logs/benchmarks")

    print("\n[3/3] 比較結果:")
    c = comparison["comparison"]
    print(f"  通過率差: {c['pass_rate_diff']:+.1%}")
    print(f"  一貫性差: {c['consistency_diff']:+.3f}")
    print(f"  安全違反差: {c['safety_diff']:+d}")
    print(f"\n  レポート保存: logs/benchmarks/comparison.json")

    # 軌跡エクスポート
    print("\n  各シナリオの状態軌跡を保存中...")
    for r in h_report.scenario_results:
        if r.trajectory and r.trajectory.size > 0:
            path = Path(f"logs/benchmarks/trajectories/{r.name}.csv")
            r.trajectory.export_csv(path)
    print("  保存完了: logs/benchmarks/trajectories/")

    print(f"\n{'=' * 60}")
    print("  完了")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
