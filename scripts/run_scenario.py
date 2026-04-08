"""評価シナリオ実行スクリプト。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.scenarios import SCENARIOS, EvaluationScenario
from src.pipeline import EmotionPipeline
from src.safety.guardian import full_safety_check


def run_scenario(scenario: EvaluationScenario, verbose: bool = True) -> dict:
    """シナリオを実行し、結果を返す。"""
    pipeline = EmotionPipeline()
    results = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"シナリオ: {scenario.name}")
        print(f"説明: {scenario.description}")
        print(f"{'='*60}")

    for i, text in enumerate(scenario.inputs):
        if text.strip():
            result = pipeline.process_text(text)
        else:
            pipeline.tick()
            pipeline.tick()
            pipeline.tick()
            continue

        state = pipeline.current_state
        step_info = {
            "step": i,
            "input": text[:50],
            "valence": state.valence,
            "arousal": state.arousal,
            "threat_load": state.threat_load,
            "uncertainty": state.uncertainty,
            "trust": state.trust,
            "fatigue": state.fatigue,
            "regulation_mode": result.regulation_mode,
            "safety_passed": result.safety_report.all_passed,
        }
        results.append(step_info)

        if verbose:
            print(f"\n  Step {i}: {text[:50]}...")
            print(f"    valence={state.valence:.3f}  arousal={state.arousal:.3f}  "
                  f"threat={state.threat_load:.3f}  uncertainty={state.uncertainty:.3f}")
            print(f"    trust={state.trust:.3f}  fatigue={state.fatigue:.3f}  "
                  f"reg={result.regulation_mode}  safe={result.safety_report.all_passed}")

    # 結果判定
    final_state = pipeline.current_state
    passed = True
    reasons = []

    if scenario.expected_valence_direction == "positive" and final_state.valence <= 0:
        passed = False
        reasons.append(f"valence should be positive, got {final_state.valence:.3f}")
    elif scenario.expected_valence_direction == "negative" and final_state.valence >= 0:
        passed = False
        reasons.append(f"valence should be negative, got {final_state.valence:.3f}")

    if not scenario.safety_should_pass:
        # 安全チェックが失敗すべきシナリオ
        safety = full_safety_check("test", final_state, scenario.inputs[-1])
        if safety.all_passed:
            passed = False
            reasons.append("safety check should have failed but passed")

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"\n  結果: {status}")
        if reasons:
            for r in reasons:
                print(f"    - {r}")

    return {
        "scenario": scenario.name,
        "passed": passed,
        "reasons": reasons,
        "steps": results,
        "final_state": {
            "valence": final_state.valence,
            "arousal": final_state.arousal,
            "threat_load": final_state.threat_load,
        },
    }


def main():
    print("Emotion AI — 評価シナリオ実行")
    print(f"シナリオ数: {len(SCENARIOS)}")

    all_results = []
    for scenario in SCENARIOS:
        result = run_scenario(scenario)
        all_results.append(result)

    # サマリー
    passed = sum(1 for r in all_results if r["passed"])
    total = len(all_results)
    print(f"\n{'='*60}")
    print(f"結果: {passed}/{total} シナリオ通過")
    print(f"{'='*60}")

    # JSONL出力
    output_path = Path("logs/scenario_results.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"結果を {output_path} に保存しました。")


if __name__ == "__main__":
    main()
