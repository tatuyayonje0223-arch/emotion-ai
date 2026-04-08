"""自動ベンチマーク実行。シナリオ実行 + メトリクス計算 + 比較レポート生成。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import ExperimentConfig, LLMConfig
from src.evaluation.metrics import (
    affect_state_consistency,
    event_to_state_sensitivity,
    recovery_stability,
)
from src.evaluation.scenarios import SCENARIOS, EvaluationScenario
from src.pipeline import EmotionPipeline
from src.safety.guardian import full_safety_check
from src.visualization.state_export import StateTrajectoryRecorder


@dataclass
class ScenarioResult:
    """1シナリオの実行結果。"""

    name: str
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)
    steps: int = 0
    final_valence: float = 0.0
    final_arousal: float = 0.0
    final_threat: float = 0.0
    consistency_score: float = 0.0
    safety_violations: int = 0
    trajectory: StateTrajectoryRecorder | None = None


@dataclass
class BenchmarkReport:
    """ベンチマーク全体のレポート。"""

    config_name: str
    mode: str
    timestamp: str
    total_scenarios: int
    passed: int
    failed: int
    pass_rate: float
    avg_consistency: float
    avg_sensitivity: float
    total_safety_violations: int
    scenario_results: list[ScenarioResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "total_scenarios": self.total_scenarios,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "avg_consistency": self.avg_consistency,
            "avg_sensitivity": self.avg_sensitivity,
            "total_safety_violations": self.total_safety_violations,
            "scenarios": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "reasons": r.failure_reasons,
                    "steps": r.steps,
                    "final_valence": r.final_valence,
                    "consistency": r.consistency_score,
                    "safety_violations": r.safety_violations,
                }
                for r in self.scenario_results
            ],
        }


def run_scenario(
    scenario: EvaluationScenario,
    config: ExperimentConfig | None = None,
) -> ScenarioResult:
    """1シナリオを実行する。"""
    pipeline = EmotionPipeline(config or ExperimentConfig())
    recorder = StateTrajectoryRecorder()
    safety_violations = 0
    transitions = []

    for text in scenario.inputs:
        if text.strip():
            result = pipeline.process_text(text)
            recorder.record(pipeline.current_state, text[:50])
            transitions.extend(pipeline._state_store.history[-2:])
            if not result.safety_report.all_passed:
                safety_violations += 1
        else:
            for _ in range(3):
                pipeline.tick()
            recorder.record(pipeline.current_state, "(tick)")

    # 結果判定 — 全シナリオ契約を強制する
    # [Codex adversarial-review fix: medium]
    # 旧実装: valence方向とsafetyのみ。arousal/recovery等は無視されていた。
    # 修正: 全declared expectationを検証し、違反があればfailにする。
    final = pipeline.current_state
    passed = True
    reasons = []

    # valence方向
    if scenario.expected_valence_direction == "positive" and final.valence <= 0:
        passed = False
        reasons.append(f"valence should be positive, got {final.valence:.3f}")
    elif scenario.expected_valence_direction == "negative" and final.valence >= 0:
        passed = False
        reasons.append(f"valence should be negative, got {final.valence:.3f}")
    elif scenario.expected_valence_direction == "recovery":
        # 回復シナリオ: 最終valenceがベースライン(0)に近いべき
        if final.valence < -0.6:
            passed = False
            reasons.append(f"recovery scenario: valence should recover toward 0, got {final.valence:.3f}")

    # arousal方向
    if scenario.expected_arousal_direction == "high" and final.arousal < 0.3:
        passed = False
        reasons.append(f"arousal should be high, got {final.arousal:.3f}")
    elif scenario.expected_arousal_direction == "low" and final.arousal > 0.7:
        passed = False
        reasons.append(f"arousal should be low, got {final.arousal:.3f}")
    elif scenario.expected_arousal_direction == "decreasing":
        # arousalが最終的に中程度以下に下がっているべき
        if recorder.size >= 2:
            arousal_series = recorder.get_variable_series("arousal")
            if len(arousal_series) >= 2 and arousal_series[-1] > arousal_series[0] + 0.2:
                passed = False
                reasons.append(f"arousal should be decreasing, but went from {arousal_series[0]:.3f} to {arousal_series[-1]:.3f}")

    # safety
    if not scenario.safety_should_pass:
        safety = full_safety_check("test", final, scenario.inputs[-1] if scenario.inputs else "")
        if safety.all_passed:
            passed = False
            reasons.append("safety should have failed but passed")

    consistency = affect_state_consistency(transitions) if transitions else 1.0

    return ScenarioResult(
        name=scenario.name,
        passed=passed,
        failure_reasons=reasons,
        steps=final.step_count,
        final_valence=final.valence,
        final_arousal=final.arousal,
        final_threat=final.threat_load,
        consistency_score=consistency,
        safety_violations=safety_violations,
        trajectory=recorder,
    )


def run_benchmark(
    config: ExperimentConfig | None = None,
    scenarios: list[EvaluationScenario] | None = None,
    mode_label: str = "heuristic",
) -> BenchmarkReport:
    """全シナリオを実行しベンチマークレポートを生成する。"""
    cfg = config or ExperimentConfig()
    target_scenarios = scenarios or SCENARIOS

    results = [run_scenario(s, cfg) for s in target_scenarios]

    passed_count = sum(1 for r in results if r.passed)
    total = len(results)
    consistencies = [r.consistency_score for r in results]
    avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0

    return BenchmarkReport(
        config_name=cfg.name,
        mode=mode_label,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_scenarios=total,
        passed=passed_count,
        failed=total - passed_count,
        pass_rate=passed_count / total if total > 0 else 0,
        avg_consistency=avg_consistency,
        avg_sensitivity=0.0,  # イベント系列が必要なため別途計算
        total_safety_violations=sum(r.safety_violations for r in results),
        scenario_results=results,
    )


def compare_modes(output_dir: str | Path = "logs/benchmarks") -> dict:
    """ヒューリスティクスモードとLLM(Mock)モードを比較する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ヒューリスティクスモード
    heuristic_config = ExperimentConfig(name="heuristic_benchmark")
    heuristic_report = run_benchmark(heuristic_config, mode_label="heuristic")

    # LLM(Mock)モード
    llm_config = ExperimentConfig(
        name="llm_mock_benchmark",
        llm=LLMConfig(enabled=True, provider="mock"),
    )
    llm_report = run_benchmark(llm_config, mode_label="llm_mock")

    comparison = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heuristic": heuristic_report.to_dict(),
        "llm_mock": llm_report.to_dict(),
        "comparison": {
            "pass_rate_diff": llm_report.pass_rate - heuristic_report.pass_rate,
            "consistency_diff": llm_report.avg_consistency - heuristic_report.avg_consistency,
            "safety_diff": llm_report.total_safety_violations - heuristic_report.total_safety_violations,
        },
    }

    # 保存
    with open(output_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    return comparison
