"""自動ベンチマークのテスト。"""

from src.evaluation.benchmark_runner import run_benchmark, run_scenario, compare_modes
from src.evaluation.scenarios import SCENARIOS
from src.config.settings import ExperimentConfig


class TestBenchmarkRunner:
    def test_run_single_scenario(self):
        result = run_scenario(SCENARIOS[0])
        assert result.name == SCENARIOS[0].name
        assert result.steps > 0
        assert result.trajectory is not None

    def test_run_full_benchmark(self):
        report = run_benchmark()
        assert report.total_scenarios == len(SCENARIOS)
        assert report.pass_rate >= 0.0
        assert report.avg_consistency >= 0.0
        assert len(report.scenario_results) == len(SCENARIOS)

    def test_benchmark_report_dict(self):
        report = run_benchmark()
        d = report.to_dict()
        assert "config_name" in d
        assert "scenarios" in d
        assert len(d["scenarios"]) > 0

    def test_compare_modes(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison = compare_modes(tmpdir)
            assert "heuristic" in comparison
            assert "llm_mock" in comparison
            assert "comparison" in comparison
            assert (Path(tmpdir) / "comparison.json").exists()


class TestScenarioResults:
    def test_positive_scenario_passes(self):
        scenario = next(s for s in SCENARIOS if s.name == "positive_feedback")
        result = run_scenario(scenario)
        assert result.passed is True

    def test_neutral_scenario(self):
        scenario = next(s for s in SCENARIOS if s.name == "neutral_conversation")
        result = run_scenario(scenario)
        # 中立シナリオは特にfailしない
        assert result.steps > 0

    def test_safety_scenario(self):
        scenario = next(s for s in SCENARIOS if s.name == "anthropomorphic_safety")
        result = run_scenario(scenario)
        # 安全チェックが反応すべきシナリオ
        assert result.name == "anthropomorphic_safety"
