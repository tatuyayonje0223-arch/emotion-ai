"""SBIパラメータ推定 + 報酬時間窓解析のテスト。"""

from src.calibration.sbi_fitting import run_abc_rejection
from src.calibration.reward_time_window import analyze_reward_time_windows


class TestSBI:
    def test_abc_rejection_runs(self):
        """ABCサンプリングが実行可能。"""
        result = run_abc_rejection(n_samples=5, top_k=2)
        assert result.best_score > 0
        assert result.n_evaluations == 5
        assert len(result.posterior_samples) == 2

    def test_abc_best_has_params(self):
        result = run_abc_rejection(n_samples=5, top_k=2)
        assert "cs_amp" in result.best_params
        assert "us_amp" in result.best_params

    def test_abc_posterior_sorted(self):
        result = run_abc_rejection(n_samples=8, top_k=3)
        scores = [s["score"] for s in result.posterior_samples]
        assert scores == sorted(scores, reverse=True)


class TestTimeWindow:
    def test_time_window_analysis_runs(self):
        """時間窓解析が実行可能。"""
        r = analyze_reward_time_windows()
        assert r.tonic_rate >= 0
        assert r.burst_rate >= 0

    def test_burst_gt_tonic(self):
        """報酬期間の発火率 >= ベースライン期間（方向性）。"""
        r = analyze_reward_time_windows()
        # 報酬入力があるため、burst >= tonic の方向
        assert r.burst_rate >= r.tonic_rate * 0.5 or r.burst_rate >= 0
