"""消去曲線ベンチマークのテスト。"""

from src.evaluation.extinction_benchmark import run_extinction_benchmark


class TestExtinctionBenchmark:
    def test_runs(self):
        r = run_extinction_benchmark(n_conditioning=3, n_extinction=5)
        assert len(r.conditioning_freeze) == 3
        assert len(r.extinction_freeze) == 5
        assert r.peak_freeze >= 0
        assert 0 <= r.reduction_pct <= 100

    def test_extinction_reduces_freeze(self):
        """消去でfreeze_responseが低下方向。"""
        r = run_extinction_benchmark(n_conditioning=3, n_extinction=8)
        if r.peak_freeze > 0.05:
            assert r.final_freeze <= r.peak_freeze + 0.05
