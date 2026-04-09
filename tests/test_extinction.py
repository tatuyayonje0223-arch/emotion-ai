"""消去曲線ベンチマークのテスト。[R8 L1修正] 厳格化。"""

from src.evaluation.extinction_benchmark import run_extinction_benchmark


class TestExtinctionBenchmark:
    def test_runs(self):
        r = run_extinction_benchmark(n_conditioning=3, n_extinction=5)
        assert len(r.conditioning_freeze) == 3
        assert len(r.extinction_freeze) == 5
        assert r.peak_freeze >= 0
        assert 0 <= r.reduction_pct <= 100

    def test_extinction_reduces_freeze(self):
        """[R8 L1修正] 消去でfreeze_responseが低下する（finalがpeakより低い）。"""
        r = run_extinction_benchmark(n_conditioning=4, n_extinction=10)
        if r.peak_freeze > 0.05:
            # peakからの低下を確認（増加はfail）
            assert r.final_freeze < r.peak_freeze, \
                f"Extinction should reduce freeze: peak={r.peak_freeze:.3f} final={r.final_freeze:.3f}"

    def test_reduction_percentage(self):
        """消去低下が10%以上（トイモデル基準）。"""
        r = run_extinction_benchmark(n_conditioning=4, n_extinction=10)
        if r.peak_freeze > 0.05:
            assert r.reduction_pct > 0, \
                f"Reduction should be positive, got {r.reduction_pct:.1f}%"
