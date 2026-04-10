"""自発的回復シナリオのテスト。"""

from src.evaluation.spontaneous_recovery import run_spontaneous_recovery


class TestSpontaneousRecovery:
    def test_runs(self):
        r = run_spontaneous_recovery(n_conditioning=3, n_extinction=5, n_rest_trials=3, n_test=2)
        assert len(r.conditioning_freeze) == 3
        assert len(r.extinction_freeze) == 5
        assert len(r.recovery_freeze) == 2
        assert 0 <= r.recovery_ratio <= 2.0

    def test_conditioning_produces_freeze(self):
        """条件付けでfreezeが発生する。"""
        r = run_spontaneous_recovery(n_conditioning=3, n_extinction=3, n_rest_trials=2, n_test=2)
        assert r.peak_freeze > 0

    def test_protocol_completes(self):
        """全プロトコル（条件付け→消去→休息→テスト）が完了する。"""
        r = run_spontaneous_recovery(n_conditioning=2, n_extinction=3, n_rest_trials=2, n_test=2)
        total = len(r.conditioning_freeze) + len(r.extinction_freeze) + len(r.recovery_freeze)
        assert total == 7  # 2 + 3 + 2
