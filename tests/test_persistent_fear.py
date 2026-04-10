"""永続型恐怖回路のテスト。真のSTDP学習を検証。"""

import brian2
brian2.prefs.codegen.target = "numpy"
from brian2 import start_scope
import pytest

from src.brian2_circuits.persistent_fear_circuit import PersistentFearCircuit


@pytest.fixture(autouse=True)
def fresh_scope():
    start_scope()
    yield


class TestPersistentFearCircuit:
    def test_runs(self):
        c = PersistentFearCircuit()
        r = c.run_trial(cs=True, us=True, phase="conditioning")
        assert r.cem_rate >= 0
        assert 0 <= r.freeze_response <= 1.0

    def test_multiple_trials(self):
        """複数試行がNetwork永続で動作。"""
        c = PersistentFearCircuit()
        for i in range(3):
            c.run_trial(cs=True, us=True, phase="conditioning", trial_num=i)
        assert len(c.all_results) == 3

    def test_stdp_weight_increases_with_conditioning(self):
        """[F-04] 条件付けでSTDP重みが初期値から増加する（真のLTP）。"""
        c = PersistentFearCircuit()
        w0 = c.la_ba_weight
        c.run_conditioning(n=3)
        w3 = c.la_ba_weight
        assert w3 > w0, f"STDP weight should increase: {w0:.4f} → {w3:.4f}"

    def test_conditioning_then_extinction(self):
        c = PersistentFearCircuit()
        cond = c.run_conditioning(n=2)
        ext = c.run_extinction(n=3)
        assert len(c.all_results) == 5

    def test_no_saturation(self):
        c = PersistentFearCircuit()
        c.run_conditioning(n=2)
        for r in c.all_results:
            assert r.cem_rate <= 200

    def test_weight_converges(self):
        """STDP重みが数試行で収束する（発散しない）。"""
        c = PersistentFearCircuit()
        c.run_conditioning(n=5)
        weights = [r.cs_la_weight for r in c.all_results]
        # 最後の2試行で重みが安定（差が小さい）
        assert abs(weights[-1] - weights[-2]) < 1.0, \
            f"Weight should converge: {weights}"
