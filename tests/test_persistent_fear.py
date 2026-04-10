"""永続型恐怖回路のテスト。Brian2 Networkが試行間で保持されることを検証。"""

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
        """複数試行がクラッシュせず実行できる（Network永続）。"""
        c = PersistentFearCircuit()
        for i in range(3):
            r = c.run_trial(cs=True, us=True, phase="conditioning", trial_num=i)
        assert len(c.all_results) == 3

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
