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

    def test_extinction_decreases_weight(self):
        """[LTD修正] 消去でSTDP重みが減少する（脱増強=depotentiation）。"""
        c = PersistentFearCircuit(decay_rate=0.05)
        # 条件付け: 重みを増加させる
        c.run_conditioning(n=5)
        w_after_cond = c.la_ba_weight
        assert w_after_cond > 0, f"Weight should be positive after conditioning: {w_after_cond}"

        # 消去: 重みが減少するはず
        c.run_extinction(n=10)
        w_after_ext = c.la_ba_weight
        assert w_after_ext < w_after_cond, \
            f"Weight should decrease during extinction: {w_after_cond:.4f} → {w_after_ext:.4f}"

    def test_decay_rate_zero_no_extinction(self):
        """decay_rate=0 では消去時に重みが減少しない（event-driven LTDのみ）。"""
        c = PersistentFearCircuit(decay_rate=0.0)
        c.run_conditioning(n=3)
        w_cond = c.la_ba_weight

        c.run_extinction(n=5)
        w_ext = c.la_ba_weight
        # decay_rate=0 では depotentiation なし。
        # event-driven LTDも効かない（BA tonic問題）ので重みは維持or微増。
        assert w_ext >= w_cond * 0.95, \
            f"With decay_rate=0, weight should not decrease significantly: {w_cond:.4f} → {w_ext:.4f}"

    def test_decay_rate_controllable(self):
        """decay_rateプロパティで減衰率を制御できる。"""
        c = PersistentFearCircuit(decay_rate=0.05)
        assert c.decay_rate == 0.05
        c.decay_rate = 0.10
        assert c.decay_rate == 0.10
        c.decay_rate = -0.5  # clamped to 0
        assert c.decay_rate == 0.0
        c.decay_rate = 1.5  # clamped to 1
        assert c.decay_rate == 1.0

    def test_higher_decay_faster_extinction(self):
        """高いdecay_rateはより速い消去をもたらす。"""
        results = {}
        for rate in [0.03, 0.10]:
            c = PersistentFearCircuit(decay_rate=rate)
            c.run_conditioning(n=5)
            c.run_extinction(n=5)
            results[rate] = c.la_ba_weight

        # 高い減衰率 → より低い重み
        assert results[0.10] < results[0.03], \
            f"Higher decay should give lower weight: rate=0.03→{results[0.03]:.4f}, rate=0.10→{results[0.10]:.4f}"
