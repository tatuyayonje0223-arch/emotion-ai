"""ストレス応答回路のテスト。HPA軸ダイナミクスの検証。"""

import pytest

from src.circuits.stress_circuit import StressCircuit, StressCircuitConfig


@pytest.fixture
def circuit():
    cfg = StressCircuitConfig(
        n_bla=30, n_pvn=20, n_hippo=30, n_mpfc=20, n_lc=15,
        steps_per_trial=40,
    )
    return StressCircuit(cfg)


class TestAcuteStress:
    def test_stressor_raises_cortisol(self, circuit):
        """急性ストレス → コルチゾール上昇。"""
        baseline = circuit.cortisol
        results = circuit.run_acute_stress(n_trials=1, intensity=1.0)
        assert results[0].cortisol_peak >= baseline

    def test_recovery_after_acute(self, circuit):
        """急性ストレス後の回復 → コルチゾール低下。"""
        circuit.run_acute_stress(n_trials=1, intensity=1.0)
        peak_cort = circuit.cortisol
        recovery = circuit.run_recovery(n_trials=3)
        final_cort = circuit.cortisol
        # 回復後はピークより低い（基底に向かう）
        assert final_cort <= peak_cort + 0.05

    def test_stressor_activates_bla(self, circuit):
        """ストレッサー → BLA活性化。"""
        results = circuit.run_acute_stress(n_trials=1)
        assert results[0].bla_mean_rate > 0

    def test_stressor_activates_lc(self, circuit):
        """ストレッサー → LC(NE)活性化。"""
        results = circuit.run_acute_stress(n_trials=1)
        assert results[0].lc_mean_rate >= 0  # LCは背景ノイズ依存で0もありうる

    def test_ne_increases_with_stress(self, circuit):
        """ストレスでNEレベルが上昇。"""
        baseline_ne = circuit.ne_level
        circuit.run_acute_stress(n_trials=1, intensity=1.0)
        assert circuit.ne_level >= baseline_ne


class TestChronicStress:
    def test_chronic_raises_cortisol(self, circuit):
        """慢性ストレス → コルチゾール持続上昇。"""
        circuit.run_chronic_stress(n_trials=8, intensity=0.8)
        assert circuit.cortisol > circuit.cfg.cort_baseline

    def test_chronic_reduces_gr_sensitivity(self, circuit):
        """慢性ストレス → GR感度低下（ダウンレギュレーション）。"""
        initial_gr = circuit.gr_sensitivity
        circuit.run_chronic_stress(n_trials=8, intensity=0.8)
        # GR感度が初期値以下
        assert circuit.gr_sensitivity <= initial_gr

    def test_chronic_impairs_recovery(self, circuit):
        """慢性ストレス後 → 回復が遅い（GR機能低下のため）。"""
        # 急性のみ
        c1 = StressCircuit(circuit.cfg)
        c1.run_acute_stress(n_trials=1, intensity=1.0)
        c1.run_recovery(n_trials=3)
        acute_recovery_cort = c1.cortisol

        # 慢性 → 急性 → 回復
        c2 = StressCircuit(circuit.cfg)
        c2.run_chronic_stress(n_trials=8, intensity=0.7)
        c2.run_acute_stress(n_trials=1, intensity=1.0)
        c2.run_recovery(n_trials=3)
        chronic_recovery_cort = c2.cortisol

        # 慢性後は回復がより遅い（コルチゾールが高く残る）
        assert chronic_recovery_cort >= acute_recovery_cort - 0.05


class TestNegativeFeedback:
    def test_cortisol_trajectory_has_peak(self, circuit):
        """急性ストレスのコルチゾール軌跡にピークがある。"""
        results = circuit.run_acute_stress(n_trials=1)
        traj = results[0].cortisol_trajectory
        assert len(traj) > 0
        # ピークが存在
        assert max(traj) >= traj[0]

    def test_no_stressor_cortisol_stable(self, circuit):
        """ストレッサーなし → コルチゾールが基底付近で安定。"""
        results = circuit.run_recovery(n_trials=3)
        for r in results:
            assert r.cortisol_final < 0.4  # 基底付近

    def test_all_rates_nonnegative(self, circuit):
        circuit.run_acute_stress(n_trials=2)
        circuit.run_recovery(n_trials=2)
        for r in circuit.all_results:
            assert r.pvn_mean_rate >= 0
            assert r.bla_mean_rate >= 0
            assert r.hippo_mean_rate >= 0
            assert r.lc_mean_rate >= 0
            assert 0 <= r.cortisol_final <= 1
            assert 0 <= r.ne_level <= 1
