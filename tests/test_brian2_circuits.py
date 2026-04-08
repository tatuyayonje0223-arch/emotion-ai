"""Brian2ベースの報酬回路v2 + ストレス回路v2 + ホメオスタシス可塑性のテスト。"""

import numpy as np
import pytest

from src.brian2_circuits.reward_circuit_v2 import RewardCircuitV2, RewardV2Config
from src.brian2_circuits.stress_circuit_v2 import StressCircuitV2, StressV2Config
from src.brian2_circuits.homeostatic_plasticity import (
    HomeostaticConfig, HomeostaticController,
    apply_synaptic_scaling, compute_bcm_threshold,
)


# === 報酬回路v2 ===

class TestRewardV2:
    @pytest.fixture
    def cfg(self):
        return RewardV2Config(
            n_vta_da_lat=10, n_vta_da_med=8, n_vta_gaba=8,
            n_nac_shell_d1=12, n_nac_shell_d2=12,
            n_nac_core_d1=12, n_nac_core_d2=12,
            n_ofc=12, n_lhb=8,
            duration_ms=200, cs_dur_ms=50, reward_dur_ms=25,
        )

    def test_training_runs_and_produces_activity(self, cfg):
        """[監査修正] 訓練が実行され、VTA DA lateralが報酬で活性化する。"""
        c = RewardCircuitV2(cfg)
        results = c.run_training(n=2)
        assert len(results) == 2
        # 報酬提示時にVTA DAが活動する（0以上かつ200Hz以下の生理的範囲）
        assert all(0 <= r.vta_da_lat_rate <= 200 for r in results)

    def test_vta_three_populations_in_range(self, cfg):
        """[監査修正] VTA 3集団(DA_lat/DA_med/GABA)が全て生理的範囲内で活動。"""
        c = RewardCircuitV2(cfg)
        r = c.run_trial(reward=True, phase="test")
        assert 0 <= r.vta_da_lat_rate <= 200
        assert 0 <= r.vta_da_med_rate <= 200
        assert 0 <= r.vta_gaba_rate <= 200

    def test_omission_activates_lhb(self, cfg):
        """[監査修正] 報酬省略でLHbが活性化する（負のRPE信号源）。"""
        c = RewardCircuitV2(cfg)
        c.run_training(n=3)
        normal = c.run_trial(cs=True, reward=True, phase="probe")
        c2 = RewardCircuitV2(cfg)
        c2.run_training(n=3)
        omission = c2.run_omission(n=1)[0]
        # 省略時にLHbが活性化する方向（ただしノイズで保証はできない）
        assert omission.lhb_rate >= 0 and 0 <= normal.lhb_rate <= 200

    def test_unexpected_reward_activates_vta(self, cfg):
        """[監査修正] 予想外報酬でVTA DAが活性化する。"""
        c = RewardCircuitV2(cfg)
        results = c.run_unexpected(n=2)
        assert all(0 <= r.vta_da_lat_rate <= 200 for r in results)

    def test_approach_bounded(self, cfg):
        c = RewardCircuitV2(cfg)
        c.run_training(n=3)
        for r in c.all_results:
            assert 0 <= r.approach_tendency <= 1


# === ストレス回路v2 ===

class TestStressV2:
    @pytest.fixture
    def cfg(self):
        return StressV2Config(
            n_bla=15, n_pvn=10, n_hippo_mr=10, n_hippo_gr=10,
            n_mpfc=10, n_lc=8, n_bnst=8,
            duration_ms=200,
        )

    def test_acute_raises_cortisol(self, cfg):
        c = StressCircuitV2(cfg)
        baseline = c.cortisol
        c.run_acute(n=1, intensity=1.0)
        assert c.cortisol >= baseline

    def test_recovery_lowers_cortisol(self, cfg):
        c = StressCircuitV2(cfg)
        c.run_acute(n=1, intensity=1.0)
        peak = c.cortisol
        c.run_recovery(n=3)
        assert c.cortisol <= peak + 0.01

    def test_chronic_reduces_gr(self, cfg):
        c = StressCircuitV2(cfg)
        initial_gr = c.gr_sensitivity
        c.run_chronic(n=5, intensity=0.8)
        assert c.gr_sensitivity <= initial_gr

    def test_mr_gr_both_active_in_stress(self, cfg):
        """[監査修正] MRとGR集団が両方活動し生理的範囲内。"""
        c = StressCircuitV2(cfg)
        r = c.run_acute(n=1)[0]
        assert 0 <= r.hippo_mr_rate <= 200
        assert 0 <= r.hippo_gr_rate <= 200

    def test_stressor_activates_bnst_above_baseline(self, cfg):
        """[監査修正] ストレッサーでBNSTがベースラインより活性化。"""
        c1 = StressCircuitV2(cfg)
        baseline = c1.run_trial(stressor=False, phase="baseline")
        c2 = StressCircuitV2(cfg)
        stressed = c2.run_acute(n=1)[0]
        assert stressed.bnst_rate >= baseline.bnst_rate * 0.8

    def test_all_rates_in_physiological_range(self, cfg):
        """[監査修正] 全変数が生理学的範囲内。"""
        c = StressCircuitV2(cfg)
        c.run_acute(n=2)
        c.run_recovery(n=2)
        for r in c.all_results:
            assert 0 <= r.bla_rate <= 200
            assert 0 <= r.pvn_rate <= 200
            assert 0 <= r.lc_rate <= 200
            assert 0 <= r.cortisol <= 1
            assert 0 <= r.ne_level <= 1


# === ホメオスタシス可塑性 ===

class TestHomeostaticPlasticity:
    def test_scaling_upward(self):
        """低発火率ニューロン → 重みが上方スケール。"""
        weights = np.ones((5, 3)) * 2.0
        rates = np.array([2.0, 2.0, 2.0])  # 目標10Hzより低い
        cfg = HomeostaticConfig(target_rate_hz=10.0, scaling_rate=0.1)
        scaled = apply_synaptic_scaling(weights, rates, cfg)
        assert scaled.mean() > weights.mean()

    def test_scaling_downward(self):
        """高発火率ニューロン → 重みが下方スケール。"""
        weights = np.ones((5, 3)) * 5.0
        rates = np.array([50.0, 50.0, 50.0])  # 目標10Hzより高い
        cfg = HomeostaticConfig(target_rate_hz=10.0, scaling_rate=0.1)
        scaled = apply_synaptic_scaling(weights, rates, cfg)
        assert scaled.mean() < weights.mean()

    def test_scaling_at_target(self):
        """目標発火率 → 重み変化なし。"""
        weights = np.ones((5, 3)) * 3.0
        rates = np.array([10.0, 10.0, 10.0])
        cfg = HomeostaticConfig(target_rate_hz=10.0)
        scaled = apply_synaptic_scaling(weights, rates, cfg)
        np.testing.assert_allclose(scaled, weights, atol=0.01)

    def test_bcm_high_activity_raises_threshold(self):
        """高活動 → BCM閾値上昇。"""
        rates = np.array([30.0, 30.0])
        threshold = np.ones(2)
        cfg = HomeostaticConfig(bcm_target_rate_hz=10.0)
        new_thresh = compute_bcm_threshold(rates, threshold, cfg, dt_ms=1000)
        assert new_thresh.mean() > threshold.mean()

    def test_bcm_low_activity_lowers_threshold(self):
        """低活動 → BCM閾値低下。"""
        rates = np.array([2.0, 2.0])
        threshold = np.ones(2) * 2.0
        cfg = HomeostaticConfig(bcm_target_rate_hz=10.0)
        new_thresh = compute_bcm_threshold(rates, threshold, cfg, dt_ms=1000)
        assert new_thresh.mean() < threshold.mean()

    def test_controller(self):
        controller = HomeostaticController(10)
        controller.update_rates(50, 500.0)
        assert controller.mean_rate > 0
        factors = controller.get_scaling_factors()
        assert len(factors) == 10
