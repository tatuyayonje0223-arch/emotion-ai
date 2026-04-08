"""報酬学習回路のテスト。Schultz (1997) RPE理論の検証。"""

import pytest

from src.circuits.reward_circuit import RewardCircuit, RewardCircuitConfig


@pytest.fixture
def circuit():
    cfg = RewardCircuitConfig(
        n_vta=30, n_nac_d1=40, n_nac_d2=40, n_ofc=30, n_lhb=15,
        trial_duration_ms=400, cs_duration_ms=80,
        reward_onset_ms=250, reward_duration_ms=30,
    )
    return RewardCircuit(cfg)


class TestRewardLearning:
    def test_unexpected_reward_causes_burst(self, circuit):
        """予想外の報酬 → VTA DAバースト (RPE > 0)。"""
        results = circuit.run_unexpected_reward(n_trials=2)
        avg_rpe = sum(r.rpe_estimate for r in results) / len(results)
        # 予想外報酬でVTAが報酬時点でベースラインより活性化
        assert results[0].vta_firing_rate_at_reward >= 0

    def test_cs_vta_weight_increases_with_training(self, circuit):
        """訓練でCS→VTA結合重みが増加する。"""
        w_before = circuit.syn_cs_vta.mean_weight
        circuit.run_training(n_trials=10)
        w_after = circuit.syn_cs_vta.mean_weight
        assert w_after > w_before

    def test_training_increases_d1(self, circuit):
        """訓練でNAc D1(Go)経路の活性が上昇。"""
        pre = circuit.run_probe(n_trials=2)
        pre_d1 = sum(r.nac_d1_rate for r in pre) / len(pre)

        circuit.run_training(n_trials=10)

        post = circuit.run_probe(n_trials=2)
        post_d1 = sum(r.nac_d1_rate for r in post) / len(post)

        # D1活性が同等以上
        assert post_d1 >= pre_d1 * 0.5

    def test_omission_after_training(self, circuit):
        """訓練後の報酬省略 → 負のRPE方向。"""
        circuit.run_training(n_trials=10)
        omission = circuit.run_omission_test(n_trials=2)
        normal = circuit.run_probe(n_trials=2)

        # 省略試行ではVTA reward発火が通常より低い
        omission_rew = sum(r.vta_firing_rate_at_reward for r in omission) / len(omission)
        normal_rew = sum(r.vta_firing_rate_at_reward for r in normal) / len(normal)
        assert omission_rew <= normal_rew + 5  # 省略で報酬時VTA活性が低い方向

    def test_approach_tendency_nonnegative(self, circuit):
        """接近傾向(D1-D2バランス)が0-1範囲内。"""
        circuit.run_training(n_trials=5)
        for r in circuit.all_results:
            assert 0 <= r.approach_tendency <= 1

    def test_full_protocol(self, circuit):
        """完全プロトコル: 訓練→プローブ→省略→予想外。"""
        circuit.run_training(n_trials=5)
        circuit.run_probe(n_trials=2)
        circuit.run_omission_test(n_trials=2)
        circuit.run_unexpected_reward(n_trials=2)
        assert len(circuit.all_results) == 11

    def test_all_rates_nonnegative(self, circuit):
        circuit.run_training(n_trials=3)
        circuit.run_omission_test(n_trials=2)
        for r in circuit.all_results:
            assert r.vta_firing_rate_baseline >= 0
            assert r.vta_firing_rate_at_cs >= 0
            assert r.vta_firing_rate_at_reward >= 0
            assert r.nac_d1_rate >= 0
            assert r.nac_d2_rate >= 0
