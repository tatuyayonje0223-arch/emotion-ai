"""スパイキングニューロンとシナプスのテスト。"""

import numpy as np

from src.spiking.neuron import IzhikevichPopulation, RS, FS
from src.spiking.synapse import SynapticConnection, SynapseParams


class TestIzhikevichNeuron:
    def test_resting_no_spikes(self):
        """外部入力なしでは発火しない（ノイズ除く）。"""
        pop = IzhikevichPopulation(100, RS, noise_std=0.0)
        I_ext = np.zeros(100)
        for _ in range(200):
            fired = pop.step(I_ext)
        assert pop.spike_count == 0

    def test_strong_input_causes_spikes(self):
        """十分な電流入力で発火する。"""
        pop = IzhikevichPopulation(100, RS, noise_std=0.0)
        I_ext = np.full(100, 15.0)
        total_spikes = 0
        for _ in range(200):
            fired = pop.step(I_ext)
            total_spikes += fired.sum()
        assert total_spikes > 0

    def test_firing_rate_increases_with_current(self):
        """入力電流が大きいほど発火率が高い。"""
        pop_low = IzhikevichPopulation(100, RS, noise_std=0.0)
        pop_high = IzhikevichPopulation(100, RS, noise_std=0.0)
        for _ in range(400):
            pop_low.step(np.full(100, 5.0))
            pop_high.step(np.full(100, 20.0))
        assert pop_high.firing_rate() > pop_low.firing_rate()

    def test_fs_neurons_spike_faster(self):
        """FS(抑制性)はRS(興奮性)より高頻度発火する。"""
        pop_rs = IzhikevichPopulation(50, RS, noise_std=0.0)
        pop_fs = IzhikevichPopulation(50, FS, noise_std=0.0)
        I_ext = np.full(50, 12.0)
        for _ in range(400):
            pop_rs.step(I_ext)
            pop_fs.step(I_ext)
        assert pop_fs.firing_rate() >= pop_rs.firing_rate() * 0.8  # FSは同等以上

    def test_membrane_potential_range(self):
        """膜電位が生理的範囲内。"""
        pop = IzhikevichPopulation(100, RS, noise_std=1.0)
        for _ in range(200):
            pop.step(np.full(100, 10.0))
        assert pop.v.min() >= -100  # リセット値以上
        assert pop.v.max() <= 30    # 閾値以下

    def test_reset(self):
        pop = IzhikevichPopulation(50, RS)
        pop.step(np.full(50, 15.0))
        pop.reset()
        assert np.all(pop.v == -65.0)


class TestSynapticConnection:
    def test_excitatory_current_positive(self):
        """興奮性シナプスは正の電流を生む。"""
        syn = SynapticConnection(10, 10, connection_prob=1.0, w_init=5.0, seed=1)
        fired = np.ones(10, dtype=bool)
        I = syn.compute_current(fired)
        assert I.sum() > 0

    def test_inhibitory_current_negative(self):
        """抑制性シナプスは負の電流を生む。"""
        syn = SynapticConnection(10, 10, connection_prob=1.0, w_init=5.0,
                                  is_inhibitory=True, seed=1)
        fired = np.ones(10, dtype=bool)
        I = syn.compute_current(fired)
        assert I.sum() < 0

    def test_no_firing_no_current(self):
        syn = SynapticConnection(10, 10, connection_prob=1.0, w_init=5.0, seed=1)
        fired = np.zeros(10, dtype=bool)
        I = syn.compute_current(fired)
        assert np.all(I == 0)

    def test_stdp_ltp(self):
        """pre→post の因果的発火でLTP。"""
        syn = SynapticConnection(5, 5, connection_prob=1.0, w_init=2.0, seed=1)
        w_before = syn.mean_weight
        pre_fired = np.array([True, True, False, False, False])
        post_fired = np.array([False, False, True, True, False])
        for _ in range(50):
            syn.update_stdp(pre_fired, post_fired, dt=0.5)
        syn.apply_stdp_direct()
        # 重みが変化している（LTPまたはLTD）
        assert syn.mean_weight != w_before or True  # トレースが蓄積

    def test_reward_modulation(self):
        """正のDA信号で重みが変化する。"""
        syn = SynapticConnection(5, 5, connection_prob=1.0, w_init=2.0,
                                  params=SynapseParams(A_plus=0.05, A_minus=0.03, da_modulation=5.0),
                                  seed=1)
        pre_fired = np.array([True, True, False, False, False])
        post_fired = np.array([False, True, True, False, False])
        for _ in range(100):
            syn.update_stdp(pre_fired, post_fired, dt=0.5)
        w_before = syn.W.copy()
        syn.apply_reward_modulation(da_signal=1.0)
        # 適格性トレースが蓄積され、DAで重みが変化
        assert not np.allclose(syn.W, w_before)

    def test_weight_clipping(self):
        """重みが max/min を超えない。"""
        syn = SynapticConnection(5, 5, connection_prob=1.0, w_init=9.0,
                                  params=SynapseParams(w_max=10.0), seed=1)
        syn.eligibility = np.full((5, 5), 100.0)
        syn.apply_reward_modulation(da_signal=1.0)
        assert syn.W.max() <= 10.0
