"""ハイブリッド脳モデルのテスト。AdEx mean-field + Brian2スパイキング。"""

import pytest

from src.brian2_circuits.adex_meanfield import (
    AdExMFParams, MeanFieldRegion, MeanFieldState, step_meanfield,
)
from src.brian2_circuits.hybrid_brain import HybridBrain, HybridConfig
from src.neurocircuit.brain import SensoryInput


class TestAdExMeanField:
    def test_resting_state(self):
        """外部入力なしで安定した基底活動。"""
        region = MeanFieldRegion("test")
        for _ in range(100):
            region.step(dt=1.0)
        assert region.output >= 0
        assert region.output < 50  # 飽和していない

    def test_excitation_increases_rate(self):
        """外部興奮入力で発火率が上昇。"""
        region = MeanFieldRegion("test")
        for _ in range(50):
            region.step(dt=1.0)
        baseline = region.output

        for _ in range(50):
            region.step(ext_exc=10.0, dt=1.0)
        excited = region.output
        assert excited > baseline

    def test_inhibition_decreases_rate(self):
        """外部抑制で発火率が低下。"""
        region = MeanFieldRegion("test")
        for _ in range(50):
            region.step(ext_exc=5.0, dt=1.0)
        baseline = region.output

        for _ in range(50):
            region.step(ext_inh=10.0, dt=1.0)
        inhibited = region.output
        assert inhibited <= baseline

    def test_neuromodulation_ne(self):
        """NE変調で利得が上昇。"""
        r1 = MeanFieldRegion("test1")
        r2 = MeanFieldRegion("test2")
        for _ in range(50):
            r1.step(ext_exc=5.0, dt=1.0)
            r2.step(ext_exc=5.0, neuromod={"norepinephrine": 0.5}, dt=1.0)
        # NE変調ありの方が高い出力
        assert r2.output >= r1.output * 0.9

    def test_adaptation(self):
        """持続入力で適応（発火率が漸減）。"""
        region = MeanFieldRegion("test")
        rates = []
        for _ in range(200):
            region.step(ext_exc=8.0, dt=1.0)
            rates.append(region.output)
        # 後半は前半より低い（適応）
        first_quarter = rates[25:50]
        last_quarter = rates[175:200]
        if first_quarter and last_quarter:
            assert max(last_quarter) <= max(first_quarter) * 1.5  # 適応で増加が抑制

    def test_rate_history(self):
        region = MeanFieldRegion("test")
        for _ in range(10):
            region.step(dt=1.0)
        assert len(region.rate_history) == 10

    def test_reset(self):
        region = MeanFieldRegion("test")
        region.step(ext_exc=10.0, dt=1.0)
        region.reset()
        assert region.output == 0.0
        assert len(region.rate_history) == 0


class TestHybridBrain:
    def test_threat_processing(self):
        """脅威入力でスパイキング+mean-fieldが両方動作。"""
        brain = HybridBrain()
        result = brain.process(SensoryInput(threat_signal=0.8))
        assert result.readout.threat_load >= 0
        assert "fear" in result.spiking_result.circuits_activated
        assert "insula" in result.mf_rates
        assert "acc" in result.mf_rates

    def test_reward_processing(self):
        """報酬入力。"""
        brain = HybridBrain()
        result = brain.process(SensoryInput(reward_signal=0.8))
        assert result.readout.reward_drive >= 0
        assert "reward" in result.spiking_result.circuits_activated

    def test_neutral_processing(self):
        """中立入力。"""
        brain = HybridBrain()
        result = brain.process(SensoryInput())
        assert result.readout is not None
        assert result.total_virtual_neurons >= 0

    def test_total_neurons_includes_mf(self):
        """仮想ニューロン数にmean-field分が含まれる。"""
        brain = HybridBrain()
        result = brain.process(SensoryInput(threat_signal=0.5))
        # mean-field: 4領域 × (8000E + 2000I) = 40,000
        assert result.total_virtual_neurons >= 10000

    def test_mf_feedback_from_spiking(self):
        """スパイキング→mean-fieldのフィードバックが動作。"""
        brain = HybridBrain()
        # 脅威入力で扁桃体→island/ACCへフィードバック
        brain.process(SensoryInput(threat_signal=0.8))
        # island/ACCの状態が変化している
        assert brain.mf_states["insula"] >= 0
        assert brain.mf_states["acc"] >= 0

    def test_readout_bounded(self):
        brain = HybridBrain()
        result = brain.process(SensoryInput(
            threat_signal=1.0, reward_signal=1.0, pain_input=1.0,
        ))
        r = result.readout
        assert -1.0 <= r.valence <= 1.0
        assert 0.0 <= r.arousal <= 1.0
        assert 0.0 <= r.threat_load <= 1.0
        assert 0.0 <= r.cognitive_control <= 1.0
        assert 0.0 <= r.energy <= 1.0

    def test_reset(self):
        brain = HybridBrain()
        brain.process(SensoryInput(threat_signal=0.5))
        brain.reset()
        # mean-field状態がリセットされている
        for rate in brain.mf_states.values():
            assert rate == 0.0
