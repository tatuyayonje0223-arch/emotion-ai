"""高度な神経修飾システムのテスト。"""

import math

from src.brian2_circuits.neuromodulation import (
    EndocannabinoidState, AcetylcholineState, ThetaOscillator,
    StructuralPlasticityState,
    update_endocannabinoid, update_acetylcholine,
    compute_theta_coherence, update_structural_plasticity,
)


class TestEndocannabinoid:
    def test_extinction_increases_aea(self):
        """消去試行でAEAトーンが上昇する。"""
        ecb = EndocannabinoidState()
        initial_aea = ecb.aea_tone
        for _ in range(100):
            ecb = update_endocannabinoid(ecb, bla_activity=0.3, pfc_activity=0.6,
                                          is_extinction_trial=True, dt_ms=1.0)
        assert ecb.aea_tone >= initial_aea

    def test_extinction_reduces_cb1r_gaba(self):
        """消去でGABA端末CB1Rがダウンレギュレーション。"""
        ecb = EndocannabinoidState(cb1r_gaba=0.5)
        for _ in range(200):
            ecb = update_endocannabinoid(ecb, 0.3, 0.6, is_extinction_trial=True, dt_ms=1.0)
        assert ecb.cb1r_gaba <= 0.5

    def test_2ag_increases_with_activity(self):
        """強い神経活動で2-AGが放出される。"""
        ecb = EndocannabinoidState(two_ag=0.0)
        for _ in range(50):
            ecb = update_endocannabinoid(ecb, bla_activity=0.8, pfc_activity=0.7,
                                          is_extinction_trial=False, dt_ms=1.0)
        assert ecb.two_ag > 0.0

    def test_all_values_bounded(self):
        ecb = EndocannabinoidState()
        for _ in range(300):
            ecb = update_endocannabinoid(ecb, 1.0, 1.0, True, 1.0)
        assert 0 <= ecb.two_ag <= 1
        assert 0 <= ecb.aea_tone <= 1
        assert 0 <= ecb.cb1r_gaba <= 1
        assert 0 <= ecb.cb1r_glu <= 1


class TestAcetylcholine:
    def test_threat_increases_nbm_ach(self):
        """脅威でNBM→BLA AChが上昇。"""
        ach = AcetylcholineState()
        for _ in range(100):
            ach = update_acetylcholine(ach, threat_signal=0.8, reward_signal=0.0,
                                       bla_activity=0.5, dt_ms=1.0)
        assert ach.nbm_ach > 0.3

    def test_reward_increases_nac_cin(self):
        """報酬でNAc CINが上昇。"""
        ach = AcetylcholineState()
        for _ in range(100):
            ach = update_acetylcholine(ach, 0.0, reward_signal=0.8, bla_activity=0.0, dt_ms=1.0)
        assert ach.nac_cin > 0.3

    def test_nicotinic_depends_on_ach(self):
        """ニコチン受容体活性がACh依存。"""
        ach = AcetylcholineState(nbm_ach=0.8, nac_cin=0.6)
        ach = update_acetylcholine(ach, 0.5, 0.5, 0.5, 1.0)
        assert ach.nicotinic_activation > 0.3


class TestThetaOscillator:
    def test_oscillation_frequency(self):
        """正しい周波数でオシレーション。"""
        theta = ThetaOscillator(frequency_hz=6.0)
        values = []
        for _ in range(1000):  # 1000ms = 1秒
            values.append(theta.step(dt_ms=1.0))
        # 1秒で約6サイクル → ゼロクロスが約12回
        crossings = sum(1 for i in range(1, len(values)) if values[i-1] * values[i] < 0)
        assert 8 <= crossings <= 16  # 6Hz → ~12ゼロクロス (許容幅あり)

    def test_phase_coherence_identical(self):
        """同位相の2オシレータ → 高い同期度。"""
        t1 = ThetaOscillator(frequency_hz=6.0)
        t2 = ThetaOscillator(frequency_hz=6.0)
        # 同じステップ数進めて同位相
        for _ in range(50):
            t1.step(1.0)
            t2.step(1.0)
        assert compute_theta_coherence(t1, t2) > 0.9

    def test_phase_coherence_opposite(self):
        """逆位相 → 低い同期度。"""
        t1 = ThetaOscillator(frequency_hz=6.0, phase=0.0)
        t2 = ThetaOscillator(frequency_hz=6.0, phase=math.pi)
        assert compute_theta_coherence(t1, t2) <= 0.1  # cos(pi)=-1→max(0,-1)=0


class TestStructuralPlasticity:
    def test_ltp_increases_spines(self):
        """LTPでスパイン密度が増加。"""
        sp = StructuralPlasticityState()
        for _ in range(100):
            sp = update_structural_plasticity(sp, ltp_occurred=True, ltd_occurred=False,
                                               amygdala_activity=0.5, dt_ms=1.0)
        assert sp.spine_density > 1.0

    def test_ltd_decreases_spines(self):
        """LTDでスパイン密度が減少（PNN未成熟時）。"""
        sp = StructuralPlasticityState(pnn_maturity=0.2)
        for _ in range(100):
            sp = update_structural_plasticity(sp, ltp_occurred=False, ltd_occurred=True,
                                               amygdala_activity=0.3, dt_ms=1.0)
        assert sp.spine_density < 1.0

    def test_pnn_blocks_ltd_pruning(self):
        """成熟PNNがLTDによるスパイン除去を制限。"""
        sp_immature = StructuralPlasticityState(pnn_maturity=0.1)
        sp_mature = StructuralPlasticityState(pnn_maturity=0.9)
        for _ in range(100):
            sp_immature = update_structural_plasticity(sp_immature, False, True, 0.3, 1.0)
            sp_mature = update_structural_plasticity(sp_mature, False, True, 0.3, 1.0)
        # 未成熟PNNの方がスパイン除去が大きい
        assert sp_immature.spine_density < sp_mature.spine_density

    def test_bdnf_rises_with_activity(self):
        """扁桃体活性でBDNFが上昇。"""
        sp = StructuralPlasticityState(bdnf_level=0.3)
        for _ in range(100):
            sp = update_structural_plasticity(sp, False, False, amygdala_activity=0.8, dt_ms=1.0)
        assert sp.bdnf_level > 0.3
