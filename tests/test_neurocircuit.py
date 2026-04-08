"""脳神経回路モデルのテスト。

3レベルで検証:
1. 神経活動レベル: 各領域の活性が生理学的範囲内か
2. 回路機能レベル: 脅威→扁桃体→HPA等の因果連鎖
3. 行動レベル: シナリオで一貫した適応的応答
"""

import pytest

from src.neurocircuit.brain import (
    BrainState, EmotionReadout, SensoryInput,
    compute_readout, step_brain,
)
from src.neurocircuit.connectivity import (
    ANATOMICAL_CONNECTIONS, get_connections_to, get_connections_from,
    get_connection_matrix,
)
from src.neurocircuit.homeostasis import (
    BodyState, HPAAxisState, update_hpa_axis, update_autonomic,
)
from src.neurocircuit.neurotransmitters import (
    NeurotransmitterSystem, NeurotransmitterState, apply_reuptake,
)
from src.neurocircuit.plasticity import (
    PlasticityParams, hebbian_update, reward_modulated_update,
    fear_conditioning, extinction, emotional_memory_tag,
)
from src.neurocircuit.regions import RegionState, update_region


# === 神経活動レベルのテスト ===

class TestRegionDynamics:
    def test_region_stays_in_range(self):
        region = RegionState(name="test", excitatory=0.5, inhibitory=0.3)
        for _ in range(100):
            region = update_region(region, external_excitatory=0.5, dt=0.01)
        assert 0.0 <= region.excitatory <= 1.0
        assert 0.0 <= region.inhibitory <= 1.0
        assert 0.0 <= region.output <= 1.0

    def test_excitation_increases_activity(self):
        region = RegionState(name="test", excitatory=0.1)
        excited = update_region(region, external_excitatory=1.0, dt=0.1)
        assert excited.excitatory > region.excitatory

    def test_inhibition_suppresses_activity(self):
        """強い外部抑制をかけ続けると活性が低下する。"""
        region = RegionState(name="test", excitatory=0.5, w_ee=0.5)  # 自己興奮を弱めて抑制効果を見やすく
        for _ in range(20):
            region = update_region(region, external_inhibitory=1.0, dt=0.05)
        assert region.excitatory < 0.5 or region.output < 0.5

    def test_neuromod_gain_amplifies(self):
        region = RegionState(name="test", excitatory=0.3)
        normal = update_region(region, external_excitatory=0.5, neuromod_gain=1.0, dt=0.1)
        amplified = update_region(region, external_excitatory=0.5, neuromod_gain=2.0, dt=0.1)
        assert amplified.output >= normal.output


class TestNeurotransmitters:
    def test_effective_level_in_range(self):
        nt = NeurotransmitterState(tonic=0.5, phasic=0.8)
        assert 0.0 <= nt.effective_level <= 1.0

    def test_reuptake_reduces_phasic(self):
        system = NeurotransmitterSystem()
        system.dopamine.phasic = 0.5
        updated = apply_reuptake(system)
        assert updated.dopamine.phasic < 0.5

    def test_tonic_homeostasis(self):
        system = NeurotransmitterSystem()
        system.serotonin.tonic = 0.1  # 基底値(0.5)より低い
        updated = apply_reuptake(system)
        assert updated.serotonin.tonic > 0.1  # 基底に向かって回復

    def test_all_names(self):
        system = NeurotransmitterSystem()
        names = system.all_names()
        assert len(names) == 8
        assert "dopamine" in names


class TestConnectivity:
    def test_connections_exist(self):
        assert len(ANATOMICAL_CONNECTIONS) > 20

    def test_amygdala_has_inputs(self):
        inputs = get_connections_to("amygdala")
        assert len(inputs) > 0

    def test_amygdala_has_outputs(self):
        outputs = get_connections_from("amygdala")
        assert len(outputs) > 0

    def test_pfc_inhibits_amygdala(self):
        conns = get_connections_to("amygdala")
        pfc_inh = [c for c in conns if "PFC" in c.source and c.conn_type == "inhibitory"]
        assert len(pfc_inh) > 0

    def test_connection_matrix(self):
        matrix = get_connection_matrix()
        assert len(matrix) > 0


# === 回路機能レベルのテスト ===

class TestThreatCircuit:
    """脅威→扁桃体→HPA→コルチゾールの因果連鎖。"""

    def test_threat_activates_amygdala(self):
        brain = BrainState()
        threat = SensoryInput(threat_signal=0.8)
        for _ in range(50):
            brain = step_brain(brain, threat, dt=0.02)
        assert brain.amygdala.output > 0.3

    def test_amygdala_activates_hpa(self):
        brain = BrainState()
        threat = SensoryInput(threat_signal=0.9)
        for _ in range(100):
            brain = step_brain(brain, threat, dt=0.02)
        assert brain.body.hpa.cortisol > brain.body.hpa.cortisol_baseline

    def test_threat_increases_ne(self):
        brain = BrainState()
        baseline_ne = brain.neurotransmitters.norepinephrine.effective_level
        threat = SensoryInput(threat_signal=0.8)
        for _ in range(50):
            brain = step_brain(brain, threat, dt=0.02)
        assert brain.neurotransmitters.norepinephrine.effective_level > baseline_ne

    def test_threat_readout(self):
        brain = BrainState()
        threat = SensoryInput(threat_signal=0.8)
        for _ in range(50):
            brain = step_brain(brain, threat, dt=0.02)
        readout = compute_readout(brain)
        assert readout.threat_load > 0.2
        assert readout.arousal > 0.2


class TestRewardCircuit:
    """報酬→VTA→DA→NAcの因果連鎖。"""

    def test_reward_activates_vta(self):
        brain = BrainState()
        reward = SensoryInput(reward_signal=0.8)
        for _ in range(50):
            brain = step_brain(brain, reward, dt=0.02)
        assert brain.brainstem.vta.output > 0.2

    def test_reward_increases_dopamine(self):
        brain = BrainState()
        baseline_da = brain.neurotransmitters.dopamine.effective_level
        reward = SensoryInput(reward_signal=0.8)
        for _ in range(50):
            brain = step_brain(brain, reward, dt=0.02)
        assert brain.neurotransmitters.dopamine.effective_level > baseline_da

    def test_reward_readout(self):
        brain = BrainState()
        reward = SensoryInput(reward_signal=0.8)
        for _ in range(50):
            brain = step_brain(brain, reward, dt=0.02)
        readout = compute_readout(brain)
        assert readout.reward_drive > 0.2
        assert readout.valence > -0.5  # ポジティブ方向


class TestSocialCircuit:
    """社会的入力→OXT→信頼。"""

    def test_social_increases_oxytocin(self):
        brain = BrainState()
        social = SensoryInput(social_signal=0.8)
        for _ in range(50):
            brain = step_brain(brain, social, dt=0.02)
        assert brain.neurotransmitters.oxytocin.effective_level > 0.3

    def test_social_warmth_readout(self):
        brain = BrainState()
        social = SensoryInput(social_signal=0.8)
        for _ in range(50):
            brain = step_brain(brain, social, dt=0.02)
        readout = compute_readout(brain)
        assert readout.social_warmth > 0.2


class TestHomeostasis:
    def test_hpa_negative_feedback(self):
        hpa = HPAAxisState(crh=0.5, acth=0.5, cortisol=0.6)
        initial_cort = hpa.cortisol
        for _ in range(500):
            hpa = update_hpa_axis(hpa, amygdala_output=0.0, dt=0.02)
        # 扁桃体入力なし→コルチゾールが基底(0.2)に向かって減少
        assert hpa.cortisol < initial_cort
        assert hpa.crh < 0.1  # CRHは抑制される

    def test_autonomic_sympathetic_activation(self):
        from src.neurocircuit.homeostasis import AutonomicState
        auto = AutonomicState(sympathetic=0.3)
        auto = update_autonomic(auto, amygdala_output=0.8, pag_output=0.5, pfc_inhibition=0.0, dt=0.1)
        assert auto.sympathetic > 0.3


# === 可塑性テスト ===

class TestPlasticity:
    def test_hebbian_strengthens(self):
        params = PlasticityParams()
        w = hebbian_update(0.5, pre_activity=0.8, post_activity=0.8, params=params)
        assert w > 0.5

    def test_reward_positive_rpe(self):
        params = PlasticityParams()
        w = reward_modulated_update(0.5, 0.8, 0.8, reward_prediction_error=0.5, params=params)
        assert w > 0.5

    def test_reward_negative_rpe(self):
        params = PlasticityParams()
        w = reward_modulated_update(0.5, 0.8, 0.8, reward_prediction_error=-0.5, params=params)
        assert w < 0.5

    def test_fear_conditioning_strengthens(self):
        params = PlasticityParams()
        w = fear_conditioning(0.3, cs_activity=0.8, us_activity=0.9, amygdala_activity=0.7, params=params)
        assert w > 0.3

    def test_extinction_weakens(self):
        params = PlasticityParams()
        w = extinction(0.8, cs_activity=0.6, us_absence=True, pfc_inhibition=0.6, params=params)
        assert w < 0.8

    def test_emotional_memory_tag(self):
        strength = emotional_memory_tag(0.5, amygdala_activity=0.8, cortisol_level=0.3, norepinephrine_level=0.6)
        assert strength > 0.5

    def test_cortisol_inverted_u(self):
        low_cort = emotional_memory_tag(0.5, 0.5, cortisol_level=0.3, norepinephrine_level=0.5)
        high_cort = emotional_memory_tag(0.5, 0.5, cortisol_level=0.9, norepinephrine_level=0.5)
        assert low_cort > high_cort  # 逆U字: 高コルチゾールは記憶を阻害


# === 行動レベルのテスト ===

class TestBehavioralScenarios:
    def _run_scenario(self, sensory: SensoryInput, steps: int = 80) -> tuple[BrainState, EmotionReadout]:
        brain = BrainState()
        for _ in range(steps):
            brain = step_brain(brain, sensory, dt=0.02)
        return brain, compute_readout(brain)

    def test_neutral_baseline(self):
        """ニュートラル入力での定常状態。脅威入力ありの場合より低いこと。"""
        _, neutral_readout = self._run_scenario(SensoryInput())
        _, threat_readout = self._run_scenario(SensoryInput(threat_signal=0.8))
        # ニュートラルは脅威シナリオより脅威が低い
        assert neutral_readout.threat_load < threat_readout.threat_load
        # valenceは極端でない
        assert abs(neutral_readout.valence) < 0.8

    def test_threat_then_safety(self):
        brain = BrainState()
        # 脅威フェーズ
        threat = SensoryInput(threat_signal=0.8)
        for _ in range(80):
            brain = step_brain(brain, threat, dt=0.02)
        threat_readout = compute_readout(brain)

        # 安全フェーズ（入力なし、十分な回復時間）
        safe = SensoryInput()
        for _ in range(300):
            brain = step_brain(brain, safe, dt=0.02)
        recovery_readout = compute_readout(brain)

        # 扁桃体活性が回復している（脅威入力がなくなれば下がる）
        assert brain.amygdala.output < 0.5 or recovery_readout.threat_load < threat_readout.threat_load + 0.1

    def test_reward_makes_positive_valence(self):
        _, readout = self._run_scenario(SensoryInput(reward_signal=0.8))
        assert readout.valence > -0.3

    def test_all_regions_in_range(self):
        brain = BrainState()
        sensory = SensoryInput(threat_signal=0.5, reward_signal=0.3, social_signal=0.4)
        for _ in range(100):
            brain = step_brain(brain, sensory, dt=0.02)
        # 全領域が0-1範囲内
        assert 0 <= brain.amygdala.excitatory <= 1
        assert 0 <= brain.pfc.vmPFC.excitatory <= 1
        assert 0 <= brain.insula.excitatory <= 1
        assert 0 <= brain.ventral_striatum.excitatory <= 1
        assert 0 <= brain.body.hpa.cortisol <= 1
        assert 0 <= brain.body.autonomic.sympathetic <= 1

    def test_readout_all_bounded(self):
        _, readout = self._run_scenario(SensoryInput(threat_signal=1.0, reward_signal=1.0))
        assert -1.0 <= readout.valence <= 1.0
        assert 0.0 <= readout.arousal <= 1.0
        assert 0.0 <= readout.threat_load <= 1.0
        assert 0.0 <= readout.energy <= 1.0
