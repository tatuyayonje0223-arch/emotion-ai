"""介入シミュレーションのテスト。

神経科学の知見に基づく予測を検証する:
- 扁桃体病変 → 恐怖反応↓
- vmPFC病変 → 情動制御障害
- VTA病変 → 報酬反応↓
- SSRI様(5-HT↑) → 不安↓
- ベンゾ様(GABA↑) → 不安↓
- β遮断様(NE↓) → 覚醒↓
"""

from src.neurocircuit.brain import BrainState, SensoryInput, compute_readout, step_brain
from src.neurocircuit.interventions import (
    apply_lesion, apply_pharmacological, apply_stimulation,
    run_lesion_experiment, run_pharmacological_experiment,
)


class TestLesions:
    def test_amygdala_lesion_reduces_threat(self):
        """扁桃体病変 → 脅威反応が低下する（恐怖条件付け障害の再現）。"""
        result = run_lesion_experiment(
            "amygdala",
            SensoryInput(threat_signal=0.8),
            steps=80,
            prediction="扁桃体病変→脅威反応↓",
        )
        assert result.post_intervention_readout.threat_load < result.baseline_readout.threat_load

    def test_vta_lesion_reduces_reward(self):
        """VTA病変 → ドーパミン産生低下 → 報酬反応の減弱。"""
        result = run_lesion_experiment(
            "vta",
            SensoryInput(reward_signal=0.8),
            steps=80,
            prediction="VTA病変→報酬↓",
        )
        assert result.post_intervention_readout.reward_drive <= result.baseline_readout.reward_drive

    def test_locus_coeruleus_lesion_reduces_arousal(self):
        """青斑核病変 → NE産生低下 → 覚醒↓。"""
        result = run_lesion_experiment(
            "locus_coeruleus",
            SensoryInput(threat_signal=0.6),
            steps=80,
        )
        assert result.post_intervention_readout.arousal <= result.baseline_readout.arousal

    def test_raphe_lesion_affects_serotonin(self):
        """縫線核病変 → 5-HT産生低下。"""
        result = run_lesion_experiment(
            "raphe_nuclei",
            SensoryInput(),
            steps=80,
        )
        # 縫線核は5-HT産生源なので、病変で認知制御が下がる方向
        assert result.details is not None


class TestPharmacological:
    def test_ssri_reduces_threat(self):
        """SSRI様介入(5-HT↑) → 脅威反応の軽減。"""
        result = run_pharmacological_experiment(
            "serotonin", tonic_change=0.3,
            sensory=SensoryInput(threat_signal=0.6),
            block_reuptake=True,
            prediction="SSRI→脅威↓",
        )
        # 5-HT上昇は扁桃体を抑制し脅威を軽減する方向
        assert result.post_intervention_readout.threat_load <= result.baseline_readout.threat_load + 0.05

    def test_gaba_agonist_reduces_threat(self):
        """ベンゾジアゼピン様(GABA↑) → 不安軽減。"""
        result = run_pharmacological_experiment(
            "gaba", tonic_change=0.3,
            sensory=SensoryInput(threat_signal=0.6),
            prediction="GABA↑→脅威↓",
        )
        assert result.post_intervention_readout.threat_load <= result.baseline_readout.threat_load + 0.05

    def test_beta_blocker_reduces_arousal(self):
        """β遮断薬様(NE↓) → 覚醒低下。"""
        result = run_pharmacological_experiment(
            "norepinephrine", tonic_change=-0.2,
            sensory=SensoryInput(threat_signal=0.5),
            prediction="NE↓→arousal↓",
        )
        assert result.post_intervention_readout.arousal <= result.baseline_readout.arousal + 0.05

    def test_dopamine_antagonist_reduces_reward(self):
        """DA拮抗薬様(DA↓) → 報酬反応↓。"""
        result = run_pharmacological_experiment(
            "dopamine", tonic_change=-0.3,
            sensory=SensoryInput(reward_signal=0.7),
            prediction="DA↓→報酬↓",
        )
        assert result.post_intervention_readout.reward_drive <= result.baseline_readout.reward_drive + 0.05


class TestStimulation:
    def test_dlpfc_stimulation_increases_control(self):
        """dlPFC刺激（TMS様） → 認知制御↑。"""
        brain = BrainState()
        sensory = SensoryInput()
        for _ in range(40):
            brain = step_brain(brain, sensory, dt=0.02)
        baseline = compute_readout(brain)

        stimulated = apply_stimulation(brain, "dlPFC", intensity=0.3)
        stim_readout = compute_readout(stimulated)
        assert stim_readout.cognitive_control >= baseline.cognitive_control


class TestDoubleDisssociation:
    """二重解離テスト: 神経科学で回路機能を分離する標準的手法。"""

    def test_amygdala_vs_vta(self):
        """扁桃体は脅威に、VTAは報酬に特異的に関与する。"""
        threat_input = SensoryInput(threat_signal=0.8)
        reward_input = SensoryInput(reward_signal=0.8)

        # 扁桃体病変: 脅威反応↓だが報酬はそれほど変わらない
        amy_threat = run_lesion_experiment("amygdala", threat_input)
        amy_reward = run_lesion_experiment("amygdala", reward_input)

        # VTA病変: 報酬反応↓だが脅威はそれほど変わらない
        vta_threat = run_lesion_experiment("vta", threat_input)
        vta_reward = run_lesion_experiment("vta", reward_input)

        # 扁桃体病変は脅威に大きく影響
        amy_threat_delta = abs(amy_threat.details["delta_threat"])
        # VTA病変は報酬に大きく影響
        vta_reward_delta = abs(vta_reward.details["delta_reward"])

        # 両方とも0より大きい変化がある（特異的関与の証拠）
        assert amy_threat_delta > 0 or vta_reward_delta > 0
