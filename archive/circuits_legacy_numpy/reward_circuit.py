"""報酬学習回路。ドーパミン報酬予測誤差(RPE)のスパイキング実装。

Schultz et al. (1997) に基づく:
  - 予想外の報酬 → DA burst (RPE > 0)
  - 予想通りの報酬 → DA変化なし (RPE ≈ 0)
  - 期待した報酬が来ない → DA dip (RPE < 0)
  - 学習後、DAバーストはCS(予測手がかり)の時点にシフトする

回路構成:
  - VTA (腹側被蓋野): DAニューロン。RPE信号を出力
  - NAc (側坐核): 報酬処理。VTAからDA入力を受ける。D1/D2受容体で分離
  - OFC (眼窩前頭皮質): 期待報酬の表象。NAc/VTAにフィードバック
  - LHb (外側手綱核): 負のRPE信号源。VTAを抑制

学習則:
  - VTA DA neurons: CS-reward連合をSTDP+eligibility traceで学習
  - NAc D1 pathway (Go): 正のRPEで強化
  - NAc D2 pathway (NoGo): 負のRPEで強化
  - OFC→VTA: 期待報酬を伝達し、実際の報酬との差(RPE)を計算

検証対象データ:
  - VTA DA burst: 報酬提示後 ~100-200ms, ~20-50Hz短時間バースト
  - RPEシフト: 10-20試行でCSへのバースト移行
  - 報酬省略: 予想報酬時点でDA dip
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.spiking.neuron import IzhikevichPopulation, RS, FS, IB, NeuronParams
from src.spiking.synapse import SynapticConnection, SynapseParams


# VTA DAニューロン: 低頻度自発発火(3-5Hz) + バースト可能
DA_NEURON = NeuronParams(a=0.02, b=0.2, c=-55, d=4, label="DA")  # IBに近い


@dataclass
class RewardCircuitConfig:
    """報酬回路パラメータ。"""

    n_vta: int = 60           # VTA DAニューロン
    n_nac_d1: int = 80        # NAc D1(Go)経路
    n_nac_d2: int = 80        # NAc D2(NoGo)経路
    n_ofc: int = 60           # OFC期待報酬ニューロン
    n_lhb: int = 30           # LHb負のRPE

    cs_current: float = 6.0   # CS(報酬予測手がかり)入力
    reward_current: float = 12.0  # 報酬入力
    background_noise: float = 2.0

    dt: float = 0.5
    trial_duration_ms: float = 600.0
    cs_onset_ms: float = 100.0
    cs_duration_ms: float = 100.0
    reward_onset_ms: float = 400.0  # CSの300ms後
    reward_duration_ms: float = 50.0


@dataclass
class RewardTrialResult:
    """報酬学習1試行の結果。"""

    trial_num: int
    phase: str
    cs_presented: bool
    reward_presented: bool
    vta_firing_rate_at_cs: float     # CS提示時のVTA発火率
    vta_firing_rate_at_reward: float  # 報酬提示時のVTA発火率
    vta_firing_rate_baseline: float   # ベースライン発火率
    nac_d1_rate: float
    nac_d2_rate: float
    ofc_rate: float
    rpe_estimate: float              # RPE推定値
    cs_vta_weight: float             # CS→VTA結合重み
    approach_tendency: float         # 接近行動傾向 (D1-D2)


class RewardCircuit:
    """報酬学習の完全回路。"""

    def __init__(self, config: RewardCircuitConfig | None = None):
        self.cfg = config or RewardCircuitConfig()
        self._build()
        self._results: list[RewardTrialResult] = []

    def _build(self) -> None:
        c = self.cfg

        # ニューロン集団
        self.vta = IzhikevichPopulation(c.n_vta, DA_NEURON, noise_std=0.8)
        self.nac_d1 = IzhikevichPopulation(c.n_nac_d1, RS, noise_std=1.0)
        self.nac_d2 = IzhikevichPopulation(c.n_nac_d2, RS, noise_std=1.0)
        self.ofc = IzhikevichPopulation(c.n_ofc, RS, noise_std=1.0)
        self.lhb = IzhikevichPopulation(c.n_lhb, RS, noise_std=0.5)

        # CS → VTA (可塑的: RPEシフトの基盤)
        self.syn_cs_vta = SynapticConnection(
            c.n_vta, c.n_vta, connection_prob=0.3, w_init=0.5,
            params=SynapseParams(A_plus=0.01, A_minus=0.005, w_max=8.0, da_modulation=2.0),
            seed=10,
        )

        # Reward → VTA (強い固定結合)
        self.syn_reward_vta = SynapticConnection(
            c.n_vta, c.n_vta, connection_prob=0.5, w_init=5.0,
            params=SynapseParams(w_max=8.0), seed=11,
        )

        # VTA → NAc D1 (DA興奮: Go経路)
        self.syn_vta_d1 = SynapticConnection(
            c.n_vta, c.n_nac_d1, connection_prob=0.4, w_init=3.0, seed=12,
        )

        # VTA → NAc D2 (DA抑制: NoGo経路)
        self.syn_vta_d2 = SynapticConnection(
            c.n_vta, c.n_nac_d2, connection_prob=0.3, w_init=2.0,
            is_inhibitory=True, seed=13,
        )

        # OFC → VTA (期待報酬→RPE計算: 抑制的)
        self.syn_ofc_vta = SynapticConnection(
            c.n_ofc, c.n_vta, connection_prob=0.3, w_init=1.0,
            is_inhibitory=True,
            params=SynapseParams(A_plus=0.005, w_max=6.0),
            seed=14,
        )

        # CS → OFC (CSから期待報酬を学習)
        self.syn_cs_ofc = SynapticConnection(
            c.n_ofc, c.n_ofc, connection_prob=0.3, w_init=0.5,
            params=SynapseParams(A_plus=0.008, A_minus=0.004, w_max=8.0, da_modulation=1.5),
            seed=15,
        )

        # LHb → VTA (負のRPE: VTA抑制)
        self.syn_lhb_vta = SynapticConnection(
            c.n_lhb, c.n_vta, connection_prob=0.4, w_init=4.0,
            is_inhibitory=True, seed=16,
        )

        # NAc D1 recurrent (報酬信号の持続)
        self.syn_d1_d1 = SynapticConnection(
            c.n_nac_d1, c.n_nac_d1, connection_prob=0.1, w_init=1.5, seed=17,
        )

    def run_trial(
        self,
        cs: bool = True,
        reward: bool = True,
        phase: str = "training",
        trial_num: int = 0,
    ) -> RewardTrialResult:
        c = self.cfg
        n_steps = int(c.trial_duration_ms / c.dt)
        cs_start = int(c.cs_onset_ms / c.dt)
        cs_end = int((c.cs_onset_ms + c.cs_duration_ms) / c.dt)
        rew_start = int(c.reward_onset_ms / c.dt)
        rew_end = int((c.reward_onset_ms + c.reward_duration_ms) / c.dt)

        # 区間別スパイクカウント
        vta_spikes_baseline = 0
        vta_spikes_cs = 0
        vta_spikes_reward = 0
        d1_spikes = 0
        d2_spikes = 0
        ofc_spikes = 0
        baseline_steps = cs_start
        cs_steps = cs_end - cs_start
        reward_steps = rew_end - rew_start

        for t in range(n_steps):
            # 入力
            I_cs_vta = np.zeros(c.n_vta)
            I_cs_ofc = np.zeros(c.n_ofc)
            I_reward = np.zeros(c.n_vta)
            I_omission = np.zeros(c.n_lhb)  # 報酬省略→LHb活性

            if cs and cs_start <= t < cs_end:
                cs_input = np.zeros(c.n_vta, dtype=bool)
                cs_input[:c.n_vta // 3] = True
                I_cs_vta = self.syn_cs_vta.compute_current(cs_input)
                ofc_input = np.zeros(c.n_ofc, dtype=bool)
                ofc_input[:c.n_ofc // 3] = True
                I_cs_ofc = self.syn_cs_ofc.compute_current(ofc_input)

            if reward and rew_start <= t < rew_end:
                rew_input = np.ones(c.n_vta, dtype=bool)
                I_reward = self.syn_reward_vta.compute_current(rew_input)

            # 報酬省略: 期待報酬時点でrewardなし→LHb活性化
            if not reward and rew_start <= t < rew_end and cs:
                I_omission = np.full(c.n_lhb, 8.0)

            # 背景
            bg = lambda n: np.random.normal(c.background_noise, 1.0, n)

            # OFC
            ofc_fired = self.ofc.step(I_cs_ofc + bg(c.n_ofc), c.dt)

            # LHb
            lhb_fired = self.lhb.step(I_omission + bg(c.n_lhb) * 0.5, c.dt)

            # VTA: 報酬 + CS入力 + OFC抑制 + LHb抑制
            I_ofc_vta = self.syn_ofc_vta.compute_current(ofc_fired)
            I_lhb_vta = self.syn_lhb_vta.compute_current(lhb_fired)
            vta_fired = self.vta.step(
                I_cs_vta + I_reward + I_ofc_vta + I_lhb_vta + bg(c.n_vta), c.dt,
            )

            # NAc D1
            I_vta_d1 = self.syn_vta_d1.compute_current(vta_fired)
            I_d1_d1 = self.syn_d1_d1.compute_current(self.nac_d1.fired)
            d1_fired = self.nac_d1.step(I_vta_d1 + I_d1_d1 + bg(c.n_nac_d1), c.dt)

            # NAc D2
            I_vta_d2 = self.syn_vta_d2.compute_current(vta_fired)
            d2_fired = self.nac_d2.step(I_vta_d2 + bg(c.n_nac_d2), c.dt)

            # STDP (CS期間)
            if cs and cs_start <= t < cs_end:
                cs_mask = np.zeros(c.n_vta, dtype=bool)
                cs_mask[:c.n_vta // 3] = True
                self.syn_cs_vta.update_stdp(cs_mask, vta_fired, c.dt)
                ofc_mask = np.zeros(c.n_ofc, dtype=bool)
                ofc_mask[:c.n_ofc // 3] = True
                self.syn_cs_ofc.update_stdp(ofc_mask, ofc_fired, c.dt)

            # スパイクカウント
            if t < cs_start:
                vta_spikes_baseline += vta_fired.sum()
            elif cs_start <= t < cs_end:
                vta_spikes_cs += vta_fired.sum()
            if rew_start <= t < rew_end:
                vta_spikes_reward += vta_fired.sum()
            if cs_start <= t < rew_end:
                d1_spikes += d1_fired.sum()
                d2_spikes += d2_fired.sum()
                ofc_spikes += ofc_fired.sum()

        # 報酬変調学習
        if reward:
            # 正のRPE: CS→VTAとCS→OFCを強化
            self.syn_cs_vta.apply_reward_modulation(da_signal=1.0)
            self.syn_cs_ofc.apply_reward_modulation(da_signal=0.5)
        elif cs and not reward:
            # 負のRPE(報酬省略): CS→VTAを弱化
            self.syn_cs_vta.apply_reward_modulation(da_signal=-0.3)

        # 発火率
        def rate(spikes, n_neurons, n_steps):
            if n_steps == 0:
                return 0.0
            return (spikes / n_neurons) / (n_steps * c.dt / 1000.0)

        vta_bl = rate(vta_spikes_baseline, c.n_vta, baseline_steps)
        vta_cs = rate(vta_spikes_cs, c.n_vta, cs_steps)
        vta_rew = rate(vta_spikes_reward, c.n_vta, reward_steps)
        total_steps = cs_end - cs_start + reward_steps
        d1_r = rate(d1_spikes, c.n_nac_d1, total_steps)
        d2_r = rate(d2_spikes, c.n_nac_d2, total_steps)
        ofc_r = rate(ofc_spikes, c.n_ofc, total_steps)

        # RPE推定: VTA(reward) - VTA(baseline)
        rpe = vta_rew - vta_bl

        result = RewardTrialResult(
            trial_num=trial_num,
            phase=phase,
            cs_presented=cs,
            reward_presented=reward,
            vta_firing_rate_at_cs=vta_cs,
            vta_firing_rate_at_reward=vta_rew,
            vta_firing_rate_baseline=vta_bl,
            nac_d1_rate=d1_r,
            nac_d2_rate=d2_r,
            ofc_rate=ofc_r,
            rpe_estimate=rpe,
            cs_vta_weight=self.syn_cs_vta.mean_weight,
            approach_tendency=max(0, min(1, (d1_r - d2_r) / max(d1_r + d2_r, 1))),
        )
        self._results.append(result)
        return result

    def run_training(self, n_trials: int = 15) -> list[RewardTrialResult]:
        """訓練: CS + Reward を繰り返す。"""
        offset = len(self._results)
        return [self.run_trial(cs=True, reward=True, phase="training", trial_num=offset + i)
                for i in range(n_trials)]

    def run_omission_test(self, n_trials: int = 3) -> list[RewardTrialResult]:
        """報酬省略テスト: CS提示だが報酬なし → 負のRPE。"""
        offset = len(self._results)
        return [self.run_trial(cs=True, reward=False, phase="omission", trial_num=offset + i)
                for i in range(n_trials)]

    def run_probe(self, n_trials: int = 3) -> list[RewardTrialResult]:
        """プローブ: CS+Reward で反応を確認。"""
        offset = len(self._results)
        return [self.run_trial(cs=True, reward=True, phase="probe", trial_num=offset + i)
                for i in range(n_trials)]

    def run_unexpected_reward(self, n_trials: int = 3) -> list[RewardTrialResult]:
        """予想外報酬: CSなしで報酬だけ → 正のRPE。"""
        offset = len(self._results)
        return [self.run_trial(cs=False, reward=True, phase="unexpected", trial_num=offset + i)
                for i in range(n_trials)]

    @property
    def all_results(self) -> list[RewardTrialResult]:
        return list(self._results)
