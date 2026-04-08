"""ストレス応答回路。HPA軸のスパイキング実装。

HPA軸カスケード（McEwen, de Kloet et al.）:
  1. ストレッサー → 扁桃体BLA活性化 → PVN(視床下部室傍核)
  2. PVN → CRH放出 → 下垂体前葉 → ACTH放出
  3. ACTH → 副腎皮質 → コルチゾール放出
  4. コルチゾール → 海馬MR/GR + PFC → PVNの負のフィードバック（抑制）
  5. 慢性ストレス: 海馬のGRダウンレギュレーション → 負のFB機能低下 → コルチゾール持続上昇

回路構成:
  - BLA: ストレッサー検出（恐怖回路と共有）
  - PVN: CRH産生ニューロン。ストレス応答の起点
  - 海馬: 負のフィードバック。MR(低CORT)とGR(高CORT)で二相性
  - mPFC: 負のフィードバック（認知制御）
  - LC (青斑核): NE産生。急性ストレスで活性化

コルチゾールの時間経過:
  - 急性: ストレス後15-30分でピーク、1-2時間で回復
  - シミュレーション上は1ステップ≈1分として近似

検証対象:
  - 急性ストレス→コルチゾール上昇→回復（逆U字型の時間経過）
  - 慢性ストレス→コルチゾール持続上昇+海馬機能低下
  - 負のフィードバック遮断→コルチゾール暴走
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.spiking.neuron import IzhikevichPopulation, RS, FS, NeuronParams
from src.spiking.synapse import SynapticConnection, SynapseParams


@dataclass
class StressCircuitConfig:
    """ストレス回路パラメータ。"""

    n_bla: int = 60          # BLA（ストレッサー検出）
    n_pvn: int = 40          # PVN（CRH産生）
    n_hippo: int = 60        # 海馬（負のFB）
    n_mpfc: int = 40         # mPFC（認知制御FB）
    n_lc: int = 30           # LC（NE産生）

    stressor_current: float = 10.0
    background_noise: float = 2.5
    dt: float = 0.5

    # 1ステップ≈1分。1試行=60ステップ≈1時間
    steps_per_trial: int = 60

    # コルチゾール動態
    crh_to_cort_gain: float = 0.02   # PVN活性→コルチゾール上昇速度
    cort_decay_rate: float = 0.01    # コルチゾール自然減衰
    cort_baseline: float = 0.15      # 基底コルチゾール
    mr_threshold: float = 0.3        # MR活性化閾値（低CORT）
    gr_threshold: float = 0.5        # GR活性化閾値（高CORT）
    gr_feedback_gain: float = 0.8    # GR負のFB強度
    chronic_gr_downreg_rate: float = 0.005  # 慢性ストレスでのGRダウンレギュレーション速度


@dataclass
class StressTrialResult:
    """ストレス応答1試行の結果。"""

    trial_num: int
    phase: str
    stressor_present: bool
    cortisol_peak: float
    cortisol_final: float
    cortisol_trajectory: list[float]
    pvn_mean_rate: float
    bla_mean_rate: float
    hippo_mean_rate: float
    lc_mean_rate: float
    ne_level: float
    gr_sensitivity: float   # GR感度（慢性で低下）


class StressCircuit:
    """HPA軸ストレス応答回路。"""

    def __init__(self, config: StressCircuitConfig | None = None):
        self.cfg = config or StressCircuitConfig()
        self._build()
        self._results: list[StressTrialResult] = []

        # 内部化学状態
        self.cortisol = self.cfg.cort_baseline
        self.ne_level = 0.3
        self.gr_sensitivity = 1.0  # 慢性ストレスで低下

    def _build(self) -> None:
        c = self.cfg

        self.bla = IzhikevichPopulation(c.n_bla, RS, noise_std=1.0)
        self.pvn = IzhikevichPopulation(c.n_pvn, RS, noise_std=0.8)
        self.hippo = IzhikevichPopulation(c.n_hippo, RS, noise_std=1.0)
        self.mpfc = IzhikevichPopulation(c.n_mpfc, RS, noise_std=1.0)
        self.lc = IzhikevichPopulation(c.n_lc, RS, noise_std=0.5)

        # BLA → PVN（ストレス→CRH駆動）
        self.syn_bla_pvn = SynapticConnection(
            c.n_bla, c.n_pvn, connection_prob=0.4, w_init=4.0, seed=20,
        )

        # BLA → LC（ストレス→NE放出）
        self.syn_bla_lc = SynapticConnection(
            c.n_bla, c.n_lc, connection_prob=0.3, w_init=3.0, seed=21,
        )

        # 海馬 → PVN（負のFB: 抑制）
        self.syn_hippo_pvn = SynapticConnection(
            c.n_hippo, c.n_pvn, connection_prob=0.4, w_init=5.0,
            is_inhibitory=True, seed=22,
        )

        # mPFC → PVN（認知制御FB: 抑制）
        self.syn_mpfc_pvn = SynapticConnection(
            c.n_mpfc, c.n_pvn, connection_prob=0.3, w_init=3.0,
            is_inhibitory=True, seed=23,
        )

        # LC → BLA（NE→脅威感度↑）
        self.syn_lc_bla = SynapticConnection(
            c.n_lc, c.n_bla, connection_prob=0.3, w_init=2.0, seed=24,
        )

    def run_trial(
        self,
        stressor: bool = True,
        stressor_intensity: float = 1.0,
        phase: str = "acute",
        trial_num: int = 0,
    ) -> StressTrialResult:
        c = self.cfg
        cort_trajectory = []
        pvn_spikes = 0
        bla_spikes = 0
        hippo_spikes = 0
        lc_spikes = 0
        cort_peak = self.cortisol

        for t in range(c.steps_per_trial):
            # ストレッサー入力
            I_stressor = np.zeros(c.n_bla)
            if stressor:
                I_stressor = np.full(c.n_bla, c.stressor_current * stressor_intensity)

            bg = lambda n: np.random.normal(c.background_noise, 1.0, n)

            # LC → BLA
            I_lc_bla = self.syn_lc_bla.compute_current(self.lc.fired)

            # BLA
            bla_fired = self.bla.step(I_stressor + I_lc_bla + bg(c.n_bla), c.dt)

            # 海馬: コルチゾールレベルに依存する活性
            # MR(低CORT時活性): tonic抑制制御
            # GR(高CORT時活性): 負のフィードバック
            hippo_input = bg(c.n_hippo)
            if self.cortisol > c.gr_threshold:
                # GR活性化 → 海馬が負のFBを強める
                hippo_input += np.full(c.n_hippo, 5.0 * self.gr_sensitivity)
            elif self.cortisol > c.mr_threshold:
                hippo_input += np.full(c.n_hippo, 2.0)
            hippo_fired = self.hippo.step(hippo_input, c.dt)

            # mPFC（ベースライン活性）
            mpfc_fired = self.mpfc.step(bg(c.n_mpfc), c.dt)

            # PVN: BLA興奮 - 海馬抑制 - mPFC抑制
            I_bla_pvn = self.syn_bla_pvn.compute_current(bla_fired)
            I_hippo_pvn = self.syn_hippo_pvn.compute_current(hippo_fired)
            I_mpfc_pvn = self.syn_mpfc_pvn.compute_current(mpfc_fired)
            pvn_fired = self.pvn.step(
                I_bla_pvn + I_hippo_pvn + I_mpfc_pvn + bg(c.n_pvn), c.dt,
            )

            # LC: BLA→LC（NE放出）
            I_bla_lc = self.syn_bla_lc.compute_current(bla_fired)
            lc_fired = self.lc.step(I_bla_lc + bg(c.n_lc) * 0.5, c.dt)

            # コルチゾール動態
            pvn_rate = pvn_fired.sum() / c.n_pvn
            cort_drive = pvn_rate * c.crh_to_cort_gain
            cort_decay = (self.cortisol - c.cort_baseline) * c.cort_decay_rate
            self.cortisol += cort_drive - cort_decay
            self.cortisol = max(0.0, min(1.0, self.cortisol))
            cort_trajectory.append(self.cortisol)
            cort_peak = max(cort_peak, self.cortisol)

            # NE動態
            lc_rate = lc_fired.sum() / c.n_lc
            self.ne_level += lc_rate * 0.01 - (self.ne_level - 0.3) * 0.02
            self.ne_level = max(0.0, min(1.0, self.ne_level))

            # スパイクカウント
            pvn_spikes += pvn_fired.sum()
            bla_spikes += bla_fired.sum()
            hippo_spikes += hippo_fired.sum()
            lc_spikes += lc_fired.sum()

        # 慢性ストレス: GRダウンレギュレーション
        if stressor and self.cortisol > c.gr_threshold:
            self.gr_sensitivity = max(0.1,
                self.gr_sensitivity - c.chronic_gr_downreg_rate * stressor_intensity)

        duration_s = c.steps_per_trial * c.dt / 1000.0
        result = StressTrialResult(
            trial_num=trial_num,
            phase=phase,
            stressor_present=stressor,
            cortisol_peak=cort_peak,
            cortisol_final=self.cortisol,
            cortisol_trajectory=cort_trajectory,
            pvn_mean_rate=(pvn_spikes / c.n_pvn) / duration_s,
            bla_mean_rate=(bla_spikes / c.n_bla) / duration_s,
            hippo_mean_rate=(hippo_spikes / c.n_hippo) / duration_s,
            lc_mean_rate=(lc_spikes / c.n_lc) / duration_s,
            ne_level=self.ne_level,
            gr_sensitivity=self.gr_sensitivity,
        )
        self._results.append(result)
        return result

    def run_acute_stress(self, n_trials: int = 1, intensity: float = 1.0) -> list[StressTrialResult]:
        offset = len(self._results)
        return [self.run_trial(True, intensity, "acute", offset + i) for i in range(n_trials)]

    def run_recovery(self, n_trials: int = 3) -> list[StressTrialResult]:
        offset = len(self._results)
        return [self.run_trial(False, 0, "recovery", offset + i) for i in range(n_trials)]

    def run_chronic_stress(self, n_trials: int = 10, intensity: float = 0.7) -> list[StressTrialResult]:
        offset = len(self._results)
        return [self.run_trial(True, intensity, "chronic", offset + i) for i in range(n_trials)]

    @property
    def all_results(self) -> list[StressTrialResult]:
        return list(self._results)
