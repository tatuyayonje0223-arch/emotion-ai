"""恐怖条件付け/消去回路。スパイキングニューロンレベルの忠実な実装。

回路構成（LeDoux, Quirk et al. に基づく）:
  - BLA (基底外側扁桃体): CS-US連合学習の場。興奮性(80%) + 抑制性(20%)
  - CeA (中心扁桃体): 恐怖出力。BLAから入力を受け、防御行動を駆動
  - mPFC-IL (内側前頭前皮質-辺縁下野): 消去学習。BLAの恐怖ニューロンを抑制
  - ITC (介在細胞塊): mPFC-IL→ITC→CeA の消去経路

入力:
  - CS (条件刺激): 例=音。感覚皮質/視床→BLA
  - US (無条件刺激): 例=電撃。痛覚→BLA

学習則:
  - BLAのCS入力シナプス: STDP + US信号による強化（恐怖条件付け）
  - mPFC-IL→ITC: CS単独提示で徐々に強化（消去学習）

検証対象データ（文献値）:
  - 条件付け前: CS誘発BLA発火率 ~5-10 Hz
  - 条件付け後: CS誘発BLA発火率 ~20-40 Hz
  - 消去後: CS誘発BLA発火率 低下（mPFC-IL抑制）
  - 条件付け: 5-10 CS-US対提示で獲得
  - 消去: 10-30 CS単独提示で消去
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.spiking.neuron import IzhikevichPopulation, RS, FS, LTS, NeuronParams
from src.spiking.synapse import SynapticConnection, SynapseParams


@dataclass
class FearCircuitConfig:
    """恐怖回路のパラメータ。"""

    # ニューロン数
    n_bla_exc: int = 200       # BLA興奮性（主細胞）
    n_bla_inh: int = 50        # BLA抑制性（局所介在）
    n_cea: int = 80            # CeA出力ニューロン
    n_mpfc_il: int = 100       # mPFC-IL
    n_itc: int = 40            # 介在細胞塊

    # 入力強度
    cs_current: float = 8.0    # CS入力電流
    us_current: float = 15.0   # US入力電流
    background_noise: float = 3.0  # 背景ノイズ

    # シミュレーション
    dt: float = 0.5            # 時間ステップ (ms)
    trial_duration_ms: float = 500.0  # 1試行の長さ
    cs_onset_ms: float = 100.0       # CS開始時刻
    cs_duration_ms: float = 300.0    # CS持続時間
    us_onset_ms: float = 350.0       # US開始時刻（CS後半に重畳）
    us_duration_ms: float = 50.0     # US持続時間


@dataclass
class TrialResult:
    """1試行の結果。"""

    trial_num: int
    phase: str  # "conditioning", "extinction", "recovery_test"
    cs_presented: bool
    us_presented: bool
    bla_firing_rate: float     # Hz
    cea_firing_rate: float     # Hz
    mpfc_firing_rate: float    # Hz
    cs_bla_weight: float       # CS→BLA平均結合重み
    mpfc_itc_weight: float     # mPFC→ITC平均結合重み
    freeze_output: float       # 凍結反応強度 (CeA発火率の正規化)


class FearCircuit:
    """恐怖条件付け/消去の完全回路。"""

    def __init__(self, config: FearCircuitConfig | None = None):
        self.cfg = config or FearCircuitConfig()
        self._build_circuit()
        self._trial_results: list[TrialResult] = []

    def _build_circuit(self) -> None:
        c = self.cfg

        # ニューロン集団
        self.bla_exc = IzhikevichPopulation(c.n_bla_exc, RS, noise_std=1.0)
        self.bla_inh = IzhikevichPopulation(c.n_bla_inh, FS, noise_std=0.5)
        self.cea = IzhikevichPopulation(c.n_cea, RS, noise_std=0.8)
        self.mpfc_il = IzhikevichPopulation(c.n_mpfc_il, RS, noise_std=1.0)
        self.itc = IzhikevichPopulation(c.n_itc, LTS, noise_std=0.5)

        # シナプス結合
        # CS → BLA_exc（可塑的: 恐怖条件付けの主要部位）
        self.syn_cs_bla = SynapticConnection(
            c.n_bla_exc, c.n_bla_exc,  # CS入力はBLA_excと同じ次元で注入
            connection_prob=0.3, w_init=1.5,
            params=SynapseParams(A_plus=0.008, A_minus=0.005, w_max=12.0),
            seed=1,
        )

        # US → BLA_exc（強い固定結合）
        self.syn_us_bla = SynapticConnection(
            c.n_bla_exc, c.n_bla_exc,
            connection_prob=0.4, w_init=4.0,
            params=SynapseParams(w_max=8.0),
            seed=2,
        )

        # BLA_exc → BLA_inh（フィードバック抑制）
        self.syn_bla_exc_inh = SynapticConnection(
            c.n_bla_exc, c.n_bla_inh,
            connection_prob=0.3, w_init=3.0, seed=3,
        )

        # BLA_inh → BLA_exc（抑制）
        self.syn_bla_inh_exc = SynapticConnection(
            c.n_bla_inh, c.n_bla_exc,
            connection_prob=0.4, w_init=4.0, is_inhibitory=True, seed=4,
        )

        # BLA_exc → CeA（恐怖出力経路）
        self.syn_bla_cea = SynapticConnection(
            c.n_bla_exc, c.n_cea,
            connection_prob=0.3, w_init=3.0, seed=5,
        )

        # mPFC-IL → ITC（消去経路）
        self.syn_mpfc_itc = SynapticConnection(
            c.n_mpfc_il, c.n_itc,
            connection_prob=0.3, w_init=1.0,
            params=SynapseParams(A_plus=0.006, A_minus=0.003, w_max=10.0),
            seed=6,
        )

        # ITC → CeA（抑制: 消去のメカニズム）
        self.syn_itc_cea = SynapticConnection(
            c.n_itc, c.n_cea,
            connection_prob=0.5, w_init=5.0, is_inhibitory=True, seed=7,
        )

        # CS → mPFC-IL（CS情報はmPFCにも届く）
        self.syn_cs_mpfc = SynapticConnection(
            c.n_mpfc_il, c.n_mpfc_il,
            connection_prob=0.2, w_init=1.0, seed=8,
        )

    def run_trial(
        self,
        cs: bool = True,
        us: bool = False,
        phase: str = "test",
        trial_num: int = 0,
    ) -> TrialResult:
        """1試行を実行する。"""
        c = self.cfg
        n_steps = int(c.trial_duration_ms / c.dt)
        cs_start = int(c.cs_onset_ms / c.dt)
        cs_end = int((c.cs_onset_ms + c.cs_duration_ms) / c.dt)
        us_start = int(c.us_onset_ms / c.dt)
        us_end = int((c.us_onset_ms + c.us_duration_ms) / c.dt)

        bla_spikes = 0
        cea_spikes = 0
        mpfc_spikes = 0
        cs_steps = 0

        for t in range(n_steps):
            # 入力電流
            I_cs = np.zeros(c.n_bla_exc)
            I_us = np.zeros(c.n_bla_exc)
            I_cs_mpfc = np.zeros(c.n_mpfc_il)

            if cs and cs_start <= t < cs_end:
                # CSは一部のBLAニューロンに入力（感覚表象）
                cs_target = np.zeros(c.n_bla_exc)
                cs_target[:c.n_bla_exc // 3] = c.cs_current  # 前1/3がCS応答
                I_cs = self.syn_cs_bla.compute_current(cs_target > 0)
                I_cs_mpfc[:c.n_mpfc_il // 3] = c.cs_current * 0.5
                cs_steps += 1

            if us and us_start <= t < us_end:
                us_target = np.full(c.n_bla_exc, c.us_current * 0.3)
                I_us = self.syn_us_bla.compute_current(us_target > 0)

            # 背景ノイズ
            bg_bla = np.random.normal(c.background_noise, 1.0, c.n_bla_exc)
            bg_inh = np.random.normal(c.background_noise * 0.5, 0.5, c.n_bla_inh)
            bg_cea = np.random.normal(c.background_noise * 0.5, 0.5, c.n_cea)
            bg_mpfc = np.random.normal(c.background_noise, 1.0, c.n_mpfc_il)
            bg_itc = np.random.normal(c.background_noise * 0.3, 0.3, c.n_itc)

            # BLA_exc: CS + US + 背景 + BLA_inh→BLA_exc抑制
            I_bla_inh = self.syn_bla_inh_exc.compute_current(self.bla_inh.fired)
            I_bla_total = I_cs + I_us + bg_bla + I_bla_inh
            bla_fired = self.bla_exc.step(I_bla_total, c.dt)

            # BLA_inh
            I_exc_to_inh = self.syn_bla_exc_inh.compute_current(bla_fired)
            self.bla_inh.step(I_exc_to_inh + bg_inh, c.dt)

            # CeA: BLA入力 + ITC抑制
            I_bla_to_cea = self.syn_bla_cea.compute_current(bla_fired)
            I_itc_to_cea = self.syn_itc_cea.compute_current(self.itc.fired)
            cea_fired = self.cea.step(I_bla_to_cea + I_itc_to_cea + bg_cea, c.dt)

            # mPFC-IL
            I_cs_to_mpfc = self.syn_cs_mpfc.compute_current(I_cs_mpfc > 0)
            mpfc_fired = self.mpfc_il.step(I_cs_to_mpfc + bg_mpfc, c.dt)

            # ITC: mPFC入力
            I_mpfc_to_itc = self.syn_mpfc_itc.compute_current(mpfc_fired)
            self.itc.step(I_mpfc_to_itc + bg_itc, c.dt)

            # STDP更新（CS期間中のみ）
            if cs and cs_start <= t < cs_end:
                cs_input_fired = np.zeros(c.n_bla_exc, dtype=bool)
                cs_input_fired[:c.n_bla_exc // 3] = True
                self.syn_cs_bla.update_stdp(cs_input_fired, bla_fired, c.dt)
                self.syn_mpfc_itc.update_stdp(mpfc_fired, self.itc.fired, c.dt)

            # スパイクカウント（CS期間中）
            if cs and cs_start <= t < cs_end:
                bla_spikes += bla_fired.sum()
                cea_spikes += cea_fired.sum()
                mpfc_spikes += mpfc_fired.sum()

        # 条件付けフェーズ: US信号でCS→BLA結合を強化
        if us:
            self.syn_cs_bla.apply_reward_modulation(da_signal=1.0)
        elif phase == "extinction":
            # 消去: mPFC→ITC結合を強化（CS単独提示で消去学習）
            self.syn_mpfc_itc.apply_stdp_direct()
            self.syn_cs_bla.apply_reward_modulation(da_signal=-0.1)  # 弱い減弱

        # 発火率計算
        cs_duration_s = (c.cs_duration_ms / 1000.0) if cs_steps > 0 else 1.0
        bla_rate = (bla_spikes / c.n_bla_exc) / cs_duration_s if cs_steps > 0 else 0
        cea_rate = (cea_spikes / c.n_cea) / cs_duration_s if cs_steps > 0 else 0
        mpfc_rate = (mpfc_spikes / c.n_mpfc_il) / cs_duration_s if cs_steps > 0 else 0

        result = TrialResult(
            trial_num=trial_num,
            phase=phase,
            cs_presented=cs,
            us_presented=us,
            bla_firing_rate=bla_rate,
            cea_firing_rate=cea_rate,
            mpfc_firing_rate=mpfc_rate,
            cs_bla_weight=self.syn_cs_bla.mean_weight,
            mpfc_itc_weight=self.syn_mpfc_itc.mean_weight,
            freeze_output=min(1.0, cea_rate / 40.0),  # 正規化
        )
        self._trial_results.append(result)
        return result

    def run_conditioning(self, n_trials: int = 10) -> list[TrialResult]:
        """条件付けフェーズ。CS + US を繰り返し提示。"""
        results = []
        offset = len(self._trial_results)
        for i in range(n_trials):
            r = self.run_trial(cs=True, us=True, phase="conditioning", trial_num=offset + i)
            results.append(r)
        return results

    def run_extinction(self, n_trials: int = 20) -> list[TrialResult]:
        """消去フェーズ。CS のみ繰り返し提示。"""
        results = []
        offset = len(self._trial_results)
        for i in range(n_trials):
            r = self.run_trial(cs=True, us=False, phase="extinction", trial_num=offset + i)
            results.append(r)
        return results

    def run_test(self, n_trials: int = 3) -> list[TrialResult]:
        """テストフェーズ。CS のみ提示して反応を計測。"""
        results = []
        offset = len(self._trial_results)
        for i in range(n_trials):
            r = self.run_trial(cs=True, us=False, phase="test", trial_num=offset + i)
            results.append(r)
        return results

    @property
    def all_results(self) -> list[TrialResult]:
        return list(self._trial_results)
