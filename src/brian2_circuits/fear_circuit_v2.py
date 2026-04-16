"""恐怖条件付け/消去回路 v2（Brian2ベース、TimedArray高速版）。

5つの主要改善:
1. BLA分割: LA + BA
2. CeA分割: CeL_SOM+ + CeL_PKCdelta+ + CeM（脱抑制メカニズム）
3. mPFC分割: PL(恐怖発現) + IL(消去)
4. VIP脱抑制: VIP→SOM/PV抑制
5. BNST: 持続不安
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import brian2
brian2.prefs.codegen.target = "numpy"
from src.brian2_circuits.neuron_models import IZH_TIMED_EQS
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, Network,
    TimedArray, ms, second, defaultclock, start_scope,
)


@dataclass
class FearV2Config:
    n_la_exc: int = 40
    n_la_pv: int = 10
    n_la_vip: int = 5
    n_ba_exc: int = 30
    n_cel_som: int = 15
    n_cel_pkcd: int = 15
    n_cem: int = 12
    n_itc: int = 15
    n_pl: int = 30
    n_il: int = 30
    n_bnst: int = 15

    cs_amp: float = 10.0
    us_amp: float = 18.0
    bg_noise: float = 3.0
    sustained_threat_amp: float = 6.0

    dt_ms: float = 0.5
    duration_ms: float = 300.0
    cs_onset_ms: float = 50.0
    cs_dur_ms: float = 150.0
    us_onset_ms: float = 160.0
    us_dur_ms: float = 30.0




@dataclass
class FearV2TrialResult:
    trial_num: int
    phase: str
    cs_presented: bool
    us_presented: bool
    la_rate: float = 0.0
    ba_rate: float = 0.0
    cel_som_rate: float = 0.0
    cel_pkcd_rate: float = 0.0
    cem_rate: float = 0.0
    pl_rate: float = 0.0
    il_rate: float = 0.0
    bnst_rate: float = 0.0
    freeze_response: float = 0.0
    anxiety_level: float = 0.0
    cs_la_weight: float = 0.0


class FearCircuitV2:
    def __init__(self, config: FearV2Config | None = None):
        self.cfg = config or FearV2Config()
        self._results: list[FearV2TrialResult] = []
        self._saved_weights: dict[str, np.ndarray] = {}
        self._extinction_trial_count: int = 0  # 消去試行カウンター（IL強化用）

    def run_trial(
        self,
        cs: bool = True,
        us: bool = False,
        sustained_threat: bool = False,
        phase: str = "test",
        trial_num: int = 0,
    ) -> FearV2TrialResult:
        c = self.cfg
        start_scope()
        defaultclock.dt = c.dt_ms * ms

        n_steps = int(c.duration_ms / c.dt_ms)
        total_n = (c.n_la_exc + c.n_la_pv + c.n_la_vip +
                   c.n_ba_exc + c.n_cel_som + c.n_cel_pkcd + c.n_cem +
                   c.n_itc + c.n_pl + c.n_il + c.n_bnst)

        # === 入力スケジュールをTimedArrayで事前構築（ノイズ込み） ===
        # [R8 M3修正] 接続用RNG(固定)とノイズ用RNG(試行依存)を分離
        rng = np.random.default_rng(12345)  # 接続パターン用（全試行で同一トポロジー）
        noise_rng = np.random.default_rng(trial_num * 7 + 42)
        drive = c.bg_noise + noise_rng.normal(0, c.bg_noise * 0.3, (n_steps, total_n))

        # インデックス範囲
        idx = {}
        offset = 0
        for name, n in [("la_exc", c.n_la_exc), ("la_pv", c.n_la_pv), ("la_vip", c.n_la_vip),
                         ("ba_exc", c.n_ba_exc), ("cel_som", c.n_cel_som), ("cel_pkcd", c.n_cel_pkcd),
                         ("cem", c.n_cem), ("itc", c.n_itc), ("pl", c.n_pl), ("il", c.n_il),
                         ("bnst", c.n_bnst)]:
            idx[name] = (offset, offset + n)
            offset += n

        cs_start = int(c.cs_onset_ms / c.dt_ms)
        cs_end = int((c.cs_onset_ms + c.cs_dur_ms) / c.dt_ms)
        us_start = int(c.us_onset_ms / c.dt_ms)
        us_end = int((c.us_onset_ms + c.us_dur_ms) / c.dt_ms)

        if cs:
            la_s, la_e = idx["la_exc"]
            cs_target = la_s + c.n_la_exc // 3
            drive[cs_start:cs_end, la_s:cs_target] += c.cs_amp
            pl_s, pl_e = idx["pl"]
            drive[cs_start:cs_end, pl_s:pl_s + c.n_pl // 4] += 4.0
            il_s, il_e = idx["il"]
            # [STDP消去] 消去フェーズではIL入力を段階的に強化（消去学習の模倣）
            il_base = 4.0
            if phase == "extinction":
                self._extinction_trial_count += 1
                il_base += self._extinction_trial_count * 1.5  # 試行ごとに+1.5
            drive[cs_start:cs_end, il_s:il_s + c.n_il // 4] += il_base

        if us:
            la_s, la_e = idx["la_exc"]
            drive[us_start:us_end, la_s:la_e] += c.us_amp
            vip_s, vip_e = idx["la_vip"]
            drive[us_start:us_end, vip_s:vip_e] += 10.0  # VIP脱抑制ゲート
            # US→CeL SOM+直接活性化（BLA経由の恐怖学習シグナル）
            som_s, som_e = idx["cel_som"]
            drive[us_start:us_end, som_s:som_e] += c.us_amp * 0.5

        if sustained_threat:
            bnst_s, bnst_e = idx["bnst"]
            drive[:, bnst_s:bnst_e] += c.sustained_threat_amp

        # CeL背景ノイズ低減: SOM+ 15Hz, PKCd+ 5Hz方向
        som_s, som_e = idx["cel_som"]
        drive[:, som_s:som_e] *= 0.5  # SOM+背景50%
        pkcd_s, pkcd_e = idx["cel_pkcd"]
        drive[:, pkcd_s:pkcd_e] *= 0.15  # PKCd+背景15%（target 5Hz / ratio 3.0）
        # [R7 H1修正] CeM tonic hackを除去。代わりにPKCd+→CeM抑制弱化+SOM+→CeM強化で対応
        # CeM背景は通常の bg_noise のみ（外部注入なし）

        I_drive = TimedArray(drive, dt=c.dt_ms * ms)

        # === ニューロン集団（単一NeuronGroup） ===
        G = NeuronGroup(
            total_n, IZH_TIMED_EQS,
            threshold="v >= 30", reset="v = c; u += d",
            method="euler", name="all_neurons",
        )
        G.v = -65 + rng.normal(0, 2, total_n)
        G.tau_inh = 5 * ms  # GABA_A default

        # 細胞タイプ別パラメータ設定
        def _set_params(start, end, a, b, c_val, d, _=0):
            G.a[start:end] = a
            G.b[start:end] = b
            G.c[start:end] = c_val
            G.d[start:end] = d

        G.u = 0.2 * G.v[:]

        # 興奮性(RS): la_exc, ba_exc, cem, pl, il, bnst
        for name in ["la_exc", "ba_exc", "cem", "pl", "il", "bnst"]:
            s, e = idx[name]
            _set_params(s, e, 0.02, 0.2, -65, 8, 0)

        # PV+(FS): la_pv
        s, e = idx["la_pv"]
        _set_params(s, e, 0.1, 0.2, -65, 2, 0)

        # VIP: la_vip
        s, e = idx["la_vip"]
        _set_params(s, e, 0.02, 0.25, -65, 2, 0)

        # SOM+/PKCd+/ITC: LTS型
        for name in ["cel_som", "cel_pkcd", "itc"]:
            s, e = idx[name]
            _set_params(s, e, 0.02, 0.25, -65, 2, 0)

        # === シナプス結合 ===
        # [NH2修正] STDP付きシナプス関数を追加
        def _conn(src_name, tgt_name, p, w, inh=False, cid=0, stdp=False):
            ss, se = idx[src_name]
            ts, te = idx[tgt_name]
            sign_val = -1.0 if inh else 1.0

            if stdp and not inh:
                # [問題1修正] STDP付きシナプス — pre-before-post=LTP, post-before-pre=LTD
                syn = Synapses(G, G, model="""
                    w : 1
                    dA_ltp/dt = -A_ltp / (20*ms) : 1 (event-driven)
                    dA_ltd/dt = -A_ltd / (20*ms) : 1 (event-driven)
                """,
                on_pre="v_post += w; A_ltp += 0.005; w = clip(w + A_ltd, 0, 15)",
                on_post="A_ltd -= 0.003; w = clip(w + A_ltp, 0, 15)",
                name=f"s{cid}_{src_name[:4]}_{tgt_name[:4]}")
            else:
                syn = Synapses(G, G, "w : 1", on_pre=f"v_post += {sign_val} * w",
                               name=f"s{cid}_{src_name[:4]}_{tgt_name[:4]}")

            conn_i, conn_j = [], []
            for i in range(ss, se):
                for j in range(ts, te):
                    if rng.random() < p:
                        conn_i.append(i)
                        conn_j.append(j)
            if conn_i:
                syn.connect(i=conn_i, j=conn_j)
                n_src = se - ss
                # 小集団(N<20)ではスケーリングなし（CeL等の抑制を保護）
                if n_src >= 20:
                    n_inputs = max(1, n_src * p)
                    w_scaled = w / np.sqrt(n_inputs)
                else:
                    w_scaled = w
                syn.w = rng.uniform(0, w_scaled, len(syn))
            return syn

        synapses = []
        cid = 0
        # LA内E-I
        synapses.append(_conn("la_exc", "la_pv", 0.3, 3.0, cid=cid)); cid += 1
        synapses.append(_conn("la_pv", "la_exc", 0.4, 4.0, inh=True, cid=cid)); cid += 1
        # VIP脱抑制
        synapses.append(_conn("la_vip", "la_pv", 0.5, 5.0, inh=True, cid=cid)); cid += 1

        # LA → BA [NH2: STDP付き — 恐怖条件付けの主要可塑的結合]
        synapses.append(_conn("la_exc", "ba_exc", 0.3, 3.0, cid=cid, stdp=True)); cid += 1

        # LA/BA → CeL_SOM+ [Step0較正: w=2.0でSOM+を15Hz方向に]
        synapses.append(_conn("la_exc", "cel_som", 0.25, 2.0, cid=cid, stdp=True)); cid += 1
        synapses.append(_conn("ba_exc", "cel_som", 0.15, 1.5, cid=cid)); cid += 1

        # CeL相互抑制 [Step0較正: SOM+→PKCd+を強化してPKCd+を抑制=脱抑制メカニズム]
        synapses.append(_conn("cel_som", "cel_pkcd", 0.7, 8.0, inh=True, cid=cid)); cid += 1
        synapses.append(_conn("cel_pkcd", "cel_som", 0.3, 3.0, inh=True, cid=cid)); cid += 1

        # [R7 H1] PKCd → CeM 抑制 (弱化: 1.5でCeM発火を許可)
        synapses.append(_conn("cel_pkcd", "cem", 0.3, 1.5, inh=True, cid=cid)); cid += 1
        # SOM+ → CeM 興奮 (強化: CeM脱抑制の主要経路)
        synapses.append(_conn("cel_som", "cem", 0.6, 8.0, cid=cid)); cid += 1
        # BA → CeM 直接経路 (Pitkänen 2000)
        synapses.append(_conn("ba_exc", "cem", 0.25, 4.0, cid=cid)); cid += 1

        # PL → LA (恐怖発現促進)
        synapses.append(_conn("pl", "la_exc", 0.2, 2.0, cid=cid)); cid += 1

        # IL → ITC → CeM (消去)
        synapses.append(_conn("il", "itc", 0.3, 2.5, cid=cid)); cid += 1
        synapses.append(_conn("itc", "cem", 0.5, 5.0, inh=True, cid=cid)); cid += 1

        # BA → BNST
        synapses.append(_conn("ba_exc", "bnst", 0.2, 2.0, cid=cid)); cid += 1
        synapses.append(_conn("cel_som", "bnst", 0.2, 1.5, cid=cid)); cid += 1

        # モニター（全ニューロンを1つのモニターで監視、後でインデックスで分離）
        spike_mon = SpikeMonitor(G, name="spike_mon")

        # [問題1] 試行間STDP重み復元
        for syn in synapses:
            if syn.name in self._saved_weights and len(syn) > 0:
                prev = self._saved_weights[syn.name]
                if len(prev) == len(syn):
                    syn.w = prev

        # ネットワーク構築・実行
        net = Network(G, spike_mon, *synapses)
        net.run(c.duration_ms * ms)

        # [問題1] 試行後STDP重み保存
        for syn in synapses:
            if len(syn) > 0 and hasattr(syn, 'A_ltp'):
                self._saved_weights[syn.name] = np.array(syn.w[:])

        # === 結果集計 ===
        dur_s = c.duration_ms / 1000.0
        all_spike_indices = np.array(spike_mon.i[:])

        def _rate(name):
            s, e = idx[name]
            n = e - s
            if n == 0:
                return 0.0
            count = np.sum((all_spike_indices >= s) & (all_spike_indices < e))
            return count / n / dur_s

        cem_r = _rate("cem")
        bnst_r = _rate("bnst")

        result = FearV2TrialResult(
            trial_num=trial_num, phase=phase,
            cs_presented=cs, us_presented=us,
            la_rate=_rate("la_exc"), ba_rate=_rate("ba_exc"),
            cel_som_rate=_rate("cel_som"), cel_pkcd_rate=_rate("cel_pkcd"),
            cem_rate=cem_r, pl_rate=_rate("pl"), il_rate=_rate("il"),
            bnst_rate=bnst_r,
            freeze_response=min(1.0, cem_r / 40.0),
            anxiety_level=min(1.0, bnst_r / 30.0),
            cs_la_weight=0.0,  # 試行間可塑性は外部で追跡
        )
        self._results.append(result)
        return result

    def run_conditioning(self, n_trials: int = 8) -> list[FearV2TrialResult]:
        offset = len(self._results)
        return [self.run_trial(cs=True, us=True, phase="conditioning", trial_num=offset + i)
                for i in range(n_trials)]

    def run_extinction(self, n_trials: int = 15) -> list[FearV2TrialResult]:
        offset = len(self._results)
        return [self.run_trial(cs=True, us=False, phase="extinction", trial_num=offset + i)
                for i in range(n_trials)]

    def run_test(self, n_trials: int = 3) -> list[FearV2TrialResult]:
        offset = len(self._results)
        return [self.run_trial(cs=True, us=False, phase="test", trial_num=offset + i)
                for i in range(n_trials)]

    def run_sustained_anxiety_test(self, n_trials: int = 3) -> list[FearV2TrialResult]:
        offset = len(self._results)
        return [self.run_trial(sustained_threat=True, phase="sustained_anxiety", trial_num=offset + i)
                for i in range(n_trials)]

    @property
    def all_results(self) -> list[FearV2TrialResult]:
        return list(self._results)
