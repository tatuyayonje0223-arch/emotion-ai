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


IZH_EQS = """
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I_drive(t, i)) / ms : 1
    du/dt = (a*(b*v - u)) / ms : 1
    a : 1 (constant)
    b : 1 (constant)
    c : 1 (constant)
    d : 1 (constant)
"""


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
        self._cs_la_weights: list[float] = []  # 試行間の重み追跡

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
        rng = np.random.default_rng(trial_num * 7 + 42)
        drive = c.bg_noise + rng.normal(0, c.bg_noise * 0.3, (n_steps, total_n))

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
            cs_target = la_s + c.n_la_exc // 3  # 前1/3がCS応答
            drive[cs_start:cs_end, la_s:cs_target] += c.cs_amp
            pl_s, pl_e = idx["pl"]
            drive[cs_start:cs_end, pl_s:pl_s + c.n_pl // 4] += 4.0
            il_s, il_e = idx["il"]
            drive[cs_start:cs_end, il_s:il_s + c.n_il // 4] += 4.0

        if us:
            la_s, la_e = idx["la_exc"]
            drive[us_start:us_end, la_s:la_e] += c.us_amp
            vip_s, vip_e = idx["la_vip"]
            drive[us_start:us_end, vip_s:vip_e] += 10.0  # VIP脱抑制ゲート

        if sustained_threat:
            bnst_s, bnst_e = idx["bnst"]
            drive[:, bnst_s:bnst_e] += c.sustained_threat_amp

        I_drive = TimedArray(drive, dt=c.dt_ms * ms)

        # === ニューロン集団（単一NeuronGroup） ===
        G = NeuronGroup(
            total_n, IZH_EQS,
            threshold="v >= 30", reset="v = c; u += d",
            method="euler", name="all_neurons",
        )
        G.v = -65 + rng.normal(0, 2, total_n)

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

        # === シナプス結合（全体NeuronGroupに対してインデックスベースで接続） ===
        def _conn(src_name, tgt_name, p, w, inh=False, cid=0):
            ss, se = idx[src_name]
            ts, te = idx[tgt_name]
            n_src = se - ss
            n_tgt = te - ts
            sign_val = -1.0 if inh else 1.0
            syn = Synapses(G, G, "w : 1", on_pre=f"v_post += {sign_val} * w",
                           name=f"s{cid}_{src_name[:4]}_{tgt_name[:4]}")
            # インデックスベースの条件付き接続
            conn_i = []
            conn_j = []
            for i in range(ss, se):
                for j in range(ts, te):
                    if rng.random() < p:
                        conn_i.append(i)
                        conn_j.append(j)
            if conn_i:
                syn.connect(i=conn_i, j=conn_j)
                # [監査P2] バランスネットワーク重みスケーリング: w / sqrt(N_pre * p)
                n_inputs = max(1, (se - ss) * p)
                w_scaled = w / np.sqrt(n_inputs)
                syn.w = rng.uniform(0, w_scaled, len(syn))
            return syn

        synapses = []
        cid = 0
        # LA内E-I
        synapses.append(_conn("la_exc", "la_pv", 0.3, 3.0, cid=cid)); cid += 1
        synapses.append(_conn("la_pv", "la_exc", 0.4, 4.0, inh=True, cid=cid)); cid += 1
        # VIP脱抑制
        synapses.append(_conn("la_vip", "la_pv", 0.5, 5.0, inh=True, cid=cid)); cid += 1

        # LA → BA
        synapses.append(_conn("la_exc", "ba_exc", 0.3, 3.0, cid=cid)); cid += 1

        # LA/BA → CeL_SOM+
        synapses.append(_conn("la_exc", "cel_som", 0.3, 3.0, cid=cid)); cid += 1
        synapses.append(_conn("ba_exc", "cel_som", 0.2, 2.0, cid=cid)); cid += 1

        # CeL相互抑制
        synapses.append(_conn("cel_som", "cel_pkcd", 0.5, 5.0, inh=True, cid=cid)); cid += 1
        synapses.append(_conn("cel_pkcd", "cel_som", 0.3, 3.0, inh=True, cid=cid)); cid += 1

        # PKCd → CeM トニック抑制
        synapses.append(_conn("cel_pkcd", "cem", 0.5, 6.0, inh=True, cid=cid)); cid += 1
        # SOM+ → CeM 弱い直接興奮
        synapses.append(_conn("cel_som", "cem", 0.2, 1.0, cid=cid)); cid += 1

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

        # ネットワーク構築・実行
        net = Network(G, spike_mon, *synapses)
        net.run(c.duration_ms * ms)

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
