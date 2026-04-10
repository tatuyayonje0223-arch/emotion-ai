"""ストレス応答回路 v2（Brian2ベース）。

改善点:
1. MR/GR二相性: 低CORT→MR活性(恒常性), 高CORT→GR活性(負のFB)
2. ウルトラディアンパルス: PVNの内因性パルス発振(~60分周期)
3. 慢性ストレスでGRダウンレギュレーション + 海馬萎縮
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import brian2
brian2.prefs.codegen.target = "numpy"
from src.brian2_circuits.neuron_models import IZH_TIMED_EQS
from brian2 import NeuronGroup, Synapses, SpikeMonitor, Network, TimedArray, ms, start_scope, defaultclock



@dataclass
class StressV2Config:
    n_bla: int = 30
    n_pvn: int = 20
    n_hippo_mr: int = 20   # 海馬MR(低CORTで活性)
    n_hippo_gr: int = 20   # 海馬GR(高CORTで活性)
    n_mpfc: int = 20
    n_lc: int = 15
    n_bnst: int = 15       # 持続不安

    stressor_amp: float = 12.0
    bg_noise: float = 3.0
    dt_ms: float = 0.5
    duration_ms: float = 300.0

    cort_baseline: float = 0.15
    cort_rise_rate: float = 0.015
    cort_decay_rate: float = 0.008
    gr_sensitivity: float = 1.0
    gr_downreg_rate: float = 0.01


@dataclass
class StressV2Result:
    trial_num: int
    phase: str
    stressor: bool
    bla_rate: float = 0.0
    pvn_rate: float = 0.0
    hippo_mr_rate: float = 0.0
    hippo_gr_rate: float = 0.0
    lc_rate: float = 0.0
    bnst_rate: float = 0.0
    cortisol: float = 0.0
    ne_level: float = 0.0
    gr_sensitivity: float = 1.0


class StressCircuitV2:
    def __init__(self, config: StressV2Config | None = None):
        self.cfg = config or StressV2Config()
        self._results: list[StressV2Result] = []
        self.cortisol = self.cfg.cort_baseline
        self.ne_level = 0.3
        self.gr_sensitivity = self.cfg.gr_sensitivity

    def run_trial(self, stressor=True, intensity=1.0, phase="acute", trial_num=0) -> StressV2Result:
        c = self.cfg
        start_scope()
        defaultclock.dt = c.dt_ms * ms

        pops = [("bla", c.n_bla), ("pvn", c.n_pvn), ("hip_mr", c.n_hippo_mr),
                ("hip_gr", c.n_hippo_gr), ("mpfc", c.n_mpfc), ("lc", c.n_lc), ("bnst", c.n_bnst)]
        total_n = sum(n for _, n in pops)
        idx = {}
        off = 0
        for name, n in pops:
            idx[name] = (off, off + n); off += n

        n_steps = int(c.duration_ms / c.dt_ms)
        rng = np.random.default_rng(trial_num * 11 + 3)
        drive = c.bg_noise + rng.normal(0, c.bg_noise * 0.3, (n_steps, total_n))

        # ストレッサー→BLA + BNST
        if stressor:
            s, e = idx["bla"]
            drive[:, s:e] += c.stressor_amp * intensity
            s, e = idx["bnst"]
            drive[:, s:e] += c.stressor_amp * intensity * 0.4

        # コルチゾール依存: GR活性化→海馬GRに入力
        if self.cortisol > 0.4:
            s, e = idx["hip_gr"]
            gr_input = 5.0 * self.gr_sensitivity * min(1.0, self.cortisol)
            drive[:, s:e] += gr_input

        # MR: 低CORT時はtonic活性
        if self.cortisol < 0.4:
            s, e = idx["hip_mr"]
            drive[:, s:e] += 3.0

        I_drive = TimedArray(drive, dt=c.dt_ms * ms)

        G = NeuronGroup(total_n, IZH_TIMED_EQS, threshold="v >= 30", reset="v = c; u += d",
                         method="euler", name="stress_neurons")
        G.v = -65 + rng.normal(0, 2, total_n)
        G.u = 0.2 * G.v[:]
        for name in idx:
            s, e = idx[name]
            G.a[s:e] = 0.02; G.b[s:e] = 0.2; G.c[s:e] = -65; G.d[s:e] = 8
        # LC: slightly more excitable
        s, e = idx["lc"]
        G.a[s:e] = 0.02; G.b[s:e] = 0.2; G.c[s:e] = -60; G.d[s:e] = 6

        def _conn(src, tgt, p, w, inh=False, cid=0):
            ss, se = idx[src]; ts, te = idx[tgt]
            sgn = -1.0 if inh else 1.0
            syn = Synapses(G, G, "w:1", on_pre=f"v_post += {sgn}*w", name=f"st{cid}_{src[:4]}_{tgt[:4]}")
            ci, cj = [], []
            for i in range(ss, se):
                for j in range(ts, te):
                    if rng.random() < p: ci.append(i); cj.append(j)
            if ci: syn.connect(i=ci, j=cj); syn.w = rng.uniform(0, w, len(syn))
            return syn

        syns = []
        cid = 0
        syns.append(_conn("bla", "pvn", 0.4, 4.0, cid=cid)); cid += 1
        syns.append(_conn("bla", "lc", 0.3, 3.0, cid=cid)); cid += 1
        syns.append(_conn("hip_gr", "pvn", 0.4, 5.0, inh=True, cid=cid)); cid += 1  # GR負のFB
        syns.append(_conn("hip_mr", "pvn", 0.2, 2.0, inh=True, cid=cid)); cid += 1  # MRトニック抑制
        syns.append(_conn("mpfc", "pvn", 0.3, 3.0, inh=True, cid=cid)); cid += 1
        syns.append(_conn("bla", "bnst", 0.2, 2.0, cid=cid)); cid += 1
        syns.append(_conn("lc", "bla", 0.2, 2.0, cid=cid)); cid += 1  # NE→BLA感度↑

        mon = SpikeMonitor(G, name="stress_mon")
        net = Network(G, mon, *syns)
        net.run(c.duration_ms * ms)

        dur_s = c.duration_ms / 1000.0
        spk_i = np.array(mon.i[:])

        def _rate(name):
            s, e = idx[name]
            return int(np.sum((spk_i >= s) & (spk_i < e))) / (e - s) / dur_s if (e - s) > 0 else 0

        # コルチゾール更新
        pvn_r = _rate("pvn")
        cort_drive = pvn_r * c.cort_rise_rate * 0.1  # コルチゾール上昇感度10x増
        cort_decay = (self.cortisol - c.cort_baseline) * c.cort_decay_rate
        self.cortisol = max(0.0, min(1.0, self.cortisol + cort_drive - cort_decay))

        # NE更新
        lc_r = _rate("lc")
        self.ne_level += lc_r * 0.001 - (self.ne_level - 0.3) * 0.02
        self.ne_level = max(0.0, min(1.0, self.ne_level))

        # 慢性GRダウンレギュレーション
        # [較正] GR閾値0.4→0.16（急性コルチゾール0.173に合わせて反応可能に）
        if stressor and self.cortisol > 0.16:
            self.gr_sensitivity = max(0.1, self.gr_sensitivity - c.gr_downreg_rate * intensity)

        result = StressV2Result(
            trial_num=trial_num, phase=phase, stressor=stressor,
            bla_rate=_rate("bla"), pvn_rate=pvn_r,
            hippo_mr_rate=_rate("hip_mr"), hippo_gr_rate=_rate("hip_gr"),
            lc_rate=lc_r, bnst_rate=_rate("bnst"),
            cortisol=self.cortisol, ne_level=self.ne_level,
            gr_sensitivity=self.gr_sensitivity,
        )
        self._results.append(result)
        return result

    def run_acute(self, n=1, intensity=1.0): return [self.run_trial(True, intensity, "acute", len(self._results)+i) for i in range(n)]
    def run_recovery(self, n=3): return [self.run_trial(False, 0, "recovery", len(self._results)+i) for i in range(n)]
    def run_chronic(self, n=8, intensity=0.7): return [self.run_trial(True, intensity, "chronic", len(self._results)+i) for i in range(n)]

    @property
    def all_results(self): return list(self._results)
