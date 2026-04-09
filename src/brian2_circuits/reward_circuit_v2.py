"""報酬学習回路 v2（Brian2ベース）。

affective-neuroscientist監査に基づく改善:
1. VTA分割: DA_medial(→mPFC, 嫌悪) + DA_lateral(→NAc, 報酬) + GABA(局所抑制)
2. NAc分割: Shell(主観的価値) + Core(予測手がかり応答)、各D1/D2
3. LHb→VTA GABA→VTA DA の間接抑制経路
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import brian2
brian2.prefs.codegen.target = "numpy"
from src.brian2_circuits.neuron_models import IZH_TIMED_EQS
from brian2 import NeuronGroup, Synapses, SpikeMonitor, Network, TimedArray, ms, start_scope, defaultclock




@dataclass
class RewardV2Config:
    n_vta_da_lat: int = 20    # VTA DA lateral (→NAc, 報酬)
    n_vta_da_med: int = 15    # VTA DA medial (→mPFC, 嫌悪)
    n_vta_gaba: int = 15      # VTA GABA (局所抑制)
    n_nac_shell_d1: int = 25  # NAc Shell D1
    n_nac_shell_d2: int = 25  # NAc Shell D2
    n_nac_core_d1: int = 25   # NAc Core D1
    n_nac_core_d2: int = 25   # NAc Core D2
    n_ofc: int = 25           # OFC
    n_lhb: int = 15           # LHb

    cs_amp: float = 8.0
    reward_amp: float = 15.0
    bg_noise: float = 3.0
    dt_ms: float = 0.5
    duration_ms: float = 400.0
    cs_onset_ms: float = 50.0
    cs_dur_ms: float = 80.0
    reward_onset_ms: float = 250.0
    reward_dur_ms: float = 40.0


@dataclass
class RewardV2Result:
    trial_num: int
    phase: str
    cs_presented: bool
    reward_presented: bool
    vta_da_lat_rate: float = 0.0
    vta_da_med_rate: float = 0.0
    vta_gaba_rate: float = 0.0
    nac_shell_d1_rate: float = 0.0
    nac_core_d1_rate: float = 0.0
    ofc_rate: float = 0.0
    lhb_rate: float = 0.0
    approach_tendency: float = 0.0


class RewardCircuitV2:
    def __init__(self, config: RewardV2Config | None = None):
        self.cfg = config or RewardV2Config()
        self._results: list[RewardV2Result] = []

    def run_trial(self, cs=True, reward=True, phase="training", trial_num=0) -> RewardV2Result:
        c = self.cfg
        start_scope()
        defaultclock.dt = c.dt_ms * ms

        pops = [
            ("vta_da_lat", c.n_vta_da_lat), ("vta_da_med", c.n_vta_da_med),
            ("vta_gaba", c.n_vta_gaba),
            ("nac_sh_d1", c.n_nac_shell_d1), ("nac_sh_d2", c.n_nac_shell_d2),
            ("nac_co_d1", c.n_nac_core_d1), ("nac_co_d2", c.n_nac_core_d2),
            ("ofc", c.n_ofc), ("lhb", c.n_lhb),
        ]
        total_n = sum(n for _, n in pops)
        idx = {}
        off = 0
        for name, n in pops:
            idx[name] = (off, off + n)
            off += n

        n_steps = int(c.duration_ms / c.dt_ms)
        rng = np.random.default_rng(trial_num * 13 + 7)
        drive = c.bg_noise + rng.normal(0, c.bg_noise * 0.3, (n_steps, total_n))

        cs_s = int(c.cs_onset_ms / c.dt_ms)
        cs_e = int((c.cs_onset_ms + c.cs_dur_ms) / c.dt_ms)
        rew_s = int(c.reward_onset_ms / c.dt_ms)
        rew_e = int((c.reward_onset_ms + c.reward_dur_ms) / c.dt_ms)

        if cs:
            s, e = idx["ofc"]
            drive[cs_s:cs_e, s:s + c.n_ofc // 3] += c.cs_amp
            s, e = idx["nac_co_d1"]
            drive[cs_s:cs_e, s:s + c.n_nac_core_d1 // 3] += c.cs_amp * 0.5

        # [DAチューニング] tonic_drive=3.8, burst_drive=10.0 (score=0.901)
        for da_name in ["vta_da_lat", "vta_da_med"]:
            s, e = idx[da_name]
            drive[:, s:e] += 3.8  # tonic: 3.5Hz目標

        # 報酬入力（burst 29.5Hz目標）
        if reward:
            s, e = idx["vta_da_lat"]
            drive[rew_s:rew_e, s:e] += 10.0  # burst_drive=10.0
            s, e = idx["nac_sh_d1"]
            drive[rew_s:rew_e, s:e] += c.reward_amp * 0.5

        if not reward and cs:  # 報酬省略→LHb活性化
            s, e = idx["lhb"]
            drive[rew_s:rew_e, s:e] += 10.0

        I_drive = TimedArray(drive, dt=c.dt_ms * ms)

        G = NeuronGroup(total_n, IZH_TIMED_EQS, threshold="v >= 30", reset="v = c; u += d",
                         method="euler", name="reward_neurons")
        G.v = -65 + rng.normal(0, 2, total_n)
        G.u = 0.2 * G.v[:]

        # DA neurons [DAチューニング: a=0.01,d=10 → tonic=3.5Hz,burst=29.5Hz,ratio=8.4x]
        for name in ["vta_da_lat", "vta_da_med"]:
            s, e = idx[name]
            G.a[s:e] = 0.01; G.b[s:e] = 0.2; G.c[s:e] = -65; G.d[s:e] = 10
        # GABA (FS)
        s, e = idx["vta_gaba"]
        G.a[s:e] = 0.1; G.b[s:e] = 0.2; G.c[s:e] = -65; G.d[s:e] = 2
        # MSN (RS)
        for name in ["nac_sh_d1", "nac_sh_d2", "nac_co_d1", "nac_co_d2", "ofc", "lhb"]:
            s, e = idx[name]
            G.a[s:e] = 0.02; G.b[s:e] = 0.2; G.c[s:e] = -65; G.d[s:e] = 8

        def _conn(src, tgt, p, w, inh=False, cid=0):
            ss, se = idx[src]; ts, te = idx[tgt]
            sgn = -1.0 if inh else 1.0
            syn = Synapses(G, G, "w:1", on_pre=f"v_post += {sgn}*w", name=f"r{cid}_{src[:5]}_{tgt[:5]}")
            ci, cj = [], []
            for i in range(ss, se):
                for j in range(ts, te):
                    if rng.random() < p: ci.append(i); cj.append(j)
            if ci: syn.connect(i=ci, j=cj); syn.w = rng.uniform(0, w, len(syn))
            return syn

        syns = []
        cid = 0
        # VTA DA_lat → NAc Shell D1 (報酬)
        syns.append(_conn("vta_da_lat", "nac_sh_d1", 0.4, 4.0, cid=cid)); cid += 1
        # VTA DA_lat → NAc Core D1
        syns.append(_conn("vta_da_lat", "nac_co_d1", 0.3, 3.0, cid=cid)); cid += 1
        # VTA DA_lat → NAc D2 (抑制的)
        syns.append(_conn("vta_da_lat", "nac_sh_d2", 0.2, 2.0, inh=True, cid=cid)); cid += 1
        # VTA GABA → VTA DA (局所抑制)
        syns.append(_conn("vta_gaba", "vta_da_lat", 0.4, 4.0, inh=True, cid=cid)); cid += 1
        syns.append(_conn("vta_gaba", "vta_da_med", 0.4, 4.0, inh=True, cid=cid)); cid += 1
        # LHb → VTA GABA (間接抑制: LHb→GABA→DA)
        syns.append(_conn("lhb", "vta_gaba", 0.4, 4.0, cid=cid)); cid += 1
        # OFC → VTA DA_lat (期待報酬: 抑制的=RPE計算)
        syns.append(_conn("ofc", "vta_da_lat", 0.3, 2.0, inh=True, cid=cid)); cid += 1
        # NAc Shell D1 → OFC (報酬信号フィードバック)
        syns.append(_conn("nac_sh_d1", "ofc", 0.2, 2.0, cid=cid)); cid += 1

        mon = SpikeMonitor(G, name="reward_mon")
        net = Network(G, mon, *syns)
        net.run(c.duration_ms * ms)

        dur_s = c.duration_ms / 1000.0
        spk_i = np.array(mon.i[:])

        def _rate(name):
            s, e = idx[name]
            return int(np.sum((spk_i >= s) & (spk_i < e))) / (e - s) / dur_s if (e - s) > 0 else 0

        d1_rate = _rate("nac_sh_d1") + _rate("nac_co_d1")
        d2_rate = _rate("nac_sh_d2") + _rate("nac_co_d2")
        approach = max(0, min(1, (d1_rate - d2_rate) / max(d1_rate + d2_rate, 1)))

        result = RewardV2Result(
            trial_num=trial_num, phase=phase, cs_presented=cs, reward_presented=reward,
            vta_da_lat_rate=_rate("vta_da_lat"), vta_da_med_rate=_rate("vta_da_med"),
            vta_gaba_rate=_rate("vta_gaba"),
            nac_shell_d1_rate=_rate("nac_sh_d1"), nac_core_d1_rate=_rate("nac_co_d1"),
            ofc_rate=_rate("ofc"), lhb_rate=_rate("lhb"),
            approach_tendency=approach,
        )
        self._results.append(result)
        return result

    def run_training(self, n=10): return [self.run_trial(True, True, "training", len(self._results)+i) for i in range(n)]
    def run_omission(self, n=3): return [self.run_trial(True, False, "omission", len(self._results)+i) for i in range(n)]
    def run_unexpected(self, n=3): return [self.run_trial(False, True, "unexpected", len(self._results)+i) for i in range(n)]

    @property
    def all_results(self): return list(self._results)
