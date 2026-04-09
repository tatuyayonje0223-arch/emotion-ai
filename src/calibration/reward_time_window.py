"""報酬回路のburst/tonic時間窓解析。

報酬提示期間のみの発火率を計算し、tonic vs burstを分離する。

[R8 M2注意] この解析はスタンドアロンNeuronGroup（シナプスなし）で実行するため、
RewardCircuitV2の実回路（GABA局所抑制等）とは数値が乖離する。
実回路のtonic: ~7.7Hz、本解析のtonic: ~19Hz（GABA抑制なし分高い）。
burst/tonic比の方向性は正しいが、絶対値の比較には注意。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import brian2
brian2.prefs.codegen.target = "numpy"
from brian2 import NeuronGroup, Synapses, SpikeMonitor, Network, TimedArray, ms, start_scope, defaultclock

from src.brian2_circuits.neuron_models import IZH_TIMED_EQS
from src.brian2_circuits.reward_circuit_v2 import RewardCircuitV2, RewardV2Config


@dataclass
class TimeWindowResult:
    """時間窓別の発火率。"""

    tonic_rate: float = 0.0       # ベースライン期間の発火率
    burst_rate: float = 0.0       # 報酬提示期間の発火率
    burst_tonic_ratio: float = 0.0
    d1_reward_rate: float = 0.0
    lhb_omission_rate: float = 0.0


def analyze_reward_time_windows(cfg: RewardV2Config | None = None) -> TimeWindowResult:
    """報酬回路の時間窓別発火率を計算する。

    Brian2のSpikeMonitorからスパイクタイミングを抽出し、
    ベースライン期間 vs 報酬期間で分離計算する。
    """
    cfg = cfg or RewardV2Config(duration_ms=300, cs_dur_ms=80, reward_onset_ms=200, reward_dur_ms=40)
    start_scope()
    defaultclock.dt = cfg.dt_ms * ms

    pops = [
        ("vta_da_lat", cfg.n_vta_da_lat), ("vta_da_med", cfg.n_vta_da_med),
        ("vta_gaba", cfg.n_vta_gaba),
        ("nac_sh_d1", cfg.n_nac_shell_d1), ("nac_sh_d2", cfg.n_nac_shell_d2),
        ("nac_co_d1", cfg.n_nac_core_d1), ("nac_co_d2", cfg.n_nac_core_d2),
        ("ofc", cfg.n_ofc), ("lhb", cfg.n_lhb),
    ]
    total_n = sum(n for _, n in pops)
    idx = {}
    off = 0
    for name, n in pops:
        idx[name] = (off, off + n)
        off += n

    n_steps = int(cfg.duration_ms / cfg.dt_ms)
    rng = np.random.default_rng(42)
    drive = cfg.bg_noise + rng.normal(0, cfg.bg_noise * 0.3, (n_steps, total_n))

    rew_s = int(cfg.reward_onset_ms / cfg.dt_ms)
    rew_e = int((cfg.reward_onset_ms + cfg.reward_dur_ms) / cfg.dt_ms)
    cs_s = int(cfg.cs_onset_ms / cfg.dt_ms)
    cs_e = int((cfg.cs_onset_ms + cfg.cs_dur_ms) / cfg.dt_ms)

    # [DAチューニング] tonic_drive=3.8, burst_drive=10.0
    for da in ["vta_da_lat", "vta_da_med"]:
        s, e = idx[da]
        drive[:, s:e] += 3.8
    s, e = idx["vta_da_lat"]
    drive[rew_s:rew_e, s:e] += 10.0

    I_drive = TimedArray(drive, dt=cfg.dt_ms * ms)

    G = NeuronGroup(total_n, IZH_TIMED_EQS, threshold="v >= 30", reset="v = c; u += d",
                     method="euler", name="tw_neurons")
    G.v = -65 + rng.normal(0, 2, total_n)
    G.u = 0.2 * G.v[:]
    for name in idx:
        s, e = idx[name]
        if name in ("vta_da_lat", "vta_da_med"):
            G.a[s:e] = 0.01; G.b[s:e] = 0.2; G.c[s:e] = -65; G.d[s:e] = 10  # DA tuned
        else:
            G.a[s:e] = 0.02; G.b[s:e] = 0.2; G.c[s:e] = -65; G.d[s:e] = 8

    mon = SpikeMonitor(G, name="tw_mon")
    net = Network(G, mon)
    net.run(cfg.duration_ms * ms)

    spk_i = np.array(mon.i[:])
    spk_t = np.array(mon.t[:] / ms)

    def _windowed_rate(pop_name, t_start_ms, t_end_ms):
        s, e = idx[pop_name]
        n = e - s
        if n == 0:
            return 0.0
        mask = (spk_i >= s) & (spk_i < e) & (spk_t >= t_start_ms) & (spk_t < t_end_ms)
        count = mask.sum()
        dur_s = (t_end_ms - t_start_ms) / 1000.0
        return count / n / dur_s if dur_s > 0 else 0.0

    # ベースライン期間: 0〜CS開始
    tonic = _windowed_rate("vta_da_lat", 0, cfg.cs_onset_ms)

    # 報酬期間のみ
    burst = _windowed_rate("vta_da_lat", cfg.reward_onset_ms, cfg.reward_onset_ms + cfg.reward_dur_ms)

    ratio = burst / max(tonic, 0.1)

    # D1 during reward
    d1_rew = _windowed_rate("nac_sh_d1", cfg.reward_onset_ms, cfg.reward_onset_ms + cfg.reward_dur_ms)

    return TimeWindowResult(
        tonic_rate=tonic,
        burst_rate=burst,
        burst_tonic_ratio=ratio,
        d1_reward_rate=d1_rew,
    )
