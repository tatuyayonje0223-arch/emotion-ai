"""CircuitSpec → Brian2ネットワーク変換エンジン。

宣言的YAML仕様からBrian2シミュレーションを自動構築する。
手配線とデータ駆動の結合を統一的に扱う。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import brian2
brian2.prefs.codegen.target = "numpy"
from brian2 import NeuronGroup, Synapses, SpikeMonitor, Network, TimedArray, ms, start_scope, defaultclock

from src.brian2_circuits.neuron_models import CELL_TYPES
from src.data_driven.circuit_spec import CircuitSpec, PopulationSpec, ConnectionSpec, InputSpec


IZH_EQS = """
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I_drive(t, i)) / ms : 1
    du/dt = (a*(b*v - u)) / ms : 1
    a : 1 (constant)
    b : 1 (constant)
    c : 1 (constant)
    d : 1 (constant)
"""


@dataclass
class SimulationResult:
    """シミュレーション結果。"""

    spike_indices: np.ndarray
    spike_times: np.ndarray
    population_rates: dict[str, float]
    total_spikes: int
    duration_ms: float


def build_and_run(
    spec: CircuitSpec,
    extra_drive: np.ndarray | None = None,
) -> SimulationResult:
    """CircuitSpecからBrian2ネットワークを構築し実行する。

    Args:
        spec: 回路仕様
        extra_drive: 追加の外部入力 (n_steps, total_n)。Noneならspec.inputsのみ

    Returns:
        SimulationResult
    """
    start_scope()
    sim = spec.simulation
    dt_ms = sim.get("dt_ms", 0.5)
    duration_ms = sim.get("duration_ms", 300.0)
    bg_noise = sim.get("background_noise", 3.0)
    defaultclock.dt = dt_ms * ms

    total_n = spec.total_neurons
    n_steps = int(duration_ms / dt_ms)

    # ニューロンインデックスマップ
    idx: dict[str, tuple[int, int]] = {}
    offset = 0
    for pop in spec.populations:
        idx[pop.name] = (offset, offset + pop.n)
        offset += pop.n

    # 入力スケジュール構築
    rng = np.random.default_rng(42)
    drive = bg_noise + rng.normal(0, bg_noise * 0.3, (n_steps, total_n))

    # spec.inputsから入力を追加
    for inp in spec.inputs:
        if inp.target_population not in idx:
            continue
        s, e = idx[inp.target_population]
        n_pop = e - s
        n_target = max(1, int(n_pop * inp.target_fraction))
        inp_start = int(inp.onset_ms / dt_ms)
        inp_end = int((inp.onset_ms + inp.duration_ms) / dt_ms)
        inp_end = min(inp_end, n_steps)
        if inp_start < n_steps:
            drive[inp_start:inp_end, s:s + n_target] += inp.amplitude

    if extra_drive is not None:
        drive += extra_drive[:n_steps, :total_n]

    I_drive = TimedArray(drive, dt=dt_ms * ms)

    # NeuronGroup
    G = NeuronGroup(
        total_n, IZH_EQS,
        threshold="v >= 30", reset="v = c; u += d",
        method="euler", name="spec_neurons",
    )
    G.v = -65 + rng.normal(0, 2, total_n)
    G.u = 0.2 * G.v[:]

    # 細胞タイプパラメータ設定
    for pop in spec.populations:
        s, e = idx[pop.name]
        params = CELL_TYPES.get(pop.cell_type, CELL_TYPES["RS"])
        G.a[s:e] = params["a"]
        G.b[s:e] = params["b"]
        G.c[s:e] = params["c"]
        G.d[s:e] = params["d"]

    # シナプス結合
    synapses = []
    for ci, conn in enumerate(spec.connections):
        if conn.source not in idx or conn.target not in idx:
            continue
        ss, se = idx[conn.source]
        ts, te = idx[conn.target]
        sgn = -1.0 if conn.conn_type == "inhibitory" else 1.0

        # インデックスベース接続
        ci_list, cj_list = [], []
        for i in range(ss, se):
            for j in range(ts, te):
                if rng.random() < conn.probability:
                    ci_list.append(i)
                    cj_list.append(j)

        if not ci_list:
            continue  # 接続0件のシナプスはスキップ（Brian2エラー回避）

        syn = Synapses(G, G, "w : 1", on_pre=f"v_post += {sgn} * w",
                       name=f"c{ci}_{conn.source[:5]}_{conn.target[:5]}")
        syn.connect(i=ci_list, j=cj_list)
        # バランスネットワーク重みスケーリング
        n_inputs = max(1, (se - ss) * conn.probability)
        w_scaled = conn.weight_mean / np.sqrt(n_inputs)
        syn.w = rng.normal(w_scaled, conn.weight_std * 0.5, len(syn))
        syn.w = np.clip(syn.w[:], 0, conn.weight_mean * 3)
        synapses.append(syn)

    # モニター
    mon = SpikeMonitor(G, name="spec_mon")

    # 実行
    net = Network(G, mon, *synapses)
    net.run(duration_ms * ms)

    # 結果集計
    spk_i = np.array(mon.i[:])
    spk_t = np.array(mon.t[:] / ms)
    dur_s = duration_ms / 1000.0

    pop_rates = {}
    for pop in spec.populations:
        s, e = idx[pop.name]
        n = e - s
        count = int(np.sum((spk_i >= s) & (spk_i < e)))
        pop_rates[pop.name] = count / n / dur_s if n > 0 else 0.0

    return SimulationResult(
        spike_indices=spk_i,
        spike_times=spk_t,
        population_rates=pop_rates,
        total_spikes=int(mon.num_spikes),
        duration_ms=duration_ms,
    )
