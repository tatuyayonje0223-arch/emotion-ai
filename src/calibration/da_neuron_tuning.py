"""VTA DAニューロンのチューニング。burst/tonic比の改善。

目標（Schultz 1997, Grace 1991, Hyland 2002）:
  - tonic: 3-5 Hz（自発的低頻度発火）
  - burst: 20-40 Hz（報酬時の短期バースト）
  - burst/tonic ratio: 4-8x

問題:
  - RS型(a=0.02,b=0.2,c=-65,d=8)は背景ノイズで高頻度発火しやすい
  - IB型(c=-55,d=4)は閾値が高く自発バーストが困難
  - 解決: DAニューロン特有のパラメータ（低自発発火+バースト可能）を探索

DAニューロンの特徴（Grace & Bunney 1984）:
  - 低tonic発火（3-5Hz）、長い活動電位（>2ms）
  - パッチクランプで測定されるA電流(IA)がtonic発火を抑制
  - NMDAバーストモード: NMDA入力で一時的に高頻度化
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import brian2
brian2.prefs.codegen.target = "numpy"
from brian2 import NeuronGroup, SpikeMonitor, Network, TimedArray, ms, start_scope, defaultclock

from src.brian2_circuits.neuron_models import IZH_TIMED_EQS


@dataclass
class DAParamSet:
    """DAニューロンパラメータセット。"""
    name: str
    a: float
    b: float
    c: float
    d: float
    tonic_drive: float  # 背景入力
    burst_drive: float  # 報酬入力の追加分


# 候補パラメータ（Izhikevichモデルの異なる領域を探索）
DA_CANDIDATES = [
    # 低tonic: aを大きくして回復を速く→tonic抑制
    DAParamSet("low_tonic_fast_recovery", a=0.1, b=0.26, c=-65, d=2, tonic_drive=0.5, burst_drive=12.0),
    # Accommodating: 発火適応が強い→tonic低下
    DAParamSet("accommodating", a=0.02, b=0.2, c=-65, d=8, tonic_drive=0.3, burst_drive=15.0),
    # Low threshold spiking + strong adaptation
    DAParamSet("lts_adapted", a=0.02, b=0.25, c=-65, d=2, tonic_drive=0.8, burst_drive=10.0),
    # FS-like but with low drive
    DAParamSet("fs_low_drive", a=0.1, b=0.2, c=-65, d=2, tonic_drive=0.3, burst_drive=12.0),
    # Custom DA: very slow recovery (low a) + strong adaptation (high d)
    DAParamSet("custom_da_v1", a=0.01, b=0.2, c=-65, d=10, tonic_drive=1.0, burst_drive=12.0),
    # Custom DA v2: moderate with low bg
    DAParamSet("custom_da_v2", a=0.02, b=0.2, c=-60, d=6, tonic_drive=0.5, burst_drive=10.0),
]


@dataclass
class TuningResult:
    name: str
    tonic_rate: float
    burst_rate: float
    ratio: float
    score: float


def test_da_params(params: DAParamSet, n_neurons: int = 20, duration_ms: float = 500.0) -> TuningResult:
    """1つのDAパラメータセットをテストする。"""
    start_scope()
    dt = 0.5
    defaultclock.dt = dt * ms
    n_steps = int(duration_ms / dt)

    rng = np.random.default_rng(42)
    drive = params.tonic_drive + rng.normal(0, 0.3, (n_steps, n_neurons))

    # burst期間: 300-400ms
    burst_start = int(300 / dt)
    burst_end = int(400 / dt)
    drive[burst_start:burst_end, :] += params.burst_drive

    I_drive = TimedArray(drive, dt=dt * ms)

    G = NeuronGroup(n_neurons, IZH_TIMED_EQS, threshold="v >= 30", reset="v = c; u += d",
                     method="euler", name="da_test")
    G.v = -65 + rng.normal(0, 2, n_neurons)
    G.u = params.b * G.v[:]
    G.a = params.a
    G.b = params.b
    G.c = params.c
    G.d = params.d

    mon = SpikeMonitor(G, name="da_mon")
    net = Network(G, mon)
    net.run(duration_ms * ms)

    spk_i = np.array(mon.i[:])
    spk_t = np.array(mon.t[:] / ms)

    # tonic: 0-300ms
    tonic_mask = spk_t < 300
    tonic_count = tonic_mask.sum()
    tonic_rate = tonic_count / n_neurons / (300 / 1000)

    # burst: 300-400ms
    burst_mask = (spk_t >= 300) & (spk_t < 400)
    burst_count = burst_mask.sum()
    burst_rate = burst_count / n_neurons / (100 / 1000)

    ratio = burst_rate / max(tonic_rate, 0.1)

    # スコア: tonic 3-5Hz + burst 20-40Hz + ratio 4-8x
    def _s(val, target, tol):
        return max(0, 1 - abs(val - target) / max(abs(target), 0.1) / tol)

    score = (_s(tonic_rate, 4.0, 1.5) + _s(burst_rate, 30.0, 1.5) + _s(ratio, 6.0, 2.0)) / 3

    return TuningResult(name=params.name, tonic_rate=tonic_rate, burst_rate=burst_rate,
                         ratio=ratio, score=score)


def run_da_tuning() -> list[TuningResult]:
    """全候補パラメータをテストし、スコア順にソートする。"""
    results = [test_da_params(p) for p in DA_CANDIDATES]
    results.sort(key=lambda r: r.score, reverse=True)
    return results
