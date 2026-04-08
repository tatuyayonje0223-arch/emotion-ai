"""Izhikevichスパイキングニューロンモデル（ベクトル化実装）。

Izhikevich (2003): 2変数で20+種類の発火パターンを再現する。
  dv/dt = 0.04v² + 5v + 140 - u + I
  du/dt = a(bv - u)
  if v >= 30mV: v = c, u = u + d

パラメータで異なる細胞タイプを表現:
  RS (regular spiking, 興奮性皮質): a=0.02, b=0.2, c=-65, d=8
  FS (fast spiking, 抑制性):        a=0.1,  b=0.2, c=-65, d=2
  IB (intrinsically bursting):      a=0.02, b=0.2, c=-55, d=4
  CH (chattering):                  a=0.02, b=0.2, c=-50, d=2
  LTS (low-threshold spiking):      a=0.02, b=0.25, c=-65, d=2

時間ステップ: 0.5ms（安定性のため0.04v²項に対応）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NeuronParams:
    """Izhikevichニューロンのパラメータセット。"""

    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    v_thresh: float = 30.0
    label: str = "RS"


# 標準的な細胞タイプ
RS = NeuronParams(a=0.02, b=0.2, c=-65, d=8, label="RS")
FS = NeuronParams(a=0.1, b=0.2, c=-65, d=2, label="FS")
IB = NeuronParams(a=0.02, b=0.2, c=-55, d=4, label="IB")
CH = NeuronParams(a=0.02, b=0.2, c=-50, d=2, label="CH")
LTS = NeuronParams(a=0.02, b=0.25, c=-65, d=2, label="LTS")


class IzhikevichPopulation:
    """N個のIzhikevichニューロンの集団（ベクトル化）。

    numpy配列で並列計算。GPUなしでも数千ニューロンをリアルタイムに近い速度で実行可能。
    """

    def __init__(self, n: int, params: NeuronParams, noise_std: float = 0.5):
        self.n = n
        self.params = params
        self.noise_std = noise_std

        # 状態変数
        self.v = np.full(n, -65.0)   # 膜電位 (mV)
        self.u = np.full(n, params.b * -65.0)  # 回復変数
        self.fired = np.zeros(n, dtype=bool)  # 直前ステップで発火したか

        # パラメータ配列（個体差を持たせるためにわずかにばらつかせる）
        rng = np.random.default_rng(42)
        self.a = np.full(n, params.a) * (1 + rng.normal(0, 0.05, n))
        self.b_arr = np.full(n, params.b) * (1 + rng.normal(0, 0.05, n))
        self.c = np.full(n, params.c) + rng.normal(0, 1.0, n)
        self.d = np.full(n, params.d) * (1 + rng.normal(0, 0.05, n))

        # 記録
        self._spike_history: list[np.ndarray] = []

    def step(self, I_ext: np.ndarray, dt: float = 0.5) -> np.ndarray:
        """1ステップ(dt ms)更新する。

        Args:
            I_ext: 外部電流 (shape: [n])
            dt: 時間ステップ (ms)。0.5ms推奨（安定性）。

        Returns:
            fired: 発火したニューロンのブールマスク
        """
        # ノイズ
        noise = np.random.normal(0, self.noise_std, self.n) if self.noise_std > 0 else 0

        # 2回の半ステップ更新（数値安定性のため）
        half_dt = dt / 2.0
        for _ in range(2):
            dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I_ext + noise) * half_dt
            self.v += dv
            du = self.a * (self.b_arr * self.v - self.u) * half_dt
            self.u += du

        # 発火判定
        self.fired = self.v >= self.params.v_thresh
        self.v[self.fired] = self.c[self.fired]
        self.u[self.fired] = self.u[self.fired] + self.d[self.fired]

        self._spike_history.append(self.fired.copy())

        return self.fired

    def firing_rate(self, window_ms: int = 100, dt: float = 0.5) -> float:
        """直近windowの平均発火率(Hz)を返す。"""
        steps = int(window_ms / dt)
        if len(self._spike_history) < steps:
            steps = len(self._spike_history)
        if steps == 0:
            return 0.0
        recent = self._spike_history[-steps:]
        total_spikes = sum(s.sum() for s in recent)
        return (total_spikes / self.n) / (steps * dt / 1000.0)

    def reset(self) -> None:
        self.v[:] = -65.0
        self.u[:] = self.params.b * -65.0
        self.fired[:] = False
        self._spike_history.clear()

    @property
    def mean_potential(self) -> float:
        return float(self.v.mean())

    @property
    def spike_count(self) -> int:
        if not self._spike_history:
            return 0
        return int(self._spike_history[-1].sum())
