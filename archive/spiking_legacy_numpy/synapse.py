"""シナプスモデル。STDP + 報酬変調学習。

STDP (Spike-Timing-Dependent Plasticity):
  Bi & Poo (2001)に基づく:
  - pre→post (因果的): Δw > 0 (LTP)、時間窓 ~20ms
  - post→pre (非因果的): Δw < 0 (LTD)、時間窓 ~20ms
  - Δw = A_plus * exp(-Δt/tau_plus)   if Δt > 0
  - Δw = -A_minus * exp(Δt/tau_minus)  if Δt < 0

報酬変調STDP (R-STDP):
  三要素則: Δw = DA_signal * STDP_trace
  ドーパミンがあるときだけSTDPが確定する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SynapseParams:
    """シナプスパラメータ。"""

    # STDP
    A_plus: float = 0.005      # LTP振幅
    A_minus: float = 0.005     # LTD振幅
    tau_plus: float = 20.0     # LTP時間窓 (ms)
    tau_minus: float = 20.0    # LTD時間窓 (ms)

    # 結合
    w_max: float = 10.0        # 最大結合重み
    w_min: float = 0.0         # 最小結合重み
    delay_ms: float = 1.0      # 伝達遅延 (ms)

    # 報酬変調
    da_modulation: float = 1.0  # DA変調強度
    eligibility_decay: float = 0.95  # 適格性トレースの減衰率


class SynapticConnection:
    """N_pre → N_post のシナプス結合（密行列）。

    重み行列 W[i,j] = pre i → post j の結合強度。
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        connection_prob: float = 0.2,
        w_init: float = 2.0,
        is_inhibitory: bool = False,
        params: SynapseParams | None = None,
        seed: int = 42,
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.params = params or SynapseParams()
        self.is_inhibitory = is_inhibitory

        rng = np.random.default_rng(seed)

        # 結合行列（ランダム接続）
        mask = rng.random((n_pre, n_post)) < connection_prob
        self.W = np.where(mask, rng.uniform(0, w_init, (n_pre, n_post)), 0.0)

        # STDPトレース
        self.pre_trace = np.zeros(n_pre)   # pre側の適格性
        self.post_trace = np.zeros(n_post)  # post側の適格性
        self.eligibility = np.zeros((n_pre, n_post))  # 報酬変調用

    def compute_current(self, pre_fired: np.ndarray) -> np.ndarray:
        """preの発火からpostへのシナプス電流を計算する。

        Returns:
            I_syn: postニューロンへの電流 (shape: [n_post])
        """
        sign = -1.0 if self.is_inhibitory else 1.0
        return sign * (pre_fired.astype(float) @ self.W)

    def update_stdp(self, pre_fired: np.ndarray, post_fired: np.ndarray, dt: float = 0.5) -> None:
        """STDPによる重み更新。

        トレースベースのオンライン実装（O(N²)だが数百ニューロンなら十分）。
        """
        p = self.params

        # トレースの減衰
        self.pre_trace *= np.exp(-dt / p.tau_plus)
        self.post_trace *= np.exp(-dt / p.tau_minus)

        # 発火でトレース更新
        self.pre_trace[pre_fired] += 1.0
        self.post_trace[post_fired] += 1.0

        # LTP: pre trace × post spike
        if post_fired.any():
            dw_ltp = p.A_plus * np.outer(self.pre_trace, post_fired.astype(float))
            self.eligibility += dw_ltp

        # LTD: post trace × pre spike
        if pre_fired.any():
            dw_ltd = -p.A_minus * np.outer(pre_fired.astype(float), self.post_trace)
            self.eligibility += dw_ltd

    def apply_reward_modulation(self, da_signal: float) -> None:
        """報酬変調: DAシグナルで適格性トレースを重みに反映する。

        三要素則: Δw = DA * eligibility
        """
        dw = self.params.da_modulation * da_signal * self.eligibility
        self.W += dw
        self.W = np.clip(self.W, self.params.w_min, self.params.w_max)

        # 適格性トレースの減衰
        self.eligibility *= self.params.eligibility_decay

    def apply_stdp_direct(self) -> None:
        """DA変調なしで直接STDPを適用する（古典的STDP）。"""
        self.W += self.eligibility * 0.1  # 直接適用は控えめに
        self.W = np.clip(self.W, self.params.w_min, self.params.w_max)
        self.eligibility *= self.params.eligibility_decay

    @property
    def mean_weight(self) -> float:
        nonzero = self.W[self.W > 0]
        return float(nonzero.mean()) if len(nonzero) > 0 else 0.0

    def reset_traces(self) -> None:
        self.pre_trace[:] = 0
        self.post_trace[:] = 0
        self.eligibility[:] = 0
