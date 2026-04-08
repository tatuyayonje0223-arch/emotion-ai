"""ホメオスタシス可塑性。スパイキング回路の安定性を保証する。

literature-cartographer調査に基づく2つのメカニズム:

1. シナプススケーリング (Homeostatic Synaptic Scaling, HSS)
   - 各ニューロンの平均発火率を追跡
   - 目標発火率との差に基づいて全入力シナプス重みを比例的に調整
   - 上方スケーリング(活動不足時) / 下方スケーリング(過活動時)
   - eLife 2024: 単調的・重みベースの負のフィードバック制御

2. BCMメタ可塑性 (Bienenstock-Cooper-Munro)
   - STDP の LTP/LTD 閾値がニューロンの活動履歴に応じてスライドする
   - 高活動 → LTD 閾値が下がる（LTD が起きやすくなる）
   - 低活動 → LTP 閾値が下がる（LTP が起きやすくなる）
   - 視覚皮質の特徴選択性発達で実験的に確認
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HomeostaticConfig:
    """ホメオスタシス可塑性パラメータ。"""

    # シナプススケーリング
    target_rate_hz: float = 10.0       # 目標発火率 (Hz)
    scaling_rate: float = 0.001        # スケーリング速度
    scaling_min_weight: float = 0.1    # 最小重み（完全消失を防ぐ）
    scaling_max_weight: float = 15.0   # 最大重み

    # BCMメタ可塑性
    bcm_tau_ms: float = 10000.0        # BCM閾値の時定数 (ms)
    bcm_target_rate_hz: float = 10.0   # BCM目標発火率

    # 内因性可塑性
    intrinsic_rate: float = 0.0001     # 興奮性パラメータの調整速度


class FiringRateTracker:
    """各ニューロンの発火率を指数移動平均で追跡する。"""

    def __init__(self, n: int, tau_ms: float = 5000.0, dt_ms: float = 0.5):
        self.n = n
        self.rates = np.zeros(n)  # 推定発火率 (Hz)
        self._alpha = dt_ms / tau_ms  # 指数移動平均の係数

    def update(self, fired: np.ndarray, dt_ms: float = 0.5) -> None:
        """発火情報で発火率推定を更新する。"""
        instantaneous = fired.astype(float) / (dt_ms / 1000.0)  # Hz換算
        self.rates = (1 - self._alpha) * self.rates + self._alpha * instantaneous

    def update_from_count(self, spike_count: int, n_neurons: int, duration_ms: float) -> None:
        """試行全体のスパイクカウントから更新する。"""
        rate = (spike_count / n_neurons) / (duration_ms / 1000.0) if n_neurons > 0 else 0
        self.rates[:] = (1 - 0.1) * self.rates + 0.1 * rate


def apply_synaptic_scaling(
    weights: np.ndarray,
    neuron_rates: np.ndarray,
    config: HomeostaticConfig,
) -> np.ndarray:
    """シナプススケーリングを適用する。

    各postニューロンの発火率と目標発火率の比を使って、
    そのニューロンへの全入力重みを比例的にスケールする。

    Args:
        weights: シナプス重み行列 (pre x post) または (n_synapses,)
        neuron_rates: 各postニューロンの推定発火率 (n_post,)
        config: パラメータ

    Returns:
        調整後の重み
    """
    target = config.target_rate_hz
    if target <= 0:
        return weights

    # 各ニューロンのスケーリング係数
    # rate < target → 上方スケーリング (factor > 1)
    # rate > target → 下方スケーリング (factor < 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(neuron_rates > 0.1, target / neuron_rates, 1.0)

    # 急激な変化を防ぐ: factorを1に近づける
    factor = 1.0 + config.scaling_rate * (ratio - 1.0)
    factor = np.clip(factor, 0.9, 1.1)  # 1ステップで最大10%変化

    # 重みをスケール
    if weights.ndim == 2:
        # (pre, post) 行列: 各列(post)ニューロンにfactorを適用
        scaled = weights * factor[np.newaxis, :]
    else:
        # 1D: post_neuron情報が必要（呼び出し側で対応）
        scaled = weights * np.mean(factor)

    return np.clip(scaled, config.scaling_min_weight, config.scaling_max_weight)


def compute_bcm_threshold(
    neuron_rates: np.ndarray,
    current_threshold: np.ndarray,
    config: HomeostaticConfig,
    dt_ms: float = 0.5,
) -> np.ndarray:
    """BCMメタ可塑性: STDP閾値のスライディング。

    高活動のニューロン → 閾値が上昇 → LTDが起きやすくなる
    低活動のニューロン → 閾値が低下 → LTPが起きやすくなる

    Returns:
        更新後の閾値 (各ニューロン)
    """
    alpha = dt_ms / config.bcm_tau_ms
    # 閾値はrate^2に比例して上昇（BCM理論）
    target_theta = (neuron_rates / config.bcm_target_rate_hz) ** 2
    new_threshold = (1 - alpha) * current_threshold + alpha * target_theta
    return np.clip(new_threshold, 0.1, 10.0)


def apply_intrinsic_plasticity(
    excitability: np.ndarray,
    neuron_rates: np.ndarray,
    target_rate: float,
    rate: float = 0.0001,
) -> np.ndarray:
    """内因性可塑性: ニューロンの興奮性パラメータを発火率に応じて調整。

    Izhikevichモデルではbパラメータを微調整:
    - 発火率が低い → bを減少（より興奮しやすく）
    - 発火率が高い → bを増加（より興奮しにくく）
    """
    error = neuron_rates - target_rate
    adjustment = -rate * error  # 負のフィードバック
    return excitability + adjustment


class HomeostaticController:
    """ホメオスタシス可塑性の統合コントローラ。"""

    def __init__(self, n_neurons: int, config: HomeostaticConfig | None = None):
        self.config = config or HomeostaticConfig()
        self.rate_tracker = FiringRateTracker(n_neurons)
        self.bcm_thresholds = np.ones(n_neurons)
        self.n_neurons = n_neurons

    def update_rates(self, spike_count: int, duration_ms: float) -> None:
        """試行後に発火率を更新する。"""
        self.rate_tracker.update_from_count(spike_count, self.n_neurons, duration_ms)

    def get_scaling_factors(self) -> np.ndarray:
        """現在の発火率に基づくスケーリング係数を返す。"""
        target = self.config.target_rate_hz
        rates = self.rate_tracker.rates
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(rates > 0.1, target / rates, 1.0)
        factor = 1.0 + self.config.scaling_rate * (ratio - 1.0)
        return np.clip(factor, 0.9, 1.1)

    def update_bcm(self, dt_ms: float = 500.0) -> None:
        """BCM閾値を更新する。"""
        self.bcm_thresholds = compute_bcm_threshold(
            self.rate_tracker.rates, self.bcm_thresholds, self.config, dt_ms,
        )

    @property
    def mean_rate(self) -> float:
        return float(self.rate_tracker.rates.mean())

    @property
    def rate_deviation(self) -> float:
        """目標発火率からの平均偏差。"""
        return float(np.abs(self.rate_tracker.rates - self.config.target_rate_hz).mean())
