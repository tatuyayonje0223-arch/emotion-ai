"""Brian2回路用データ駆動リードアウト（PCA/クラスタリング）。

[監査Fix2] 手動設計の線形結合(valence=DA*0.3-CORT*0.3...)を、
データ駆動のPCA+クラスタリングに置き換える。

回路のスパイキング活動パターンから、情動に対応する
状態空間を自動的に発見する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field


@dataclass
class ReadoutTrainingData:
    """リードアウト学習用データ。"""

    rates_matrix: np.ndarray  # (n_samples, n_populations)
    labels: list[str]         # 各サンプルの条件ラベル
    population_names: list[str]


class SpikingReadout:
    """スパイキング活動からのデータ駆動リードアウト。

    PCAで次元削減→上位2-3成分を情動軸にマッピング。
    """

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self._fitted = False
        self._mean: np.ndarray | None = None
        self._components: np.ndarray | None = None
        self._explained_variance: np.ndarray | None = None
        self._cluster_centers: np.ndarray | None = None
        self._cluster_labels: list[str] = []

    def fit(self, data: ReadoutTrainingData) -> None:
        """活動パターンからPCAを学習する。"""
        X = data.rates_matrix
        if X.shape[0] < self.n_components + 1:
            return  # データ不足

        # PCA（numpy実装、sklearn不要）
        self._mean = X.mean(axis=0)
        X_centered = X - self._mean
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 降順ソート
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        k = min(self.n_components, len(eigenvalues))
        self._components = eigenvectors[:, :k].T  # (k, n_populations)
        self._explained_variance = eigenvalues[:k] / eigenvalues.sum()

        # ラベル別クラスタ中心を計算
        projected = X_centered @ self._components.T  # (n_samples, k)
        unique_labels = sorted(set(data.labels))
        centers = []
        for label in unique_labels:
            mask = [i for i, l in enumerate(data.labels) if l == label]
            if mask:
                centers.append(projected[mask].mean(axis=0))
        if centers:
            self._cluster_centers = np.array(centers)
            self._cluster_labels = unique_labels

        self._fitted = True

    def transform(self, rates: np.ndarray) -> np.ndarray:
        """活動ベクトルをPC空間に射影する。"""
        if not self._fitted or self._mean is None or self._components is None:
            return rates[:self.n_components] if len(rates) >= self.n_components else rates
        centered = rates - self._mean
        return centered @ self._components.T

    def classify(self, rates: np.ndarray) -> dict[str, float]:
        """活動ベクトルを情動状態に分類する。

        Returns:
            {状態ラベル: 類似度スコア(0-1)} の辞書
        """
        if not self._fitted or self._cluster_centers is None:
            return {"unknown": 1.0}

        projected = self.transform(rates)
        # 各クラスタ中心との距離を計算
        distances = np.linalg.norm(self._cluster_centers - projected, axis=1)
        # 距離を類似度に変換（ソフトマックス的）
        if distances.min() == distances.max():
            scores = np.ones(len(distances)) / len(distances)
        else:
            inv_dist = 1.0 / (distances + 1e-6)
            scores = inv_dist / inv_dist.sum()

        return {label: float(score) for label, score in zip(self._cluster_labels, scores)}

    def to_emotion_readout(self, rates: np.ndarray) -> dict[str, float]:
        """活動ベクトルから情動リードアウトを計算する。

        PC1→valence軸、PC2→arousal軸として解釈。
        """
        if not self._fitted or self._components is None:
            return {"valence": 0.0, "arousal": 0.0, "state": "unfitted"}

        projected = self.transform(rates)
        n = len(projected)

        result = {
            "valence": float(np.tanh(projected[0] * 0.1)) if n > 0 else 0.0,
            "arousal": float(np.clip(projected[1] * 0.1 + 0.5, 0, 1)) if n > 1 else 0.5,
        }

        # クラスタ分類を追加
        classification = self.classify(rates)
        result["classification"] = classification
        result["dominant_state"] = max(classification, key=classification.get)

        return result

    @property
    def explained_variance_ratio(self) -> list[float]:
        if self._explained_variance is None:
            return []
        return self._explained_variance.tolist()

    @property
    def is_fitted(self) -> bool:
        return self._fitted


def collect_fear_training_data(n_per_condition: int = 5) -> ReadoutTrainingData:
    """恐怖回路から学習用データを収集する。"""
    from src.brian2_circuits.fear_circuit_v2 import FearCircuitV2, FearV2Config

    cfg = FearV2Config(duration_ms=200, cs_dur_ms=100, us_onset_ms=130, us_dur_ms=20)
    all_rates = []
    all_labels = []
    pop_names = ["la_exc", "ba_exc", "cel_som", "cel_pkcd", "cem", "pl", "il", "bnst"]

    conditions = [
        ("baseline", {"cs": False, "us": False}),
        ("cs_only", {"cs": True, "us": False}),
        ("cs_us", {"cs": True, "us": True}),
        ("sustained_threat", {"cs": False, "us": False, "sustained_threat": True}),
    ]

    for label, kwargs in conditions:
        for i in range(n_per_condition):
            circuit = FearCircuitV2(cfg)
            result = circuit.run_trial(**kwargs, phase=label, trial_num=i)
            rates = [
                result.la_rate, result.ba_rate,
                result.cel_som_rate, result.cel_pkcd_rate,
                result.cem_rate, result.pl_rate, result.il_rate, result.bnst_rate,
            ]
            all_rates.append(rates)
            all_labels.append(label)

    return ReadoutTrainingData(
        rates_matrix=np.array(all_rates),
        labels=all_labels,
        population_names=pop_names,
    )
