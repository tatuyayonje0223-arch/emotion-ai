"""データマネージャ。Allen APIキャッシュ + 文献フォールバック。

[監査Fix3] Allen APIからの実測データを取得し、キャッシュする。
API不可時は文献ベースにフォールバック。
"""

from __future__ import annotations

from pathlib import Path

from src.data_driven.allen_connectivity import (
    AllenConnectivityMatrix, build_literature_matrix, save_matrix, load_matrix,
)

_DEFAULT_CACHE = Path("data/connectivity/allen_real_matrix.json")


def get_or_build_matrix(cache_path: str | Path = _DEFAULT_CACHE, force_api: bool = False) -> AllenConnectivityMatrix:
    """最良の結合マトリクスを取得する。

    1. キャッシュが存在し、force_api=Falseなら読み込み
    2. Allen APIで実測データを取得
    3. API失敗なら文献ベースにフォールバック
    """
    cache = Path(cache_path)

    if cache.exists() and not force_api:
        try:
            return load_matrix(cache)
        except Exception:
            pass

    # Allen APIを試行
    try:
        from src.data_driven.allen_api import build_allen_api_matrix
        matrix = build_allen_api_matrix(fallback_to_literature=True)

        # APIデータが含まれているか確認
        api_count = sum(1 for p in matrix.projections if "allen_api" in p.data_source)
        if api_count > 0:
            matrix.notes = f"Allen API実測データ{api_count}件 + 文献フォールバック"
        else:
            matrix.notes = "Allen API接続成功だがデータなし。文献ベースで全補完"

        save_matrix(matrix, cache)
        return matrix

    except Exception:
        # 完全フォールバック
        matrix = build_literature_matrix()
        matrix.notes = "Allen API接続失敗。文献ベース100%"
        return matrix
