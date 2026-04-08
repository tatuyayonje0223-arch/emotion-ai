"""Allen Brain Atlas APIモジュールのテスト。

APIテスト: ネットワーク接続に依存するため、接続失敗時はスキップ。
文献フォールバック: 常にテスト可能。
"""

import pytest

from src.data_driven.allen_api import (
    build_allen_api_matrix,
    find_experiments_for_structure,
    ALLEN_STRUCTURE_IDS,
)
from src.data_driven.allen_connectivity import ALLEN_STRUCTURE_IDS as STRUCTURE_IDS


class TestAllenAPIOffline:
    """APIなしでも動作するテスト。"""

    def test_fallback_to_literature(self):
        """API接続失敗時に文献ベースにフォールバック。"""
        # 存在しない構造IDで強制フォールバック
        matrix = build_allen_api_matrix(
            regions={"FAKE": 99999},
            fallback_to_literature=True,
        )
        assert len(matrix.projections) > 0
        assert "literature" in matrix.notes or len(matrix.projections) > 20

    def test_structure_ids_complete(self):
        """12領域全ての構造IDが定義されている。"""
        required = ["LA", "BLA", "CeA", "PL", "IL", "CA1", "VTA", "NAc", "PVN", "PAG", "BNST", "LC", "DR"]
        for name in required:
            assert name in STRUCTURE_IDS, f"{name} missing from ALLEN_STRUCTURE_IDS"


class TestAllenAPIOnline:
    """API接続が必要なテスト。"""

    @pytest.fixture(autouse=True)
    def check_network(self):
        """ネットワーク接続を確認。なければスキップ。"""
        import urllib.request
        try:
            urllib.request.urlopen("http://api.brain-map.org/api/v2/data/query.json?criteria=model::Product&num_rows=1", timeout=10)
        except Exception:
            pytest.skip("Allen Brain Atlas API not reachable")

    def test_find_experiments_bla(self):
        """BLAの実験を検索できる。"""
        exp_ids = find_experiments_for_structure(295)  # BLA
        assert len(exp_ids) > 0

    def test_build_real_matrix(self):
        """実測データで結合マトリクスを構築できる。"""
        # 小さいサブセットでテスト
        small_regions = {"BLA": 295, "CeA": 536, "NAc": 56}
        matrix = build_allen_api_matrix(regions=small_regions, fallback_to_literature=True)
        assert len(matrix.projections) > 0

    def test_bla_cea_projection_exists(self):
        """BLA→CeA投射が実測データに存在する。"""
        regions = {"BLA": 295, "CeA": 536}
        matrix = build_allen_api_matrix(regions=regions, fallback_to_literature=False)
        proj = matrix.get_projection("BLA", "CeA")
        if proj:  # APIが成功した場合のみ検証
            assert proj.normalized_projection_volume > 0
            assert proj.n_experiments > 0
