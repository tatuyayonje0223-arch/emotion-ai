"""Allen Mouse Brain Connectivity Atlas REST API直接アクセス。

allensdk不要。Python 3.14でurllib+jsonのみで動作。
API: http://api.brain-map.org/api/v2/

リサーチ結果に基づく実装:
- Product ID 5 = Mouse Connectivity Projection (2331実験)
- ProjectionStructureUnionize = 各実験の領域別投射データ
- normalized_projection_volume = 投射強度の標準指標 (Oh et al. 2014)
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from src.data_driven.allen_connectivity import (
    ALLEN_STRUCTURE_IDS, AllenConnectivityMatrix, ProjectionData,
    build_literature_matrix, save_matrix,
)

API_BASE = "http://api.brain-map.org/api/v2"


def _api_query(criteria: str, num_rows: int = 2000) -> list[dict]:
    """Allen Brain Atlas APIにクエリを送信する。"""
    url = f"{API_BASE}/data/query.json?criteria={criteria}&num_rows={num_rows}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            if data.get("success"):
                return data.get("msg", [])
    except Exception:
        pass
    return []


def find_experiments_for_structure(structure_id: int) -> list[int]:
    """指定構造に注入された実験のIDリストを返す。

    ProjectionStructureUnionize で is_injection=true かつ該当構造のレコードから逆引き。
    """
    criteria = (
        f"model::ProjectionStructureUnionize,"
        f"rma::criteria,"
        f"[structure_id$eq{structure_id}],"
        f"[is_injection$eqtrue],"
        f"[hemisphere_id$eq3]"
    )
    results = _api_query(criteria, num_rows=50)
    exp_ids = list(set(r["section_data_set_id"] for r in results if "section_data_set_id" in r))
    return exp_ids


def get_projection_data(
    experiment_id: int,
    target_structure_ids: list[int],
) -> list[dict[str, Any]]:
    """1実験の指定ターゲット領域への投射データを取得する。"""
    ids_str = ",".join(str(sid) for sid in target_structure_ids)
    criteria = (
        f"model::ProjectionStructureUnionize,"
        f"rma::criteria,"
        f"[section_data_set_id$eq{experiment_id}],"
        f"[structure_id$in{ids_str}],"
        f"[is_injection$eqfalse],"
        f"[hemisphere_id$eq3]"
    )
    return _api_query(criteria, num_rows=200)


def build_allen_api_matrix(
    regions: dict[str, int] | None = None,
    fallback_to_literature: bool = True,
) -> AllenConnectivityMatrix:
    """Allen APIから実測データで結合マトリクスを構築する。

    API接続失敗時は文献ベースにフォールバック。
    """
    if regions is None:
        regions = ALLEN_STRUCTURE_IDS

    target_ids = list(regions.values())
    id_to_name = {v: k for k, v in regions.items()}
    projections: list[ProjectionData] = []
    api_success = False

    # 各ソース領域から実験を検索
    for src_name, src_id in regions.items():
        exp_ids = find_experiments_for_structure(src_id)
        if not exp_ids:
            continue

        api_success = True

        # 各実験の投射データを取得し、平均化
        tgt_data: dict[int, list[dict]] = {tid: [] for tid in target_ids}

        for exp_id in exp_ids[:5]:  # 最大5実験
            records = get_projection_data(exp_id, target_ids)
            for rec in records:
                sid = rec.get("structure_id")
                if sid in tgt_data:
                    tgt_data[sid].append(rec)

        # ターゲットごとに平均
        for tgt_id, records in tgt_data.items():
            if not records or tgt_id == src_id:
                continue
            tgt_name = id_to_name.get(tgt_id, str(tgt_id))

            avg_vol = sum(r.get("normalized_projection_volume", 0) for r in records) / len(records)
            avg_dens = sum(r.get("projection_density", 0) for r in records) / len(records)

            if avg_vol > 0.0001 or avg_dens > 0.0001:
                projections.append(ProjectionData(
                    source_region=src_name,
                    target_region=tgt_name,
                    source_structure_id=src_id,
                    target_structure_id=tgt_id,
                    normalized_projection_volume=round(avg_vol, 6),
                    projection_density=round(avg_dens, 6),
                    n_experiments=len(records),
                    data_source=f"allen_api:avg_{len(records)}_experiments",
                    confidence=min(1.0, 0.5 + len(records) * 0.1),
                ))

    if not api_success and fallback_to_literature:
        return build_literature_matrix()

    # API成功だが一部欠損 → 文献データで補完
    if fallback_to_literature:
        lit_matrix = build_literature_matrix()
        existing_pairs = {(p.source_region, p.target_region) for p in projections}
        for lit_proj in lit_matrix.projections:
            if (lit_proj.source_region, lit_proj.target_region) not in existing_pairs:
                lit_proj.data_source = f"literature_fallback:{lit_proj.data_source}"
                lit_proj.confidence *= 0.7  # フォールバックは信頼度を下げる
                projections.append(lit_proj)

    return AllenConnectivityMatrix(
        regions=sorted(regions.keys()),
        projections=projections,
        notes=f"Allen API実測データ + 文献フォールバック。API成功={api_success}",
    )


def download_precomputed_csv(output_dir: str | Path = "data/connectivity") -> Path | None:
    """neuroinformatics.nlから事前計算済みCSVをダウンロードする。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "connectivity": "https://neuroinformatics.nl/HBP/ABA_mouse/ABA_connectivityR.csv",
        "structures": "https://neuroinformatics.nl/HBP/ABA_mouse/ABA_structures.csv",
        "injections": "https://neuroinformatics.nl/HBP/ABA_mouse/ABA_injections.csv",
    }

    downloaded = {}
    for name, url in urls.items():
        path = output_dir / f"allen_{name}.csv"
        try:
            urllib.request.urlretrieve(url, str(path))
            downloaded[name] = path
        except Exception:
            continue

    return downloaded.get("connectivity")
