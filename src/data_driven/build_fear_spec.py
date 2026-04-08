"""データ駆動版の恐怖回路仕様を構築する。

文献ベース結合マトリクスを使って、hand-tunedの結合重みをデータ駆動に置換する。
"""

from __future__ import annotations

from src.data_driven.allen_connectivity import AllenConnectivityMatrix
from src.data_driven.data_manager import get_or_build_matrix
from src.data_driven.circuit_spec import CircuitSpec, PopulationSpec, ConnectionSpec, InputSpec


def build_fear_circuit_spec(
    matrix: AllenConnectivityMatrix | None = None,
    scale: float = 1.0,
) -> CircuitSpec:
    """恐怖回路の仕様をデータ駆動で構築する。

    [監査Fix3] デフォルトでAllen API実測データを使用（キャッシュ付き）。
    API不可時は文献ベースにフォールバック。
    """
    if matrix is None:
        matrix = get_or_build_matrix()

    def n(base: int) -> int:
        return max(5, int(base * scale))

    populations = [
        # BLA分割
        PopulationSpec(name="la_exc", cell_type="RS", n=n(40), region="LA",
                       description="外側核 興奮性主細胞", allen_structure_id=131),
        PopulationSpec(name="la_pv", cell_type="PV", n=n(10), region="LA",
                       description="外側核 PV+抑制性"),
        PopulationSpec(name="la_vip", cell_type="VIP", n=n(5), region="LA",
                       description="外側核 VIP+(脱抑制)"),
        PopulationSpec(name="ba_exc", cell_type="RS", n=n(30), region="BLA",
                       description="基底核 興奮性", allen_structure_id=295),

        # CeA分割
        PopulationSpec(name="cel_som", cell_type="CeL_SOM", n=n(15), region="CeA",
                       description="中心核外側部 SOM+(fear-ON)", allen_structure_id=536),
        PopulationSpec(name="cel_pkcd", cell_type="PKCd", n=n(15), region="CeA",
                       description="中心核外側部 PKCdelta+(fear-OFF)"),
        PopulationSpec(name="cem", cell_type="RS", n=n(12), region="CeA",
                       description="中心核内側部(恐怖出力)"),

        # ITC
        PopulationSpec(name="itc", cell_type="LTS", n=n(15), region="ITC",
                       description="介在細胞塊"),

        # mPFC分割
        PopulationSpec(name="pl", cell_type="RS", n=n(30), region="PL",
                       description="前辺縁皮質(恐怖発現促進)", allen_structure_id=972),
        PopulationSpec(name="il", cell_type="RS", n=n(30), region="IL",
                       description="辺縁下皮質(消去)", allen_structure_id=44),

        # BNST
        PopulationSpec(name="bnst", cell_type="RS", n=n(15), region="BNST",
                       description="分界条床核(持続不安)", allen_structure_id=351),
    ]

    # データ駆動結合の構築
    connections = []

    # LA内E-I回路（局所回路のためAllenデータなし→文献ベース固定）
    connections.extend([
        ConnectionSpec(source="la_exc", target="la_pv", probability=0.3, weight_mean=3.0,
                       conn_type="excitatory", data_source="literature:local_circuit", confidence=0.7),
        ConnectionSpec(source="la_pv", target="la_exc", probability=0.4, weight_mean=4.0,
                       conn_type="inhibitory", data_source="literature:local_circuit", confidence=0.7),
        ConnectionSpec(source="la_vip", target="la_pv", probability=0.5, weight_mean=5.0,
                       conn_type="inhibitory", data_source="literature:VIP_disinhibition",
                       confidence=0.8, notes="Krabbe2019 Nature Neuroscience"),
    ])

    # LA → BA → CeL (データ駆動)
    la_bla = matrix.get_projection("LA", "BLA")
    connections.append(ConnectionSpec(
        source="la_exc", target="ba_exc",
        probability=matrix.to_probability("LA", "BLA") or 0.3,
        weight_mean=matrix.to_weight("LA", "BLA", 6.0) or 3.0,
        conn_type="excitatory",
        data_source=f"allen_literature:{la_bla.data_source if la_bla else 'Pitkänen2000'}",
        confidence=la_bla.confidence if la_bla else 0.5,
    ))

    # BLA → CeA (データ駆動)
    bla_cea = matrix.get_projection("BLA", "CeA")
    for target in ["cel_som", "cel_pkcd"]:
        connections.append(ConnectionSpec(
            source="ba_exc" if target == "cel_som" else "la_exc",
            target=target,
            probability=matrix.to_probability("BLA", "CeA") or 0.25,
            weight_mean=matrix.to_weight("BLA", "CeA", 6.0) or 3.0,
            conn_type="excitatory",
            data_source=f"allen_literature:{bla_cea.data_source if bla_cea else 'LeDoux2007'}",
            confidence=bla_cea.confidence if bla_cea else 0.5,
        ))

    # CeL内相互抑制
    connections.extend([
        ConnectionSpec(source="cel_som", target="cel_pkcd", probability=0.5, weight_mean=5.0,
                       conn_type="inhibitory", data_source="literature:Ciocchi2010", confidence=0.9),
        ConnectionSpec(source="cel_pkcd", target="cel_som", probability=0.3, weight_mean=3.0,
                       conn_type="inhibitory", data_source="literature:Ciocchi2010", confidence=0.9),
        ConnectionSpec(source="cel_pkcd", target="cem", probability=0.5, weight_mean=6.0,
                       conn_type="inhibitory", data_source="literature:Ciocchi2010",
                       confidence=0.9, notes="PKCd+→CeM tonic inhibition (disinhibition gate)"),
        ConnectionSpec(source="cel_som", target="cem", probability=0.2, weight_mean=1.0,
                       conn_type="excitatory", data_source="literature:Li2013", confidence=0.7),
    ])

    # PFC → BLA (データ駆動)
    pl_bla = matrix.get_projection("PL", "BLA")
    connections.append(ConnectionSpec(
        source="pl", target="la_exc",
        probability=matrix.to_probability("PL", "BLA") or 0.2,
        weight_mean=matrix.to_weight("PL", "BLA", 6.0) or 2.0,
        conn_type="excitatory",
        data_source=f"allen_literature:{pl_bla.data_source if pl_bla else 'Vertes2004'}",
        confidence=pl_bla.confidence if pl_bla else 0.5,
    ))

    # IL → ITC → CeM (消去経路)
    il_cea = matrix.get_projection("IL", "CeA")
    connections.extend([
        ConnectionSpec(source="il", target="itc",
                       probability=matrix.to_probability("IL", "CeA") or 0.3,
                       weight_mean=matrix.to_weight("IL", "CeA", 6.0) or 2.5,
                       conn_type="excitatory",
                       data_source=f"allen_literature:{il_cea.data_source if il_cea else 'Quirk2003'}",
                       confidence=il_cea.confidence if il_cea else 0.5),
        ConnectionSpec(source="itc", target="cem", probability=0.5, weight_mean=5.0,
                       conn_type="inhibitory", data_source="literature:Likhtik2008", confidence=0.8),
    ])

    # BLA → BNST (データ駆動)
    bla_bnst = matrix.get_projection("BLA", "BNST")
    connections.append(ConnectionSpec(
        source="ba_exc", target="bnst",
        probability=matrix.to_probability("BLA", "BNST") or 0.2,
        weight_mean=matrix.to_weight("BLA", "BNST", 6.0) or 2.0,
        conn_type="excitatory",
        data_source=f"allen_literature:{bla_bnst.data_source if bla_bnst else 'Kim2013'}",
        confidence=bla_bnst.confidence if bla_bnst else 0.5,
    ))
    connections.append(ConnectionSpec(
        source="cel_som", target="bnst", probability=0.2, weight_mean=1.5,
        conn_type="excitatory", data_source="literature:Dong2001", confidence=0.7,
    ))

    # 入力定義
    inputs = [
        InputSpec(name="CS", target_population="la_exc", target_fraction=0.33,
                  amplitude=8.0, onset_ms=50.0, duration_ms=150.0),
        InputSpec(name="CS_to_PL", target_population="pl", target_fraction=0.25,
                  amplitude=4.0, onset_ms=50.0, duration_ms=150.0),
        InputSpec(name="CS_to_IL", target_population="il", target_fraction=0.25,
                  amplitude=4.0, onset_ms=50.0, duration_ms=150.0),
    ]

    return CircuitSpec(
        name="fear_circuit_data_driven",
        version="2.0",
        description="データ駆動版恐怖条件付け/消去回路。"
                    "文献ベース投射データ + CeA脱抑制 + VIP + BNST + PL/IL",
        populations=populations,
        connections=connections,
        inputs=inputs,
        simulation={"dt_ms": 0.5, "duration_ms": 300.0, "background_noise": 3.0},
        metadata={
            "data_sources": ["Allen Mouse Brain Connectivity Atlas (literature-derived)",
                             "Pitkänen2000", "LeDoux2007", "Quirk2003", "Ciocchi2010"],
        },
    )
