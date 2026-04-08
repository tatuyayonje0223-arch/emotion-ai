"""脳領域間の解剖学的結合マトリクス。

コネクトーム研究に基づく結合パターンを定義する。
各結合は (source, target, weight, type) で表現。
type: 'excitatory' or 'inhibitory'
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Connection(BaseModel):
    """1つの領域間結合。"""

    source: str
    target: str
    weight: float = Field(0.5, ge=0.0, le=2.0)
    conn_type: str = "excitatory"  # "excitatory" or "inhibitory"
    neuromodulator: str | None = None  # この結合を修飾する伝達物質


# 解剖学的結合の定義（神経科学文献に基づく）
ANATOMICAL_CONNECTIONS: list[Connection] = [
    # === 扁桃体の結合 ===
    # 扁桃体 → 視床下部（恐怖→自律神経応答）
    Connection(source="amygdala", target="hypothalamus", weight=0.8, conn_type="excitatory"),
    # 扁桃体 → PAG（恐怖→凍結/逃走）
    Connection(source="amygdala", target="pag", weight=0.7, conn_type="excitatory"),
    # 扁桃体 → NAc（情動的価値→動機づけ）
    Connection(source="amygdala", target="ventral_striatum", weight=0.5, conn_type="excitatory"),
    # 扁桃体 → 海馬（情動タグ付け記憶）
    Connection(source="amygdala", target="hippocampus", weight=0.6, conn_type="excitatory"),
    # 扁桃体 → ACC（情動→コンフリクト信号）
    Connection(source="amygdala", target="acc", weight=0.4, conn_type="excitatory"),
    # 扁桃体 → 島皮質（情動→内受容統合）
    Connection(source="amygdala", target="insula", weight=0.4, conn_type="excitatory"),
    # 扁桃体 → 青斑核（恐怖→NE放出↑）
    Connection(source="amygdala", target="locus_coeruleus", weight=0.6, conn_type="excitatory"),

    # === PFC → 扁桃体（トップダウン抑制） ===
    Connection(source="vmPFC", target="amygdala", weight=0.7, conn_type="inhibitory"),
    # vmPFC → NAc（価値→報酬処理）
    Connection(source="vmPFC", target="ventral_striatum", weight=0.5, conn_type="excitatory"),
    # dlPFC → ACC（認知制御→コンフリクト解消）
    Connection(source="dlPFC", target="acc", weight=0.5, conn_type="excitatory"),
    # OFC → 扁桃体（報酬期待→情動調整）
    Connection(source="OFC", target="amygdala", weight=0.4, conn_type="inhibitory"),
    # OFC → NAc（報酬評価→動機づけ）
    Connection(source="OFC", target="ventral_striatum", weight=0.6, conn_type="excitatory"),

    # === 島皮質 ===
    # 島皮質 → ACC（内受容→コンフリクト/感情意識）
    Connection(source="insula", target="acc", weight=0.5, conn_type="excitatory"),
    # 島皮質 → 扁桃体（身体状態→情動評価）
    Connection(source="insula", target="amygdala", weight=0.4, conn_type="excitatory"),
    # 島皮質 → PFC（内受容→意思決定）
    Connection(source="insula", target="vmPFC", weight=0.3, conn_type="excitatory"),

    # === ACC ===
    # ACC → dlPFC（制御要求→認知制御起動）
    Connection(source="acc", target="dlPFC", weight=0.6, conn_type="excitatory"),
    # ACC → PAG（痛み→防御反応）
    Connection(source="acc", target="pag", weight=0.3, conn_type="excitatory"),

    # === 海馬 ===
    # 海馬 → PFC（文脈記憶→意思決定）
    Connection(source="hippocampus", target="vmPFC", weight=0.5, conn_type="excitatory"),
    # 海馬 → 扁桃体（文脈→恐怖記憶の再活性化）
    Connection(source="hippocampus", target="amygdala", weight=0.5, conn_type="excitatory"),

    # === 視床下部 ===
    # 視床下部 → 脳幹（自律神経→覚醒）
    Connection(source="hypothalamus", target="locus_coeruleus", weight=0.4, conn_type="excitatory"),
    # 視床下部 → PAG（ホメオスタシス→防御行動）
    Connection(source="hypothalamus", target="pag", weight=0.3, conn_type="excitatory"),

    # === PAG ===
    # PAG → 脳幹（防御反応→運動出力）
    Connection(source="pag", target="locus_coeruleus", weight=0.3, conn_type="excitatory"),

    # === 腹側線条体/NAc ===
    # NAc → vmPFC（報酬信号→価値更新）
    Connection(source="ventral_striatum", target="vmPFC", weight=0.4, conn_type="excitatory"),

    # === 脳幹核 ===
    # VTA → NAc（ドーパミン投射：報酬予測誤差）
    Connection(source="vta", target="ventral_striatum", weight=0.8, conn_type="excitatory",
               neuromodulator="dopamine"),
    # 青斑核 → 扁桃体（NE→脅威感度↑）
    Connection(source="locus_coeruleus", target="amygdala", weight=0.5, conn_type="excitatory",
               neuromodulator="norepinephrine"),
    # 縫線核 → 扁桃体（5-HT→抑制調整）
    Connection(source="raphe_nuclei", target="amygdala", weight=0.4, conn_type="inhibitory",
               neuromodulator="serotonin"),
    # 縫線核 → PFC（5-HT→衝動抑制）
    Connection(source="raphe_nuclei", target="vmPFC", weight=0.3, conn_type="excitatory",
               neuromodulator="serotonin"),
]


def get_connections_to(target: str) -> list[Connection]:
    """指定領域への入力結合を返す。"""
    return [c for c in ANATOMICAL_CONNECTIONS if c.target == target]


def get_connections_from(source: str) -> list[Connection]:
    """指定領域からの出力結合を返す。"""
    return [c for c in ANATOMICAL_CONNECTIONS if c.source == source]


def get_connection_matrix() -> dict[str, dict[str, float]]:
    """全結合を行列形式で返す。正=興奮性、負=抑制性。"""
    matrix: dict[str, dict[str, float]] = {}
    for c in ANATOMICAL_CONNECTIONS:
        if c.source not in matrix:
            matrix[c.source] = {}
        sign = 1.0 if c.conn_type == "excitatory" else -1.0
        matrix[c.source][c.target] = c.weight * sign
    return matrix
