"""宣言的回路仕様フォーマット。NeuroML/SONATA思想に基づくYAML/JSON記述。

手配線の結合をデータ駆動に置換するための中間表現。
- 領域定義(名前、細胞タイプ、ニューロン数)
- 結合定義(ソース、ターゲット、接続確率、重み、タイプ)
- データソース参照(Allen Brain Atlas ID等)

YAML形式で記述し、Brian2回路に自動変換する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class PopulationSpec(BaseModel):
    """ニューロン集団の仕様。"""

    name: str
    cell_type: str  # RS, PV, SOM, VIP, LTS, IB, D1_MSN, D2_MSN, etc.
    n: int
    region: str  # 解剖学的領域名
    description: str = ""
    allen_structure_id: int | None = None  # Allen Brain Atlas構造ID


class ConnectionSpec(BaseModel):
    """結合の仕様。"""

    source: str  # PopulationSpec.name
    target: str
    probability: float = Field(0.2, ge=0.0, le=1.0)
    weight_mean: float = 2.0
    weight_std: float = 0.5
    conn_type: Literal["excitatory", "inhibitory"] = "excitatory"
    plasticity: bool = False
    neurotransmitter: str = ""  # "glutamate", "GABA", "dopamine", etc.
    data_source: str = ""  # "allen_api", "literature:Smith2020", "hand_tuned"
    confidence: float = Field(0.5, ge=0.0, le=1.0)  # データの信頼度
    notes: str = ""


class InputSpec(BaseModel):
    """外部入力の仕様。"""

    name: str
    target_population: str
    target_fraction: float = Field(0.33, ge=0.0, le=1.0)
    amplitude: float = 8.0
    onset_ms: float = 50.0
    duration_ms: float = 100.0


class CircuitSpec(BaseModel):
    """回路全体の仕様。"""

    name: str
    version: str = "1.0"
    description: str = ""
    populations: list[PopulationSpec]
    connections: list[ConnectionSpec]
    inputs: list[InputSpec] = Field(default_factory=list)
    simulation: dict[str, Any] = Field(default_factory=lambda: {
        "dt_ms": 0.5,
        "duration_ms": 300.0,
        "background_noise": 3.0,
    })
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CircuitSpec:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def get_population(self, name: str) -> PopulationSpec | None:
        return next((p for p in self.populations if p.name == name), None)

    def get_connections_to(self, target: str) -> list[ConnectionSpec]:
        return [c for c in self.connections if c.target == target]

    def get_connections_from(self, source: str) -> list[ConnectionSpec]:
        return [c for c in self.connections if c.source == source]

    @property
    def total_neurons(self) -> int:
        return sum(p.n for p in self.populations)

    @property
    def data_driven_fraction(self) -> float:
        """データ駆動の結合の割合。"""
        if not self.connections:
            return 0.0
        data_count = sum(1 for c in self.connections if c.data_source.startswith("allen"))
        return data_count / len(self.connections)
