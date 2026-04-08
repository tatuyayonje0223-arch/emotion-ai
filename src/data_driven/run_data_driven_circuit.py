"""data_driven → Brian2 の統合実行。

[監査H5修正] CircuitSpec(YAML) → Brian2自動変換 → 実行 → 結果返却。
build_fear_spec()で生成した仕様を即座にBrian2で動かす。
"""

from __future__ import annotations

from src.data_driven.build_fear_spec import build_fear_circuit_spec
from src.data_driven.spec_to_brian2 import build_and_run, SimulationResult
from src.data_driven.allen_connectivity import AllenConnectivityMatrix


def run_data_driven_fear(
    scale: float = 1.0,
    matrix: AllenConnectivityMatrix | None = None,
    extra_cs_amp: float = 0.0,
) -> SimulationResult:
    """データ駆動版恐怖回路をBrian2で実行する。

    1. Allen APIまたは文献マトリクスから結合データ取得
    2. CircuitSpecを自動構築
    3. Brian2ネットワークに変換
    4. シミュレーション実行
    5. 各集団の発火率を返す
    """
    import numpy as np

    spec = build_fear_circuit_spec(matrix=matrix, scale=scale)

    # CS入力の強度調整
    if extra_cs_amp > 0:
        for inp in spec.inputs:
            if inp.name == "CS":
                inp.amplitude += extra_cs_amp

    result = build_and_run(spec)
    return result


def compare_hand_vs_data_driven() -> dict:
    """手配線とデータ駆動の結果を比較する。"""
    from src.brian2_circuits.fear_circuit_v2 import FearCircuitV2, FearV2Config

    # 手配線版
    hand = FearCircuitV2(FearV2Config(duration_ms=200, cs_dur_ms=100, us_onset_ms=130, us_dur_ms=20))
    hand_result = hand.run_trial(cs=True, us=False, phase="test")

    # データ駆動版
    dd_result = run_data_driven_fear(scale=0.5)

    return {
        "hand_wired": {
            "la_rate": hand_result.la_rate,
            "cem_rate": hand_result.cem_rate,
            "freeze": hand_result.freeze_response,
        },
        "data_driven": {
            "total_spikes": dd_result.total_spikes,
            "population_rates": dd_result.population_rates,
        },
    }
