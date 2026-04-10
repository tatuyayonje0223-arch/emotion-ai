"""自発的回復シナリオ。

消去後に時間が経過すると、恐怖反応が部分的に回復する現象。
Pavlov (1927), Rescorla (2004) で確立。

プロトコル:
1. 条件付け (CS+US × 5)
2. 消去 (CS only × 10)
3. 休息（時間経過、sleepリプレイ）
4. テスト (CS only) → 自発的回復の確認
"""

from __future__ import annotations

from dataclasses import dataclass

import brian2
brian2.prefs.codegen.target = "numpy"
from brian2 import start_scope

from src.brian2_circuits.persistent_fear_circuit import PersistentFearCircuit


@dataclass
class SpontaneousRecoveryResult:
    conditioning_freeze: list[float]
    extinction_freeze: list[float]
    recovery_freeze: list[float]
    peak_freeze: float
    post_extinction_freeze: float
    recovery_freeze_mean: float
    recovery_ratio: float  # recovery / peak
    shows_recovery: bool


def run_spontaneous_recovery(
    n_conditioning: int = 5,
    n_extinction: int = 8,
    n_rest_trials: int = 5,
    n_test: int = 3,
) -> SpontaneousRecoveryResult:
    """自発的回復プロトコルを実行する。

    PersistentFearCircuitを使用し、STDP重みが試行間で保持される。
    「休息」は入力なし試行として模倣（実際の時間経過の簡略化）。
    """
    start_scope()
    circuit = PersistentFearCircuit()

    # 1. 条件付け
    cond_results = circuit.run_conditioning(n=n_conditioning)
    cond_freeze = [r.freeze_response for r in cond_results]
    peak = max(cond_freeze) if cond_freeze else 0

    # 2. 消去
    ext_results = circuit.run_extinction(n=n_extinction)
    ext_freeze = [r.freeze_response for r in ext_results]
    post_ext = ext_freeze[-1] if ext_freeze else 0

    # 3. 休息（入力なし試行 — 時間経過の模倣）
    for i in range(n_rest_trials):
        circuit.run_trial(cs=False, us=False, phase="rest",
                          trial_num=n_conditioning + n_extinction + i)

    # 4. テスト（CS only → 自発的回復の確認）
    test_results = circuit.run_test(n=n_test)
    rec_freeze = [r.freeze_response for r in test_results]
    rec_mean = sum(rec_freeze) / len(rec_freeze) if rec_freeze else 0

    # 自発的回復 = テスト時のfreezeが消去後より高い
    shows_recovery = rec_mean > post_ext

    return SpontaneousRecoveryResult(
        conditioning_freeze=cond_freeze,
        extinction_freeze=ext_freeze,
        recovery_freeze=rec_freeze,
        peak_freeze=peak,
        post_extinction_freeze=post_ext,
        recovery_freeze_mean=rec_mean,
        recovery_ratio=rec_mean / max(peak, 0.01),
        shows_recovery=shows_recovery,
    )
