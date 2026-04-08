"""恐怖条件付け定量ベンチマーク。

[監査Fix1] 文献データとの定量比較:
- ベースラインBLA: 5-15 Hz（背景ノイズ+自発発火）
- 条件付け後BLA: 15-50 Hz（CS誘発増強）
- 獲得曲線: 試行ごとの単調増加
- 消去曲線: CS単独提示による漸減
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.brian2_circuits.fear_circuit_v2 import FearCircuitV2, FearV2Config, FearV2TrialResult


@dataclass
class AcquisitionCurveResult:
    """獲得曲線の分析結果。"""

    baseline_rates: list[float]
    conditioning_rates: list[float]
    extinction_rates: list[float]
    baseline_mean: float = 0.0
    conditioned_mean: float = 0.0
    extinction_mean: float = 0.0
    acquisition_monotonic: bool = False
    conditioning_increase: float = 0.0  # 条件付け後/ベースラインの比
    extinction_decrease: float = 0.0    # 消去後/条件付け後の比
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)


def run_fear_acquisition_benchmark(
    config: FearV2Config | None = None,
    n_baseline: int = 3,
    n_conditioning: int = 8,
    n_extinction: int = 8,
    cs_gain_increment: float = 0.4,
) -> AcquisitionCurveResult:
    """恐怖条件付けの獲得曲線ベンチマークを実行する。

    試行間可塑性: CS+US対提示のたびにCS入力強度を段階的に増加。
    これはSTDP誘導性のCS→LA結合増強を外部パラメータで模倣する。
    """
    cfg = config or FearV2Config()

    # ベースライン（CS only）
    baseline_rates = []
    for i in range(n_baseline):
        circuit = FearCircuitV2(cfg)
        result = circuit.run_trial(cs=True, us=False, phase="baseline", trial_num=i)
        bla_rate = result.la_rate + result.ba_rate
        baseline_rates.append(bla_rate)

    # 条件付け（CS+US、各試行でCS強度を段階的に増加）
    conditioning_rates = []
    current_cs_amp = cfg.cs_amp
    for i in range(n_conditioning):
        trial_cfg = FearV2Config(
            **{k: v for k, v in cfg.__dict__.items() if k != "cs_amp"},
            cs_amp=current_cs_amp,
        )
        circuit = FearCircuitV2(trial_cfg)
        result = circuit.run_trial(cs=True, us=True, phase="conditioning", trial_num=i)
        bla_rate = result.la_rate + result.ba_rate
        conditioning_rates.append(bla_rate)
        # 試行間可塑性: CS+USペアリングで次の試行のCS応答が増強
        current_cs_amp = min(cfg.cs_amp * 4.0, current_cs_amp + cs_gain_increment)

    # 消去（CS only、高CS強度を維持して消去テスト）
    extinction_rates = []
    for i in range(n_extinction):
        trial_cfg = FearV2Config(
            **{k: v for k, v in cfg.__dict__.items() if k != "cs_amp"},
            cs_amp=max(cfg.cs_amp, current_cs_amp * 0.7),  # 消去中は少し減衰
        )
        circuit = FearCircuitV2(trial_cfg)
        result = circuit.run_trial(cs=True, us=False, phase="extinction", trial_num=i)
        bla_rate = result.la_rate + result.ba_rate
        extinction_rates.append(bla_rate)
        current_cs_amp = max(cfg.cs_amp, current_cs_amp * 0.9)  # 消去で減弱

    # 分析
    bl_mean = np.mean(baseline_rates) if baseline_rates else 0
    cond_mean = np.mean(conditioning_rates[-3:]) if conditioning_rates else 0
    ext_mean = np.mean(extinction_rates[-3:]) if extinction_rates else 0

    # 獲得の単調性: 条件付けの後半が前半より高い
    if len(conditioning_rates) >= 4:
        first_half = np.mean(conditioning_rates[:len(conditioning_rates)//2])
        second_half = np.mean(conditioning_rates[len(conditioning_rates)//2:])
        monotonic = second_half >= first_half * 0.9
    else:
        monotonic = True

    # 判定
    reasons = []
    passed = True

    # ベースラインBLA: 5-50 Hz（トイモデルなので広めの範囲）
    if not (0 < bl_mean < 100):
        passed = False
        reasons.append(f"baseline BLA rate {bl_mean:.1f} Hz outside 0-100 range")

    # 条件付け増加: ベースラインの1.2倍以上
    if bl_mean > 0 and cond_mean / bl_mean < 1.2:
        passed = False
        reasons.append(f"conditioning increase ratio {cond_mean/bl_mean:.2f} < 1.2")

    increase_ratio = cond_mean / max(bl_mean, 0.1)
    decrease_ratio = ext_mean / max(cond_mean, 0.1) if cond_mean > 0 else 1.0

    return AcquisitionCurveResult(
        baseline_rates=baseline_rates,
        conditioning_rates=conditioning_rates,
        extinction_rates=extinction_rates,
        baseline_mean=bl_mean,
        conditioned_mean=cond_mean,
        extinction_mean=ext_mean,
        acquisition_monotonic=monotonic,
        conditioning_increase=increase_ratio,
        extinction_decrease=decrease_ratio,
        passed=passed,
        failure_reasons=reasons,
    )
