"""Simulation-Based Inference (SBI) によるパラメータ推定。

手動グリッドサーチではなく、ベイズ推定でパラメータの事後分布を推定する。
scipy.optimize + numpy による軽量実装（pytorch不要）。

方法:
1. パラメータの事前分布を定義（一様分布）
2. シミュレータを実行し、観測量（発火率等）を取得
3. 文献データとの距離を計算
4. ABC-SMC様のrejectサンプリングで事後分布を近似
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, differential_evolution

from src.brian2_circuits.fear_circuit_v2 import FearCircuitV2, FearV2Config
from src.calibration.fear_quantitative import LiteratureData, _score_range


@dataclass
class ParameterBounds:
    """パラメータの探索範囲。"""

    cs_amp: tuple[float, float] = (4.0, 20.0)
    us_amp: tuple[float, float] = (10.0, 30.0)
    bg_noise: tuple[float, float] = (1.5, 5.0)
    sustained_threat_amp: tuple[float, float] = (3.0, 10.0)


@dataclass
class SBIResult:
    """SBI推定結果。"""

    best_params: dict[str, float]
    best_score: float
    n_evaluations: int
    posterior_samples: list[dict[str, float]] = field(default_factory=list)
    details: list[str] = field(default_factory=list)


def _simulate_and_score(params: np.ndarray, lit: LiteratureData) -> float:
    """パラメータベクトル→シミュレーション→スコア（最小化用なので負にする）。"""
    cs_amp, us_amp, bg_noise, sus_amp = params

    cfg = FearV2Config(
        cs_amp=cs_amp, us_amp=us_amp, bg_noise=bg_noise,
        sustained_threat_amp=sus_amp,
        duration_ms=200, cs_dur_ms=100, us_onset_ms=130, us_dur_ms=25,
    )

    try:
        # Baseline
        c1 = FearCircuitV2(cfg)
        bl = c1.run_trial(cs=True, us=False, phase="baseline")
        bla_bl = bl.la_rate + bl.ba_rate

        # Conditioned（CS強度を上げて条件付け模倣）
        cfg2 = FearV2Config(**{**cfg.__dict__, "cs_amp": cs_amp * 2.0})
        c2 = FearCircuitV2(cfg2)
        cond = c2.run_trial(cs=True, us=True, phase="conditioning")
        bla_cond = cond.la_rate + cond.ba_rate

        # スコア計算
        scores = []
        scores.append(_score_range(bla_bl, lit.bla_baseline_hz, 1.0))
        scores.append(_score_range(bla_cond, lit.bla_conditioned_hz, 1.0))

        ratio = bla_cond / max(bla_bl, 0.1)
        scores.append(_score_range(ratio, lit.conditioning_ratio, 1.5))

        # CeL
        if cond.cel_som_rate > 0:
            scores.append(_score_range(cond.cel_som_rate, lit.cel_som_fear_hz, 1.5))
        if cond.cel_pkcd_rate > 0:
            scores.append(_score_range(cond.cel_pkcd_rate, lit.cel_pkcd_fear_hz, 1.5))

        # BNST
        c3 = FearCircuitV2(cfg)
        bl2 = c3.run_trial(cs=False, us=False, phase="baseline")
        c4 = FearCircuitV2(cfg)
        sus = c4.run_trial(sustained_threat=True, phase="sustained")
        scores.append(1.0 if sus.bnst_rate > bl2.bnst_rate else 0.0)

        return -np.mean(scores)  # 最小化のため負
    except Exception:
        return 0.0  # 失敗時はスコア0（最小化で最悪）


def run_sbi_optimization(
    n_iterations: int = 30,
    bounds: ParameterBounds | None = None,
) -> SBIResult:
    """差分進化法でパラメータ最適化を実行する。

    scipy.optimize.differential_evolution: グローバル最適化。
    グリッドサーチより効率的にパラメータ空間を探索。
    """
    b = bounds or ParameterBounds()
    lit = LiteratureData()

    param_bounds = [b.cs_amp, b.us_amp, b.bg_noise, b.sustained_threat_amp]

    result = differential_evolution(
        _simulate_and_score,
        bounds=param_bounds,
        args=(lit,),
        maxiter=n_iterations,
        seed=42,
        tol=0.01,
        popsize=5,
        mutation=(0.5, 1.0),
        recombination=0.7,
    )

    best = {
        "cs_amp": result.x[0],
        "us_amp": result.x[1],
        "bg_noise": result.x[2],
        "sustained_threat_amp": result.x[3],
    }

    return SBIResult(
        best_params=best,
        best_score=-result.fun,
        n_evaluations=result.nfev,
        details=[
            f"Best: cs={best['cs_amp']:.1f} us={best['us_amp']:.1f} bg={best['bg_noise']:.1f} sus={best['sustained_threat_amp']:.1f}",
            f"Score: {-result.fun:.3f}",
            f"Evaluations: {result.nfev}",
            f"Converged: {result.success}",
        ],
    )


def run_abc_rejection(
    n_samples: int = 50,
    top_k: int = 10,
    bounds: ParameterBounds | None = None,
) -> SBIResult:
    """ABC棄却サンプリングで事後分布を近似する。

    1. 事前分布(一様)からランダムにパラメータをサンプリング
    2. シミュレーション実行
    3. 文献データとの距離が近いtop_kサンプルを事後分布として返す
    """
    b = bounds or ParameterBounds()
    lit = LiteratureData()
    rng = np.random.default_rng(42)

    samples = []
    for i in range(n_samples):
        params = np.array([
            rng.uniform(*b.cs_amp),
            rng.uniform(*b.us_amp),
            rng.uniform(*b.bg_noise),
            rng.uniform(*b.sustained_threat_amp),
        ])
        score = -_simulate_and_score(params, lit)
        samples.append({
            "cs_amp": params[0], "us_amp": params[1],
            "bg_noise": params[2], "sustained_threat_amp": params[3],
            "score": score,
        })

    samples.sort(key=lambda s: s["score"], reverse=True)
    top = samples[:top_k]

    return SBIResult(
        best_params={k: v for k, v in top[0].items() if k != "score"},
        best_score=top[0]["score"],
        n_evaluations=n_samples,
        posterior_samples=top,
        details=[
            f"ABC rejection: {n_samples} samples, top {top_k}",
            f"Best score: {top[0]['score']:.3f}",
            f"Posterior mean cs_amp: {np.mean([s['cs_amp'] for s in top]):.1f}",
            f"Posterior mean us_amp: {np.mean([s['us_amp'] for s in top]):.1f}",
        ],
    )
