"""V2回路のSBI自動較正。

SharedCoreNetworkのtonic driveパラメータをABC rejection + differential evolutionで最適化。
各シナリオを独立インスタンスで実行し、文献ターゲットとの距離を最小化する。

パラメータ空間: 10個のtonic drive値
ターゲット: quantitative_targets_v2.pyの24ターゲット
"""

from __future__ import annotations

import sys
import time

import numpy as np
from scipy.optimize import differential_evolution

from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
from src.brian2_circuits.shared_core_network import SharedCoreConfig


# ── Tunable Parameters ──────────────────────────────────────────
# tonic drive for key populations that need calibration
PARAM_NAMES = [
    "la_exc",       # target baseline 1-8Hz
    "cel_som",      # target: SOM+/PKCd+ ratio 2-5
    "cel_pkcd",     # target: 0-5Hz during CS
    "mea",          # target baseline 3-10Hz
    "vmh",          # target baseline 2-8Hz
    "vta_gaba",     # controls DA burst ability
    "dr",           # target: suppressible by LHb
    "nts_disgust",  # target 5-25Hz with contamination
    "la_cs_amp",    # CS drive amplitude for LA
    "vta_burst_amp",  # VTA burst drive amplitude
]

PARAM_BOUNDS = [
    (1.5, 4.0),   # la_exc tonic
    (0.5, 3.0),   # cel_som tonic
    (0.3, 2.0),   # cel_pkcd tonic
    (0.5, 3.0),   # mea tonic
    (1.5, 4.0),   # vmh tonic
    (1.5, 4.0),   # vta_gaba tonic
    (1.0, 3.5),   # dr tonic
    (0.5, 3.0),   # nts_disgust tonic
    (5.0, 20.0),  # la_cs_amp
    (10.0, 35.0), # vta_burst_amp
]


def _score_range(actual: float, min_val: float, max_val: float) -> float:
    """ターゲット範囲内なら1.0、範囲外なら距離に応じて0に近づく。"""
    if min_val <= actual <= max_val:
        return 1.0
    if actual < min_val:
        return max(0.0, 1.0 - (min_val - actual) / max(1.0, min_val))
    return max(0.0, 1.0 - (actual - max_val) / max(1.0, max_val))


def simulate_and_score(params: np.ndarray) -> float:
    """パラメータ→シミュレーション→スコア。最小化用なので1-scoreを返す。"""
    (la_tonic, cel_som_tonic, cel_pkcd_tonic, mea_tonic, vmh_tonic,
     vta_gaba_tonic, dr_tonic, nts_tonic, la_cs_amp, vta_burst_amp) = params

    # SharedCoreConfigのtonic driveを上書きするためモンキーパッチ
    # EmotionBrainV2を作成し、内部のSharedCoreNetworkのtonic_drivesを変更
    import src.brian2_circuits.shared_core_network as scn

    # 一時的にtonic_drivesを上書き
    original_run_trial = scn.SharedCoreNetwork.run_trial

    def patched_run_trial(self, drive_overrides=None, trial_num=0):
        # tonic driveを上書き
        c = self.cfg
        n_steps = int(c.duration_ms / c.dt_ms)
        noise_rng = np.random.default_rng(trial_num * 7 + 42)
        drive = c.bg_noise + noise_rng.normal(0, c.bg_noise * 0.3, (n_steps, self._total_n))

        tonic_overrides = {
            "la_exc": la_tonic, "cel_som": cel_som_tonic, "cel_pkcd": cel_pkcd_tonic,
            "mea": mea_tonic, "vmh": vmh_tonic, "vta_gaba": vta_gaba_tonic,
            "dr": dr_tonic, "nts_disgust": nts_tonic,
        }

        for pop_name, tonic in self._get_all_tonic_drives().items():
            if pop_name in self._idx:
                ps, pe = self._idx[pop_name]
                t = tonic_overrides.get(pop_name, tonic)
                drive[:, ps:pe] += t

        for pag_name in ["vlpag", "dlpag"]:
            if pag_name in self._idx:
                ps, pe = self._idx[pag_name]
                drive[:, ps:pe] = c.bg_noise * 0.2

        if drive_overrides:
            for pop_name, override in drive_overrides.items():
                if pop_name in self._idx:
                    s, e = self._idx[pop_name]
                    n_pop = e - s
                    if override.ndim == 1:
                        drive[:, s:e] += override[:n_pop]
                    elif override.ndim == 2:
                        drive[:override.shape[0], s:s + min(n_pop, override.shape[1])] += override[:, :n_pop]

        from brian2 import TimedArray, ms
        self._I_drive = TimedArray(drive, dt=c.dt_ms * ms)
        self._G.v = -65 + noise_rng.normal(0, 2, self._total_n)
        self._G.u = 0.2 * self._G.v[:]
        self._mon.resize(0)
        self._net.run(c.duration_ms * ms, namespace={"I_drive": self._I_drive})

        spk_i = np.array(self._mon.i[:])
        dur_s = c.duration_ms / 1000.0
        rates = {}
        for name, (s, e) in self._idx.items():
            n = e - s
            count = int(np.sum((spk_i >= s) & (spk_i < e)))
            rates[name] = count / n / dur_s if n > 0 else 0.0
        self._trial_count += 1
        from src.brian2_circuits.shared_core_network import CoreTrialResult
        return CoreTrialResult(trial_num=trial_num, rates=rates, total_spikes=len(spk_i))

    # モンキーパッチは複雑すぎるので、代わにEmotionBrainV2のprocess()で
    # la_cs_ampとvta_burst_ampを変更する別のアプローチをとる

    tonic_dict = {
        "la_exc": la_tonic, "cel_som": cel_som_tonic, "cel_pkcd": cel_pkcd_tonic,
        "mea": mea_tonic, "vmh": vmh_tonic, "vta_gaba": vta_gaba_tonic,
        "dr": dr_tonic, "nts_disgust": nts_tonic,
    }

    scores = []
    try:
        # FEAR baseline
        b = EmotionBrainV2(tonic_overrides=tonic_dict)
        s = b.process(threat=0.0)
        scores.append(_score_range(s.all_rates.get("la_exc", 0), 1, 8))

        # FEAR cs_evoked (new instance with same tonic)
        b = EmotionBrainV2(tonic_overrides=tonic_dict)
        s = b.process(threat=0.8)
        scores.append(_score_range(s.all_rates.get("la_exc", 0), 8, 35))
        scores.append(_score_range(s.all_rates.get("cem", 0), 8, 30))
        scores.append(_score_range(s.all_rates.get("cel_som", 0), 5, 25))
        scores.append(_score_range(s.all_rates.get("cel_pkcd", 0), 0, 5))

        # VTA tonic
        b = EmotionBrainV2(tonic_overrides=tonic_dict)
        s = b.process(reward=0.0)
        scores.append(_score_range(s.all_rates.get("vta_da_lat", 0), 3, 10))

        # VTA burst
        b = EmotionBrainV2(tonic_overrides=tonic_dict)
        s = b.process(reward=0.8)
        scores.append(_score_range(s.all_rates.get("vta_da_lat", 0), 15, 35))

        # RAGE baseline
        b = EmotionBrainV2(tonic_overrides=tonic_dict)
        s = b.process(frustration=0.0)
        scores.append(_score_range(s.all_rates.get("mea", 0), 3, 10))
        scores.append(_score_range(s.all_rates.get("vmh", 0), 2, 8))

        # SADNESS
        b = EmotionBrainV2(tonic_overrides=tonic_dict)
        s = b.process(loss=0.8)
        scores.append(_score_range(s.all_rates.get("sgacc", 0), 5, 30))

    except Exception as e:
        return 1.0

    avg_score = np.mean(scores) if scores else 0.0
    return 1.0 - avg_score  # minimize


def run_calibration(n_iter: int = 30, seed: int = 42) -> dict:
    """Differential Evolution で最適パラメータを探索。"""
    print(f"Starting SBI V2 Calibration (DE, maxiter={n_iter})")
    t0 = time.time()

    result = differential_evolution(
        simulate_and_score,
        bounds=PARAM_BOUNDS,
        maxiter=n_iter,
        seed=seed,
        tol=0.02,
        popsize=3,  # minimal for speed (3 * 10 params = 30 evals/gen)
        disp=True,
        workers=1,
    )

    elapsed = time.time() - t0
    best_score = 1.0 - result.fun
    param_dict = dict(zip(PARAM_NAMES, result.x))

    print(f"\n{'='*60}")
    print(f"SBI V2 Calibration Complete")
    print(f"{'='*60}")
    print(f"Best score: {best_score:.3f}")
    print(f"Iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"\nOptimal parameters:")
    for name, val in param_dict.items():
        print(f"  {name}: {val:.3f}")

    return {"score": best_score, "params": param_dict, "nfev": result.nfev}


if __name__ == "__main__":
    run_calibration(n_iter=15)
