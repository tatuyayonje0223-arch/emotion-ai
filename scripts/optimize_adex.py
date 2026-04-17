"""AdEx automatic parameter optimization using differential evolution.

Optimizes per-population AdEx tonic drives to maximize validation target pass rate.
Uses scipy.optimize.differential_evolution for gradient-free multi-objective optimization.

Architecture:
  - tonic_overrides replaces default tonic drive values in SharedCoreNetwork.run_trial()
  - Each evaluation creates a fresh EmotionBrainV2 per scenario (no STDP accumulation)
  - Objective = -(pass_count + smoothness_bonus) for scipy minimization

Usage:
    PYTHONPATH=. python scripts/optimize_adex.py [--max-iter 50] [--pop-size 15] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

# ═══════════════════════════════════════════════════════════
# Parameters to optimize: AdEx tonic drives that affect failing targets.
# Each entry: (population_name, current_adex_default, min_bound, max_bound)
#
# Values from SharedCoreNetwork adex_tonic dict (shared_core_network.py:546-580).
# Bounds chosen as +/- ~40% around current defaults, clamped to physiological range.
# ═══════════════════════════════════════════════════════════

PARAM_SPEC = [
    # Shared regions
    ("vta_da_lat", 2.5, 1.5, 4.0),    # IB(g_L=0.2): Grace 2007 tonic DA
    ("bnst", 2.0, 1.0, 3.5),          # Davis 2010: sustained anxiety
    ("lc", 2.8, 2.0, 4.0),            # Sara & Bouret 2012: NE tonic
    ("dr", 3.0, 2.0, 4.0),            # Celada 2001: 5-HT tonic
    ("habenula", 2.5, 1.5, 3.5),      # Matsumoto 2007: reward omission
    ("putamen", 4.5, 3.5, 6.0),       # Calder 2000: disgust recognition
    # Fear circuit
    ("il", 3.5, 2.5, 4.5),            # Quirk 2002: extinction recall
    ("cel_som", 3.5, 2.5, 4.5),       # Ciocchi 2010: SOM+ CS-evoked
    ("cem", 4.5, 3.5, 5.5),           # Ciocchi 2010: fear expression
    # Play/Lust/Surprise
    ("pfa_thalamus", 3.5, 2.5, 4.5),  # Siviy & Panksepp 2011: social play
    ("lust_mpoa", 3.5, 2.5, 4.5),     # Dominguez & Hull 2005: sexual arousal
    ("surprise_amygdala", 3.5, 2.5, 4.5),  # Sara & Bouret 2012: novelty
]

PARAM_NAMES = [p[0] for p in PARAM_SPEC]
PARAM_DEFAULTS = np.array([p[1] for p in PARAM_SPEC])
PARAM_BOUNDS = [(p[2], p[3]) for p in PARAM_SPEC]

# ═══════════════════════════════════════════════════════════
# Validation scenarios — mirrors run_v2_validation.py + quantitative_targets_v2.py
#
# Format: {emotion: [(condition_name, process_kwargs, [(region, min_hz, max_hz, typical_hz), ...]), ...]}
# ═══════════════════════════════════════════════════════════

SCENARIOS = {
    "fear": [
        ("baseline", {"threat": 0.0},
         [("la_exc", 2, 5, 3.0)]),
        ("cs_evoked", {"threat": 0.8},
         [("la_exc", 14, 26, 20.0), ("cel_som", 8, 16, 12.0),
          ("cel_pkcd", 0, 2, 0.5), ("cem", 10, 20, 15.0)]),
        ("fear_burst", {"threat": 0.8},
         [("pl", 17, 33, 25.0)]),
        ("freezing", {"threat": 0.8},
         [("vlpag", 7, 13, 10.0)]),
        ("extinction_recall", {"threat": 0.1},
         [("il", 7, 13, 10.0)]),
    ],
    "rage": [
        ("baseline", {"frustration": 0.0},
         [("mea", 3, 7, 5.0), ("vmh", 2, 5, 3.5)]),
        ("social_encounter", {"frustration": 0.3},
         [("mea", 10, 20, 15.0)]),
        ("investigation", {"frustration": 0.5},
         [("vmh", 7, 13, 10.0)]),
        ("attack", {"frustration": 0.8},
         [("vmh", 24, 46, 35.0), ("dlpag", 17, 33, 25.0)]),
    ],
    "seeking": [
        ("tonic", {"reward": 0.0},
         [("vta_da_lat", 3, 7, 5.0)]),
        ("phasic_burst", {"reward": 0.8},
         [("vta_da_lat", 17, 33, 25.0)]),
        ("pause", {"loss": 0.5},
         [("vta_da_lat", 0, 1, 0.0)]),
        ("reward", {"reward": 0.8},
         [("nac_shell_d1", 8, 16, 12.0)]),
    ],
    "sadness": [
        ("depression", {"loss": 0.8},
         [("sgacc", 14, 20, 16.0)]),
        ("reward_omission", {"loss": 0.5},
         [("habenula", 10, 20, 15.0)]),
        ("sadness_suppressed", {"loss": 0.8},
         [("dr", 2, 4, 3.0)]),
    ],
    "disgust": [
        ("disgust_stimulus", {"contamination": 0.8},
         [("aic", 10, 20, 15.0), ("nts_disgust", 10, 20, 15.0)]),
        ("disgust_recognition", {"contamination": 0.8},
         [("putamen", 7, 13, 10.0)]),
    ],
    "care": [
        ("social_bonding", {"social": 0.8, "attachment_need": 0.6},
         [("mpoa", 7, 13, 10.0), ("pvn_oxt", 5, 11, 8.0),
          ("vta_da_lat", 7, 15, 10.0)]),
    ],
    "panic_grief": [
        ("separation", {"loss": 0.8, "attachment_need": 0.8},
         [("dacc", 10, 20, 15.0), ("bnst", 7, 13, 10.0),
          ("pvn_crh", 5, 15, 10.0)]),
    ],
    "play": [
        ("social_play", {"social": 0.7, "reward": 0.5, "novelty": 0.3},
         [("pfa_thalamus", 7, 13, 10.0), ("play_cortex", 5, 15, 10.0)]),
    ],
    "lust": [
        ("sexual_arousal", {"social": 0.7, "reward": 0.4},
         [("lust_mpoa", 8, 16, 12.0), ("vta_da_lat", 7, 15, 10.0)]),
    ],
    "surprise": [
        ("novelty_burst", {"novelty": 0.9},
         [("lc", 8, 16, 12.0)]),
        ("novelty", {"novelty": 0.9},
         [("surprise_amygdala", 7, 13, 10.0)]),
    ],
}


def _count_targets() -> int:
    """Total number of firing rate targets across all scenarios."""
    return sum(
        len(targets)
        for scenarios in SCENARIOS.values()
        for _, _, targets in scenarios
    )


TOTAL_TARGETS = _count_targets()


def evaluate(params: np.ndarray, *, verbose: bool = False) -> float:
    """Evaluate AdEx configuration with given tonic drive parameters.

    Returns negative (pass_count + smoothness_bonus) for scipy minimization.
    The smoothness bonus (0..1) rewards near-miss targets to guide the optimizer
    toward the basin of attraction even when pass count is tied.
    """
    # Lazy imports to avoid Brian2 startup cost at module level
    from src.brian2_circuits.emotion_circuits_v2 import EmotionBrainV2
    from src.brian2_circuits.shared_core_network import SharedCoreConfig

    cfg = SharedCoreConfig(use_adex=True)

    # Build tonic overrides dict from parameter vector
    overrides: dict[str, float] = {}
    for i, name in enumerate(PARAM_NAMES):
        overrides[name] = float(params[i])

    passed = 0
    total = 0
    penalty = 0.0
    details: list[str] = []

    for emotion, scenarios in SCENARIOS.items():
        for cond_name, inputs, targets in scenarios:
            # Fresh brain per scenario (no STDP accumulation)
            brain = EmotionBrainV2(config=cfg, tonic_overrides=overrides)
            result = brain.process(**inputs)
            rates = result.all_rates

            for pop, lo, hi, typical in targets:
                rate = rates.get(pop, 0.0)
                total += 1
                in_range = lo <= rate <= hi
                if in_range:
                    passed += 1
                    status = "PASS"
                else:
                    status = "FAIL"
                    # Normalized distance from target range (0 = on boundary, 1+ = far)
                    if rate < lo:
                        penalty += (lo - rate) / max(typical, 1.0)
                    else:
                        penalty += (rate - hi) / max(typical, 1.0)

                if verbose:
                    details.append(
                        f"  {status}  {emotion}/{cond_name}/{pop}: "
                        f"{rate:.1f} Hz  [{lo}-{hi}]"
                    )

    if verbose:
        for d in details:
            print(d)
        print(f"\n  Total: {passed}/{total} PASS, penalty={penalty:.2f}")

    # Objective: maximize passes, break ties with distance penalty
    # smoothness_bonus in [0, 1]: 1.0 when penalty=0, decays toward 0
    smoothness_bonus = max(0.0, 1.0 - penalty * 0.1)
    return -(passed + smoothness_bonus)


def _callback_factory(start_time: float, total_targets: int):
    """Create a callback that logs progress (no re-evaluation — uses convergence)."""
    state = {"gen": 0, "best_convergence": 1.0}

    def callback(xk, convergence):
        state["gen"] += 1
        elapsed = time.time() - start_time
        improved = convergence < state["best_convergence"]
        if improved:
            state["best_convergence"] = convergence
        marker = " *" if improved else ""
        print(
            f"  Gen {state['gen']:3d} | "
            f"convergence={convergence:.4f} | "
            f"elapsed={elapsed:.0f}s{marker}"
        )

    return callback


def main():
    parser = argparse.ArgumentParser(
        description="AdEx tonic drive optimization via differential evolution"
    )
    parser.add_argument("--max-iter", type=int, default=50,
                        help="Maximum DE generations (default: 50)")
    parser.add_argument("--pop-size", type=int, default=15,
                        help="DE population size multiplier (default: 15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-target results for initial/final evaluation")
    parser.add_argument("--save", type=str, default=None,
                        help="Save optimal parameters to JSON file")
    args = parser.parse_args()

    n_params = len(PARAM_SPEC)
    print("=" * 70)
    print("  AdEx Tonic Drive Optimization (Differential Evolution)")
    print("=" * 70)
    print(f"  Parameters:  {n_params}")
    print(f"  Targets:     {TOTAL_TARGETS}")
    print(f"  Max iter:    {args.max_iter}")
    print(f"  Pop size:    {args.pop_size} (total pop = {args.pop_size * n_params})")
    print(f"  Seed:        {args.seed}")
    print()

    # ── Initial evaluation with current defaults ──
    print("Initial evaluation (current AdEx defaults):")
    initial_score = evaluate(PARAM_DEFAULTS, verbose=args.verbose)
    initial_pass = round(-initial_score)
    print(f"  => {initial_pass}/{TOTAL_TARGETS} PASS (score={-initial_score:.2f})")
    print()

    # ── Run differential evolution ──
    print("Starting optimization...")
    t0 = time.time()

    callback = _callback_factory(t0, TOTAL_TARGETS)

    result = differential_evolution(
        evaluate,
        bounds=PARAM_BOUNDS,
        maxiter=args.max_iter,
        popsize=args.pop_size,
        seed=args.seed,
        tol=0.01,
        x0=PARAM_DEFAULTS,  # start from current best
        callback=callback,
        disp=False,  # we use our own callback for progress
        workers=1,   # Brian2 is not thread-safe
    )

    elapsed = time.time() - t0

    # ── Results ──
    print()
    print("=" * 70)
    print(f"  Optimization completed in {elapsed:.0f}s ({result.nit} generations)")
    print(f"  DE status: {result.message}")
    print("=" * 70)

    final_score = evaluate(result.x, verbose=args.verbose)
    final_pass = round(-final_score)

    print(f"\n  Result: {initial_pass}/{TOTAL_TARGETS} -> {final_pass}/{TOTAL_TARGETS} PASS")
    print(f"  Improvement: +{final_pass - initial_pass} targets")

    print(f"\n  Optimal tonic drives:")
    print(f"  {'Population':<25s} {'Default':>8s} {'Optimal':>8s} {'Delta':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    optimal_params = {}
    for i, (name, default, lo, hi) in enumerate(PARAM_SPEC):
        opt_val = result.x[i]
        delta = opt_val - default
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<25s} {default:8.2f} {opt_val:8.2f} {sign}{delta:7.2f}")
        optimal_params[name] = round(float(opt_val), 4)

    # ── Save results ──
    if args.save:
        save_path = Path(args.save)
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "initial_pass": initial_pass,
            "final_pass": final_pass,
            "total_targets": TOTAL_TARGETS,
            "generations": result.nit,
            "elapsed_seconds": round(elapsed, 1),
            "optimal_tonic_drives": optimal_params,
            "config": {
                "max_iter": args.max_iter,
                "pop_size": args.pop_size,
                "seed": args.seed,
            },
        }
        save_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\n  Results saved to: {save_path}")

    # ── Code snippet for applying results ──
    print("\n  To apply these values, update adex_tonic in shared_core_network.py:")
    print("  " + "-" * 50)
    for name, val in optimal_params.items():
        print(f'    "{name}": {val},')
    print("  " + "-" * 50)

    return final_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
