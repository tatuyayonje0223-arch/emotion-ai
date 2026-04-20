"""Statistical significance for lesion specificity claims (Phase 9.6 + 9.10).

v4 audit finding: "4/10 emotions show specificity" is statistically unsupported
without testing. Apply Fisher exact on the 2x2 contingency of
before-lesion vs after-lesion accuracy.

For each emotion:
  H0: lesion has no effect on accuracy of true-X instances
  2x2 table:
    before-correct / after-correct
    before-wrong   / after-wrong
"""
from __future__ import annotations

from scipy.stats import fisher_exact


def lesion_fisher_test(baseline_correct: int, baseline_n: int,
                       lesion_correct: int, lesion_n: int,
                       alternative: str = "greater") -> dict:
    """Fisher exact test for lesion accuracy change.

    baseline: before lesion, lesion: after lesion.
    alternative='greater' tests H1: baseline > lesion (accuracy dropped after lesion).
    """
    # 2x2 table:
    #            correct  wrong
    # baseline     a        b
    # lesion       c        d
    a = baseline_correct
    b = baseline_n - baseline_correct
    c = lesion_correct
    d = lesion_n - lesion_correct
    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative=alternative)
    return {
        "baseline_acc": a / baseline_n if baseline_n else 0,
        "lesion_acc": c / lesion_n if lesion_n else 0,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "significant_05": p_value < 0.05,
        "significant_001": p_value < 0.001,
    }


# Phase 9.6 + 9.10 raw data from docs/phase9_*.md
# (baseline correct / n, lesion correct / n) per emotion:
PHASE9_LESION_DATA = {
    # Phase 9.6 input-level lesion (n=50 subset)
    "input_FEAR_n3":     {"base_c": 2, "base_n": 3,  "les_c": 0, "les_n": 3},
    "input_RAGE_n11":    {"base_c": 3, "base_n": 11, "les_c": 0, "les_n": 11},  # 27.3%→0
    "input_SADNESS_n7":  {"base_c": 2, "base_n": 7,  "les_c": 0, "les_n": 7},   # 28.6%→0

    # Phase 9.10 pop-level lesion (n=50 subset)
    "pop_RAGE_n11":      {"base_c": 3, "base_n": 11, "les_c": 0, "les_n": 11},
    "pop_SEEKING_n8":    {"base_c": 1, "base_n": 8,  "les_c": 0, "les_n": 8},   # 12.5%→0
    "pop_SADNESS_n7":    {"base_c": 2, "base_n": 7,  "les_c": 0, "les_n": 7},
    "pop_FEAR_n3":       {"base_c": 2, "base_n": 3,  "les_c": 2, "les_n": 3},   # no drop
}


if __name__ == "__main__":
    print("Fisher exact test for Phase 9.6 + 9.10 lesion specificity claims\n")
    print(f"{'test':<25s} {'base':>10s} {'lesion':>10s} {'OR':>6s} {'p':>8s} {'p<0.05':>8s} {'p<0.001':>8s}")
    for name, d in PHASE9_LESION_DATA.items():
        r = lesion_fisher_test(d["base_c"], d["base_n"], d["les_c"], d["les_n"])
        sig05 = "✓" if r["significant_05"] else "✗"
        sig001 = "✓" if r["significant_001"] else "✗"
        or_str = f"{r['odds_ratio']:.2f}" if r["odds_ratio"] != float("inf") else "inf"
        print(f"  {name:<23s} {r['baseline_acc']:>7.1%} {r['lesion_acc']:>9.1%} {or_str:>6s} "
              f"{r['p_value']:>8.4f} {sig05:>7s} {sig001:>8s}")

    print("\nInterpretation:")
    print("  ✓ p<0.05 means lesion-drop is statistically significant")
    print("  If p>0.05, the specificity claim has no statistical support at that n")
