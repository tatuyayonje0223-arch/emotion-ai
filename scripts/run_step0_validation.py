"""Step 0: 恐怖回路の定量検証レポートを生成する。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.fear_quantitative import validate_fear_circuit, parameter_sweep, LiteratureData
from src.brian2_circuits.fear_circuit_v2 import FearV2Config


def main():
    print("=" * 65)
    print("  Step 0: Fear Circuit Quantitative Validation")
    print("  Literature: Quirk 1995, Ciocchi 2010, Herry 2008, Davis 2010")
    print("=" * 65)

    # デフォルト設定での検証
    print("\n--- Default Config ---")
    cfg = FearV2Config(cs_amp=10.0, us_amp=20.0, duration_ms=250, cs_dur_ms=120, us_onset_ms=150, us_dur_ms=25)
    result = validate_fear_circuit(cfg, n_conditioning=5, n_extinction=5)

    for d in result.details:
        print(f"  {d}")
    print(f"\n  Overall score: {result.score:.3f} ({'PASS' if result.passed else 'FAIL'})")

    # パラメータ探索
    print(f"\n--- Parameter Sweep (12 configs) ---")
    results = parameter_sweep()
    print(f"  Best score: {results[0].score:.3f}")
    print(f"  Best config: cs_amp={results[0].config.cs_amp}, us_amp={results[0].config.us_amp}")
    for d in results[0].details[:5]:
        print(f"    {d}")

    # メトリクスサマリー
    best = results[0]
    m = best.metrics
    print(f"\n--- Best Config Metrics ---")
    print(f"  BLA baseline:    {m['bla_baseline']:.1f} Hz (target: 8 Hz)")
    print(f"  BLA conditioned: {m['bla_conditioned']:.1f} Hz (target: 25 Hz)")
    print(f"  Ratio:           {m['conditioning_ratio']:.2f}x (target: 3x)")
    print(f"  CeL SOM+:        {m['cel_som_rate']:.1f} Hz (target: 15 Hz)")
    print(f"  CeL PKCd+:       {m['cel_pkcd_rate']:.1f} Hz (target: 5 Hz)")
    print(f"  BNST sustained:  {m['bnst_sustained']:.1f} Hz")
    print(f"  Acquisition:     {[f'{r:.1f}' for r in m.get('acquisition_rates', [])]}")

    print(f"\n{'=' * 65}")
    print(f"  Score: {best.score:.3f} / 1.0")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
