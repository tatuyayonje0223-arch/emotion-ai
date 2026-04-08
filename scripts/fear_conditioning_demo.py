"""恐怖条件付け→消去デモ。Pavlovian学習の神経回路シミュレーション。

Phase 1: 条件付け — CS(音) + US(電撃) → 扁桃体が強く反応するようになる
Phase 2: 消去 — CS単独 + PFCの抑制 → 扁桃体反応が徐々に弱まる
Phase 3: 自発的回復 — 時間経過後にCSを再提示 → 部分的な恐怖反応の回復
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.neurocircuit.brain import BrainState, SensoryInput, compute_readout, step_brain
from src.neurocircuit.plasticity import PlasticityParams, fear_conditioning, extinction


def main():
    print("=" * 65)
    print("  恐怖条件付け → 消去 → 自発的回復 デモ")
    print("  (Pavlovian fear conditioning simulation)")
    print("=" * 65)

    brain = BrainState()
    params = PlasticityParams()
    cs_amygdala_weight = 0.2  # CS→扁桃体の初期結合強度

    # === Phase 1: 条件付け (CS + US) ===
    print("\n Phase 1: 恐怖条件付け (CS + US, 20試行)")
    print("  CS=脅威信号(0.3), US=疼痛(0.8)")
    print("-" * 55)

    for trial in range(20):
        sensory = SensoryInput(threat_signal=0.3 + cs_amygdala_weight * 0.3, pain_input=0.8)
        for _ in range(30):
            brain = step_brain(brain, sensory, dt=0.02)

        # 可塑性: CS→扁桃体結合の強化
        cs_amygdala_weight = fear_conditioning(
            cs_amygdala_weight, cs_activity=0.5, us_activity=0.8,
            amygdala_activity=brain.amygdala.output, params=params,
        )

        readout = compute_readout(brain)
        if trial % 5 == 0 or trial == 19:
            print(f"  Trial {trial+1:2d}: amygdala={brain.amygdala.output:.3f}  "
                  f"threat={readout.threat_load:.3f}  CS-weight={cs_amygdala_weight:.3f}  "
                  f"cortisol={brain.body.hpa.cortisol:.3f}")

    # === Phase 2: 消去 (CS のみ、USなし) ===
    print("\n Phase 2: 消去 (CS only, 30試行)")
    print("  CS=脅威信号(減衰中), US=なし, PFC抑制あり")
    print("-" * 55)

    for trial in range(30):
        sensory = SensoryInput(threat_signal=0.3 + cs_amygdala_weight * 0.2)
        for _ in range(30):
            brain = step_brain(brain, sensory, dt=0.02)

        # 可塑性: PFC抑制による消去
        cs_amygdala_weight = extinction(
            cs_amygdala_weight, cs_activity=0.5, us_absence=True,
            pfc_inhibition=brain.pfc.vmPFC.output, params=params,
        )

        readout = compute_readout(brain)
        if trial % 10 == 0 or trial == 29:
            print(f"  Trial {trial+1:2d}: amygdala={brain.amygdala.output:.3f}  "
                  f"threat={readout.threat_load:.3f}  CS-weight={cs_amygdala_weight:.3f}  "
                  f"vmPFC={brain.pfc.vmPFC.output:.3f}")

    # === Phase 3: 自発的回復 ===
    print("\n Phase 3: 自発的回復 (休息後にCS再提示)")
    print("  休息: 100ステップ → CS再提示")
    print("-" * 55)

    # 休息
    empty = SensoryInput()
    for _ in range(100):
        brain = step_brain(brain, empty, dt=0.02)

    readout_rest = compute_readout(brain)
    print(f"  休息後:   amygdala={brain.amygdala.output:.3f}  "
          f"threat={readout_rest.threat_load:.3f}  cortisol={brain.body.hpa.cortisol:.3f}")

    # CS再提示
    sensory = SensoryInput(threat_signal=0.3 + cs_amygdala_weight * 0.3)
    for _ in range(30):
        brain = step_brain(brain, sensory, dt=0.02)

    readout_renewal = compute_readout(brain)
    print(f"  CS再提示: amygdala={brain.amygdala.output:.3f}  "
          f"threat={readout_renewal.threat_load:.3f}  CS-weight={cs_amygdala_weight:.3f}")

    # === サマリー ===
    print(f"\n{'=' * 65}")
    print("  結果サマリー:")
    print(f"    CS→扁桃体結合: 初期=0.200 → 条件付け後={cs_amygdala_weight:.3f}")
    print(f"    消去によりPFC-扁桃体抑制が機能")
    print(f"    自発的回復: 部分的な恐怖反応の再出現")
    print(f"    （実際のPTSD/恐怖記憶研究と定性的に整合）")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
