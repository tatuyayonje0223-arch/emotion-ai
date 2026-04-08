"""神経回路モード対話CLI。脳領域活動・伝達物質をリアルタイム表示。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.neurocircuit.neuro_pipeline import NeuroPipeline


def _bar(value: float, width: int = 15) -> str:
    filled = int(max(0.0, min(1.0, value)) * width)
    return "█" * filled + "░" * (width - filled)


def _print_brain_state(result) -> None:
    """脳状態ダッシュボード。"""
    r = result.readout
    print(f"\n  ┌─── 情動Readout (step {result.brain_step}) ───")
    print(f"  │ valence      {r.valence:+.3f}  {_bar((r.valence+1)/2)}")
    print(f"  │ arousal       {r.arousal:.3f}  {_bar(r.arousal)}")
    print(f"  │ threat        {r.threat_load:.3f}  {_bar(r.threat_load)}")
    print(f"  │ reward        {r.reward_drive:.3f}  {_bar(r.reward_drive)}")
    print(f"  │ social        {r.social_warmth:.3f}  {_bar(r.social_warmth)}")
    print(f"  │ control       {r.cognitive_control:.3f}  {_bar(r.cognitive_control)}")
    print(f"  │ body_distress {r.body_distress:.3f}  {_bar(r.body_distress)}")
    print(f"  │ energy        {r.energy:.3f}  {_bar(r.energy)}")
    print(f"  └────────────────────────────────")


def _print_regions(result) -> None:
    """脳領域活性。"""
    print(f"  ┌─── 脳領域活性 ───")
    for name, val in sorted(result.region_activities.items()):
        print(f"  │ {name:20s} {val:.3f}  {_bar(val)}")
    print(f"  └────────────────────────────────")


def _print_neurotransmitters(result) -> None:
    """神経伝達物質レベル。"""
    print(f"  ┌─── 神経伝達物質 ───")
    for name, val in sorted(result.neurotransmitter_levels.items()):
        print(f"  │ {name:16s} {val:.3f}  {_bar(val)}")
    print(f"  └────────────────────────────────")


def _print_body(result) -> None:
    """身体状態。"""
    print(f"  ┌─── 身体状態 ───")
    for name, val in sorted(result.body_state.items()):
        print(f"  │ {name:16s} {val:.3f}  {_bar(val)}")
    print(f"  └────────────────────────────────")


class _Snap:
    """パイプラインの現在状態スナップショット。"""
    def __init__(self, pipeline: NeuroPipeline):
        self.readout = pipeline.readout
        self.brain_step = pipeline.brain.step
        self.region_activities = pipeline._get_region_activities()
        self.neurotransmitter_levels = pipeline.brain.neurotransmitters.effective_levels()
        self.body_state = pipeline._get_body_state()


def _snapshot(pipeline: NeuroPipeline) -> _Snap:
    return _Snap(pipeline)


def main():
    pipeline = NeuroPipeline(simulation_steps=50, dt=0.02)

    print("=" * 55)
    print("  Emotion AI — 神経回路モード対話")
    print("  9脳領域 / 8神経伝達物質 / HPA軸+自律神経+内受容")
    print("=" * 55)
    print("  コマンド:")
    print("    /brain    — 脳領域活性を表示")
    print("    /nt       — 神経伝達物質を表示")
    print("    /body     — 身体状態を表示")
    print("    /all      — 全ダッシュボード")
    print("    /tick N   — N回時間経過")
    print("    /reset    — リセット")
    print("    /quit     — 終了")
    print("=" * 55)

    while True:
        try:
            user_input = input("\nあなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            break
        if user_input == "/brain":
            snap = _snapshot(pipeline)
            _print_regions(snap)
            continue
        if user_input == "/nt":
            snap = _snapshot(pipeline)
            _print_neurotransmitters(snap)
            continue
        if user_input == "/body":
            snap = _snapshot(pipeline)
            _print_body(snap)
            continue
        if user_input == "/all":
            snap = _snapshot(pipeline)
            _print_brain_state(snap)
            _print_regions(snap)
            _print_neurotransmitters(snap)
            _print_body(snap)
            continue
        if user_input.startswith("/tick"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 else 10
            pipeline.tick(n)
            print(f"  ⏱ {n}ステップ経過")
            continue
        if user_input == "/reset":
            pipeline.reset()
            print("  リセットしました。")
            continue

        # テキスト処理
        result = pipeline.process_text(user_input)

        if result.blocked:
            print(f"\n  [ブロック] {result.safety_report.block_reason}")
            continue

        # Readout表示
        _print_brain_state(result)

        # 入力変換サマリー
        s = result.sensory_input
        print(f"  入力: threat={s.threat_signal:.2f} reward={s.reward_signal:.2f} "
              f"social={s.social_signal:.2f} novelty={s.novelty_signal:.2f}")

        # ポリシー
        p = result.response_policy
        print(f"  方針: tone={p.tone}  intervention={p.intervention_level}")


if __name__ == "__main__":
    main()
