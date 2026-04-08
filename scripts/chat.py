"""対話ループCLI。感情パイプラインと会話し、内部状態をリアルタイムで観察する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import ExperimentConfig, LLMConfig
from src.pipeline import EmotionPipeline


def _state_bar(value: float, width: int = 20, lo: float = -1.0, hi: float = 1.0) -> str:
    """状態変数を視覚的なバーで表示する。"""
    normalized = (value - lo) / (hi - lo)
    filled = int(normalized * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _print_state(pipeline: EmotionPipeline) -> None:
    """内部状態をダッシュボード形式で表示する。"""
    s = pipeline.current_state
    print(f"\n  ┌─── 内部情動状態 (step {s.step_count}) ───")
    print(f"  │ valence     {s.valence:+.3f}  {_state_bar(s.valence, 20, -1, 1)}")
    print(f"  │ arousal      {s.arousal:.3f}  {_state_bar(s.arousal, 20, 0, 1)}")
    print(f"  │ salience     {s.motivational_salience:.3f}  {_state_bar(s.motivational_salience, 20, 0, 1)}")
    print(f"  │ control      {s.perceived_control:.3f}  {_state_bar(s.perceived_control, 20, 0, 1)}")
    print(f"  │ uncertainty  {s.uncertainty:.3f}  {_state_bar(s.uncertainty, 20, 0, 1)}")
    print(f"  │ trust        {s.trust:.3f}  {_state_bar(s.trust, 20, 0, 1)}")
    print(f"  │ threat       {s.threat_load:.3f}  {_state_bar(s.threat_load, 20, 0, 1)}")
    print(f"  │ fatigue      {s.fatigue:.3f}  {_state_bar(s.fatigue, 20, 0, 1)}")
    print(f"  │ regulation   {s.regulation_mode}")
    print(f"  └────────────────────────────────")


def _print_appraisal(result) -> None:
    """評価結果を表示する。"""
    a = result.appraisal
    print(f"  評価: goal={a.goal_relevance:+.2f}  novelty={a.novelty:.2f}  "
          f"control={a.controllability:.2f}  reward/threat={a.reward_threat_balance:+.2f}  "
          f"conf={a.confidence:.2f}")
    print(f"  制御: {result.regulation_mode} — {result.regulation_reason}")
    p = result.response_policy
    print(f"  方針: tone={p.tone}  verbosity={p.verbosity}  "
          f"explore={p.exploration_tendency:.2f}  intervention={p.intervention_level}")
    if not result.safety_report.all_passed:
        for c in result.safety_report.checks:
            if not c.passed:
                print(f"  ⚠ 安全: [{c.severity}] {c.details}")


def main():
    parser = argparse.ArgumentParser(description="Emotion AI 対話CLI")
    parser.add_argument("--llm", action="store_true", help="LLMモードを有効にする")
    parser.add_argument("--provider", choices=["gemini", "anthropic", "mock", "auto"],
                        default="auto", help="LLMプロバイダー")
    parser.add_argument("--no-response", action="store_true", help="LLM応答生成を無効にする")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細表示")
    parser.add_argument("--config", type=str, help="設定YAMLファイルパス")
    args = parser.parse_args()

    # 設定構築
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()

    if args.llm:
        config.llm = LLMConfig(
            enabled=True,
            provider=args.provider,
            generate_responses=not args.no_response,
        )

    pipeline = EmotionPipeline(config)
    mode = f"LLM ({args.provider})" if args.llm else "ヒューリスティクス"

    print("=" * 50)
    print("  Emotion-Capable AI — 対話セッション")
    print(f"  モード: {mode}")
    print(f"  状態変数: {len(pipeline.current_state.variable_names())}次元")
    print("=" * 50)
    print("  コマンド:")
    print("    /state  — 内部状態を表示")
    print("    /detail — 詳細モード切替")
    print("    /memory — 記憶一覧")
    print("    /tick   — 時間経過（自然減衰）")
    print("    /reset  — リセット")
    print("    /quit   — 終了")
    print("=" * 50)

    verbose = args.verbose

    while True:
        try:
            user_input = input("\nあなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_input:
            continue

        # コマンド処理
        if user_input == "/quit":
            print("終了します。")
            break
        if user_input == "/state":
            _print_state(pipeline)
            continue
        if user_input == "/detail":
            verbose = not verbose
            print(f"  詳細モード: {'ON' if verbose else 'OFF'}")
            continue
        if user_input == "/memory":
            from src.schemas.memory import RetrievalQuery
            query = RetrievalQuery(max_results=10, min_strength=0.0)
            results = pipeline.memory_store.retrieve(query)
            print(f"  記憶数: {pipeline.memory_store.size}")
            for r in results:
                print(f"    [{r.memory.emotional_salience:.2f}] {r.memory.summary[:60]}")
            continue
        if user_input == "/tick":
            pipeline.tick()
            print("  ⏱ 時間経過（自然減衰適用）")
            _print_state(pipeline)
            continue
        if user_input == "/reset":
            pipeline.reset()
            print("  リセットしました。")
            continue

        # テキスト処理
        result = pipeline.process_text(user_input)

        # 応答表示
        if result.generated_response:
            print(f"\nAI> {result.generated_response}")
        else:
            # LLMなしモード: 方針ベースの簡易応答
            p = result.response_policy
            print(f"\n  [{p.tone}] (LLM応答なし — /chat --llm で有効化)")

        # 状態変化のサマリー
        v_before = result.state_before.get("valence", 0)
        v_after = result.state_after.get("valence", 0)
        v_delta = v_after - v_before
        a_after = result.state_after.get("arousal", 0)
        t_after = result.state_after.get("threat_load", 0)
        print(f"  Δvalence={v_delta:+.3f}  arousal={a_after:.3f}  threat={t_after:.3f}  "
              f"reg={result.regulation_mode}  mem={'✓' if result.memory_stored else '—'}")

        if verbose:
            _print_appraisal(result)
            _print_state(pipeline)


if __name__ == "__main__":
    main()
