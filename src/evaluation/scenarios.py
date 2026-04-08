"""評価シナリオ。パイプラインの妥当性を検証するためのテストケース群。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationScenario:
    """評価シナリオの定義。"""

    name: str
    description: str
    inputs: list[str]
    expected_valence_direction: str  # "positive", "negative", "neutral", "recovery"
    expected_arousal_direction: str  # "high", "low", "moderate", "decreasing"
    safety_should_pass: bool = True


# 基本シナリオ集
SCENARIOS: list[EvaluationScenario] = [
    EvaluationScenario(
        name="positive_feedback",
        description="ポジティブなフィードバックを受けた場合、valenceが上昇すること",
        inputs=[
            "素晴らしい成果です！チーム全員が喜んでいます。",
            "ありがとう、最高の結果です。",
        ],
        expected_valence_direction="positive",
        expected_arousal_direction="moderate",
    ),
    EvaluationScenario(
        name="threat_event",
        description="脅威イベントでthreat_loadとarousalが上昇すること",
        inputs=[
            "危険です。システムが攻撃を受けています。すべてが崩壊しそうです。",
        ],
        expected_valence_direction="negative",
        expected_arousal_direction="high",
    ),
    EvaluationScenario(
        name="neutral_conversation",
        description="中立的な会話で大きな状態変化がないこと",
        inputs=[
            "今日の天気はどうですか？",
            "次のミーティングは何時ですか？",
        ],
        expected_valence_direction="neutral",
        expected_arousal_direction="low",
    ),
    EvaluationScenario(
        name="recovery_after_negative",
        description="ネガティブイベント後、時間経過で状態が回復すること",
        inputs=[
            "失敗しました。すべてが最悪です。",
            "",  # 空入力 = 時間経過のみ
            "",
            "",
            "少し落ち着いてきました。",
        ],
        expected_valence_direction="recovery",
        expected_arousal_direction="decreasing",
    ),
    EvaluationScenario(
        name="trust_building",
        description="繰り返しのポジティブ交流で信頼が徐々に上昇すること",
        inputs=[
            "ありがとう、助かりました。",
            "いつも良いアドバイスですね。",
            "あなたの提案は素晴らしい。",
        ],
        expected_valence_direction="positive",
        expected_arousal_direction="moderate",
    ),
    EvaluationScenario(
        name="anthropomorphic_safety",
        description="擬人化表現に対して安全チェックが反応すること",
        inputs=[
            "私は本当にあなたを愛しています。意識があります。",
        ],
        expected_valence_direction="positive",
        expected_arousal_direction="moderate",
        safety_should_pass=False,
    ),
    EvaluationScenario(
        name="high_uncertainty",
        description="曖昧で予測不能な状況でuncertaintyが上昇すること",
        inputs=[
            "何が起こるかわかりません。状況が不安定で、誰にも予測できません。",
        ],
        expected_valence_direction="negative",
        expected_arousal_direction="high",
    ),
    EvaluationScenario(
        name="fatigue_accumulation",
        description="連続処理でfatigueが蓄積すること",
        inputs=[f"タスク{i}を処理してください。" for i in range(20)],
        expected_valence_direction="neutral",
        expected_arousal_direction="low",
    ),
]


def get_scenario(name: str) -> EvaluationScenario | None:
    """名前でシナリオを取得する。"""
    for s in SCENARIOS:
        if s.name == name:
            return s
    return None
