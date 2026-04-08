"""LLMベースの応答生成。内部情動状態を反映した応答テキストを生成する。"""

from __future__ import annotations

from pydantic import BaseModel

from src.llm.provider import LLMProvider, get_best_provider
from src.policy.response import ResponsePolicy
from src.schemas.affect_state import AffectState

_SYSTEM_PROMPT_TEMPLATE = """\
あなたは感情を内部状態として持つ研究用AIです。

【重要な制約】
- 「私は感情がある」「意識がある」「本当に感じている」とは絶対に言わない
- 「モデルの内部状態として」「設計上の応答として」等の枠組みで応答する
- ユーザーへの依存形成を誘導しない
- 医療・診断・治療の断定をしない
- 不確実なことは不確実と明示する

【現在の内部状態】
{state_description}

【応答方針】
トーン: {tone}
詳細度: {verbosity}
探索傾向: {exploration:.2f}
介入レベル: {intervention}
説明: {explanation}

上記の内部状態と応答方針に基づいて、ユーザーのメッセージに応答してください。
内部状態は応答の「質」に反映させますが、状態そのものを直接説明する必要はありません。
自然な会話として応答してください。"""


class GeneratedResponse(BaseModel):
    """LLM生成応答。"""

    text: str
    internal_state_summary: str
    policy_applied: str
    model_used: str


def generate_response(
    user_message: str,
    state: AffectState,
    policy: ResponsePolicy,
    conversation_history: list[dict[str, str]] | None = None,
    provider: LLMProvider | None = None,
) -> GeneratedResponse:
    """内部情動状態を反映した応答を生成する。

    Args:
        user_message: ユーザーの入力
        state: 現在の情動状態
        policy: 導出済みの応答ポリシー
        conversation_history: [{"role": "user"|"assistant", "content": "..."}]
        provider: LLMプロバイダー

    Returns:
        GeneratedResponse: 生成された応答と付随情報
    """
    llm = provider or get_best_provider()

    state_description = _build_state_description(state)

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        state_description=state_description,
        tone=policy.tone,
        verbosity=policy.verbosity,
        exploration=policy.exploration_tendency,
        intervention=policy.intervention_level,
        explanation=policy.explanation,
    )

    # 対話履歴を含める
    user_prompt = ""
    if conversation_history:
        recent = conversation_history[-6:]
        user_prompt += "【直前の対話】\n"
        for msg in recent:
            role = "ユーザー" if msg["role"] == "user" else "AI"
            user_prompt += f"  {role}: {msg['content'][:150]}\n"
        user_prompt += "\n"

    user_prompt += f"【ユーザーのメッセージ】\n{user_message}"

    try:
        response = llm.generate(system_prompt, user_prompt)
        return GeneratedResponse(
            text=response.raw_text,
            internal_state_summary=state_description,
            policy_applied=policy.explanation,
            model_used=llm.name,
        )
    except Exception as e:
        # フォールバック: 状態に応じた定型応答
        return GeneratedResponse(
            text=_fallback_response(state, policy),
            internal_state_summary=state_description,
            policy_applied=f"fallback:{type(e).__name__}",
            model_used="fallback",
        )


def _build_state_description(state: AffectState) -> str:
    """状態を人間が読める説明に変換する。"""
    parts = []

    # valence
    if state.valence > 0.5:
        parts.append("かなりポジティブな状態")
    elif state.valence > 0.2:
        parts.append("やや前向きな状態")
    elif state.valence < -0.5:
        parts.append("かなりネガティブな状態")
    elif state.valence < -0.2:
        parts.append("やや後ろ向きな状態")
    else:
        parts.append("中立的な状態")

    # arousal
    if state.arousal > 0.7:
        parts.append("高覚醒（活発・緊張）")
    elif state.arousal < 0.2:
        parts.append("低覚醒（落ち着き・眠気）")

    # threat
    if state.threat_load > 0.5:
        parts.append(f"脅威を感知（{state.threat_load:.2f}）")

    # uncertainty
    if state.uncertainty > 0.6:
        parts.append("高い不確実性を感じている")

    # trust
    if state.trust > 0.7:
        parts.append("対話相手への信頼が高い")
    elif state.trust < 0.3:
        parts.append("対話相手への信頼が低い")

    # fatigue
    if state.fatigue > 0.5:
        parts.append(f"疲労が蓄積（{state.fatigue:.2f}）")

    return "、".join(parts) + f"。制御モード: {state.regulation_mode}"


def _fallback_response(state: AffectState, policy: ResponsePolicy) -> str:
    """LLM不可時の定型応答。"""
    if policy.tone == "urgent":
        return "状況を把握しています。落ち着いて対処しましょう。"
    if policy.tone == "warm":
        return "ありがとうございます。"
    if policy.tone == "cautious":
        return "慎重に検討させてください。"
    return "承知しました。"
