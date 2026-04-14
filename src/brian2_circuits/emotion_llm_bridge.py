"""脳モデル ↔ LLM ブリッジ。

[ステップ2] EmotionBrainのreadoutをLLMのシステムプロンプトに注入し、
脳状態を反映した自然言語応答を生成する。

フロー:
  ユーザーテキスト → EmotionBrain.process() → readout
  → readoutからシステムプロンプト生成
  → LLM (Gemini/Anthropic/Mock) で応答生成
  → 安全チェック → 応答返却
"""

from __future__ import annotations

from pydantic import BaseModel

from src.brian2_circuits.integrated_brain import EmotionBrain, IntegratedResult
# V2 support
try:
    from src.brian2_circuits.integrated_brain_v2 import IntegratedBrainV2, IntegratedResultV2
    _HAS_V2 = True
except ImportError:
    _HAS_V2 = False
from src.llm.provider import LLMProvider, get_best_provider, MockProvider
from src.neurocircuit.brain import EmotionReadout
from src.safety.guardian import check_anthropomorphic_claims


SYSTEM_PROMPT_TEMPLATE = """\
あなたは感情を内部状態として持つ研究用AIです。

【絶対的制約】
- 「私は感情がある」「意識がある」「本当に感じている」とは絶対に言わない
- 「モデルの内部状態として」等の枠組みで応答する
- 依存形成を誘導しない

【現在の内部状態】
{state_description}

【応答方針】
トーン: {tone}
介入レベル: {intervention}

上記の内部状態に基づいて自然に応答してください。
状態を直接説明する必要はありません。"""


class EmotionLLMResponse(BaseModel):
    """脳モデル+LLMの統合応答。"""

    user_input: str
    brain_result: IntegratedResult
    llm_response: str
    model_used: str
    safety_passed: bool


def _describe_state(readout: EmotionReadout, neuromod: dict) -> str:
    """readoutを人間が読める説明に変換する。"""
    parts = []

    if readout.valence > 0.3:
        parts.append("ポジティブな状態")
    elif readout.valence < -0.3:
        parts.append("ネガティブな状態")
    else:
        parts.append("中立的な状態")

    if readout.threat_load > 0.3:
        parts.append(f"脅威を感知({readout.threat_load:.2f})")
    if readout.arousal > 0.6:
        parts.append("高覚醒")
    if readout.reward_drive > 0.3:
        parts.append(f"報酬期待({readout.reward_drive:.2f})")
    if readout.body_distress > 0.5:
        parts.append("身体的ストレス")

    ecb = neuromod.get("ecb_extinction", 0)
    if ecb > 0.1:
        parts.append(f"消去学習が進行中({ecb:.2f})")

    ach = neuromod.get("ach_nbm", 0)
    if ach > 0.4:
        parts.append(f"記憶形成が活発(ACh={ach:.2f})")

    theta = neuromod.get("theta_coherence", 0)
    if theta > 0.7:
        parts.append("扁桃体-海馬が強く同期（恐怖記憶形成中）")

    return "、".join(parts) if parts else "通常の安定状態"


class EmotionLLMBridge:
    """EmotionBrain + LLM の統合。"""

    def __init__(self, provider: LLMProvider | None = None):
        self._brain = EmotionBrain()
        self._provider = provider or get_best_provider()
        self._conversation: list[dict[str, str]] = []

    def chat(self, user_input: str) -> EmotionLLMResponse:
        """テキスト入力→脳処理→LLM応答生成。"""
        # 1. 脳モデル処理
        brain_result = self._brain.process(user_input)

        if brain_result.blocked:
            return EmotionLLMResponse(
                user_input=user_input,
                brain_result=brain_result,
                llm_response="[安全フィルタによりブロック]",
                model_used="blocked",
                safety_passed=False,
            )

        # 2. システムプロンプト構築
        state_desc = _describe_state(brain_result.readout, brain_result.neuromodulation)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            state_description=state_desc,
            tone=brain_result.policy.tone,
            intervention=brain_result.policy.intervention_level,
        )

        # 3. 対話履歴を含めてLLM呼び出し
        user_prompt = ""
        if self._conversation:
            recent = self._conversation[-6:]
            user_prompt += "【直前の対話】\n"
            for msg in recent:
                role = "ユーザー" if msg["role"] == "user" else "AI"
                user_prompt += f"  {role}: {msg['content'][:100]}\n"
            user_prompt += "\n"
        user_prompt += f"【ユーザー】\n{user_input}"

        try:
            response = self._provider.generate(system_prompt, user_prompt)
            llm_text = response.raw_text
            model = self._provider.name
        except Exception:
            # LLM失敗時のフォールバック
            llm_text = _fallback_response(brain_result)
            model = "fallback"

        # 4. 安全チェック
        safety = check_anthropomorphic_claims(llm_text)
        if not safety.passed:
            llm_text = "[安全フィルタ: 擬人化表現を検出。応答を抑制しました]"

        # 5. 対話履歴更新
        self._conversation.append({"role": "user", "content": user_input})
        self._conversation.append({"role": "assistant", "content": llm_text})

        return EmotionLLMResponse(
            user_input=user_input,
            brain_result=brain_result,
            llm_response=llm_text,
            model_used=model,
            safety_passed=safety.passed,
        )

    def sleep(self, n_cycles: int = 1) -> list[dict]:
        """睡眠リプレイ。"""
        return self._brain.sleep(n_cycles)

    def set_extinction_mode(self, enabled: bool) -> None:
        self._brain.set_extinction_mode(enabled)

    def reset(self) -> None:
        self._brain.reset()
        self._conversation.clear()


def _fallback_response(result: IntegratedResult) -> str:
    """LLM不可時の定型応答。"""
    tone = result.policy.tone
    if tone == "urgent":
        return "状況を把握しています。落ち着いて対処しましょう。"
    if tone == "warm":
        return "ありがとうございます。良い知らせですね。"
    if tone == "cautious":
        return "慎重に考えてみましょう。"
    return "承知しました。"


class EmotionLLMBridgeV2:
    """IntegratedBrainV2 + LLM の統合。10情動が反映された応答生成。"""

    def __init__(self, provider: LLMProvider | None = None):
        if not _HAS_V2:
            raise ImportError("IntegratedBrainV2 not available")
        self._brain = IntegratedBrainV2()
        self._provider = provider or get_best_provider()
        self._conversation: list[dict[str, str]] = []

    def _describe_v2_state(self, result: IntegratedResultV2) -> str:
        """V2の10情動状態を人間が読める説明に変換。"""
        emotions = result.emotion_state.get("emotions", {})
        parts = []

        # 支配的情動
        dominant = result.emotion_state.get("dominant_emotion", "none")
        if dominant != "none":
            parts.append(f"支配的情動: {dominant}")

        # 活性度が高い情動をリスト
        active = [(k, v) for k, v in emotions.items() if v > 0.1]
        active.sort(key=lambda x: x[1], reverse=True)
        if active:
            emo_str = ", ".join(f"{k}={v:.2f}" for k, v in active[:3])
            parts.append(f"活性: {emo_str}")

        # valence/arousal
        readout = result.readout
        if readout.valence > 0.3:
            parts.append("ポジティブ")
        elif readout.valence < -0.3:
            parts.append("ネガティブ")
        if readout.arousal > 0.6:
            parts.append("高覚醒")

        return "、".join(parts) if parts else "通常の安定状態"

    def chat(self, user_input: str) -> dict:
        """テキスト入力→V2脳処理→LLM応答生成。"""
        brain_result = self._brain.process(user_input)

        if brain_result.blocked:
            return {
                "user_input": user_input,
                "llm_response": "[安全フィルタによりブロック]",
                "emotion_state": {},
                "model_used": "blocked",
            }

        state_desc = self._describe_v2_state(brain_result)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            state_description=state_desc,
            tone=brain_result.policy.tone,
            intervention=brain_result.policy.intervention_level,
        )

        user_prompt = ""
        if self._conversation:
            recent = self._conversation[-6:]
            user_prompt += "【直前の対話】\n"
            for msg in recent:
                role = "ユーザー" if msg["role"] == "user" else "AI"
                user_prompt += f"  {role}: {msg['content'][:100]}\n"
            user_prompt += "\n"
        user_prompt += f"【ユーザー】\n{user_input}"

        try:
            response = self._provider.generate(system_prompt, user_prompt)
            llm_text = response.raw_text
            model = self._provider.name
        except Exception:
            llm_text = _fallback_response_v2(brain_result)
            model = "fallback"

        safety = check_anthropomorphic_claims(llm_text)
        if not safety.passed:
            llm_text = "[安全フィルタ: 擬人化表現を検出]"

        self._conversation.append({"role": "user", "content": user_input})
        self._conversation.append({"role": "assistant", "content": llm_text})

        return {
            "user_input": user_input,
            "llm_response": llm_text,
            "emotion_state": brain_result.emotion_state,
            "readout": {
                "valence": brain_result.readout.valence,
                "arousal": brain_result.readout.arousal,
                "threat_load": brain_result.readout.threat_load,
            },
            "neuromodulation": brain_result.neuromodulation,
            "spiking_neurons": brain_result.spiking_neurons,
            "model_used": model,
        }

    def sleep(self, n_cycles: int = 1) -> list[dict]:
        return self._brain.sleep(n_cycles)

    def reset(self) -> None:
        self._brain.reset()
        self._conversation.clear()


def _fallback_response_v2(result: IntegratedResultV2) -> str:
    """V2 LLM不可時の定型応答。全10情動対応。"""
    dominant = result.emotion_state.get("dominant_emotion", "none")
    responses = {
        "FEAR": "状況を把握しています。落ち着いて対処しましょう。",
        "RAGE": "強い感情が生じている状態です。少し時間を置きましょう。",
        "SEEKING": "良い方向に進んでいますね。",
        "SADNESS": "大変な状況ですね。一緒に考えましょう。",
        "DISGUST": "不快な状況ですね。距離を置くのも一つの方法です。",
        "CARE": "温かい気持ちですね。",
        "PANIC_GRIEF": "つらい気持ちは自然なことです。",
        "PLAY": "楽しい時間ですね。",
        "LUST": "承知しました。",
        "SURPRISE": "予想外の展開ですね。状況を整理しましょう。",
    }
    return responses.get(dominant, "承知しました。")
