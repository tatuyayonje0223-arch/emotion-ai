"""LLMベースのテキスト分析。文脈・皮肉・暗示を理解した情動信号抽出。"""

from __future__ import annotations

from src.llm.provider import LLMProvider, get_best_provider
from src.perception.text_analyzer import analyze_text as keyword_analyze
from src.schemas.events import PerceptionSignal

_SYSTEM_PROMPT = """\
あなたは感情分析の専門家です。
ユーザーのテキストを分析し、以下のJSON形式で情動信号を返してください。

重要なルール:
- 表面的な言葉だけでなく、文脈・皮肉・暗示・行間を読む
- 確信が低い場合はconfidenceを下げる
- 短いテキストでは過剰な推定をしない
- 文化的文脈（日本語の婉曲表現等）を考慮する

```json
{
  "sentiment": <-1.0〜1.0 快不快>,
  "arousal": <0.0〜1.0 覚醒度>,
  "confidence": <0.0〜1.0 推定の確信度>,
  "threat_level": <0.0〜1.0 脅威の程度>,
  "social_cues": <0.0〜1.0 社会的手がかりの強さ>,
  "context_cues": [<検出した文脈的手がかりのリスト>],
  "implicit_emotion": "<表面に現れていない潜在感情があれば記述、なければnull>",
  "reasoning": "<判断の根拠を1-2文で>"
}
```"""


def analyze_text_llm(
    text: str,
    provider: LLMProvider | None = None,
    conversation_history: list[str] | None = None,
) -> PerceptionSignal:
    """LLMを使ったテキスト情動信号抽出。

    Args:
        text: 分析対象テキスト
        provider: LLMプロバイダー（Noneなら自動選択）
        conversation_history: 直前の対話履歴（文脈理解用）

    Returns:
        PerceptionSignal: 不確実性付きの知覚信号
    """
    if not text.strip():
        return PerceptionSignal(
            modality="text",
            sentiment_score=0.0,
            arousal_estimate=0.0,
            confidence=0.1,
            features={"method": "llm_empty_input"},
        )

    llm = provider or get_best_provider()

    # 対話履歴があれば文脈として含める
    user_prompt = ""
    if conversation_history:
        recent = conversation_history[-5:]  # 直近5件
        user_prompt += "【直前の対話】\n"
        for i, msg in enumerate(recent):
            user_prompt += f"  {i+1}. {msg[:100]}\n"
        user_prompt += "\n"

    user_prompt += f"【分析対象テキスト】\n{text}"

    try:
        response = llm.generate(_SYSTEM_PROMPT, user_prompt)

        if response.parsed_json:
            data = response.parsed_json
            return PerceptionSignal(
                modality="text",
                sentiment_score=_clamp(data.get("sentiment", 0.0), -1.0, 1.0),
                arousal_estimate=_clamp(data.get("arousal", 0.3), 0.0, 1.0),
                confidence=_clamp(data.get("confidence", 0.5), 0.0, 1.0),
                features={
                    "method": f"llm:{llm.name}",
                    "threat_level": data.get("threat_level", 0.0),
                    "social_cues": data.get("social_cues", 0.0),
                    "context_cues": data.get("context_cues", []),
                    "implicit_emotion": data.get("implicit_emotion"),
                    "reasoning": data.get("reasoning", ""),
                    "threat_hits": 1 if data.get("threat_level", 0) > 0.3 else 0,
                },
            )
        else:
            # JSON解析失敗 → キーワードフォールバック
            return _fallback(text, reason="llm_json_parse_failed")

    except Exception as e:
        return _fallback(text, reason=f"llm_error:{type(e).__name__}")


def _fallback(text: str, reason: str) -> PerceptionSignal:
    """LLM失敗時のキーワードフォールバック。"""
    result = keyword_analyze(text)
    result.features["fallback_reason"] = reason
    result.features["method"] = "keyword_fallback"
    # フォールバック時は信頼度を下げる
    result.confidence = min(result.confidence, 0.4)
    return result


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
