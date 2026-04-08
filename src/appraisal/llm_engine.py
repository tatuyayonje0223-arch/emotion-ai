"""LLMベースの評価エンジン。文脈を理解した多次元評価を行う。"""

from __future__ import annotations

import json

from src.llm.provider import LLMProvider, get_best_provider
from src.appraisal.engine import appraise as heuristic_appraise
from src.schemas.affect_state import AffectState
from src.schemas.events import AppraisalResult, EmotionEvent

_SYSTEM_PROMPT = """\
あなたは認知評価（cognitive appraisal）の専門家です。
イベントを受け取り、現在の内部状態を考慮して、以下の評価次元をJSON形式で返してください。

評価次元の定義:
- goal_relevance: イベントが目標を促進(+1)か阻害(-1)か (範囲: -1.0〜1.0)
- novelty: 予想外の度合い (範囲: 0.0〜1.0)
- controllability: 対処可能性 (範囲: 0.0〜1.0)
- uncertainty_change: 不確実性の増減 (範囲: -1.0〜1.0, 正=増加)
- social_significance: 社会的重要度 (範囲: 0.0〜1.0)
- reward_threat_balance: 報酬と脅威のバランス (範囲: -1.0〜1.0, 正=報酬)
- confidence: この評価全体の確信度 (範囲: 0.0〜1.0)

重要なルール:
- 文脈（対話履歴、状態）を考慮する
- 日本語の婉曲表現や暗示に注意する
- 確信が低い場合は中立値に寄せ、confidenceを下げる
- 皮肉や反語を検出する

```json
{
  "goal_relevance": 0.0,
  "novelty": 0.0,
  "controllability": 0.5,
  "uncertainty_change": 0.0,
  "social_significance": 0.0,
  "reward_threat_balance": 0.0,
  "confidence": 0.5,
  "reasoning": ""
}
```"""


def appraise_llm(
    event: EmotionEvent,
    current_state: AffectState,
    memory_context: list[dict] | None = None,
    provider: LLMProvider | None = None,
) -> AppraisalResult:
    """LLMを使ったイベント評価。

    文脈を理解した多次元認知評価を行う。
    失敗時はヒューリスティクスにフォールバック。
    """
    llm = provider or get_best_provider()

    # 現在の内部状態を文脈として提供
    state_context = (
        f"valence={current_state.valence:.2f}, "
        f"arousal={current_state.arousal:.2f}, "
        f"threat={current_state.threat_load:.2f}, "
        f"uncertainty={current_state.uncertainty:.2f}, "
        f"trust={current_state.trust:.2f}, "
        f"control={current_state.perceived_control:.2f}"
    )

    user_prompt = f"【現在の内部状態】\n{state_context}\n\n"

    if memory_context:
        user_prompt += "【関連する過去の記憶】\n"
        for mem in memory_context[:3]:
            user_prompt += f"  - {mem.get('summary', '')[:80]}\n"
        user_prompt += "\n"

    # 知覚信号のサマリー
    if event.perception_signals:
        signals_summary = []
        for s in event.perception_signals:
            signals_summary.append(
                f"{s.modality}: sentiment={s.sentiment_score:.2f}, "
                f"arousal={s.arousal_estimate:.2f}, confidence={s.confidence:.2f}"
            )
        user_prompt += "【知覚信号】\n" + "\n".join(f"  {s}" for s in signals_summary) + "\n\n"

    user_prompt += f"【イベント】\nタイプ: {event.event_type}\n内容: {event.raw_content}"

    try:
        response = llm.generate(_SYSTEM_PROMPT, user_prompt)

        if response.parsed_json:
            data = response.parsed_json
            return AppraisalResult(
                goal_relevance=_clamp(data.get("goal_relevance", 0.0), -1.0, 1.0),
                novelty=_clamp(data.get("novelty", 0.0), 0.0, 1.0),
                controllability=_clamp(data.get("controllability", 0.5), 0.0, 1.0),
                uncertainty_change=_clamp(data.get("uncertainty_change", 0.0), -1.0, 1.0),
                social_significance=_clamp(data.get("social_significance", 0.0), 0.0, 1.0),
                reward_threat_balance=_clamp(data.get("reward_threat_balance", 0.0), -1.0, 1.0),
                confidence=_clamp(data.get("confidence", 0.5), 0.0, 1.0),
            )
        else:
            return heuristic_appraise(event, current_state, memory_context)

    except Exception:
        return heuristic_appraise(event, current_state, memory_context)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
