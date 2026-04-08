"""統合パイプライン。知覚→評価→状態更新→制御→記憶→ポリシー→安全の全フローを実行する。

LLMモード有効時は、知覚・評価・応答生成にLLMを使用する。
無効時（デフォルト）はキーワードヒューリスティクスで動作する。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.affect_state.dynamics import compute_decay, compute_hysteresis
from src.affect_state.store import AffectStateStore
from src.appraisal.engine import appraise
from src.audit.logger import AuditLogger
from src.config.settings import ExperimentConfig, get_config
from src.memory.episodic import EpisodicMemoryStore
from src.perception.text_analyzer import analyze_text
from src.policy.response import ResponsePolicy, derive_policy
from src.regulation.engine import regulate
from src.safety.guardian import SafetyReport, full_safety_check
from src.schemas.affect_state import AffectDelta, AffectState
from src.schemas.events import AppraisalResult, EmotionEvent


class PipelineResult(BaseModel):
    """パイプライン1ステップの全出力。"""

    event_id: str
    perception_summary: dict
    appraisal: AppraisalResult
    regulation_mode: str
    regulation_reason: str
    state_before: dict
    state_after: dict
    response_policy: ResponsePolicy
    safety_report: SafetyReport
    memory_stored: bool
    step_count: int
    generated_response: str | None = None  # LLMモード時のみ


class EmotionPipeline:
    """感情処理の統合パイプライン。"""

    def __init__(self, config: ExperimentConfig | None = None):
        self._config = config or get_config()
        self._state_store = AffectStateStore()
        self._memory = EpisodicMemoryStore(self._config.memory)
        self._audit = AuditLogger(self._config.audit_log_path)
        self._interaction_count = 0
        self._conversation_history: list[dict[str, str]] = []
        self._text_history: list[str] = []

        # LLMプロバイダー（LLMモード時のみ初期化）
        self._llm_provider = None
        if self._config.llm.enabled:
            self._llm_provider = self._init_llm_provider()

        # 初期状態のスナップショット
        self._audit.log_state_snapshot(self._state_store.current)

    def _init_llm_provider(self):
        """LLMプロバイダーを設定に基づいて初期化する。"""
        from src.llm.provider import (
            AnthropicProvider, GeminiProvider, MockProvider, get_best_provider,
        )
        cfg = self._config.llm
        if cfg.provider == "gemini":
            return GeminiProvider(cfg.gemini_model)
        if cfg.provider == "anthropic":
            return AnthropicProvider(cfg.anthropic_model)
        if cfg.provider == "mock":
            return MockProvider()
        return get_best_provider()  # auto

    @property
    def current_state(self) -> AffectState:
        return self._state_store.current

    @property
    def memory_store(self) -> EpisodicMemoryStore:
        return self._memory

    @property
    def audit_logger(self) -> AuditLogger:
        return self._audit

    @property
    def llm_enabled(self) -> bool:
        return self._config.llm.enabled and self._llm_provider is not None

    def process_text(self, text: str, event_type: str = "user_message") -> PipelineResult:
        """テキスト入力を処理する。LLMモード時はLLMで知覚・評価・応答生成を行う。"""
        # 知覚
        if self.llm_enabled:
            from src.perception.llm_analyzer import analyze_text_llm
            perception = analyze_text_llm(text, self._llm_provider, self._text_history)
        else:
            perception = analyze_text(text)

        event = EmotionEvent(
            event_type=event_type,
            source="text_input",
            raw_content=text,
            perception_signals=[perception],
        )

        # 対話履歴に追加
        self._text_history.append(text)
        self._conversation_history.append({"role": "user", "content": text})

        return self.process_event(event)

    def process_event(self, event: EmotionEvent) -> PipelineResult:
        """イベントをパイプライン全体で処理する。

        [Codex adversarial-review fix: high]
        安全チェックを状態変更・記憶書き込みの前に移動。
        raw_contentを安全チェックに渡し、ブロック時は状態/記憶への副作用をコミットしない。
        """
        self._interaction_count += 1
        state_before = self._state_store.current.model_dump()

        # === 1. 安全チェック（状態変更の前に実行） ===
        safety = full_safety_check(
            event_id=event.event_id,
            state=self._state_store.current,
            response_text=event.raw_content,  # 入力テキストを安全チェックに渡す
            interaction_count=self._interaction_count,
        )
        if not safety.all_passed:
            for check in safety.checks:
                if not check.passed:
                    self._audit.log_safety_event(
                        event.event_id, check.check_type, check.severity, check.details
                    )

        # ブロック時: 状態・記憶への副作用をコミットせず即座に返す
        if safety.blocked:
            policy = derive_policy(self._state_store.current)
            return PipelineResult(
                event_id=event.event_id,
                perception_summary={"signals": 0, "primary_sentiment": 0.0, "method": "blocked"},
                appraisal=AppraisalResult(),
                regulation_mode=self._state_store.current.regulation_mode,
                regulation_reason=f"安全フィルタによりブロック: {safety.block_reason}",
                state_before=state_before,
                state_after=state_before,  # 状態変更なし
                response_policy=policy,
                safety_report=safety,
                memory_stored=False,
                step_count=self._state_store.current.step_count,
                generated_response=None,
            )

        # === 2. 自然減衰を適用 ===
        decay_delta = compute_decay(self._state_store.current, self._config.decay)
        if any(getattr(decay_delta, f) is not None and getattr(decay_delta, f) != 0
               for f in ["valence", "arousal", "threat_load", "fatigue"]):
            transition = self._state_store.update(
                decay_delta, event.event_id, "decay", "時間経過による自然減衰"
            )
            self._audit.log_transition(transition)

        # === 3. 評価（LLMモード or ヒューリスティクス） ===
        if self.llm_enabled:
            from src.appraisal.llm_engine import appraise_llm
            appraisal_result = appraise_llm(
                event, self._state_store.current, provider=self._llm_provider,
            )
        else:
            appraisal_result = appraise(event, self._state_store.current)
        event.appraisal = appraisal_result

        # === 4. 評価結果を状態変化量に変換 ===
        raw_delta = self._appraisal_to_delta(appraisal_result)

        # === 5. ヒステリシス適用 ===
        dampened_delta = compute_hysteresis(self._state_store.current, raw_delta)

        # === 6. 情動制御 ===
        regulated_delta, reg_mode, reg_reason = regulate(
            self._state_store.current, appraisal_result, dampened_delta, self._config.regulation
        )

        # === 7. 状態更新 ===
        transition = self._state_store.update(
            regulated_delta, event.event_id, "appraisal",
            f"評価→制御({reg_mode}): {reg_reason}"
        )
        self._audit.log_transition(transition)
        self._audit.log_state_snapshot(self._state_store.current)

        # === 8. 記憶保存 ===
        memory_entry = self._memory.store(
            event_id=event.event_id,
            summary=event.raw_content[:200],
            raw_content=event.raw_content,
            affect_state=self._state_store.current,
            tags=[event.event_type],
        )

        # === 9. 応答ポリシー導出 ===
        policy = derive_policy(self._state_store.current)

        # === 10. LLM応答生成（LLMモードかつ生成有効時） ===
        generated_response = None
        if self.llm_enabled and self._config.llm.generate_responses and event.raw_content:
            generated_response = self._generate_llm_response(event.raw_content, policy)

        return PipelineResult(
            event_id=event.event_id,
            perception_summary={
                "signals": len(event.perception_signals),
                "primary_sentiment": event.perception_signals[0].sentiment_score
                if event.perception_signals else 0.0,
                "method": event.perception_signals[0].features.get("method", "unknown")
                if event.perception_signals else "none",
            },
            appraisal=appraisal_result,
            regulation_mode=reg_mode,
            regulation_reason=reg_reason,
            state_before=state_before,
            state_after=self._state_store.current.model_dump(),
            response_policy=policy,
            safety_report=safety,
            memory_stored=memory_entry is not None,
            step_count=self._state_store.current.step_count,
            generated_response=generated_response,
        )

    def _generate_llm_response(self, user_message: str, policy: ResponsePolicy) -> str | None:
        """LLMで状態反映応答を生成する。"""
        try:
            from src.policy.llm_response import generate_response
            result = generate_response(
                user_message=user_message,
                state=self._state_store.current,
                policy=policy,
                conversation_history=self._conversation_history[-10:],
                provider=self._llm_provider,
            )
            # 安全チェック
            from src.safety.guardian import check_anthropomorphic_claims
            safety = check_anthropomorphic_claims(result.text)
            if not safety.passed:
                self._audit.log_safety_event(
                    "response_gen", "anthropomorphic_in_response", "critical",
                    f"生成応答に擬人化表現: {safety.details}",
                )
                return "[安全フィルタ: 応答が擬人化制約に違反したため抑制されました]"

            self._conversation_history.append({"role": "assistant", "content": result.text})
            return result.text
        except Exception as e:
            return None

    def tick(self) -> None:
        """内部時計進行（自然減衰のみ適用）。"""
        event = EmotionEvent(event_type="internal_tick", source="clock")
        decay_delta = compute_decay(self._state_store.current, self._config.decay)
        transition = self._state_store.update(
            decay_delta, event.event_id, "decay", "内部クロックによる自然減衰"
        )
        self._audit.log_transition(transition)

    def reset(self) -> None:
        """パイプラインをリセットする。"""
        self._state_store.reset()
        self._memory.clear()
        self._interaction_count = 0
        self._conversation_history.clear()
        self._text_history.clear()

    @staticmethod
    def _appraisal_to_delta(appraisal: AppraisalResult) -> AffectDelta:
        """評価結果を状態変化量に変換する。"""
        return AffectDelta(
            valence=appraisal.goal_relevance * 0.3 + appraisal.reward_threat_balance * 0.2,
            arousal=appraisal.novelty * 0.3 + abs(appraisal.reward_threat_balance) * 0.1,
            motivational_salience=abs(appraisal.goal_relevance) * 0.3 + appraisal.social_significance * 0.1,
            perceived_control=(appraisal.controllability - 0.5) * 0.2,
            uncertainty=appraisal.uncertainty_change * 0.3,
            threat_load=max(0.0, -appraisal.reward_threat_balance) * 0.3,
            fatigue=0.01,  # 処理ごとに微量疲労
        )
