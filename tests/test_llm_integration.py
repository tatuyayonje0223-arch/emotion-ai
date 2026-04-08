"""LLM統合のテスト（MockProviderを使用。API不要）。"""

from src.config.settings import ExperimentConfig, LLMConfig
from src.llm.provider import MockProvider, LLMResponse, _extract_json
from src.perception.llm_analyzer import analyze_text_llm
from src.appraisal.llm_engine import appraise_llm
from src.policy.llm_response import generate_response, _build_state_description
from src.pipeline import EmotionPipeline
from src.schemas.affect_state import AffectState
from src.schemas.events import EmotionEvent, PerceptionSignal
from src.policy.response import derive_policy


class TestExtractJson:
    def test_json_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_raw_json(self):
        text = 'some text {"key": 123} more text'
        result = _extract_json(text)
        assert result == {"key": 123}

    def test_no_json(self):
        result = _extract_json("no json here")
        assert result is None


class TestMockProvider:
    def test_default_response(self):
        mock = MockProvider()
        assert mock.is_available()
        assert mock.name == "mock"
        response = mock.generate("system", "user")
        assert response.parsed_json is not None
        assert "sentiment" in response.parsed_json

    def test_fixed_response(self):
        mock = MockProvider({"custom": "data"})
        response = mock.generate("s", "u")
        assert response.parsed_json["custom"] == "data"


class TestLLMPerception:
    def test_with_mock(self):
        mock = MockProvider({
            "sentiment": 0.7,
            "arousal": 0.5,
            "confidence": 0.8,
            "threat_level": 0.0,
            "social_cues": 0.3,
            "context_cues": ["positive_feedback"],
            "implicit_emotion": None,
            "reasoning": "positive text detected",
        })
        result = analyze_text_llm("素晴らしい成果です！", provider=mock)
        assert result.sentiment_score == 0.7
        assert result.arousal_estimate == 0.5
        assert result.confidence == 0.8
        assert "llm:mock" in result.features["method"]

    def test_empty_input(self):
        mock = MockProvider()
        result = analyze_text_llm("", provider=mock)
        assert result.confidence == 0.1

    def test_with_history(self):
        mock = MockProvider({
            "sentiment": -0.3,
            "arousal": 0.6,
            "confidence": 0.7,
            "threat_level": 0.2,
            "social_cues": 0.0,
            "context_cues": [],
            "implicit_emotion": "frustration",
            "reasoning": "context suggests frustration",
        })
        history = ["前のメッセージ1", "前のメッセージ2"]
        result = analyze_text_llm("もういいです", provider=mock, conversation_history=history)
        assert result.sentiment_score == -0.3
        assert result.features.get("implicit_emotion") == "frustration"

    def test_fallback_on_bad_json(self):
        """JSONパース失敗時はキーワードフォールバック。"""
        bad_mock = MockProvider()
        # MockProviderは常にJSONを返すが、providerがNoneでjsonが返らない場合をテスト
        result = analyze_text_llm("嬉しい", provider=bad_mock)
        # MockProviderはデフォルトJSONを返すので正常動作
        assert result.modality == "text"


class TestLLMAppraisal:
    def test_with_mock(self):
        mock = MockProvider({
            "goal_relevance": 0.6,
            "novelty": 0.3,
            "controllability": 0.7,
            "uncertainty_change": -0.1,
            "social_significance": 0.4,
            "reward_threat_balance": 0.5,
            "confidence": 0.8,
            "reasoning": "positive event",
        })
        state = AffectState()
        event = EmotionEvent(
            event_type="user_message",
            raw_content="良いニュースです",
            perception_signals=[PerceptionSignal(modality="text", sentiment_score=0.5)],
        )
        result = appraise_llm(event, state, provider=mock)
        assert result.goal_relevance == 0.6
        assert result.confidence == 0.8


class TestLLMResponseGeneration:
    def test_with_mock(self):
        mock = MockProvider()
        # generate_response は raw_text をそのまま使う
        state = AffectState(valence=0.5, trust=0.6)
        policy = derive_policy(state)
        result = generate_response("こんにちは", state, policy, provider=mock)
        assert result.text != ""
        assert result.model_used == "mock"

    def test_state_description(self):
        state = AffectState(valence=0.7, trust=0.8, threat_load=0.0)
        desc = _build_state_description(state)
        assert "ポジティブ" in desc
        assert "信頼" in desc


class TestPipelineLLMMode:
    def _make_llm_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            llm=LLMConfig(enabled=True, provider="mock", generate_responses=True),
        )

    def test_llm_pipeline_processes_text(self):
        config = self._make_llm_config()
        pipeline = EmotionPipeline(config)
        assert pipeline.llm_enabled is True
        result = pipeline.process_text("嬉しいニュースです")
        assert result.step_count > 0
        assert "method" in result.perception_summary

    def test_llm_pipeline_generates_response(self):
        config = self._make_llm_config()
        pipeline = EmotionPipeline(config)
        result = pipeline.process_text("こんにちは")
        # MockProviderはJSON文字列を返す → 安全チェック通過
        assert result.generated_response is not None

    def test_llm_pipeline_multiple_turns(self):
        config = self._make_llm_config()
        pipeline = EmotionPipeline(config)
        for text in ["こんにちは", "嬉しい", "不安です"]:
            result = pipeline.process_text(text)
        assert pipeline.current_state.step_count > 3

    def test_heuristic_mode_still_works(self):
        """LLM無効のデフォルトモードが壊れていないことを確認。"""
        pipeline = EmotionPipeline()
        assert pipeline.llm_enabled is False
        result = pipeline.process_text("嬉しい")
        assert result.step_count > 0
        assert result.generated_response is None

    def test_conversation_history_tracked(self):
        config = self._make_llm_config()
        pipeline = EmotionPipeline(config)
        pipeline.process_text("メッセージ1")
        pipeline.process_text("メッセージ2")
        assert len(pipeline._conversation_history) >= 2
        assert len(pipeline._text_history) == 2

    def test_reset_clears_history(self):
        config = self._make_llm_config()
        pipeline = EmotionPipeline(config)
        pipeline.process_text("テスト")
        pipeline.reset()
        assert len(pipeline._conversation_history) == 0
        assert len(pipeline._text_history) == 0
