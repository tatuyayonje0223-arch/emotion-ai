"""セッション管理。複数の対話セッションを独立して管理する。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from src.config.settings import ExperimentConfig, get_config
from src.pipeline import EmotionPipeline, PipelineResult
from src.safety.session_monitor import SessionSafetyMonitor
from src.self_model.model import SelfModel, StableTraits
from src.visualization.state_export import StateTrajectoryRecorder


class SessionInfo(BaseModel):
    """セッション情報。"""

    session_id: str
    created_at: datetime
    interaction_count: int = 0
    llm_enabled: bool = False
    has_critical_alerts: bool = False


class ConversationSession:
    """1つの対話セッション。パイプライン+安全監視+自己モデル+軌跡記録を統合。"""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        traits: StableTraits | None = None,
        session_id: str | None = None,
    ):
        self.session_id = session_id or str(uuid4())
        self.created_at = datetime.now(timezone.utc)

        self._config = config or get_config()
        self.pipeline = EmotionPipeline(self._config)
        self.safety_monitor = SessionSafetyMonitor()
        self.self_model = SelfModel(traits)
        self.trajectory = StateTrajectoryRecorder()

        self._interaction_count = 0

    @property
    def info(self) -> SessionInfo:
        return SessionInfo(
            session_id=self.session_id,
            created_at=self.created_at,
            interaction_count=self._interaction_count,
            llm_enabled=self.pipeline.llm_enabled,
            has_critical_alerts=self.safety_monitor.has_critical,
        )

    def process(self, text: str) -> dict:
        """テキストを処理し、統合結果を返す。"""
        self._interaction_count += 1

        # パイプライン処理
        result = self.pipeline.process_text(text)

        # 状態記録
        self.trajectory.record(self.pipeline.current_state, text[:50])

        # セッション安全監視
        session_alerts = self.safety_monitor.record_state(self.pipeline.current_state)

        # 自己モデル更新
        self.self_model.update_situation(interaction_count=self._interaction_count)
        assessment = self.self_model.assess(self.pipeline.current_state)

        # 統合結果
        return {
            "session_id": self.session_id,
            "event_id": result.event_id,
            "step": result.step_count,
            "response": result.generated_response,
            "state": {
                name: getattr(self.pipeline.current_state, name)
                for name in self.pipeline.current_state.variable_names()
            },
            "regulation": result.regulation_mode,
            "policy": {
                "tone": result.response_policy.tone,
                "verbosity": result.response_policy.verbosity,
                "intervention": result.response_policy.intervention_level,
            },
            "safety": {
                "pipeline_passed": result.safety_report.all_passed,
                "session_alerts": len(session_alerts),
                "critical": self.safety_monitor.has_critical,
                "new_alerts": [
                    {"type": a.alert_type, "severity": a.severity, "message": a.message}
                    for a in session_alerts
                ],
            },
            "self_assessment": {
                "summary": assessment.current_state_summary,
                "confidence": assessment.confidence_in_assessment,
                "goals": assessment.active_goals,
            },
            "memory_stored": result.memory_stored,
        }

    def get_trajectory_chart(self, variable: str = "valence") -> str:
        """状態変数のASCIIチャートを返す。"""
        return self.trajectory.get_ascii_chart(variable)

    def get_stats(self) -> dict:
        """セッション統計を返す。"""
        return {
            **self.trajectory.get_summary_stats(),
            **self.safety_monitor.get_session_summary(),
            "self_model_assessments": len(self.self_model._assessment_history),
        }

    def export_trajectory(self, path: str, format: str = "csv") -> str:
        """軌跡をエクスポートする。"""
        if format == "csv":
            return str(self.trajectory.export_csv(path))
        return str(self.trajectory.export_jsonl(path))

    def reset(self) -> None:
        self.pipeline.reset()
        self.safety_monitor.reset()
        self.trajectory.clear()
        self._interaction_count = 0


class SessionManager:
    """複数セッションの管理。"""

    def __init__(self):
        self._sessions: dict[str, ConversationSession] = {}

    def create_session(
        self,
        config: ExperimentConfig | None = None,
        traits: StableTraits | None = None,
    ) -> ConversationSession:
        session = ConversationSession(config, traits)
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> ConversationSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[SessionInfo]:
        return [s.info for s in self._sessions.values()]

    def remove_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
