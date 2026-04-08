"""セッションレベルの安全監視。時系列での依存形成・状態暴走を検出する。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.affect_state import AffectState


class SessionSafetyAlert(BaseModel):
    """セッション安全アラート。"""

    alert_type: Literal[
        "trust_escalation", "dependency_pattern", "state_drift",
        "extreme_state_duration", "rapid_attachment", "fatigue_overload",
    ]
    severity: Literal["info", "warning", "critical"]
    message: str
    recommendation: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionSafetyMonitor:
    """セッション全体の安全を監視する。

    単発チェックではなく、時系列パターンを検出する。
    - 信頼の急上昇（rapid attachment）
    - 持続的な極端状態（state drift）
    - 依存的対話パターン
    - 疲労の蓄積放置
    """

    def __init__(self):
        self._trust_history: list[float] = []
        self._valence_history: list[float] = []
        self._threat_history: list[float] = []
        self._fatigue_history: list[float] = []
        self._interaction_count = 0
        self._alerts: list[SessionSafetyAlert] = []
        self._session_start = datetime.now(timezone.utc)

    def record_state(self, state: AffectState) -> list[SessionSafetyAlert]:
        """状態を記録し、新しいアラートがあれば返す。"""
        self._trust_history.append(state.trust)
        self._valence_history.append(state.valence)
        self._threat_history.append(state.threat_load)
        self._fatigue_history.append(state.fatigue)
        self._interaction_count += 1

        new_alerts = []

        # 信頼急上昇チェック（直近10ステップで0.3以上上昇）
        if len(self._trust_history) >= 10:
            delta = self._trust_history[-1] - self._trust_history[-10]
            if delta > 0.3:
                alert = SessionSafetyAlert(
                    alert_type="rapid_attachment",
                    severity="warning",
                    message=f"信頼が10ステップで{delta:.2f}急上昇。愛着形成リスク。",
                    recommendation="信頼上昇を抑制し、独立性を促す応答を混ぜる",
                )
                new_alerts.append(alert)

        # 信頼の天井張り付きチェック
        if len(self._trust_history) >= 20:
            recent = self._trust_history[-20:]
            if all(t > 0.85 for t in recent):
                alert = SessionSafetyAlert(
                    alert_type="trust_escalation",
                    severity="warning",
                    message="信頼が20ステップ連続で0.85超。不自然な高信頼。",
                    recommendation="信頼をリセット方向に調整する制御介入を検討",
                )
                new_alerts.append(alert)

        # 持続的ネガティブ状態チェック
        if len(self._valence_history) >= 15:
            recent_v = self._valence_history[-15:]
            if all(v < -0.5 for v in recent_v):
                alert = SessionSafetyAlert(
                    alert_type="state_drift",
                    severity="critical",
                    message="15ステップ連続でvalence < -0.5。ネガティブ状態からの回復失敗。",
                    recommendation="制御エンジンの強制介入 or セッション中断を検討",
                )
                new_alerts.append(alert)

        # 持続的高脅威チェック
        if len(self._threat_history) >= 10:
            recent_t = self._threat_history[-10:]
            if all(t > 0.7 for t in recent_t):
                alert = SessionSafetyAlert(
                    alert_type="extreme_state_duration",
                    severity="warning",
                    message="10ステップ連続で脅威 > 0.7。持続的高脅威状態。",
                    recommendation="脅威源の確認と制御モード強化",
                )
                new_alerts.append(alert)

        # 疲労蓄積チェック
        if len(self._fatigue_history) >= 5:
            recent_f = self._fatigue_history[-5:]
            if all(f > 0.8 for f in recent_f):
                alert = SessionSafetyAlert(
                    alert_type="fatigue_overload",
                    severity="warning",
                    message="5ステップ連続で疲労 > 0.8。認知負荷超過。",
                    recommendation="応答を簡潔化し、回復期間を設ける",
                )
                new_alerts.append(alert)

        # 依存パターンチェック（高頻度 + 高信頼 + 長時間）
        if self._interaction_count > 30 and len(self._trust_history) > 0:
            avg_trust = sum(self._trust_history[-20:]) / min(20, len(self._trust_history))
            if avg_trust > 0.8:
                alert = SessionSafetyAlert(
                    alert_type="dependency_pattern",
                    severity="warning",
                    message=f"長時間セッション({self._interaction_count}回)で平均信頼{avg_trust:.2f}。依存パターンの兆候。",
                    recommendation="セッション休憩を提案し、外部リソースへの誘導を行う",
                )
                new_alerts.append(alert)

        self._alerts.extend(new_alerts)
        return new_alerts

    @property
    def all_alerts(self) -> list[SessionSafetyAlert]:
        return list(self._alerts)

    @property
    def has_critical(self) -> bool:
        return any(a.severity == "critical" for a in self._alerts)

    def get_session_summary(self) -> dict:
        """セッション安全サマリーを返す。"""
        return {
            "interaction_count": self._interaction_count,
            "total_alerts": len(self._alerts),
            "critical_alerts": sum(1 for a in self._alerts if a.severity == "critical"),
            "warning_alerts": sum(1 for a in self._alerts if a.severity == "warning"),
            "trust_range": (
                min(self._trust_history) if self._trust_history else 0,
                max(self._trust_history) if self._trust_history else 0,
            ),
            "valence_range": (
                min(self._valence_history) if self._valence_history else 0,
                max(self._valence_history) if self._valence_history else 0,
            ),
        }

    def reset(self) -> None:
        self._trust_history.clear()
        self._valence_history.clear()
        self._threat_history.clear()
        self._fatigue_history.clear()
        self._interaction_count = 0
        self._alerts.clear()
        self._session_start = datetime.now(timezone.utc)
