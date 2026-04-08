"""情動状態ストア。状態の保持・更新・クランプ・履歴管理を担当する。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from src.schemas.affect_state import AffectDelta, AffectState, StateTransition


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_delta(state: AffectState, delta: AffectDelta, event_id: str, trigger: str, reason: str) -> tuple[AffectState, StateTransition]:
    """現在の状態に差分を適用し、新しい状態と遷移ログを返す。純関数。"""
    previous_id = state.state_id
    new_state = deepcopy(state)
    new_state.state_id = uuid4()
    new_state.timestamp = datetime.now(timezone.utc)
    new_state.step_count = state.step_count + 1
    new_state.last_event_id = event_id

    if delta.valence is not None:
        new_state.valence = _clamp(state.valence + delta.valence, -1.0, 1.0)
    if delta.arousal is not None:
        new_state.arousal = _clamp(state.arousal + delta.arousal, 0.0, 1.0)
    if delta.motivational_salience is not None:
        new_state.motivational_salience = _clamp(
            state.motivational_salience + delta.motivational_salience, 0.0, 1.0
        )
    if delta.perceived_control is not None:
        new_state.perceived_control = _clamp(
            state.perceived_control + delta.perceived_control, 0.0, 1.0
        )
    if delta.uncertainty is not None:
        new_state.uncertainty = _clamp(state.uncertainty + delta.uncertainty, 0.0, 1.0)
    if delta.trust is not None:
        new_state.trust = _clamp(state.trust + delta.trust, 0.0, 1.0)
    if delta.threat_load is not None:
        new_state.threat_load = _clamp(state.threat_load + delta.threat_load, 0.0, 1.0)
    if delta.fatigue is not None:
        new_state.fatigue = _clamp(state.fatigue + delta.fatigue, 0.0, 1.0)
    if delta.regulation_mode is not None:
        new_state.regulation_mode = delta.regulation_mode

    transition = StateTransition(
        previous_state_id=previous_id,
        new_state_id=new_state.state_id,
        event_id=event_id,
        trigger=trigger,
        delta_applied=delta,
        reason=reason,
    )

    return new_state, transition


class AffectStateStore:
    """情動状態の保持と履歴管理。"""

    def __init__(self, initial_state: AffectState | None = None):
        self._current = initial_state or AffectState()
        self._history: list[StateTransition] = []

    @property
    def current(self) -> AffectState:
        return self._current

    @property
    def history(self) -> list[StateTransition]:
        return list(self._history)

    def update(self, delta: AffectDelta, event_id: str, trigger: str, reason: str) -> StateTransition:
        """状態を更新し、遷移ログを返す。"""
        new_state, transition = apply_delta(self._current, delta, event_id, trigger, reason)
        self._current = new_state
        self._history.append(transition)
        return transition

    def get_trajectory(self, variable: str, last_n: int = 50) -> list[float]:
        """指定変数の最近N件の値推移を返す。"""
        if variable not in AffectState.variable_names():
            raise ValueError(f"Unknown variable: {variable}")
        # 現在値を含む簡易実装（完全履歴は audit log から復元）
        return [getattr(self._current, variable)]

    def reset(self, state: AffectState | None = None) -> None:
        """状態をリセットする。"""
        self._current = state or AffectState()
        self._history.clear()
