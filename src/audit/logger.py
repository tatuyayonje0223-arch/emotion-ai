"""JSONL形式の監査ログ。全状態遷移を記録する。"""

from __future__ import annotations

import json
from pathlib import Path

from src.schemas.affect_state import AffectState, StateTransition


class AuditLogger:
    """状態遷移をJSONLファイルに記録する。"""

    def __init__(self, log_path: str | Path = "logs/audit.jsonl"):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []

    def log_transition(self, transition: StateTransition) -> None:
        """遷移をバッファに追加し、ファイルに書き出す。"""
        record = {
            "type": "state_transition",
            "transition_id": str(transition.transition_id),
            "timestamp": transition.timestamp.isoformat(),
            "previous_state_id": str(transition.previous_state_id),
            "new_state_id": str(transition.new_state_id),
            "event_id": transition.event_id,
            "trigger": transition.trigger,
            "reason": transition.reason,
            "delta": transition.delta_applied.model_dump(exclude_none=True),
        }
        self._write(record)

    def log_state_snapshot(self, state: AffectState) -> None:
        """現在の状態スナップショットを記録する。"""
        record = {
            "type": "state_snapshot",
            "state_id": str(state.state_id),
            "timestamp": state.timestamp.isoformat(),
            "step_count": state.step_count,
            "variables": {name: getattr(state, name) for name in state.variable_names()},
            "regulation_mode": state.regulation_mode,
        }
        self._write(record)

    def log_safety_event(self, event_id: str, check_type: str, result: str, details: str) -> None:
        """安全チェックの結果を記録する。"""
        from datetime import datetime, timezone

        record = {
            "type": "safety_check",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": event_id,
            "check_type": check_type,
            "result": result,
            "details": details,
        }
        self._write(record)

    def _write(self, record: dict) -> None:
        self._buffer.append(record)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_buffer(self) -> list[dict]:
        return list(self._buffer)

    def clear_buffer(self) -> None:
        self._buffer.clear()
