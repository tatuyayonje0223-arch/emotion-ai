"""状態軌跡のエクスポートと可視化データ生成。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timezone

from src.schemas.affect_state import AffectState, StateTransition


class StateTrajectoryRecorder:
    """状態変数の時系列をCSV/JSONLでエクスポートする。"""

    def __init__(self):
        self._snapshots: list[dict] = []

    def record(self, state: AffectState, event_summary: str = "") -> None:
        """状態スナップショットを記録する。"""
        self._snapshots.append({
            "step": state.step_count,
            "timestamp": state.timestamp.isoformat(),
            "valence": state.valence,
            "arousal": state.arousal,
            "motivational_salience": state.motivational_salience,
            "perceived_control": state.perceived_control,
            "uncertainty": state.uncertainty,
            "trust": state.trust,
            "threat_load": state.threat_load,
            "fatigue": state.fatigue,
            "regulation_mode": state.regulation_mode,
            "event_summary": event_summary[:100],
        })

    @property
    def size(self) -> int:
        return len(self._snapshots)

    def export_csv(self, path: str | Path) -> Path:
        """CSV形式でエクスポートする。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self._snapshots:
            return path

        fieldnames = list(self._snapshots[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._snapshots)

        return path

    def export_jsonl(self, path: str | Path) -> Path:
        """JSONL形式でエクスポートする。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for snap in self._snapshots:
                f.write(json.dumps(snap, ensure_ascii=False) + "\n")

        return path

    def get_variable_series(self, variable: str) -> list[float]:
        """指定変数の時系列を返す。"""
        return [s[variable] for s in self._snapshots if variable in s]

    def get_summary_stats(self) -> dict:
        """全変数のサマリー統計を返す。"""
        if not self._snapshots:
            return {}

        variables = AffectState.variable_names()
        stats = {}
        for var in variables:
            values = [s[var] for s in self._snapshots]
            if values:
                stats[var] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "final": values[-1],
                    "range": max(values) - min(values),
                }
        stats["total_steps"] = len(self._snapshots)
        return stats

    def get_ascii_chart(self, variable: str, width: int = 60, height: int = 15) -> str:
        """ターミナル用のASCIIチャートを生成する。"""
        values = self.get_variable_series(variable)
        if not values:
            return f"No data for {variable}"

        # 範囲を決定
        if variable == "valence":
            v_min, v_max = -1.0, 1.0
        else:
            v_min, v_max = 0.0, 1.0

        # データをリサンプル
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values

        lines = []
        lines.append(f"  {variable} (steps: {len(values)})")
        lines.append(f"  {v_max:+.1f} ┤")

        for row in range(height - 2, -1, -1):
            threshold = v_min + (v_max - v_min) * row / (height - 1)
            chars = []
            for v in sampled:
                if v >= threshold:
                    chars.append("█")
                else:
                    chars.append(" ")
            label = f"{threshold:+.1f}" if row == height // 2 else "     "
            lines.append(f"  {label} │{''.join(chars)}")

        lines.append(f"  {v_min:+.1f} ┤{'─' * len(sampled)}")
        lines.append(f"       0{' ' * (len(sampled) - 6)}step {len(values)}")

        return "\n".join(lines)

    def clear(self) -> None:
        self._snapshots.clear()
