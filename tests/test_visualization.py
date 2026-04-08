"""状態可視化のテスト。"""

import tempfile
from pathlib import Path

from src.schemas.affect_state import AffectState
from src.visualization.state_export import StateTrajectoryRecorder


class TestStateTrajectoryRecorder:
    def test_record_and_size(self):
        recorder = StateTrajectoryRecorder()
        recorder.record(AffectState(valence=0.1), "event1")
        recorder.record(AffectState(valence=0.2), "event2")
        assert recorder.size == 2

    def test_get_variable_series(self):
        recorder = StateTrajectoryRecorder()
        for v in [0.1, 0.3, 0.5]:
            recorder.record(AffectState(valence=v))
        series = recorder.get_variable_series("valence")
        assert series == [0.1, 0.3, 0.5]

    def test_export_csv(self):
        recorder = StateTrajectoryRecorder()
        recorder.record(AffectState(valence=0.5, arousal=0.3))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = recorder.export_csv(Path(tmpdir) / "test.csv")
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "valence" in content
            assert "0.5" in content

    def test_export_jsonl(self):
        recorder = StateTrajectoryRecorder()
        recorder.record(AffectState())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = recorder.export_jsonl(Path(tmpdir) / "test.jsonl")
            assert path.exists()

    def test_summary_stats(self):
        recorder = StateTrajectoryRecorder()
        for v in [-0.3, 0.0, 0.5]:
            recorder.record(AffectState(valence=v))
        stats = recorder.get_summary_stats()
        assert stats["valence"]["min"] == -0.3
        assert stats["valence"]["max"] == 0.5
        assert stats["total_steps"] == 3

    def test_ascii_chart(self):
        recorder = StateTrajectoryRecorder()
        for i in range(20):
            recorder.record(AffectState(valence=i * 0.05 - 0.5))
        chart = recorder.get_ascii_chart("valence")
        assert "valence" in chart
        assert "█" in chart

    def test_clear(self):
        recorder = StateTrajectoryRecorder()
        recorder.record(AffectState())
        recorder.clear()
        assert recorder.size == 0
