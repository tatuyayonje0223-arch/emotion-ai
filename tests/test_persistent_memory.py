"""永続記憶ストアのテスト。"""

import tempfile
from pathlib import Path

import pytest

from src.config.settings import MemoryConfig
from src.memory.persistent import PersistentMemoryStore
from src.schemas.affect_state import AffectState
from src.schemas.memory import RetrievalQuery


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        s = PersistentMemoryStore(db_path, MemoryConfig(salience_threshold=0.2))
        yield s
        s.close()


class TestPersistentMemoryStore:
    def test_store_and_retrieve(self, store):
        state = AffectState(valence=0.7, arousal=0.6, motivational_salience=0.5)
        entry = store.store("e1", "良いこと", "良いことがあった", state, ["positive"])
        assert entry is not None
        assert store.size == 1

        results = store.retrieve(RetrievalQuery(query_text="良い"))
        assert len(results) > 0
        assert "良い" in results[0].memory.summary

    def test_skip_low_salience(self, store):
        state = AffectState(valence=0.0, arousal=0.05, motivational_salience=0.0)
        entry = store.store("e1", "普通", "普通のこと", state)
        assert entry is None
        assert store.size == 0

    def test_affect_biased_retrieval(self, store):
        pos_state = AffectState(valence=0.8, arousal=0.6, motivational_salience=0.5)
        neg_state = AffectState(valence=-0.8, arousal=0.6, motivational_salience=0.5)
        store.store("e1", "良い記憶", "良い記憶", pos_state, ["test"])
        store.store("e2", "悪い記憶", "悪い記憶", neg_state, ["test"])

        query = RetrievalQuery(current_valence=0.7, affect_bias_weight=0.8, query_tags=["test"])
        results = store.retrieve(query)
        assert len(results) == 2

    def test_persistence_across_connections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "persist_test.db"
            s1 = PersistentMemoryStore(db_path, MemoryConfig(salience_threshold=0.1))
            state = AffectState(valence=0.5, arousal=0.5, motivational_salience=0.5)
            s1.store("e1", "永続テスト", "永続テスト", state)
            s1.close()

            s2 = PersistentMemoryStore(db_path, MemoryConfig(salience_threshold=0.1))
            assert s2.size == 1
            results = s2.retrieve(RetrievalQuery(query_text="永続"))
            assert len(results) == 1
            s2.close()

    def test_semantic_memory(self, store):
        store.store_semantic("user", "prefers_formal", "preference", "e1", 0.6)
        store.store_semantic("user", "prefers_formal", "preference", "e2", 0.6)
        patterns = store.get_semantic_patterns("user")
        assert len(patterns) == 1
        assert patterns[0]["evidence_count"] == 2
        assert patterns[0]["confidence"] > 0.6

    def test_capacity_enforcement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "cap_test.db"
            s = PersistentMemoryStore(db_path, MemoryConfig(max_episodic_entries=3, salience_threshold=0.1))
            state = AffectState(valence=0.5, arousal=0.5, motivational_salience=0.5)
            for i in range(5):
                s.store(f"e{i}", f"記憶{i}", f"記憶{i}", state)
            assert s.size <= 3
            s.close()

    def test_clear(self, store):
        state = AffectState(valence=0.5, arousal=0.5, motivational_salience=0.5)
        store.store("e1", "test", "test", state)
        store.store_semantic("x", "y", "t", "e1")
        store.clear()
        assert store.size == 0
        assert len(store.get_semantic_patterns()) == 0
