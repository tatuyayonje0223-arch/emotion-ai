"""記憶モジュールのテスト。"""

from src.memory.episodic import EpisodicMemoryStore
from src.config.settings import MemoryConfig
from src.schemas.affect_state import AffectState
from src.schemas.memory import RetrievalQuery


def _make_config(**kwargs) -> MemoryConfig:
    defaults = {"max_episodic_entries": 100, "salience_threshold": 0.3}
    defaults.update(kwargs)
    return MemoryConfig(**defaults)


class TestEpisodicMemoryStore:
    def test_store_high_salience(self):
        store = EpisodicMemoryStore(_make_config(salience_threshold=0.2))
        state = AffectState(valence=0.8, arousal=0.7, motivational_salience=0.5)
        entry = store.store("e1", "良いことがあった", "良いことがあった", state)
        assert entry is not None
        assert entry.emotional_salience > 0.0
        assert store.size == 1

    def test_skip_low_salience(self):
        store = EpisodicMemoryStore(_make_config(salience_threshold=0.9))
        state = AffectState(valence=0.0, arousal=0.1, motivational_salience=0.0)
        entry = store.store("e1", "普通のこと", "普通のこと", state)
        assert entry is None
        assert store.size == 0

    def test_retrieve_by_text(self):
        store = EpisodicMemoryStore(_make_config(salience_threshold=0.1))
        state = AffectState(valence=0.5, arousal=0.5, motivational_salience=0.5)
        store.store("e1", "嬉しいニュース", "嬉しいニュース", state, tags=["positive"])
        store.store("e2", "悲しいニュース", "悲しいニュース", state, tags=["negative"])

        query = RetrievalQuery(query_text="嬉しい")
        results = store.retrieve(query)
        assert len(results) > 0
        assert "嬉しい" in results[0].memory.summary

    def test_affect_biased_retrieval(self):
        store = EpisodicMemoryStore(_make_config(salience_threshold=0.1))

        pos_state = AffectState(valence=0.8, arousal=0.5, motivational_salience=0.5)
        neg_state = AffectState(valence=-0.8, arousal=0.5, motivational_salience=0.5)

        store.store("e1", "良い記憶", "良い記憶", pos_state, tags=["test"])
        store.store("e2", "悪い記憶", "悪い記憶", neg_state, tags=["test"])

        # ポジティブ状態で検索 → ポジティブ記憶が上位
        query = RetrievalQuery(
            query_tags=["test"],
            current_valence=0.7,
            affect_bias_weight=0.8,
        )
        results = store.retrieve(query)
        assert len(results) == 2
        # ポジティブな記憶のaffect_matchが高いはず
        pos_result = [r for r in results if "良い" in r.memory.summary][0]
        neg_result = [r for r in results if "悪い" in r.memory.summary][0]
        assert pos_result.affect_match_score > neg_result.affect_match_score

    def test_capacity_eviction(self):
        store = EpisodicMemoryStore(_make_config(max_episodic_entries=3, salience_threshold=0.1))
        state = AffectState(valence=0.5, arousal=0.5, motivational_salience=0.5)
        for i in range(5):
            store.store(f"e{i}", f"記憶{i}", f"記憶{i}", state)
        assert store.size <= 3

    def test_should_store_threshold(self):
        store = EpisodicMemoryStore(_make_config(salience_threshold=0.5))
        assert store.should_store(0.6, 0.5) is True
        assert store.should_store(0.1, 0.1) is False

    def test_clear(self):
        store = EpisodicMemoryStore(_make_config(salience_threshold=0.1))
        state = AffectState(valence=0.5, arousal=0.5, motivational_salience=0.5)
        store.store("e1", "test", "test", state)
        assert store.size == 1
        store.clear()
        assert store.size == 0
