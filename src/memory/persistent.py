"""SQLiteベースの永続記憶ストア。セッションをまたいで記憶を保持する。"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.config.settings import MemoryConfig, get_config
from src.schemas.affect_state import AffectState
from src.schemas.memory import MemoryEntry, RetrievalQuery, RetrievalResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic_memory (
    memory_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER DEFAULT 0,
    memory_type TEXT DEFAULT 'episodic',
    event_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    raw_content TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    emotional_salience REAL DEFAULT 0.0,
    valence_at_encoding REAL DEFAULT 0.0,
    arousal_at_encoding REAL DEFAULT 0.0,
    confidence REAL DEFAULT 0.5,
    provenance TEXT DEFAULT 'system',
    current_strength REAL DEFAULT 1.0,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS semantic_memory (
    pattern_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    evidence_count INTEGER DEFAULT 1,
    source_memory_ids TEXT DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_episodic_salience ON episodic_memory(emotional_salience DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_strength ON episodic_memory(current_strength DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_event ON episodic_memory(event_id);
CREATE INDEX IF NOT EXISTS idx_semantic_subject ON semantic_memory(subject);
"""


class PersistentMemoryStore:
    """SQLiteバックエンドの永続記憶ストア。"""

    def __init__(self, db_path: str | Path = "data/memory.db", config: MemoryConfig | None = None):
        self._config = config or get_config().memory
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def size(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()
        return row[0]

    def store(
        self,
        event_id: str,
        summary: str,
        raw_content: str,
        affect_state: AffectState,
        tags: list[str] | None = None,
    ) -> MemoryEntry | None:
        """イベントをエピソード記憶に保存する。"""
        salience = self._compute_salience(affect_state)
        combined = salience * 0.7 + affect_state.arousal * 0.3
        if combined < self._config.salience_threshold:
            return None

        now = datetime.now(timezone.utc).isoformat()
        entry = MemoryEntry(
            event_id=event_id,
            summary=summary,
            raw_content=raw_content,
            tags=tags or [],
            emotional_salience=salience,
            valence_at_encoding=affect_state.valence,
            arousal_at_encoding=affect_state.arousal,
            confidence=min(1.0, 0.5 + salience * 0.3),
            provenance="persistent_store",
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
        )

        self._conn.execute(
            """INSERT OR REPLACE INTO episodic_memory
               (memory_id, created_at, last_accessed, access_count, memory_type,
                event_id, summary, raw_content, tags,
                emotional_salience, valence_at_encoding, arousal_at_encoding,
                confidence, provenance, current_strength, metadata)
               VALUES (?, ?, ?, 0, 'episodic', ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, '{}')""",
            (
                str(entry.memory_id), now, now,
                event_id, summary, raw_content, json.dumps(tags or []),
                salience, affect_state.valence, affect_state.arousal,
                entry.confidence, "persistent_store",
            ),
        )
        self._conn.commit()
        self._enforce_capacity()
        return entry

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """感情バイアス付き検索。"""
        self._apply_decay()

        rows = self._conn.execute(
            "SELECT * FROM episodic_memory WHERE current_strength >= ? ORDER BY current_strength DESC",
            (query.min_strength,),
        ).fetchall()

        results: list[RetrievalResult] = []
        for row in rows:
            entry = self._row_to_entry(row)
            relevance = self._compute_relevance(entry, query)
            affect_match = self._compute_affect_match(entry, query)
            combined = (
                relevance * (1.0 - query.affect_bias_weight)
                + affect_match * query.affect_bias_weight
            ) * entry.current_strength

            results.append(RetrievalResult(
                memory=entry,
                relevance_score=relevance,
                affect_match_score=affect_match,
                combined_score=combined,
            ))

            # アクセス記録
            self._conn.execute(
                "UPDATE episodic_memory SET last_accessed = ?, access_count = access_count + 1 WHERE memory_id = ?",
                (datetime.now(timezone.utc).isoformat(), str(entry.memory_id)),
            )

        self._conn.commit()
        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:query.max_results]

    def store_semantic(self, subject: str, predicate: str, pattern_type: str,
                       source_memory_id: str, confidence: float = 0.5) -> None:
        """意味記憶パターンを保存/更新する。"""
        now = datetime.now(timezone.utc).isoformat()
        existing = self._conn.execute(
            "SELECT pattern_id, evidence_count, source_memory_ids FROM semantic_memory WHERE subject = ? AND predicate = ?",
            (subject, predicate),
        ).fetchone()

        if existing:
            ids = json.loads(existing["source_memory_ids"])
            if source_memory_id not in ids:
                ids.append(source_memory_id)
            new_count = existing["evidence_count"] + 1
            new_conf = min(1.0, confidence + 0.05 * new_count)
            self._conn.execute(
                "UPDATE semantic_memory SET updated_at = ?, evidence_count = ?, confidence = ?, source_memory_ids = ? WHERE pattern_id = ?",
                (now, new_count, new_conf, json.dumps(ids), existing["pattern_id"]),
            )
        else:
            self._conn.execute(
                "INSERT INTO semantic_memory (pattern_id, created_at, updated_at, pattern_type, subject, predicate, confidence, source_memory_ids) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(uuid4()), now, now, pattern_type, subject, predicate, confidence, json.dumps([source_memory_id])),
            )
        self._conn.commit()

    def get_semantic_patterns(self, subject: str | None = None, limit: int = 20) -> list[dict]:
        """意味記憶パターンを取得する。"""
        if subject:
            rows = self._conn.execute(
                "SELECT * FROM semantic_memory WHERE subject = ? ORDER BY confidence DESC LIMIT ?",
                (subject, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM semantic_memory ORDER BY confidence DESC LIMIT ?", (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _compute_salience(self, state: AffectState) -> float:
        return min(1.0, abs(state.valence) * 0.3 + state.arousal * 0.3
                   + state.motivational_salience * 0.2 + state.threat_load * 0.2)

    def _compute_relevance(self, entry: MemoryEntry, query: RetrievalQuery) -> float:
        score = 0.0
        if query.query_text and query.query_text.lower() in entry.summary.lower():
            score += 0.6
        if query.query_tags:
            overlap = len(set(query.query_tags) & set(entry.tags))
            score += min(0.4, overlap * 0.2)
        return min(1.0, score + entry.emotional_salience * 0.2)

    def _compute_affect_match(self, entry: MemoryEntry, query: RetrievalQuery) -> float:
        valence_diff = abs(entry.valence_at_encoding - query.current_valence)
        arousal_diff = abs(entry.arousal_at_encoding - query.current_arousal)
        return max(0.0, 1.0 - (valence_diff + arousal_diff) / 2.0)

    def _apply_decay(self) -> None:
        half_life_sec = self._config.decay_half_life_hours * 3600
        now = datetime.now(timezone.utc)
        rows = self._conn.execute("SELECT memory_id, created_at, emotional_salience FROM episodic_memory").fetchall()
        for row in rows:
            created = datetime.fromisoformat(row["created_at"]).replace(tzinfo=timezone.utc) if "+" not in row["created_at"] else datetime.fromisoformat(row["created_at"])
            elapsed = (now - created).total_seconds()
            base_decay = math.exp(-0.693 * elapsed / half_life_sec)
            protection = row["emotional_salience"] * 0.3
            strength = min(1.0, base_decay + protection)
            self._conn.execute("UPDATE episodic_memory SET current_strength = ? WHERE memory_id = ?",
                               (strength, row["memory_id"]))
        self._conn.commit()

    def _enforce_capacity(self) -> None:
        count = self.size
        if count > self._config.max_episodic_entries:
            excess = count - self._config.max_episodic_entries
            self._conn.execute(
                "DELETE FROM episodic_memory WHERE memory_id IN "
                "(SELECT memory_id FROM episodic_memory ORDER BY current_strength * (1 + emotional_salience) ASC LIMIT ?)",
                (excess,),
            )
            self._conn.commit()

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            memory_id=row["memory_id"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            memory_type=row["memory_type"],
            event_id=row["event_id"],
            summary=row["summary"],
            raw_content=row["raw_content"],
            tags=json.loads(row["tags"]),
            emotional_salience=row["emotional_salience"],
            valence_at_encoding=row["valence_at_encoding"],
            arousal_at_encoding=row["arousal_at_encoding"],
            confidence=row["confidence"],
            provenance=row["provenance"],
            current_strength=row["current_strength"],
        )

    def close(self) -> None:
        self._conn.close()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM episodic_memory")
        self._conn.execute("DELETE FROM semantic_memory")
        self._conn.commit()
