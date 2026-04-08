"""エピソード記憶ストア。情動的重要度に基づく書き込み・減衰・検索。"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from src.config.settings import MemoryConfig, get_config
from src.schemas.affect_state import AffectState
from src.schemas.memory import MemoryEntry, RetrievalQuery, RetrievalResult


class EpisodicMemoryStore:
    """エピソード記憶の保持と検索。"""

    def __init__(self, config: MemoryConfig | None = None):
        self._config = config or get_config().memory
        self._entries: list[MemoryEntry] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def should_store(self, emotional_salience: float, arousal: float) -> bool:
        """保存すべきかどうかを判定する。情動的重要度が閾値以上で保存。"""
        combined = emotional_salience * 0.7 + arousal * 0.3
        return combined >= self._config.salience_threshold

    def store(
        self,
        event_id: str,
        summary: str,
        raw_content: str,
        affect_state: AffectState,
        tags: list[str] | None = None,
    ) -> MemoryEntry | None:
        """イベントをエピソード記憶に保存する。保存判定を通過しない場合はNone。"""
        salience = self._compute_salience(affect_state)

        if not self.should_store(salience, affect_state.arousal):
            return None

        entry = MemoryEntry(
            event_id=event_id,
            summary=summary,
            raw_content=raw_content,
            tags=tags or [],
            emotional_salience=salience,
            valence_at_encoding=affect_state.valence,
            arousal_at_encoding=affect_state.arousal,
            confidence=min(1.0, 0.5 + salience * 0.3),
            provenance="episodic_store",
        )

        self._entries.append(entry)

        # 容量超過時は最も弱いエントリを削除
        if len(self._entries) > self._config.max_episodic_entries:
            self._evict_weakest()

        return entry

    def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """感情バイアス付き検索を実行する。"""
        self._apply_decay()

        results: list[RetrievalResult] = []
        for entry in self._entries:
            if entry.current_strength < query.min_strength:
                continue

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

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[: query.max_results]

    def _compute_salience(self, state: AffectState) -> float:
        """現在の情動状態から記憶の重要度を計算する。"""
        return min(1.0, (
            abs(state.valence) * 0.3
            + state.arousal * 0.3
            + state.motivational_salience * 0.2
            + state.threat_load * 0.2
        ))

    def _compute_relevance(self, entry: MemoryEntry, query: RetrievalQuery) -> float:
        """テキストとタグの一致度を計算する。"""
        score = 0.0
        if query.query_text and query.query_text.lower() in entry.summary.lower():
            score += 0.6
        if query.query_tags:
            overlap = len(set(query.query_tags) & set(entry.tags))
            score += min(0.4, overlap * 0.2)
        return min(1.0, score + entry.emotional_salience * 0.2)

    def _compute_affect_match(self, entry: MemoryEntry, query: RetrievalQuery) -> float:
        """現在の感情状態と記憶時の感情の類似度を計算する。

        気分一致効果: 現在ネガティブなら過去のネガティブ記憶が引き出されやすい。
        """
        valence_diff = abs(entry.valence_at_encoding - query.current_valence)
        arousal_diff = abs(entry.arousal_at_encoding - query.current_arousal)
        distance = (valence_diff + arousal_diff) / 2.0
        return max(0.0, 1.0 - distance)

    def _apply_decay(self) -> None:
        """全エントリに時間減衰を適用する。"""
        now = datetime.now(timezone.utc)
        half_life_seconds = self._config.decay_half_life_hours * 3600

        for entry in self._entries:
            elapsed = (now - entry.created_at).total_seconds()
            # 指数減衰 + 情動的重要度による保護
            base_decay = math.exp(-0.693 * elapsed / half_life_seconds)
            protection = entry.emotional_salience * 0.3
            entry.current_strength = min(1.0, base_decay + protection)

    def _evict_weakest(self) -> None:
        """最も弱いエントリを1件削除する。"""
        if not self._entries:
            return
        weakest = min(self._entries, key=lambda e: e.current_strength * (1 + e.emotional_salience))
        self._entries.remove(weakest)

    def clear(self) -> None:
        self._entries.clear()
