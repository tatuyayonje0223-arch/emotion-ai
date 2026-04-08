"""睡眠リプレイと記憶固定化。

PLOSCompBiol 2025 (文脈恐怖学習の睡眠固定化モデル) に基づく:
  - 覚醒中: 恐怖条件付け → 短期記憶(BLA活性)
  - NREM睡眠: SWR(鋭波リプル) → 海馬-皮質間リプレイ → 記憶固定化
  - REM睡眠: シータ振動 → BLA-海馬同期 → 情動記憶の強化
  - ホメオスタシス: シナプススケーリング → 弱い記憶の選好的強化

プロセス:
  1. 覚醒フェーズで学習（恐怖/報酬/ストレス）
  2. NREM: リプレイ→結合強化（高salience記憶優先）
  3. REM: シータ同期→情動タグ付き記憶の選択的強化
  4. 朝: スケーリング→全体バランス回復
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.brian2_circuits.neuromodulation import (
    ThetaOscillator, StructuralPlasticityState,
    update_structural_plasticity,
)


@dataclass
class MemoryTrace:
    """1つの記憶トレース。"""

    event_id: str
    content: str
    emotional_salience: float   # 情動的重要度 (0-1)
    valence: float              # 快/不快 (-1 to 1)
    encoding_strength: float    # エンコード強度 (0-1)
    consolidation: float = 0.0  # 固定化度合い (0-1)
    replayed_count: int = 0     # リプレイ回数
    age_cycles: int = 0         # 経過した睡眠サイクル数


@dataclass
class SleepState:
    """睡眠状態。"""

    stage: str = "awake"  # "awake", "nrem1", "nrem2", "nrem3", "rem"
    cycle: int = 0
    time_in_stage_ms: float = 0.0
    total_sleep_ms: float = 0.0

    # SWR (鋭波リプル)
    swr_count: int = 0
    swr_amplitude: float = 0.0

    # REM theta
    rem_theta: ThetaOscillator = field(default_factory=lambda: ThetaOscillator(frequency_hz=6.0))

    # コルチゾール（睡眠中は低い）
    cortisol_during_sleep: float = 0.1


class SleepReplayEngine:
    """睡眠リプレイエンジン。"""

    def __init__(self):
        self._memories: list[MemoryTrace] = []
        self._sleep = SleepState()
        self._structural = StructuralPlasticityState()

    def add_memory(self, event_id: str, content: str,
                   salience: float, valence: float, strength: float) -> None:
        """覚醒中に記憶を追加する。"""
        self._memories.append(MemoryTrace(
            event_id=event_id,
            content=content,
            emotional_salience=salience,
            valence=valence,
            encoding_strength=strength,
        ))

    def run_sleep_cycle(self) -> dict:
        """1睡眠サイクル（NREM→REM→NREM→REM...）を実行する。

        ~90分サイクル:
          NREM1: 5min → NREM2: 20min → NREM3: 30min → REM: 20min → NREM2: 15min
        """
        self._sleep.cycle += 1
        results = {
            "cycle": self._sleep.cycle,
            "replayed": [],
            "consolidated": [],
            "scaling_applied": False,
        }

        # === NREM3 (徐波睡眠): SWR → リプレイ ===
        self._sleep.stage = "nrem3"
        self._sleep.cortisol_during_sleep = 0.05  # NREM中はコルチゾール最低

        # 高salience記憶を優先的にリプレイ
        sorted_memories = sorted(self._memories, key=lambda m: m.emotional_salience, reverse=True)
        n_replay = min(5, len(sorted_memories))  # 1サイクルで最大5記憶をリプレイ

        for mem in sorted_memories[:n_replay]:
            # SWRによるリプレイ
            self._sleep.swr_count += 1
            replay_boost = 0.1 * (1.0 + mem.emotional_salience)

            # リプレイで結合強化
            mem.encoding_strength = min(1.0, mem.encoding_strength + replay_boost)
            mem.replayed_count += 1
            results["replayed"].append(mem.event_id)

            # 構造的可塑性: リプレイ→LTP→スパイン新生
            self._structural = update_structural_plasticity(
                self._structural, ltp_occurred=True, ltd_occurred=False,
                amygdala_activity=mem.emotional_salience, dt_ms=100.0,
            )

        # === REM: シータ同期 → 情動記憶の選択的強化 ===
        self._sleep.stage = "rem"
        self._sleep.cortisol_during_sleep = 0.15  # REMで少し上昇

        for mem in self._memories:
            if mem.emotional_salience > 0.5:  # 情動的に重要な記憶のみ
                # REMシータ中: 情動タグ付き記憶を固定化
                consolidation_boost = 0.35 * mem.emotional_salience * mem.encoding_strength
                mem.consolidation = min(1.0, mem.consolidation + consolidation_boost)
                if mem.consolidation > 0.5 and mem.event_id not in results["consolidated"]:
                    results["consolidated"].append(mem.event_id)

        # === 朝: ホメオスタシックスケーリング ===
        # 全記憶の強度を正規化（弱い記憶を選好的に強化）
        if self._memories:
            strengths = [m.encoding_strength for m in self._memories]
            mean_strength = np.mean(strengths)
            if mean_strength > 0:
                for mem in self._memories:
                    # 弱い記憶は相対的に強化、強い記憶は相対的に弱化
                    scaling = mean_strength / max(mem.encoding_strength, 0.01)
                    mem.encoding_strength *= (1.0 + 0.05 * (scaling - 1.0))
                    mem.encoding_strength = max(0.01, min(1.0, mem.encoding_strength))
            results["scaling_applied"] = True

        # 年齢更新
        for mem in self._memories:
            mem.age_cycles += 1

        # 忘却: 固定化されていない古い記憶は減衰
        for mem in self._memories:
            if mem.consolidation < 0.3 and mem.age_cycles > 3:
                mem.encoding_strength *= 0.8

        return results

    def get_consolidated_memories(self, min_consolidation: float = 0.5) -> list[MemoryTrace]:
        """固定化された記憶を返す。"""
        return [m for m in self._memories if m.consolidation >= min_consolidation]

    def get_memory_stats(self) -> dict:
        """記憶統計。"""
        if not self._memories:
            return {"count": 0}
        return {
            "count": len(self._memories),
            "consolidated": sum(1 for m in self._memories if m.consolidation > 0.5),
            "mean_strength": float(np.mean([m.encoding_strength for m in self._memories])),
            "mean_consolidation": float(np.mean([m.consolidation for m in self._memories])),
            "total_replays": sum(m.replayed_count for m in self._memories),
            "sleep_cycles": self._sleep.cycle,
            "spine_density": self._structural.spine_density,
        }

    @property
    def memories(self) -> list[MemoryTrace]:
        return list(self._memories)

    @property
    def sleep_state(self) -> SleepState:
        return self._sleep
