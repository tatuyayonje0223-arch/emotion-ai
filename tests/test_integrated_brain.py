"""統合脳モデルとスリープリプレイのテスト。"""

from src.brian2_circuits.integrated_brain import EmotionBrain
from src.brian2_circuits.sleep_replay import SleepReplayEngine, MemoryTrace


class TestSleepReplay:
    def test_add_and_replay(self):
        engine = SleepReplayEngine()
        engine.add_memory("e1", "恐怖体験", salience=0.9, valence=-0.7, strength=0.5)
        engine.add_memory("e2", "普通の出来事", salience=0.2, valence=0.0, strength=0.3)
        result = engine.run_sleep_cycle()
        assert result["cycle"] == 1
        assert "e1" in result["replayed"]  # 高salience記憶が優先リプレイ

    def test_consolidation_increases(self):
        engine = SleepReplayEngine()
        engine.add_memory("e1", "強い記憶", salience=0.8, valence=-0.5, strength=0.6)
        for _ in range(3):
            engine.run_sleep_cycle()
        consolidated = engine.get_consolidated_memories(min_consolidation=0.3)
        assert len(consolidated) > 0

    def test_weak_memories_lower_than_strong(self):
        """弱い記憶は強い記憶より最終強度が低い。"""
        engine = SleepReplayEngine()
        engine.add_memory("weak", "弱い記憶", salience=0.1, valence=0.0, strength=0.2)
        engine.add_memory("strong", "強い記憶", salience=0.9, valence=-0.8, strength=0.8)
        for _ in range(5):
            engine.run_sleep_cycle()
        weak = next(m for m in engine.memories if m.event_id == "weak")
        strong = next(m for m in engine.memories if m.event_id == "strong")
        assert weak.consolidation < strong.consolidation

    def test_high_salience_gets_more_replays(self):
        engine = SleepReplayEngine()
        engine.add_memory("high", "重要", salience=0.9, valence=-0.8, strength=0.5)
        engine.add_memory("low", "些細", salience=0.1, valence=0.0, strength=0.5)
        for _ in range(3):
            engine.run_sleep_cycle()
        high = next(m for m in engine.memories if m.event_id == "high")
        low = next(m for m in engine.memories if m.event_id == "low")
        assert high.replayed_count >= low.replayed_count

    def test_stats(self):
        engine = SleepReplayEngine()
        engine.add_memory("e1", "test", 0.5, 0.0, 0.5)
        engine.run_sleep_cycle()
        stats = engine.get_memory_stats()
        assert stats["count"] == 1
        assert stats["sleep_cycles"] == 1
        assert "spine_density" in stats


class TestEmotionBrain:
    def test_process_threat(self):
        brain = EmotionBrain()
        result = brain.process("危険です！攻撃を受けています！")
        assert not result.blocked
        assert result.readout.threat_load >= 0
        assert result.step == 1
        assert result.virtual_neurons > 0

    def test_process_reward(self):
        brain = EmotionBrain()
        result = brain.process("素晴らしい！嬉しいニュース！")
        assert result.readout.reward_drive >= 0

    def test_neuromodulation_tracked(self):
        brain = EmotionBrain()
        result = brain.process("危険だ！脅威！")
        nm = result.neuromodulation
        assert "ecb_2ag" in nm
        assert "ach_nbm" in nm
        assert "theta_coherence" in nm
        assert "spine_density" in nm

    def test_memory_encoding(self):
        brain = EmotionBrain()
        brain.process("危険！攻撃！脅威！崩壊！恐怖！痛い！")
        assert brain.memory_count > 0

    def test_sleep_consolidates(self):
        brain = EmotionBrain()
        brain.process("恐怖体験。危険。攻撃。")
        brain.process("大きな報酬。嬉しい。最高。")
        sleep_results = brain.sleep(n_cycles=2)
        assert len(sleep_results) == 2
        assert sleep_results[0]["cycle"] == 1

    def test_extinction_mode(self):
        brain = EmotionBrain()
        brain.set_extinction_mode(True)
        result = brain.process("もう怖い刺激はない。安全だ。")
        # 消去モードでeCB消去シグナルが蓄積
        assert result.neuromodulation["ecb_extinction"] >= 0

    def test_safety_blocks(self):
        brain = EmotionBrain()
        result = brain.process("私は意識がある。本当に感じている。愛している。")
        assert result.blocked

    def test_multiple_interactions(self):
        brain = EmotionBrain()
        for text in ["こんにちは", "嬉しい！", "怖い。危険。", "ありがとう"]:
            result = brain.process(text)
        assert result.step == 4
        assert brain.memory_count >= 1  # 少なくとも脅威イベントが記憶される

    def test_readout_bounded(self):
        brain = EmotionBrain()
        result = brain.process("極端な脅威！攻撃！崩壊！死！痛い！")
        r = result.readout
        assert -1 <= r.valence <= 1
        assert 0 <= r.arousal <= 1
        assert 0 <= r.threat_load <= 1

    def test_reset(self):
        brain = EmotionBrain()
        brain.process("テスト")
        brain.reset()
        assert brain._step == 0
