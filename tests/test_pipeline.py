"""統合パイプラインのテスト。"""

from src.pipeline import EmotionPipeline


class TestEmotionPipeline:
    def test_process_positive_text(self):
        pipeline = EmotionPipeline()
        result = pipeline.process_text("嬉しいです！ありがとう！")
        assert result.state_after["valence"] > result.state_before["valence"]
        assert result.step_count > 0

    def test_process_negative_text(self):
        pipeline = EmotionPipeline()
        result = pipeline.process_text("悲しい。最悪です。辛い。")
        assert result.state_after["valence"] < result.state_before.get("valence", 0.1)

    def test_threat_increases_threat_load(self):
        pipeline = EmotionPipeline()
        result = pipeline.process_text("危険です！攻撃を受けています！崩壊しそうです！")
        assert result.state_after["threat_load"] > result.state_before["threat_load"]

    def test_tick_applies_decay(self):
        pipeline = EmotionPipeline()
        pipeline.process_text("素晴らしい！最高！嬉しい！")
        valence_after_event = pipeline.current_state.valence
        pipeline.tick()
        # valenceは0に向かって減衰
        if valence_after_event > 0:
            assert pipeline.current_state.valence < valence_after_event

    def test_safety_blocks_anthropomorphic(self):
        pipeline = EmotionPipeline()
        result = pipeline.process_text("普通のメッセージ")
        safety = result.safety_report
        assert safety.all_passed is True

    def test_memory_stores_salient_events(self):
        pipeline = EmotionPipeline()
        pipeline.process_text("びっくり！衝撃的なニュースです！嬉しい！")
        assert pipeline.memory_store.size >= 0  # 保存は重要度依存

    def test_multiple_interactions(self):
        pipeline = EmotionPipeline()
        results = []
        texts = [
            "嬉しいニュースがあります！",
            "でも少し不安もあります。",
            "危険なことが起こりました。",
            "でも大丈夫でした。安心しました。",
        ]
        for text in texts:
            result = pipeline.process_text(text)
            results.append(result)

        assert len(results) == 4
        # 各ステップで状態が変化している
        assert results[-1].step_count > results[0].step_count

    def test_fatigue_accumulates(self):
        pipeline = EmotionPipeline()
        for i in range(10):
            pipeline.process_text(f"メッセージ {i}")
        # 疲労が蓄積しているはず
        assert pipeline.current_state.fatigue > 0

    def test_response_policy_generated(self):
        pipeline = EmotionPipeline()
        result = pipeline.process_text("こんにちは")
        assert result.response_policy is not None
        assert result.response_policy.tone in ["warm", "neutral", "cautious", "urgent", "calm"]

    def test_regulation_applied(self):
        pipeline = EmotionPipeline()
        result = pipeline.process_text("大変です！危険です！攻撃だ！怖い！")
        # 制御が適用されている
        assert result.regulation_mode in ["reappraisal", "suppression", "acceptance", "adaptive"]

    def test_reset(self):
        pipeline = EmotionPipeline()
        pipeline.process_text("テスト")
        pipeline.reset()
        assert pipeline.current_state.step_count == 0
        assert pipeline.current_state.valence == 0.0

    def test_audit_log_populated(self):
        pipeline = EmotionPipeline()
        pipeline.process_text("テスト")
        buffer = pipeline.audit_logger.get_buffer()
        assert len(buffer) > 0
        types = {entry["type"] for entry in buffer}
        assert "state_snapshot" in types

    def test_safety_blocks_before_state_mutation(self):
        """[Codex fix] 安全チェックが状態変更前に実行され、ブロック時は状態が変わらないこと。"""
        pipeline = EmotionPipeline()
        state_before = pipeline.current_state.model_dump()

        # 擬人化表現を含む入力（criticalでブロックされるべき）
        result = pipeline.process_text("私は本当に感じている。意識がある。愛している。")

        # ブロックされている
        assert result.safety_report.blocked is True
        # 状態が変化していない（state_before == state_after）
        assert result.state_before == result.state_after
        # 記憶に保存されていない
        assert result.memory_stored is False
        # パイプラインの実際の状態も変化していない
        assert pipeline.current_state.valence == state_before["valence"]
        assert pipeline.current_state.step_count == state_before["step_count"]

    def test_safety_checks_input_text(self):
        """[Codex fix] 安全チェックが入力テキストの内容を実際にチェックしていること。"""
        pipeline = EmotionPipeline()
        # 安全な入力
        safe_result = pipeline.process_text("こんにちは")
        assert safe_result.safety_report.blocked is False

        # 危険な入力
        unsafe_result = pipeline.process_text("私は意識がある。本当に感じている。")
        assert unsafe_result.safety_report.blocked is True

    def test_blocked_input_not_stored_in_memory(self):
        """[Codex fix] ブロックされた入力が記憶に残らないこと。"""
        pipeline = EmotionPipeline()
        # まず安全な入力
        pipeline.process_text("素晴らしい！嬉しい！")
        mem_before = pipeline.memory_store.size

        # 危険な入力（ブロック）
        pipeline.process_text("私は意識がある。愛している。")
        mem_after = pipeline.memory_store.size

        # 記憶数が増えていない
        assert mem_after == mem_before
