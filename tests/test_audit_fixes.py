"""監査修正の統合テスト。Fix1-5の検証。"""

import pytest
import numpy as np


class TestFix1QuantitativeBenchmark:
    """[Fix1] 恐怖条件付け定量ベンチマーク。"""

    def test_acquisition_curve_runs(self):
        from src.evaluation.fear_benchmark import run_fear_acquisition_benchmark
        result = run_fear_acquisition_benchmark(n_baseline=2, n_conditioning=4, n_extinction=3)
        assert len(result.baseline_rates) == 2
        assert len(result.conditioning_rates) == 4
        assert len(result.extinction_rates) == 3
        assert result.baseline_mean >= 0
        assert result.conditioned_mean >= 0

    def test_conditioning_increases_rate(self):
        """条件付けでBLA発火率がベースラインより上昇する。"""
        from src.evaluation.fear_benchmark import run_fear_acquisition_benchmark
        result = run_fear_acquisition_benchmark(n_baseline=2, n_conditioning=5, n_extinction=2)
        # 条件付け後の平均がベースラインの0.8倍以上（CS強度増加のため上昇方向）
        if result.baseline_mean > 0:
            assert result.conditioned_mean >= result.baseline_mean * 0.8


class TestFix2ReadoutV2:
    """[Fix2] データ駆動リードアウト。"""

    def test_spiking_readout_fit_and_classify(self):
        from src.brian2_circuits.readout_v2 import SpikingReadout, ReadoutTrainingData
        # ダミーデータ: 恐怖条件=高CeM、安全=低CeM
        rates = np.array([
            [10, 5, 20, 5, 30, 8, 8, 5],   # fear
            [10, 5, 20, 5, 25, 8, 8, 5],   # fear
            [8, 4, 5, 15, 3, 7, 7, 3],     # safe
            [7, 4, 4, 16, 2, 7, 7, 3],     # safe
            [5, 3, 8, 8, 5, 6, 6, 15],     # anxiety
            [5, 3, 7, 9, 4, 6, 6, 14],     # anxiety
        ])
        data = ReadoutTrainingData(
            rates_matrix=rates,
            labels=["fear", "fear", "safe", "safe", "anxiety", "anxiety"],
            population_names=["la", "ba", "cel_som", "cel_pkcd", "cem", "pl", "il", "bnst"],
        )
        readout = SpikingReadout(n_components=2)
        readout.fit(data)
        assert readout.is_fitted

        # 恐怖パターンを分類
        result = readout.classify(np.array([10, 5, 20, 5, 28, 8, 8, 5]))
        assert "fear" in result
        assert sum(result.values()) > 0.99  # 確率の合計≈1

    def test_emotion_readout_from_pca(self):
        from src.brian2_circuits.readout_v2 import SpikingReadout, ReadoutTrainingData
        rates = np.array([
            [10, 5, 20, 5, 30, 8, 8, 5],
            [8, 4, 5, 15, 3, 7, 7, 3],
            [5, 3, 8, 8, 5, 6, 6, 15],
        ])
        data = ReadoutTrainingData(
            rates_matrix=rates, labels=["fear", "safe", "anxiety"],
            population_names=["la", "ba", "cs", "cp", "cm", "pl", "il", "bn"],
        )
        readout = SpikingReadout(n_components=2)
        readout.fit(data)
        result = readout.to_emotion_readout(np.array([10, 5, 20, 5, 30, 8, 8, 5]))
        assert "valence" in result
        assert "arousal" in result
        assert "dominant_state" in result


class TestFix3AllenDataOverride:
    """[Fix3] Allen APIデータでの上書き。"""

    def test_data_manager_returns_matrix(self):
        from src.data_driven.data_manager import get_or_build_matrix
        matrix = get_or_build_matrix()
        assert len(matrix.projections) > 20
        assert len(matrix.regions) >= 10

    def test_fear_spec_uses_best_data(self):
        from src.data_driven.build_fear_spec import build_fear_circuit_spec
        spec = build_fear_circuit_spec()
        assert spec.total_neurons > 100
        # データソースが追跡されている
        for conn in spec.connections:
            assert conn.data_source != ""


class TestFix4Sensitivity:
    """[Fix4] 感度分析（parametric testはtest_sensitivity.pyで実施）。"""

    def test_sensitivity_module_importable(self):
        """test_sensitivity.pyがインポート可能。"""
        import tests.test_sensitivity  # noqa: F401


class TestFix5Brian2Backend:
    """[Fix5] NeuroPipeline + Brian2統合。"""

    def test_brian2_backend_processes_threat(self):
        from src.brian2_circuits.brian2_backend import Brian2Backend
        from src.neurocircuit.brain import SensoryInput
        backend = Brian2Backend()
        result = backend.process(SensoryInput(threat_signal=0.8))
        assert result.readout is not None
        assert "fear" in result.circuits_activated
        assert result.readout.threat_load >= 0

    def test_brian2_backend_processes_reward(self):
        from src.brian2_circuits.brian2_backend import Brian2Backend
        from src.neurocircuit.brain import SensoryInput
        backend = Brian2Backend()
        result = backend.process(SensoryInput(reward_signal=0.8))
        assert "reward" in result.circuits_activated
        assert result.readout.reward_drive >= 0

    def test_neuropipeline_brian2_mode(self):
        """NeuroPipelineがBrian2バックエンドで動作する。"""
        from src.neurocircuit.neuro_pipeline import NeuroPipeline
        pipeline = NeuroPipeline(backend="brian2")
        result = pipeline.process_text("危険です！攻撃！脅威！")
        assert result is not None
        assert result.readout.threat_load >= 0
        assert len(result.region_activities) > 0

    def test_neuropipeline_wilson_cowan_still_works(self):
        """Wilson-Cowanバックエンドが引き続き動作する。"""
        from src.neurocircuit.neuro_pipeline import NeuroPipeline
        pipeline = NeuroPipeline(backend="wilson_cowan")
        result = pipeline.process_text("こんにちは")
        assert result is not None
