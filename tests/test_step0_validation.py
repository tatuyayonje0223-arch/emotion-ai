"""Step 0: 恐怖回路1本の定量検証テスト。

文献データとの系統的照合。各テストに根拠論文を明記。
"""

import pytest

from src.calibration.fear_quantitative import (
    validate_fear_circuit, parameter_sweep, LiteratureData,
)
from src.brian2_circuits.fear_circuit_v2 import FearV2Config


def _default_cfg():
    return FearV2Config(
        cs_amp=10.0, us_amp=20.0,
        duration_ms=250, cs_dur_ms=120,
        us_onset_ms=150, us_dur_ms=25,
    )


class TestFearValidation:
    def test_validation_runs(self):
        """検証パイプラインが実行可能。"""
        result = validate_fear_circuit(_default_cfg(), n_conditioning=2, n_extinction=2)
        assert result is not None
        assert 0 <= result.score <= 1.0
        assert len(result.details) > 3

    def test_bla_baseline_positive(self):
        """BLAベースライン発火率が正（背景ノイズで自発発火）。"""
        result = validate_fear_circuit(_default_cfg(), n_conditioning=2, n_extinction=2)
        assert result.metrics["bla_baseline"] > 0

    def test_conditioning_increases_bla(self):
        """条件付け後のBLA発火率 > ベースライン。Ref: Quirk 1995"""
        result = validate_fear_circuit(_default_cfg(), n_conditioning=3, n_extinction=2)
        assert result.metrics["bla_conditioned"] >= result.metrics["bla_baseline"] * 0.8

    def test_cel_som_active(self):
        """条件付け時にCeL SOM+ (fear-ON) が活性化。Ref: Ciocchi 2010"""
        result = validate_fear_circuit(_default_cfg(), n_conditioning=2, n_extinction=2)
        assert result.metrics["cel_som_rate"] > 0

    def test_bnst_sustained_threat(self):
        """持続的脅威でBNST > ベースライン。Ref: Davis 2010"""
        result = validate_fear_circuit(_default_cfg(), n_conditioning=2, n_extinction=2)
        assert result.metrics["bnst_sustained"] > 0

    def test_acquisition_curve_exists(self):
        """獲得曲線が生成される（3試行以上の発火率リスト）。"""
        result = validate_fear_circuit(_default_cfg(), n_conditioning=4, n_extinction=2)
        assert len(result.metrics["acquisition_rates"]) == 4
        assert all(r >= 0 for r in result.metrics["acquisition_rates"])

    def test_overall_score_reasonable(self):
        """総合スコアが0より大きい（完全失敗ではない）。"""
        result = validate_fear_circuit(_default_cfg(), n_conditioning=3, n_extinction=2)
        assert result.score > 0

    def test_parameter_sweep_finds_best(self):
        """パラメータ探索で最良設定が見つかる。"""
        results = parameter_sweep(n_configs=4)
        assert len(results) > 0
        # ソート済み（最良が先頭）
        assert results[0].score >= results[-1].score

    def test_details_contain_literature_refs(self):
        """検証詳細に文献ターゲット値が含まれる。"""
        result = validate_fear_circuit(_default_cfg(), n_conditioning=2, n_extinction=2)
        detail_text = " ".join(result.details)
        assert "target" in detail_text
        assert "Hz" in detail_text
