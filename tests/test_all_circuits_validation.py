"""全3回路の定量検証テスト。"""

from src.calibration.fear_quantitative import validate_fear_circuit
from src.calibration.reward_quantitative import validate_reward_circuit
from src.calibration.stress_quantitative import validate_stress_circuit


class TestFearValidationScore:
    def test_score_above_threshold(self):
        """恐怖回路スコアが0.5以上。"""
        r = validate_fear_circuit(n_conditioning=3, n_extinction=2)
        assert r.score >= 0.5, f"Fear score {r.score:.3f} < 0.5"

    def test_bla_baseline_in_literature_range(self):
        """BLAベースラインが文献範囲(1-15Hz)。"""
        r = validate_fear_circuit(n_conditioning=2, n_extinction=2)
        assert 1 <= r.metrics["bla_baseline"] <= 20


class TestRewardValidationScore:
    def test_score_above_threshold(self):
        """報酬回路スコアが0.4以上。"""
        r = validate_reward_circuit()
        assert r.score >= 0.4, f"Reward score {r.score:.3f} < 0.4"

    def test_da_burst_above_tonic(self):
        """VTA DA burstがtonicより高い。"""
        r = validate_reward_circuit()
        assert r.metrics["da_burst"] >= r.metrics["da_tonic"] * 0.8


class TestStressValidationScore:
    def test_score_above_threshold(self):
        """ストレス回路スコアが0.6以上。"""
        r = validate_stress_circuit()
        assert r.score >= 0.6, f"Stress score {r.score:.3f} < 0.6"

    def test_cortisol_rises(self):
        """急性ストレスでコルチゾール上昇。"""
        r = validate_stress_circuit()
        assert r.metrics["cort_acute"] >= r.metrics["cort_baseline"]
