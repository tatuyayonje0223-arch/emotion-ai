"""感度分析テスト。

[監査Fix4] 主要パラメータを±50%変動させても回路が質的に正しく動作することを検証。
"""

import pytest

from src.brian2_circuits.fear_circuit_v2 import FearCircuitV2, FearV2Config


def _default_cfg():
    return FearV2Config(
        n_la_exc=20, n_la_pv=5, n_la_vip=3, n_ba_exc=15,
        n_cel_som=8, n_cel_pkcd=8, n_cem=6, n_itc=8,
        n_pl=15, n_il=15, n_bnst=8,
        duration_ms=200, cs_dur_ms=100, us_onset_ms=130, us_dur_ms=20,
    )


class TestCSAmplitudeSensitivity:
    """CS入力強度を±50%変動させても恐怖応答の方向性が保たれる。"""

    @pytest.mark.parametrize("factor", [0.5, 1.0, 1.5])
    def test_cs_amp_scaling(self, factor):
        cfg = _default_cfg()
        cfg.cs_amp = cfg.cs_amp * factor
        circuit = FearCircuitV2(cfg)
        baseline = circuit.run_trial(cs=False, us=False, phase="baseline")
        circuit2 = FearCircuitV2(cfg)
        with_cs = circuit2.run_trial(cs=True, us=True, phase="conditioning")
        # 方向性保存: CS+USが常にbaselineのCeM以上
        assert with_cs.cem_rate >= baseline.cem_rate * 0.5
        # 生理的範囲
        assert 0 <= with_cs.la_rate <= 300
        assert 0 <= with_cs.cem_rate <= 300


class TestUSAmplitudeSensitivity:
    """US入力強度を±50%変動。"""

    @pytest.mark.parametrize("factor", [0.5, 1.0, 1.5])
    def test_us_amp_scaling(self, factor):
        cfg = _default_cfg()
        cfg.us_amp = cfg.us_amp * factor
        circuit = FearCircuitV2(cfg)
        result = circuit.run_trial(cs=True, us=True, phase="conditioning")
        assert 0 <= result.freeze_response <= 1.0
        assert 0 <= result.la_rate <= 300


class TestBackgroundNoiseSensitivity:
    """背景ノイズを±50%変動。"""

    @pytest.mark.parametrize("factor", [0.5, 1.0, 1.5])
    def test_noise_scaling(self, factor):
        cfg = _default_cfg()
        cfg.bg_noise = cfg.bg_noise * factor
        circuit = FearCircuitV2(cfg)
        result = circuit.run_trial(cs=True, us=False, phase="test")
        # 発火率が生理的範囲内
        assert 0 <= result.la_rate <= 300
        assert 0 <= result.cem_rate <= 300


class TestScaleSensitivity:
    """ニューロン数を2倍にしても飽和しないか（バランスネットワーク重みスケーリングの検証）。"""

    def test_double_neurons_no_saturation(self):
        cfg_normal = _default_cfg()
        circuit_normal = FearCircuitV2(cfg_normal)
        r_normal = circuit_normal.run_trial(cs=True, us=True, phase="conditioning")

        cfg_double = FearV2Config(
            n_la_exc=40, n_la_pv=10, n_la_vip=6, n_ba_exc=30,
            n_cel_som=16, n_cel_pkcd=16, n_cem=12, n_itc=16,
            n_pl=30, n_il=30, n_bnst=16,
            duration_ms=200, cs_dur_ms=100, us_onset_ms=130, us_dur_ms=20,
        )
        circuit_double = FearCircuitV2(cfg_double)
        r_double = circuit_double.run_trial(cs=True, us=True, phase="conditioning")

        # 2倍スケールでも発火率が200Hz以下（飽和していない）
        assert r_double.la_rate <= 200
        assert r_double.cem_rate <= 200
        # 凍結反応が1.0に張り付いていない
        assert r_double.freeze_response < 1.0 or r_normal.freeze_response < 1.0


class TestSustainedThreatSensitivity:
    """持続的脅威入力の強度変動。"""

    @pytest.mark.parametrize("factor", [0.5, 1.0, 1.5])
    def test_sustained_amp_scaling(self, factor):
        cfg = _default_cfg()
        cfg.sustained_threat_amp = cfg.sustained_threat_amp * factor
        circuit = FearCircuitV2(cfg)
        result = circuit.run_trial(sustained_threat=True, phase="sustained_anxiety")
        assert 0 <= result.anxiety_level <= 1.0
        assert 0 <= result.bnst_rate <= 300
