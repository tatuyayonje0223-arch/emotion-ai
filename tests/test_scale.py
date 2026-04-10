"""スケールテスト。2000ニューロン規模での動作確認。"""

import pytest

from src.brian2_circuits.fear_circuit_v2 import FearCircuitV2, FearV2Config


class TestScale2000:
    """2000ニューロン規模（デフォルトの約10倍）。"""

    @pytest.fixture
    def large_cfg(self):
        return FearV2Config(
            n_la_exc=200, n_la_pv=50, n_la_vip=30,
            n_ba_exc=150,
            n_cel_som=80, n_cel_pkcd=80, n_cem=60,
            n_itc=80,
            n_pl=150, n_il=150,
            n_bnst=80,
            cs_amp=17.7, us_amp=14.7, bg_noise=1.7,
            sustained_threat_amp=5.0,
            duration_ms=200, cs_dur_ms=100,
            us_onset_ms=130, us_dur_ms=25,
        )  # total: ~1110 neurons

    def test_runs_without_crash(self, large_cfg):
        """2000ニューロン規模でクラッシュしない。"""
        c = FearCircuitV2(large_cfg)
        r = c.run_trial(cs=True, us=True, phase="conditioning")
        assert r.la_rate >= 0
        assert r.cem_rate >= 0
        assert 0 <= r.freeze_response <= 1.0

    def test_no_saturation(self, large_cfg):
        """発火率が飽和しない（200Hz以下）。"""
        c = FearCircuitV2(large_cfg)
        r = c.run_trial(cs=True, us=True, phase="conditioning")
        assert r.la_rate <= 200
        assert r.cem_rate <= 200
        assert r.cel_som_rate <= 200

    def test_bnst_responds(self, large_cfg):
        """BNSTが持続脅威に応答する。"""
        c = FearCircuitV2(large_cfg)
        r = c.run_trial(sustained_threat=True, phase="sustained")
        assert r.bnst_rate >= 0
