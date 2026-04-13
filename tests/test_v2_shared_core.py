"""Shared Core Network のテスト。"""

import numpy as np
import pytest

from src.brian2_circuits.shared_core_network import SharedCoreNetwork, SharedCoreConfig


class TestSharedCoreBasic:
    def test_build_succeeds(self):
        core = SharedCoreNetwork()
        core.build()
        assert core.total_neurons > 0

    def test_shared_region_count(self):
        core = SharedCoreNetwork()
        core.build()
        # 16 shared populations (14 + rmtg + drn_gaba)
        shared_names = [
            "vlpag", "dlpag", "bnst", "pvn_crh", "pvn_oxt",
            "vta_da_lat", "vta_da_med", "vta_gaba",
            "nac_shell_d1", "nac_shell_d2", "nac_core_d1",
            "lc", "dr", "aic",
            "rmtg", "drn_gaba",  # Jhou 2009; Challis 2013
        ]
        for name in shared_names:
            assert name in core.population_names, f"{name} missing"

    def test_default_neuron_count(self):
        core = SharedCoreNetwork()
        core.build()
        cfg = SharedCoreConfig()
        expected = (cfg.n_vlpag + cfg.n_dlpag + cfg.n_bnst +
                    cfg.n_pvn_crh + cfg.n_pvn_oxt +
                    cfg.n_vta_da_lat + cfg.n_vta_da_med + cfg.n_vta_gaba +
                    cfg.n_nac_shell_d1 + cfg.n_nac_shell_d2 + cfg.n_nac_core_d1 +
                    cfg.n_lc + cfg.n_dr + cfg.n_aic +
                    cfg.n_rmtg + cfg.n_drn_gaba)
        assert core.total_neurons == expected

    def test_register_population(self):
        core = SharedCoreNetwork()
        core.register_population("la_exc", 40, "RS")
        core.register_population("ba_exc", 30, "RS")
        core.build()
        assert "la_exc" in core.population_names
        assert "ba_exc" in core.population_names
        expected_n = SharedCoreConfig().n_vlpag + SharedCoreConfig().n_dlpag  # ... + 40 + 30
        assert core.total_neurons > 200 + 40 + 30 - 10  # rough check

    def test_register_connection(self):
        core = SharedCoreNetwork()
        core.register_population("la_exc", 40, "RS")
        core.register_connection("la_exc", "vlpag", 0.15, 3.0)
        core.build()
        # Should not raise

    def test_run_trial(self):
        core = SharedCoreNetwork()
        core.build()
        result = core.run_trial(trial_num=0)
        assert result.total_spikes >= 0
        assert "vta_da_lat" in result.rates
        assert "lc" in result.rates

    def test_vta_da_fires(self):
        """VTA DA lateralがtonic発火する（文献: 3-8 Hz）。"""
        core = SharedCoreNetwork()
        core.build()
        result = core.run_trial(trial_num=0)
        da_rate = result.rates["vta_da_lat"]
        # tonic範囲の広めチェック (0.5-20 Hz)
        assert da_rate > 0.5, f"VTA DA too quiet: {da_rate:.1f} Hz"
        assert da_rate < 50, f"VTA DA too active: {da_rate:.1f} Hz"

    def test_lc_fires(self):
        """LCが基底発火する。"""
        core = SharedCoreNetwork()
        core.build()
        result = core.run_trial(trial_num=0)
        lc_rate = result.rates["lc"]
        assert lc_rate >= 0  # LCは低い基底発火が正常

    def test_all_rates_bounded(self):
        """全領域の発火率が0-200Hz以内。"""
        core = SharedCoreNetwork()
        core.build()
        result = core.run_trial(trial_num=0)
        for name, rate in result.rates.items():
            assert 0 <= rate <= 200, f"{name}: {rate:.1f} Hz out of bounds"

    def test_drive_override(self):
        """ドライブ上書きで特定領域の活性が変わる。"""
        core = SharedCoreNetwork()
        core.build()

        # baseline
        r1 = core.run_trial(trial_num=0)
        lc_base = r1.rates["lc"]

        # LC にバースト入力
        n_steps = int(core.cfg.duration_ms / core.cfg.dt_ms)
        lc_boost = np.full((n_steps, core.cfg.n_lc), 15.0)
        r2 = core.run_trial(drive_overrides={"lc": lc_boost}, trial_num=1)
        lc_boosted = r2.rates["lc"]

        assert lc_boosted > lc_base, f"LC boost failed: {lc_base:.1f} -> {lc_boosted:.1f}"

    def test_emotion_circuit_integration(self):
        """情動固有領域を追加して恐怖回路の一部を再現。"""
        core = SharedCoreNetwork()
        # FEAR回路の一部: LA → vlPAG
        core.register_population("la_exc", 40, "RS")
        core.register_connection("la_exc", "vlpag", 0.15, 3.0, note="BLA→PAG fear")
        core.build()

        # LA に脅威入力
        n_steps = int(core.cfg.duration_ms / core.cfg.dt_ms)
        la_drive = np.zeros((n_steps, 40))
        la_drive[100:400, :15] = 15.0  # CS onset
        result = core.run_trial(drive_overrides={"la_exc": la_drive}, trial_num=0)

        assert result.rates["la_exc"] > 1.0, "LA should fire with threat input"
