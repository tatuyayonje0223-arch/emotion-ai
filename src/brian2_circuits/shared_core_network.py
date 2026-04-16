"""Shared Core Network — 全情動回路が共有する脳領域のBrian2実装。

文献準拠パラメータ: data/connectivity/literature_circuit_params.yaml
232検証済み論文のDOI+アブストラクト照合済みパラメータを使用。

アーキテクチャ:
  1つのBrian2 NeuronGroup に全共有領域 + 情動固有領域を配置。
  PersistentFearCircuitで実証済みのパターン:
    - IZH_TIMED_EQS（TimedArray駆動）
    - idx辞書でpopulation境界を管理
    - 試行ごとにTimedArray再生成、v/uリセット、STDP重みは保持

共有領域 (19領域, ~312ニューロン):
  PAG (vlPAG + dlPAG)  — 凍結/逃走/攻撃
  BNST                 — 持続不安/CRF
  PVN (CRH + OXT)      — HPA軸/社会結合
  VTA (DA_lat + DA_med + GABA) — 報酬RPE
  NAc (shell_D1 + shell_D2 + core_D1) — 報酬処理
  LC                   — NE覚醒/驚き
  DR                   — 5-HT気分調整
  aIC                  — 内受容/嫌悪
  RMTg                 — GABAergic relay for DA pause (Jhou 2009)
  DRN_GABA             — DRN internal GABA (Challis 2013)
  PPTg                 — VTA DA tonic excitation (Grace 2007)
  dHPC                 — 文脈記憶 (Maren 2001)
  vHPC                 — 不安調節 (Adhikari 2010)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import brian2
brian2.prefs.codegen.target = "numpy"
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, Network,
    TimedArray, ms, start_scope, defaultclock,
)

from src.brian2_circuits.neuron_models import (
    IZH_TIMED_EQS, CELL_TYPES,
    ADEX_TIMED_EQS, ADEX_THRESHOLD, ADEX_RESET, ADEX_CELL_TYPES,
)


# ─── 追加セルタイプ（neuron_models.pyに将来追加予定）─────────────
EXTENDED_CELL_TYPES: dict[str, dict[str, float]] = {
    **CELL_TYPES,
    "OXT_neuron": {"a": 0.02, "b": 0.2, "c": -55, "d": 4},  # Bhatt 2019 Neuron: OXT burst firing → IB-like
    "CRH_neuron": {"a": 0.02, "b": 0.2, "c": -65, "d": 8},
    "5HT_neuron": {"a": 0.02, "b": 0.2, "c": -65, "d": 8},
    "NE_neuron":  {"a": 0.02, "b": 0.2, "c": -65, "d": 10},
    "DA_medial":  {"a": 0.01, "b": 0.2, "c": -65, "d": 10},
}


@dataclass
class SharedCoreConfig:
    """Shared Core Networkの設定。"""

    dt_ms: float = 0.5
    duration_ms: float = 300.0  # 300ms standard (quantization: 1spike/10neurons = 0.33Hz)
    bg_noise: float = 1.7  # 較正済み値と同じ

    # 共有領域のニューロン数 (literature_circuit_params.yaml準拠)
    n_vlpag: int = 25  # increased for finer rate resolution
    n_dlpag: int = 20
    n_bnst: int = 15
    n_pvn_crh: int = 10
    n_pvn_oxt: int = 10
    n_vta_da_lat: int = 30
    n_vta_da_med: int = 10
    n_vta_gaba: int = 15
    n_nac_shell_d1: int = 25
    n_nac_shell_d2: int = 25
    n_nac_core_d1: int = 15
    n_lc: int = 15
    n_dr: int = 15
    n_aic: int = 20
    n_rmtg: int = 10    # Jhou 2009: RMTg GABAergic relay for DA pause
    n_drn_gaba: int = 10  # Challis 2013: DRN internal GABA interneurons
    n_pptg: int = 15     # Grace 2007; Mena-Segovia 2008: PPTg tonic excitation to VTA DA
    n_dhpc: int = 15     # Maren 2001: dorsal hippocampus context encoding
    n_vhpc: int = 12     # Adhikari 2010; Fanselow 2010: ventral hippocampus anxiety modulation

    # Neuron model selection
    use_adex: bool = False  # True: AdEx (Brette & Gerstner 2005), False: Izhikevich (default)


@dataclass
class PopulationDef:
    """Population定義。"""
    name: str
    n: int
    cell_type: str


@dataclass
class CoreTrialResult:
    """Shared Core Network試行結果。"""
    trial_num: int = 0
    phase: str = "test"
    rates: dict[str, float] = field(default_factory=dict)
    total_spikes: int = 0


class SharedCoreNetwork:
    """全情動回路が共有する脳領域のBrian2 Network。

    使い方:
      1. core = SharedCoreNetwork(config)
      2. core.register_population("la_exc", 40, "RS")  # 情動固有領域を追加
      3. core.register_connection("la_exc", "vlpag", 0.15, 3.0)
      4. core.build()  # Brian2 Networkを構築
      5. result = core.run_trial(drive_overrides={...})
    """

    def __init__(self, config: SharedCoreConfig | None = None):
        self.cfg = config or SharedCoreConfig()
        self._built = False
        self._trial_count = 0

        # 共有領域の定義 (YAMLから固定)
        self._shared_pops: list[PopulationDef] = [
            PopulationDef("vlpag", self.cfg.n_vlpag, "RS"),
            PopulationDef("dlpag", self.cfg.n_dlpag, "RS"),
            PopulationDef("bnst", self.cfg.n_bnst, "LTS"),  # Davis 2010: BNST GABAergic LTS neurons
            PopulationDef("pvn_crh", self.cfg.n_pvn_crh, "CRH_neuron"),
            PopulationDef("pvn_oxt", self.cfg.n_pvn_oxt, "OXT_neuron"),
            PopulationDef("vta_da_lat", self.cfg.n_vta_da_lat, "IB"),
            PopulationDef("vta_da_med", self.cfg.n_vta_da_med, "DA_medial"),
            PopulationDef("vta_gaba", self.cfg.n_vta_gaba, "PV"),
            PopulationDef("nac_shell_d1", self.cfg.n_nac_shell_d1, "D1_MSN"),
            PopulationDef("nac_shell_d2", self.cfg.n_nac_shell_d2, "D2_MSN"),
            PopulationDef("nac_core_d1", self.cfg.n_nac_core_d1, "D1_MSN"),
            PopulationDef("lc", self.cfg.n_lc, "NE_neuron"),
            PopulationDef("dr", self.cfg.n_dr, "5HT_neuron"),
            PopulationDef("aic", self.cfg.n_aic, "RS"),
            # Jhou 2009 J Neurosci: RMTg = principal GABAergic afferent to VTA DA
            PopulationDef("rmtg", self.cfg.n_rmtg, "PV"),
            # Challis 2013 J Neurosci: DRN GABA interneurons inhibit 5-HT neurons
            PopulationDef("drn_gaba", self.cfg.n_drn_gaba, "PV"),
            # Grace 2007 Trends Neurosci: PPTg provides tonic glutamatergic drive to VTA DA
            # Mena-Segovia 2008 J Neurosci: PPTg cholinergic/glutamatergic → VTA
            PopulationDef("pptg", self.cfg.n_pptg, "RS"),
            # Hippocampus: shared context memory (Maren 2001; Adhikari 2010; Fanselow 2010)
            PopulationDef("dhpc", self.cfg.n_dhpc, "RS"),   # dorsal HPC: context encoding
            PopulationDef("vhpc", self.cfg.n_vhpc, "RS"),   # ventral HPC: anxiety modulation
        ]

        # 情動固有領域 (register_populationで追加)
        self._extra_pops: list[PopulationDef] = []

        # 接続定義 (register_connectionで追加)
        self._conn_defs: list[dict[str, Any]] = []

        # 共有領域間の内部結合 (文献準拠)
        self._init_shared_connections()

    def _init_shared_connections(self) -> None:
        """共有領域間の文献準拠結合を定義。

        Source: brain_connectome_literature.md Connection Matrix
        """
        # VTA内部
        self._conn_defs.append({"src": "vta_gaba", "tgt": "vta_da_lat", "p": 0.3, "w": 3.0, "inh": True,
                                "note": "GABA→DA inhibition; Cohen 2012 (reduced for burst)"})
        self._conn_defs.append({"src": "vta_gaba", "tgt": "vta_da_med", "p": 0.2, "w": 2.0, "inh": True})

        # VTA → NAc (DA modulation)
        self._conn_defs.append({"src": "vta_da_lat", "tgt": "nac_shell_d1", "p": 0.15, "w": 3.0,
                                "note": "mesolimbic DA; Schultz 1997"})
        self._conn_defs.append({"src": "vta_da_lat", "tgt": "nac_core_d1", "p": 0.10, "w": 2.0})

        # NAc → VTA (feedback inhibition)
        self._conn_defs.append({"src": "nac_shell_d1", "tgt": "vta_gaba", "p": 0.15, "w": 3.0, "inh": True,
                                "note": "NAc→VTA GABA; Haber & Knutson 2010"})

        # LC → 広域 (NE arousal)
        self._conn_defs.append({"src": "lc", "tgt": "aic", "p": 0.10, "w": 2.0,
                                "note": "LC→insula NE; Sara & Bouret 2012"})
        self._conn_defs.append({"src": "lc", "tgt": "bnst", "p": 0.10, "w": 1.5})

        # DR → 広域 (5-HT modulation)
        self._conn_defs.append({"src": "dr", "tgt": "aic", "p": 0.10, "w": 1.5, "inh": True,
                                "note": "5-HT→insula; de Jong 2022 Neuron"})
        self._conn_defs.append({"src": "dr", "tgt": "dlpag", "p": 0.10, "w": 2.0, "inh": True,
                                "note": "5-HT inhibits aggression; de Boer 2009"})

        # RMTg: GABAergic relay for DA pause (Jhou 2009 J Neurosci; Barrot 2012 TINS)
        # tau_inh=10ms (midbrain) doubles g_inh accumulation → weights halved vs 5ms model
        self._conn_defs.append({"src": "rmtg", "tgt": "vta_da_lat", "p": 0.30, "w": 2.5, "inh": True, "shunting": True,
                                "note": "RMTg→VTA DA: principal GABAergic brake; Jhou 2009"})
        self._conn_defs.append({"src": "rmtg", "tgt": "vta_da_med", "p": 0.20, "w": 1.8, "inh": True, "shunting": True,
                                "note": "RMTg→VTA DA medial"})

        # RMTg → PPTg: inhibits PPTg during aversive states (Jhou 2009)
        # PPTg tau_inh=10ms → halved weight vs 5ms model
        self._conn_defs.append({"src": "rmtg", "tgt": "pptg", "p": 0.30, "w": 2.0, "inh": True, "shunting": True,
                                "note": "RMTg→PPTg GABA shunting; Jhou 2009: RMTg inhibits PPTg"})

        # DRN_GABA: internal inhibition of 5-HT (Challis 2013; Varga 2001)
        # DR tau_inh=10ms → halved weight for partial suppression (2-4Hz target)
        self._conn_defs.append({"src": "drn_gaba", "tgt": "dr", "p": 0.40, "w": 1.5, "inh": True, "shunting": True,
                                "note": "DRN GABA→5-HT: ~40% of DRN neurons are GABAergic; Varga 2001"})

        # PPTg → VTA DA: tonic glutamatergic excitation (Grace 2007; Mena-Segovia 2008)
        # PPTg provides tonic excitatory drive maintaining VTA DA tonic firing.
        # VTA DA intrinsic tonic reduced to 1.5; PPTg provides remaining ~1.3 via synaptic excitation.
        # Weight/prob set so that PPTg at ~5-10Hz yields effective ~1.3 additional I to VTA DA.
        self._conn_defs.append({"src": "pptg", "tgt": "vta_da_lat", "p": 0.30, "w": 10.0,
                                "note": "PPTg→VTA DA tonic excitation; Grace 2007 Trends Neurosci"})
        self._conn_defs.append({"src": "pptg", "tgt": "vta_da_med", "p": 0.20, "w": 8.0,
                                "note": "PPTg→VTA DA medial; Mena-Segovia 2008"})

        # BNST → PVN (HPA activation)
        self._conn_defs.append({"src": "bnst", "tgt": "pvn_crh", "p": 0.15, "w": 2.5,
                                "note": "BNST→PVN CRH; Lebow & Chen 2016"})

        # PVN_OXT → BNST (anxiolytic)
        self._conn_defs.append({"src": "pvn_oxt", "tgt": "bnst", "p": 0.10, "w": 2.0, "inh": True,
                                "note": "OXT→BNST anxiolytic; Knobloch 2012"})

    def register_population(self, name: str, n: int, cell_type: str) -> None:
        """情動固有のpopulationを追加登録する。build()前に呼ぶ。"""
        if self._built:
            raise RuntimeError("Cannot register after build()")
        self._extra_pops.append(PopulationDef(name, n, cell_type))

    def register_connection(self, src: str, tgt: str, p: float, w: float,
                            inh: bool = False, stdp: bool = False,
                            shunting: bool = False,
                            note: str = "") -> None:
        """結合を追加登録する。build()前に呼ぶ。"""
        if self._built:
            raise RuntimeError("Cannot register after build()")
        self._conn_defs.append({"src": src, "tgt": tgt, "p": p, "w": w,
                                "inh": inh, "stdp": stdp, "shunting": shunting,
                                "note": note})

    def build(self) -> None:
        """Brian2 Networkを構築する。"""
        if self._built:
            return

        start_scope()
        c = self.cfg
        defaultclock.dt = c.dt_ms * ms

        # 全populationsを統合
        all_pops = self._shared_pops + self._extra_pops
        self._total_n = sum(p.n for p in all_pops)
        self._idx: dict[str, tuple[int, int]] = {}
        off = 0
        for p in all_pops:
            self._idx[p.name] = (off, off + p.n)
            off += p.n

        # TimedArray（初期ドライブ）
        n_steps = int(c.duration_ms / c.dt_ms)
        drive = np.full((n_steps, self._total_n), c.bg_noise)
        self._I_drive = TimedArray(drive, dt=c.dt_ms * ms)

        # NeuronGroup — model selection
        rng = np.random.default_rng(12345)

        if c.use_adex:
            # AdEx model (Brette & Gerstner 2005)
            self._G = NeuronGroup(
                self._total_n, ADEX_TIMED_EQS,
                threshold=ADEX_THRESHOLD, reset=ADEX_RESET,
                method="euler", name="core_neurons",
            )
            self._G.v = -65 + rng.normal(0, 2, self._total_n)
            self._G.w_adex = 0
            self._G.tau_inh = 5 * ms

            # AdEx per-population parameters
            for p in all_pops:
                s, e = self._idx[p.name]
                ct = ADEX_CELL_TYPES.get(p.cell_type)
                if ct is None:
                    ct = ADEX_CELL_TYPES["RS"]  # fallback to RS
                self._G.g_L[s:e] = ct["g_L"]
                self._G.E_L[s:e] = ct["E_L"]
                self._G.dT[s:e] = ct["dT"]
                self._G.V_T[s:e] = ct["V_T"]
                self._G.tau_m[s:e] = ct["tau_m_ms"] * ms
                self._G.a_sub[s:e] = ct["a_sub"]
                self._G.b_spike[s:e] = ct["b_spike"]
                self._G.V_r[s:e] = ct["V_r"]
                self._G.tau_w[s:e] = ct["tau_w_ms"] * ms

                # VTA DA: strong adaptation for tonic ~5Hz (b=8, tw=100ms)
                if p.name == "vta_da_lat":
                    self._G.g_L[s:e] = 0.2
                    self._G.a_sub[s:e] = 0.005
                    self._G.b_spike[s:e] = 9  # balanced: tonic ~3Hz, burst ~32Hz, pause ~1Hz
                    self._G.tau_w[s:e] = 100 * ms
                elif p.name in ("vlpag", "dlpag"):
                    self._G.b_spike[s:e] = 8  # strong adaptation for PAG
        else:
            # Izhikevich model (default)
            self._G = NeuronGroup(
                self._total_n, IZH_TIMED_EQS,
                threshold="v >= 30", reset="v = c; u += d",
                method="euler", name="core_neurons",
            )
            self._G.v = -65 + rng.normal(0, 2, self._total_n)
            self._G.u = 0.2 * self._G.v[:]
            self._G.tau_inh = 5 * ms

            # Izhikevich per-population parameters
            for p in all_pops:
                s, e = self._idx[p.name]
                ct = EXTENDED_CELL_TYPES.get(p.cell_type, CELL_TYPES.get(p.cell_type))
                if ct is None:
                    raise ValueError(f"Unknown cell type: {p.cell_type}")

                if p.name == "vta_da_lat":
                    self._G.a[s:e] = 0.01
                    self._G.b[s:e] = 0.2
                    self._G.c[s:e] = -65
                    self._G.d[s:e] = 10
                elif p.name in ("vlpag", "dlpag"):
                    self._G.a[s:e] = 0.02
                    self._G.b[s:e] = 0.2
                    self._G.c[s:e] = -65
                    self._G.d[s:e] = 12
                elif p.cell_type in ("D1_MSN", "D2_MSN"):
                    self._G.a[s:e] = ct["a"]
                    self._G.b[s:e] = ct["b"]
                    self._G.c[s:e] = -80
                    self._G.d[s:e] = ct["d"]
                else:
                    self._G.a[s:e] = ct["a"]
                    self._G.b[s:e] = ct["b"]
                    self._G.c[s:e] = ct["c"]
                    self._G.d[s:e] = ct["d"]

        # ── Region-specific GABA_A parameters ──
        # Midbrain: slower GABA_A kinetics (Tan et al. 2010 J Physiol)
        for mname in ("vta_da_lat", "vta_da_med", "dr", "pptg"):
            if mname in self._idx:
                ms_, me_ = self._idx[mname]
                self._G.tau_inh[ms_:me_] = 10 * ms  # 10ms midbrain GABA_A
        # E_GABA: -75mV uniform (hardcoded in equations)
        # Literature: -65 to -80mV range; -75 provides best calibration balance

        # Synapses
        self._synapses: list[Synapses] = []
        self._stdp_synapses: dict[str, Synapses] = {}
        conn_rng = np.random.default_rng(12345)
        _uid = [0]

        for cdef in self._conn_defs:
            src_name = cdef["src"]
            tgt_name = cdef["tgt"]

            if src_name not in self._idx or tgt_name not in self._idx:
                continue  # 未登録のpopulationはスキップ

            ss, se = self._idx[src_name]
            ts, te = self._idx[tgt_name]
            sgn = -1.0 if cdef.get("inh") else 1.0
            uid = _uid[0]; _uid[0] += 1
            prob = cdef["p"]
            w_base = cdef["w"]

            # AdEx: per-connection shunting weight scaling
            # AdEx linear leak makes shunting less effective at same weights.
            # CeA circuit needs stronger shunting for disinhibition to work.
            if c.use_adex and cdef.get("shunting"):
                key = f"{src_name}__{tgt_name}"
                # Only CeA shunting needs scaling — RMTg/DRN_GABA already calibrated via tonic
                adex_shunting_scale = {
                    "cel_som__cel_pkcd": 4.0,  # CeA disinhibition: must silence PKCd during CS
                }
                w_base *= adex_shunting_scale.get(key, 1.0)

            if cdef.get("stdp") and not cdef.get("inh"):
                syn = Synapses(self._G, self._G, model="""
                    w : 1
                    dA_ltp/dt = -A_ltp / (100*ms) : 1 (event-driven)
                    dA_ltd/dt = -A_ltd / (100*ms) : 1 (event-driven)
                """,
                on_pre="v_post += w; A_ltp += 0.5; w = clip(w + A_ltd, 0, 15)",
                on_post="A_ltd -= 0.3; w = clip(w + A_ltp, 0, 15)",
                name=f"cs{uid}")
            elif cdef.get("shunting"):
                # True conductance-based (shunting) inhibition
                # (Mitchell & Silver 2003 PNAS; Bartos 2007 Nat Rev Neurosci)
                # Each pre-spike increments g_inh conductance (GABA_A, tau=5ms).
                # I_inh = g_inh * clip(v+75, 0, 200) in neuron equations.
                # NOTE: `inh=True` flag is NOT used here — inhibition is via g_inh
                # conductance (always positive), not via sign inversion of v_post.
                syn = Synapses(self._G, self._G, "w : 1",
                               on_pre="g_inh_post += w",
                               name=f"cs{uid}")
            else:
                syn = Synapses(self._G, self._G, "w : 1",
                               on_pre=f"v_post += {sgn} * w",
                               name=f"cs{uid}")

            # 接続生成
            ci, cj = [], []
            for i in range(ss, se):
                for j in range(ts, te):
                    if conn_rng.random() < prob:
                        ci.append(i); cj.append(j)
            if ci:
                syn.connect(i=ci, j=cj)
                n_src = se - ss
                w_s = w_base / np.sqrt(max(1, n_src * prob)) if n_src >= 20 else w_base
                syn.w = conn_rng.uniform(0, w_s, len(syn))

            self._synapses.append(syn)

            if cdef.get("stdp"):
                key = f"{src_name}__{tgt_name}"
                self._stdp_synapses[key] = syn

        # Monitor
        self._mon = SpikeMonitor(self._G, name="core_mon")

        # Network
        self._net = Network(self._G, self._mon, *self._synapses)
        self._built = True

    def set_tonic_override(self, overrides: dict[str, float]) -> None:
        """tonic driveの外部上書き（SBI較正用）。"""
        self._tonic_overrides = overrides

    def run_trial(self, drive_overrides: dict[str, np.ndarray] | None = None,
                  trial_num: int = 0) -> CoreTrialResult:
        """1試行を実行する。

        Args:
            drive_overrides: {population_name: (n_steps, n_neurons) array}
                             指定された領域のドライブを上書きする。
            trial_num: 試行番号（ノイズシード用）。
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        c = self.cfg
        n_steps = int(c.duration_ms / c.dt_ms)
        noise_rng = np.random.default_rng(trial_num * 7 + 42)

        # ドライブ構築
        drive = c.bg_noise + noise_rng.normal(0, c.bg_noise * 0.3, (n_steps, self._total_n))

        # ── Population-specific tonic drives (literature-based, bg_noise-adjusted) ──
        #
        # Izhikevich (2003) IEEE Trans Neural Networks 14(6):1569-1572
        # RS rheobase ≈ 3.78, LTS rheobase ≈ 2.5, PV/FS rheobase ≈ 3.0
        #
        # bg_noise = 1.7 is already applied to ALL neurons.
        # tonic = target_total_I - bg_noise
        #
        # Target total I by baseline rate:
        #   1-5Hz  → I ≈ 4.0  → tonic = 4.0 - 1.7 = 2.3
        #   3-8Hz  → I ≈ 4.5  → tonic = 4.5 - 1.7 = 2.8
        #   <1Hz   → I ≈ 3.5  → tonic = 3.5 - 1.7 = 1.8
        #   LTS 3-5Hz → I ≈ 2.8 → tonic = 2.8 - 1.7 = 1.1
        #   PV 40-60Hz → I ≈ 5.0 → tonic = 5.0 - 1.7 = 3.3
        #
        tonic_drives = {
            # ── Shared regions ──
            "vta_da_lat": 1.2,       # Grace 2007: reduced intrinsic tonic; PPTg provides remaining excitation
            "vta_da_med": 1.2,       # Same reduction; PPTg→VTA_med compensates
            "vta_gaba": 2.3,         # Cohen 2012: PV type but moderate tonic (I=4.0)
            "bnst": 1.0,             # Davis 2010: LTS rheobase~0, bg_noise alone (I=1.7) → 3-8Hz
            "lc": 2.3,               # Sara & Bouret 2012: tonic 1-3Hz (I=4.0)
            "dr": 1.9,               # de Jong 2022: reduced intrinsic; PL→DR provides remaining excitation (Celada 2001)
            "rmtg": 1.8,             # Jhou 2009: PV rheobase≈3.0, I=3.5 → baseline ~5-8Hz
            "drn_gaba": 1.8,         # Challis 2013: PV type, I=3.5 → baseline ~5-8Hz
            "pptg": 2.3,             # Grace 2007; Mena-Segovia 2004: PPTg tonic 5-10Hz (I=4.0)
            "aic": 2.3,              # Craig 2009: baseline (I=4.0)
            "pvn_crh": 2.3,
            "pvn_oxt": 1.5,           # Bhatt 2019: OXT burst, reduce tonic for IB type
            "nac_shell_d1": 2.3,
            "nac_shell_d2": 2.3,
            "nac_core_d1": 2.3,
            # ── FEAR ──
            "la_exc": 2.0,           # Quirk 2002: baseline 1-5Hz; Izhikevich 2003 rheobase=3.0, I=3.7
            "ba_exc": 2.8,           # Duvarci & Pare 2014: baseline 3-8Hz (I=4.5)
            "cel_som": 1.0,          # CeL SOM+ LTS: rheobase~0, bg_noise alone
            "cel_pkcd": 0.0,         # CeL PKCd+ LTS-like: rheobase~0, bg_noise alone
            "cem": 2.6,              # Ciocchi 2010: baseline 2-5Hz, need input-driven to reach 10Hz              # baseline 2-5Hz (I=4.0)
            "itc": 1.0,              # ITC LTS: rheobase~0
            "pb": 2.3,               # Li 2013: parabrachial nociceptor relay (I=4.0)
            "cel_crf": 0.5,          # CeL CRF+ LTS: low tonic, mainly input-driven (Pomrenze 2015)
            "cel_vip": 0.3,          # VIP: silent baseline, input-gated (McCullough 2018)
            "cea_pv": 2.5,           # PV: above rheobase(≈4.0) for fast inhibition (I=4.2)
            "pl": 2.3,               # Courtin 2014: baseline (I=4.0)
            "il": 2.3,               # Quirk 2002: baseline (I=4.0)
            "la_pv": 3.3,            # PV fast-spiking (I=5.0)
            "la_vip": 2.3,           # VIP: baseline (I=4.0)
            # ── HIPPOCAMPUS ──
            "dhpc": 2.3,             # Maren 2001: baseline context activity (I=4.0)
            "vhpc": 2.0,             # Fanselow 2010: moderate baseline (I=3.7)
            # ── RAGE ──
            "mea": 0.8,              # Hong 2014: LTS rheobase~0, bg_noise alone → 3-8Hz
            "vmh": 1.8,              # Lee 2014: baseline 2-5Hz; I=3.5
            # ── SEEKING ──
            "ofc_reward": 2.3,
            "vmpfc_value": 2.3,
            "vp": 1.0,               # LTS: rheobase~0
            "lhb": 2.3,
            # ── SADNESS ──
            "sgacc": 2.3,            # Mayberg 1999: baseline ~12Hz in healthy
            "habenula": 2.0,         # Matsumoto 2007: baseline, target 10-20Hz with loss drive
            # ── DISGUST ──
            "nts_disgust": 2.3,
            "putamen": 2.3,
            # ── CARE ──
            "mpoa": 2.3,             # Kohl 2018
            "care_bnst": 1.0,        # LTS: rheobase~0
            # ── PANIC/GRIEF ──
            "dacc": 2.3,             # Eisenberger 2003
            "grief_pag": 2.3,
            # ── PLAY ──
            "pfa_thalamus": 2.3,     # Siviy & Panksepp 2011
            "play_cortex": 2.3,
            # ── LUST ──
            "lust_mpoa": 2.3,        # Dominguez & Hull 2005
            "lust_hypo": 2.3,
            # ── SURPRISE ──
            "surprise_amygdala": 2.3, # Sara & Bouret 2012
            "surprise_pfc": 2.3,
        }
        # ── AdEx: per-population calibrated tonic drives ──
        # AdEx linear leak requires different tonic per cell type (no quadratic boost).
        # Each value calibrated from rheobase: g_L*(V_T-E_L) for that cell type.
        # RS(g_L=0.15): rheo=3.0, target 5Hz: I≈4.5 → tonic=2.8
        # LTS(g_L=0.12,V_T=-55): rheo=1.8, target 3-5Hz: I≈2.5 → tonic=0.8
        # PV(g_L=0.12): rheo=2.4, target 20+Hz: I≈4.5 → tonic=2.8
        # IB/DA(g_L=0.2): rheo=4.0, target 5Hz: I≈5.5 → tonic=3.8
        if c.use_adex:
            adex_tonic = {
                # Shared — calibrated per cell type rheobase
                "vta_da_lat": 2.3, "vta_da_med": 2.3,  # IB(g_L=0.2): tonic=3, burst=31, pause=1.4
                "vta_gaba": 2.8,                        # PV
                "bnst": 1.2, "lc": 2.8, "dr": 3.0,
                "rmtg": 2.0, "drn_gaba": 2.0,          # PV: low baseline, habenula-driven
                "pptg": 3.0, "aic": 3.2,
                "pvn_crh": 3.0, "pvn_oxt": 2.5,
                "nac_shell_d1": 4.5, "nac_shell_d2": 4.5, "nac_core_d1": 4.5,  # MSN(g_L=0.18): rheo=5.4
                "dhpc": 3.0, "vhpc": 2.8,
                # FEAR
                "la_exc": 2.0, "ba_exc": 3.2,
                "cel_som": 3.0, "cel_pkcd": 0.0,        # LTS(g_L=0.12,V_T=-55): rheo=1.8, higher for threat response
                "cem": 4.0, "itc": 1.5,
                "pb": 3.0, "cel_crf": 0.5, "cel_vip": 0.3,
                "cea_pv": 3.0, "pl": 3.2, "il": 3.0,
                "la_pv": 3.5, "la_vip": 2.0,
                # RAGE
                "mea": 1.2, "vmh": 2.8,
                # SEEKING
                "ofc_reward": 3.0, "vmpfc_value": 3.0,
                "vp": 1.0, "lhb": 3.0,
                # SADNESS
                "sgacc": 4.5, "habenula": 3.2,
                # DISGUST
                "nts_disgust": 3.0, "putamen": 4.0,
                # CARE
                "mpoa": 3.5, "care_bnst": 1.2,
                # PANIC/GRIEF
                "dacc": 3.0, "grief_pag": 3.0,
                # PLAY/LUST/SURPRISE
                "pfa_thalamus": 3.0, "play_cortex": 3.0,
                "lust_mpoa": 3.0, "lust_hypo": 3.0,
                "surprise_amygdala": 3.0, "surprise_pfc": 3.0,
            }
            for pop_name in list(tonic_drives.keys()):
                if pop_name in adex_tonic:
                    tonic_drives[pop_name] = adex_tonic[pop_name]

        # SBI較正用override適用
        overrides = getattr(self, '_tonic_overrides', {})
        for pop_name, tonic in tonic_drives.items():
            if pop_name in self._idx:
                ps, pe = self._idx[pop_name]
                actual_tonic = overrides.get(pop_name, tonic)
                drive[:, ps:pe] += actual_tonic

        # PAG: input-driven only. bg_noise * 0.2 as before (no tonic drive)
        for pag_name in ["vlpag", "dlpag"]:
            if pag_name in self._idx:
                ps, pe = self._idx[pag_name]
                drive[:, ps:pe] = c.bg_noise * 0.2

        # 情動固有のドライブ上書き
        adex_scale = 1.8 if c.use_adex else 1.0  # scale for emotion-specific overrides
        if drive_overrides:
            for pop_name, override in drive_overrides.items():
                if pop_name in self._idx:
                    s, e = self._idx[pop_name]
                    n_pop = e - s
                    scaled = override * adex_scale
                    if scaled.ndim == 1:
                        drive[:, s:e] += scaled[:n_pop]
                    elif scaled.ndim == 2:
                        drive[:scaled.shape[0], s:s + min(n_pop, scaled.shape[1])] += scaled[:, :n_pop]

        # 新TimedArray (Brian2キャッシュバグ回避)
        self._I_drive = TimedArray(drive, dt=c.dt_ms * ms)

        # v/adaptation/g_inh リセット
        self._G.v = -65 + noise_rng.normal(0, 2, self._total_n)
        if c.use_adex:
            self._G.w_adex = 0
        else:
            self._G.u = 0.2 * self._G.v[:]
        self._G.g_inh = 0  # conductance reset

        # スパイクモニターリセット
        self._mon.resize(0)

        # 実行
        self._net.run(c.duration_ms * ms, namespace={"I_drive": self._I_drive})

        # 集計
        spk_i = np.array(self._mon.i[:])
        dur_s = c.duration_ms / 1000.0
        rates = {}
        for name, (s, e) in self._idx.items():
            n = e - s
            count = int(np.sum((spk_i >= s) & (spk_i < e)))
            rates[name] = count / n / dur_s if n > 0 else 0.0

        self._trial_count += 1

        return CoreTrialResult(
            trial_num=trial_num,
            rates=rates,
            total_spikes=len(spk_i),
        )

    @property
    def idx(self) -> dict[str, tuple[int, int]]:
        """Population名→(start, end)インデックス。"""
        return dict(self._idx)

    @property
    def total_neurons(self) -> int:
        return self._total_n

    @property
    def population_names(self) -> list[str]:
        return list(self._idx.keys())

    @property
    def stdp_synapses(self) -> dict[str, Synapses]:
        return dict(self._stdp_synapses)

    def get_rate(self, name: str, spk_i: np.ndarray, dur_s: float) -> float:
        """指定populationの発火率を計算。"""
        s, e = self._idx[name]
        n = e - s
        count = int(np.sum((spk_i >= s) & (spk_i < e)))
        return count / n / dur_s if n > 0 else 0.0
