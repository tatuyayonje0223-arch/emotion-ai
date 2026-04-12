"""Shared Core Network — 全情動回路が共有する脳領域のBrian2実装。

文献準拠パラメータ: data/connectivity/literature_circuit_params.yaml
232検証済み論文のDOI+アブストラクト照合済みパラメータを使用。

アーキテクチャ:
  1つのBrian2 NeuronGroup に全共有領域 + 情動固有領域を配置。
  PersistentFearCircuitで実証済みのパターン:
    - IZH_TIMED_EQS（TimedArray駆動）
    - idx辞書でpopulation境界を管理
    - 試行ごとにTimedArray再生成、v/uリセット、STDP重みは保持

共有領域 (8領域, ~200ニューロン):
  PAG (vlPAG + dlPAG)  — 凍結/逃走/攻撃
  BNST                 — 持続不安/CRF
  PVN (CRH + OXT)      — HPA軸/社会結合
  VTA (DA_lat + DA_med + GABA) — 報酬RPE
  NAc (shell_D1 + shell_D2 + core_D1) — 報酬処理
  LC                   — NE覚醒/驚き
  DR                   — 5-HT気分調整
  aIC                  — 内受容/嫌悪
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

from src.brian2_circuits.neuron_models import IZH_TIMED_EQS, CELL_TYPES


# ─── 追加セルタイプ（neuron_models.pyに将来追加予定）─────────────
EXTENDED_CELL_TYPES: dict[str, dict[str, float]] = {
    **CELL_TYPES,
    "OXT_neuron": {"a": 0.02, "b": 0.2, "c": -65, "d": 8},  # RS-like for stable rates
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
    n_vlpag: int = 20
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
            PopulationDef("bnst", self.cfg.n_bnst, "RS"),  # RS for stable rates (was LTS)
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
                            note: str = "") -> None:
        """結合を追加登録する。build()前に呼ぶ。"""
        if self._built:
            raise RuntimeError("Cannot register after build()")
        self._conn_defs.append({"src": src, "tgt": tgt, "p": p, "w": w,
                                "inh": inh, "stdp": stdp, "note": note})

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

        # NeuronGroup
        self._G = NeuronGroup(
            self._total_n, IZH_TIMED_EQS,
            threshold="v >= 30", reset="v = c; u += d",
            method="euler", name="core_neurons",
        )
        rng = np.random.default_rng(12345)
        self._G.v = -65 + rng.normal(0, 2, self._total_n)
        self._G.u = 0.2 * self._G.v[:]

        # セルタイプ別パラメータ設定
        for p in all_pops:
            s, e = self._idx[p.name]
            ct = EXTENDED_CELL_TYPES.get(p.cell_type, CELL_TYPES.get(p.cell_type))
            if ct is None:
                raise ValueError(f"Unknown cell type: {p.cell_type}")

            # VTA DA lateral: 較正済みパラメータを使用 (da_neuron_tuning.py)
            if p.name == "vta_da_lat":
                self._G.a[s:e] = 0.01
                self._G.b[s:e] = 0.2
                self._G.c[s:e] = -65
                self._G.d[s:e] = 10
            # PAG: 高d値で連続発火を抑制（u recovery強化）
            elif p.name in ("vlpag", "dlpag"):
                self._G.a[s:e] = 0.02
                self._G.b[s:e] = 0.2
                self._G.c[s:e] = -65
                self._G.d[s:e] = 12  # 高d = 強い適応 → 連続発火を抑制
            # PKCd+: 高aで速い適応、高dでスパイク後回復 → 低頻度発火
            elif p.name == "cel_pkcd":
                self._G.a[s:e] = 0.1   # fast adaptation (like PV)
                self._G.b[s:e] = 0.2
                self._G.c[s:e] = -65
                self._G.d[s:e] = 8     # strong post-spike recovery
            # D1/D2-MSN: 閾値が高い（down-state）→ 追加電流が必要
            elif p.cell_type in ("D1_MSN", "D2_MSN"):
                self._G.a[s:e] = ct["a"]
                self._G.b[s:e] = ct["b"]
                self._G.c[s:e] = -80  # deeper reset for MSN down-state
                self._G.d[s:e] = ct["d"]
            else:
                self._G.a[s:e] = ct["a"]
                self._G.b[s:e] = ct["b"]
                self._G.c[s:e] = ct["c"]
                self._G.d[s:e] = ct["d"]

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

            if cdef.get("stdp") and not cdef.get("inh"):
                syn = Synapses(self._G, self._G, model="""
                    w : 1
                    dA_ltp/dt = -A_ltp / (100*ms) : 1 (event-driven)
                    dA_ltd/dt = -A_ltd / (100*ms) : 1 (event-driven)
                """,
                on_pre="v_post += w; A_ltp += 0.5; w = clip(w + A_ltd, 0, 15)",
                on_post="A_ltd -= 0.3; w = clip(w + A_ltp, 0, 15)",
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

        # ── Population-specific tonic drives (文献準拠) ──
        # Izhikevich RS threshold ~I=4-5 for onset, ~8-10 for 10Hz
        tonic_drives = {
            # 共有領域
            "vta_gaba": 1.54,     # SBI V2 calibrated
            "vta_da_lat": 2.13,   # SBI strict calibrated (typical 5Hz)
            "vta_da_med": 3.0,
            "bnst": 2.0,          # Davis 2010: baseline 3-5Hz (RS now, slight increase)
            "lc": 4.0,            # Sara 2012: tonic 1-3Hz
            "dr": 2.18,           # manual best (DR needs to be suppressible)
            "aic": 3.5,
            "pvn_crh": 3.0,
            "pvn_oxt": 1.5,       # OXT neuron: low tonic (target 3-20Hz with social drive)
            # FEAR: 既存較正値と同等のtonic
            "la_exc": 2.08,       # SBI V2 calibrated (score=0.881)
            "ba_exc": 4.0,
            "cel_som": 0.57,      # SBI strict calibrated
            "cel_pkcd": 0.30,     # minimal tonic, must stay 0-5Hz during CS
            "cem": 3.5,           # baseline 2-5Hz
            "itc": 3.0,
            "pl": 4.0,            # Courtin 2014
            "il": 4.0,            # Quirk 2002
            "nac_shell_d1": 3.5,  # D1-MSN needs more drive (deep reset)
            "nac_shell_d2": 3.0,
            "nac_core_d1": 3.0,
            "la_pv": 5.0,         # PV fast-spiking: higher threshold
            "la_vip": 4.0,
            # RAGE
            "mea": 1.96,          # SBI V2 calibrated
            "vmh": 1.74,          # SBI strict calibrated
            # SEEKING
            "ofc_reward": 4.0,
            "vmpfc_value": 3.5,
            "vp": 4.0,
            "lhb": 3.5,
            # SADNESS
            "sgacc": 3.5,
            "habenula": 3.5,
            # DISGUST
            "nts_disgust": 2.32,  # manual best
            "putamen": 4.0,
            # CARE
            "mpoa": 3.5,
            "care_bnst": 4.0,
            # PANIC
            "dacc": 3.5,
            "grief_pag": 3.0,
            # PLAY
            "pfa_thalamus": 3.5,
            "play_cortex": 3.5,
            # LUST
            "lust_mpoa": 3.0,
            "lust_hypo": 3.0,
            # SURPRISE
            "surprise_amygdala": 3.5,
            "surprise_pfc": 3.5,
        }
        # SBI較正用override適用
        overrides = getattr(self, '_tonic_overrides', {})
        for pop_name, tonic in tonic_drives.items():
            if pop_name in self._idx:
                ps, pe = self._idx[pop_name]
                actual_tonic = overrides.get(pop_name, tonic)
                drive[:, ps:pe] += actual_tonic

        # PAG: 入力駆動のみ。tonic driveを追加しない（上のdictに含めない）
        for pag_name in ["vlpag", "dlpag"]:
            if pag_name in self._idx:
                ps, pe = self._idx[pag_name]
                drive[:, ps:pe] = c.bg_noise * 0.2  # 非常に低い背景のみ

        # 情動固有のドライブ上書き
        if drive_overrides:
            for pop_name, override in drive_overrides.items():
                if pop_name in self._idx:
                    s, e = self._idx[pop_name]
                    n_pop = e - s
                    if override.ndim == 1:
                        drive[:, s:e] += override[:n_pop]
                    elif override.ndim == 2:
                        drive[:override.shape[0], s:s + min(n_pop, override.shape[1])] += override[:, :n_pop]

        # 新TimedArray (Brian2キャッシュバグ回避)
        self._I_drive = TimedArray(drive, dt=c.dt_ms * ms)

        # v/uリセット
        self._G.v = -65 + noise_rng.normal(0, 2, self._total_n)
        self._G.u = 0.2 * self._G.v[:]

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
