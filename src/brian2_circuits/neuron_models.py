"""Brian2ベースのニューロンモデル定義。

細胞タイプ別のIzhikevichパラメータと方程式テンプレート。
affective-neuroscientistの監査に基づき、情動回路に必須の細胞タイプを網羅:
  - RS (regular spiking): 皮質/BLA興奮性主細胞
  - FS/PV+ (fast spiking parvalbumin): 周細胞体抑制、ガンマ振動
  - SOM+ (somatostatin): 遠位樹状突起抑制、可塑性制御
  - VIP+ (VIP expressing): SOM/PVを抑制する脱抑制回路
  - LTS (low-threshold spiking): ITC介在細胞
  - IB/DA (intrinsically bursting): VTA DAニューロン
  - D1-MSN: NAc直接路（Gs結合）
  - D2-MSN: NAc間接路（Gi結合）
"""

from __future__ import annotations

# Brian2のcodegen warningを抑制
import brian2
brian2.prefs.codegen.target = "numpy"

from brian2 import NeuronGroup, Synapses, SpikeMonitor, StateMonitor, ms, mV, Hz
import numpy as np

# === Izhikevich方程式テンプレート ===

# TimedArray駆動版（v2回路で使用）— conductance-based inhibition対応
# g_inh: GABA_A conductance state variable (Mitchell & Silver 2003 PNAS; Chance 2002 Neuron)
# True shunting inhibition: I_inh = g_inh * (v - E_GABA)
# e_rev: per-neuron E_GABA reversal potential (default -75mV for all regions)
#   Literature range: -65 to -80mV depending on KCC2 expression
#   -75mV provides optimal calibration across emotion circuits
# tau_inh: per-neuron GABA_A decay time constant
#   - Cortical/amygdala: 5ms (Bartos 2007 Nat Rev Neurosci)
#   - Midbrain (VTA DA, DR, PPTg): 10ms (Tan et al. 2010 J Physiol)
# clip(): prevents reversal below E_GABA (Izhikevich dynamics instability)
# g_inh is mathematically non-negative: starts at 0, only receives positive
#   increments (on_pre="g_inh_post += w"), exponential decay preserves sign.
IZH_TIMED_EQS = """
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I_drive(t, i) - g_inh*clip(v + 75, 0, 200)) / ms : 1
    du/dt = (a*(b*v - u)) / ms : 1
    dg_inh/dt = -g_inh / tau_inh : 1
    a : 1 (constant)
    b : 1 (constant)
    c : 1 (constant)
    d : 1 (constant)
    tau_inh : second (constant)
"""

# Default tau_inh value for backward compatibility
DEFAULT_TAU_INH_MS = 5.0  # cortical GABA_A default (Bartos 2007)

# Synaptic入力版（neuron_models.pyのcreate_populationで使用）
IZHIKEVICH_EQS = """
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I_syn + I_ext + I_noise) / ms : 1
    du/dt = (a*(b*v - u)) / ms : 1
    I_syn : 1
    I_ext : 1
    I_noise : 1
    a : 1
    b : 1
    c : 1 (constant)
    d : 1 (constant)
"""

IZHIKEVICH_THRESHOLD = "v >= 30"
IZHIKEVICH_RESET = "v = c; u += d"


# === 細胞タイプ別パラメータ ===

CELL_TYPES: dict[str, dict[str, float]] = {
    # 興奮性
    "RS": {"a": 0.02, "b": 0.2, "c": -65, "d": 8},      # 皮質/BLA主細胞
    "IB": {"a": 0.02, "b": 0.2, "c": -55, "d": 4},      # バースト発火（VTA DA）
    "D1_MSN": {"a": 0.02, "b": 0.2, "c": -65, "d": 8},  # NAc D1-MSN
    "D2_MSN": {"a": 0.02, "b": 0.2, "c": -65, "d": 8},  # NAc D2-MSN

    # 抑制性
    "PV": {"a": 0.1, "b": 0.2, "c": -65, "d": 2},       # parvalbumin fast-spiking
    # Lopez de Armentia 2004: CeL "adapting" type shows strong spike-frequency adaptation
    # Hammack 2007: BNST Type II has low-threshold bursting with adaptation
    # b=0.20 (reduced sensitivity), d=6 (strong adaptation) matches adapting phenotype
    # Lopez de Armentia 2004: CeL "adapting" type. d=4 (moderate adaptation)
    # b=0.22 (between RS 0.20 and standard LTS 0.25)
    "SOM": {"a": 0.02, "b": 0.22, "c": -65, "d": 4},    # somatostatin — adapted
    "VIP": {"a": 0.02, "b": 0.22, "c": -65, "d": 4},    # VIP
    "LTS": {"a": 0.02, "b": 0.22, "c": -65, "d": 4},    # low-threshold (ITC)
    "PKCd": {"a": 0.02, "b": 0.22, "c": -65, "d": 4},   # CeL PKCdelta+
    "CeL_SOM": {"a": 0.02, "b": 0.22, "c": -65, "d": 4}, # CeL SOM+
}


def create_population(
    name: str,
    n: int,
    cell_type: str,
    noise_std: float = 1.0,
) -> NeuronGroup:
    """指定細胞タイプのNeuronGroupを作成する。"""
    params = CELL_TYPES[cell_type]
    rng = np.random.default_rng(hash(name) % 2**32)

    group = NeuronGroup(
        n, IZHIKEVICH_EQS,
        threshold=IZHIKEVICH_THRESHOLD,
        reset=IZHIKEVICH_RESET,
        method="euler",
        name=name,
    )
    group.v = -65 + rng.normal(0, 2, n)
    group.u = params["b"] * group.v[:]
    group.a = params["a"] * (1 + rng.normal(0, 0.05, n))
    group.b = params["b"] * (1 + rng.normal(0, 0.05, n))
    group.c = params["c"] + rng.normal(0, 1, n)
    group.d = params["d"] * (1 + rng.normal(0, 0.05, n))
    group.I_ext = 0
    group.I_syn = 0
    group.I_noise = noise_std * rng.standard_normal(n)

    return group


# === STDP + 報酬変調シナプスモデル ===

def create_synapses(
    source: NeuronGroup,
    target: NeuronGroup,
    conn_prob: float = 0.2,
    w_init: float = 2.0,
    w_max: float = 10.0,
    is_inhibitory: bool = False,
    plasticity: bool = False,
    name: str = "syn",
) -> Synapses:
    """シナプス結合を作成する。

    plasticity=True: STDP + 適格性トレース + 報酬変調
    Brian2の予約語(_pre/_post サフィックス)を回避した変数名を使用。
    """
    if plasticity:
        syn_sign_val = -1.0 if is_inhibitory else 1.0
        syn = Synapses(
            source, target,
            model="""
                w : 1
                dA_ltp/dt = -A_ltp / tau_ltp : 1 (event-driven)
                dA_ltd/dt = -A_ltd / tau_ltd : 1 (event-driven)
                delig/dt = -elig / tau_el : 1 (clock-driven)
                tau_ltp : second (constant)
                tau_ltd : second (constant)
                tau_el : second (constant)
                amp_ltp : 1 (constant)
                amp_ltd : 1 (constant)
                wmax : 1 (constant)
                ssign : 1 (constant)
            """,
            on_pre="I_syn_post += ssign * w; A_ltp += amp_ltp; elig += A_ltd",
            on_post="A_ltd += amp_ltd; elig += A_ltp",
            method="euler",
            name=name,
        )
    else:
        syn_sign = -1.0 if is_inhibitory else 1.0
        syn = Synapses(
            source, target,
            model="w : 1",
            on_pre=f"I_syn_post += {syn_sign} * w",
            name=name,
        )

    syn.connect(p=conn_prob)

    rng = np.random.default_rng(hash(name) % 2**32)
    syn.w = rng.uniform(0, w_init, len(syn))

    if plasticity:
        syn.tau_ltp = 20 * ms
        syn.tau_ltd = 20 * ms
        syn.tau_el = 1000 * ms
        syn.amp_ltp = 0.005
        syn.amp_ltd = -0.005
        syn.wmax = w_max
        syn.ssign = syn_sign_val

    return syn


def apply_da_modulation(syn: Synapses, da_signal: float) -> None:
    """報酬変調: DA信号で適格性トレースを重みに反映する。"""
    if hasattr(syn, 'elig'):
        dw = da_signal * syn.elig[:]
        new_w = syn.w[:] + dw
        syn.w = np.clip(new_w, 0, syn.wmax[:] if hasattr(syn, 'wmax') else 10.0)
        syn.elig = syn.elig[:] * 0.5
