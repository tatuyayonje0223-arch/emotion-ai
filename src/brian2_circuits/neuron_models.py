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


# === AdEx (Adaptive Exponential Integrate-and-Fire) 方程式テンプレート ===
# Brette & Gerstner 2005 J Comput Neurosci; Naud et al. 2008 Biol Cybern
# DIMENSIONLESS formulation: tau_m=1ms matches Izhikevich implicit time scale.
# g_L is an abstract leak coefficient (NOT true conductance = C/tau_m).
# Parameters cannot be directly compared to Brette & Gerstner 2005 Table 1.
# Same g_inh conductance mechanism for backward-compatible shunting inhibition.
# Calibration: 9/16 quick targets PASS (iterative tuning in progress).
ADEX_TIMED_EQS = """
    dv/dt = (-g_L*(v - E_L) + g_L*dT*exp(clip((v - V_T)/dT, -10, 10)) - w_adex + I_drive(t, i) - g_inh*clip(v + 75, 0, 200)) / tau_m : 1
    dw_adex/dt = (a_sub*(v - E_L) - w_adex) / tau_w : 1
    dg_inh/dt = -g_inh / tau_inh : 1
    g_L : 1 (constant)
    E_L : 1 (constant)
    dT : 1 (constant)
    V_T : 1 (constant)
    tau_m : second (constant)
    a_sub : 1 (constant)
    b_spike : 1 (constant)
    V_r : 1 (constant)
    tau_w : second (constant)
    tau_inh : second (constant)
"""

ADEX_THRESHOLD = "v >= V_T + 5*dT"
ADEX_RESET = "v = V_r; w_adex += b_spike"

# AdEx cell type parameters (Naud et al. 2008; Brette & Gerstner 2005)
# Calibrated to match Izhikevich voltage scale and approximate firing patterns
ADEX_CELL_TYPES: dict[str, dict[str, float]] = {
    # Calibrated for Izhikevich drive scale (bg_noise=1.7, tonic~2.3)
    # g_L=0.15: I_thresh ≈ g_L*(V_T-E_L) ≈ 0.15*20 = 3.0 → I=4.0 gives ~5Hz tonic
    # tau_m=1ms: matches Izhikevich implicit time scale
    # g_L, E_L, dT, V_T, tau_m(ms), a_sub, b_spike, V_r, tau_w(ms)
    "RS":       {"g_L": 0.15, "E_L": -70, "dT": 2, "V_T": -50, "tau_m_ms": 1.0,
                 "a_sub": 0.01, "b_spike": 5,  "V_r": -58, "tau_w_ms": 200},
    "IB":       {"g_L": 0.15, "E_L": -70, "dT": 2, "V_T": -50, "tau_m_ms": 1.0,
                 "a_sub": 0.005, "b_spike": 3, "V_r": -55, "tau_w_ms": 300},
    "PV":       {"g_L": 0.12, "E_L": -70, "dT": 1, "V_T": -50, "tau_m_ms": 0.5,
                 "a_sub": 0.05, "b_spike": 2, "V_r": -58, "tau_w_ms": 50},
    "LTS":      {"g_L": 0.12, "E_L": -70, "dT": 2, "V_T": -55, "tau_m_ms": 1.0,
                 "a_sub": 0.01, "b_spike": 4, "V_r": -58, "tau_w_ms": 300},
    "SOM":      {"g_L": 0.12, "E_L": -70, "dT": 2, "V_T": -55, "tau_m_ms": 1.0,
                 "a_sub": 0.01, "b_spike": 4, "V_r": -58, "tau_w_ms": 300},
    "CeL_SOM":  {"g_L": 0.12, "E_L": -70, "dT": 2, "V_T": -55, "tau_m_ms": 1.0,
                 "a_sub": 0.01, "b_spike": 4, "V_r": -58, "tau_w_ms": 300},
    "VIP":      {"g_L": 0.12, "E_L": -70, "dT": 2, "V_T": -55, "tau_m_ms": 1.0,
                 "a_sub": 0.01, "b_spike": 4, "V_r": -58, "tau_w_ms": 300},
    "PKCd":     {"g_L": 0.12, "E_L": -70, "dT": 2, "V_T": -55, "tau_m_ms": 1.0,
                 "a_sub": 0.01, "b_spike": 4, "V_r": -58, "tau_w_ms": 300},
    "D1_MSN":   {"g_L": 0.18, "E_L": -80, "dT": 2, "V_T": -50, "tau_m_ms": 1.0,
                 "a_sub": 0.01, "b_spike": 5, "V_r": -70, "tau_w_ms": 200},
    "D2_MSN":   {"g_L": 0.18, "E_L": -80, "dT": 2, "V_T": -50, "tau_m_ms": 1.0,
                 "a_sub": 0.01, "b_spike": 5, "V_r": -70, "tau_w_ms": 200},
    "OXT_neuron": {"g_L": 0.15, "E_L": -70, "dT": 2, "V_T": -50, "tau_m_ms": 1.0,
                   "a_sub": 0.005, "b_spike": 3, "V_r": -55, "tau_w_ms": 200},
    "CRH_neuron": {"g_L": 0.15, "E_L": -70, "dT": 2, "V_T": -50, "tau_m_ms": 1.0,
                   "a_sub": 0.01, "b_spike": 5, "V_r": -58, "tau_w_ms": 200},
    "5HT_neuron": {"g_L": 0.15, "E_L": -70, "dT": 2, "V_T": -50, "tau_m_ms": 1.0,
                   "a_sub": 0.01, "b_spike": 5, "V_r": -58, "tau_w_ms": 200},
    "NE_neuron":  {"g_L": 0.15, "E_L": -70, "dT": 2, "V_T": -50, "tau_m_ms": 1.0,
                   "a_sub": 0.01, "b_spike": 6, "V_r": -58, "tau_w_ms": 200},
    "DA_medial":  {"g_L": 0.15, "E_L": -70, "dT": 2, "V_T": -50, "tau_m_ms": 1.0,
                   "a_sub": 0.005, "b_spike": 6, "V_r": -58, "tau_w_ms": 300},
}


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
