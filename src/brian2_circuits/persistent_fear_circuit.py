"""永続型恐怖回路。Brian2 Networkを試行間で保持し、真のSTDP学習を実現する。

従来のFearCircuitV2は毎試行start_scope()でリセットされるため、
STDP重みの試行間引き継ぎが_saved_weightsハックに依存していた。

この実装では:
1. __init__でBrian2 Networkを1回だけ構築
2. 各試行はTimedArrayの入力を差し替えてnetwork.run()を繰り返す
3. STDP重みは自然に蓄積され、消去学習が回路レベルで機能する
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import brian2
brian2.prefs.codegen.target = "numpy"
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, Network,
    TimedArray, ms, start_scope, defaultclock,
)

from src.brian2_circuits.neuron_models import IZH_TIMED_EQS
from src.brian2_circuits.fear_circuit_v2 import FearV2Config, FearV2TrialResult


class PersistentFearCircuit:
    """永続型恐怖回路。STDP重みが試行間で自然に蓄積する。

    消去学習: STDP LTDはevent-drivenトレースだけでは不十分（BA tonic発火で
    post→preタイミングが得られない）。生物学的な「脱増強（depotentiation）」を
    モデル化するため、消去試行ごとにシナプス重みの緩やかな減衰を適用する。
    """

    def __init__(self, config: FearV2Config | None = None,
                 decay_rate: float = 0.05):
        """
        Args:
            config: 回路パラメータ。Noneで較正済みデフォルト。
            decay_rate: 消去時の重み減衰率（試行あたり）。
                        0.0=減衰なし、1.0=1試行で全減衰。
                        デフォルト0.05: 消去10試行で約40%減衰。
        """
        # [F-03修正] CALIBRATED_CONFIGをデフォルトに統一
        from src.calibration.calibrated_configs import CALIBRATED_FEAR_CONFIG
        self.cfg = config or CALIBRATED_FEAR_CONFIG
        self._results: list[FearV2TrialResult] = []
        self._trial_count = 0
        self._extinction_count = 0
        self._decay_rate = decay_rate  # 消去時の重み減衰率

        # Brian2 Networkを1回だけ構築
        start_scope()
        self._build_network()

    def _build_network(self) -> None:
        c = self.cfg
        defaultclock.dt = c.dt_ms * ms

        pops = [
            ("la_exc", c.n_la_exc), ("la_pv", c.n_la_pv), ("la_vip", c.n_la_vip),
            ("ba_exc", c.n_ba_exc), ("cel_som", c.n_cel_som), ("cel_pkcd", c.n_cel_pkcd),
            ("cem", c.n_cem), ("itc", c.n_itc), ("pl", c.n_pl), ("il", c.n_il),
            ("bnst", c.n_bnst),
        ]
        self._total_n = sum(n for _, n in pops)
        self._idx = {}
        off = 0
        for name, n in pops:
            self._idx[name] = (off, off + n)
            off += n

        # 初期ドライブ（後で差し替える）
        n_steps = int(c.duration_ms / c.dt_ms)
        drive = np.full((n_steps, self._total_n), c.bg_noise)
        self._drive_array = drive
        self._I_drive = TimedArray(drive, dt=c.dt_ms * ms)

        # NeuronGroup
        self._G = NeuronGroup(
            self._total_n, IZH_TIMED_EQS,
            threshold="v >= 30", reset="v = c; u += d",
            method="euler", name="pf_neurons",
        )
        rng = np.random.default_rng(12345)
        self._G.v = -65 + rng.normal(0, 2, self._total_n)
        self._G.u = 0.2 * self._G.v[:]
        self._G.tau_inh = 5 * ms  # GABA_A default

        # パラメータ設定
        idx = self._idx
        for name in ["la_exc", "ba_exc", "cem", "pl", "il", "bnst"]:
            s, e = idx[name]
            self._G.a[s:e] = 0.02; self._G.b[s:e] = 0.2; self._G.c[s:e] = -65; self._G.d[s:e] = 8
        s, e = idx["la_pv"]
        self._G.a[s:e] = 0.1; self._G.b[s:e] = 0.2; self._G.c[s:e] = -65; self._G.d[s:e] = 2
        s, e = idx["la_vip"]
        self._G.a[s:e] = 0.02; self._G.b[s:e] = 0.25; self._G.c[s:e] = -65; self._G.d[s:e] = 2
        for name in ["cel_som", "cel_pkcd", "itc"]:
            s, e = idx[name]
            self._G.a[s:e] = 0.02; self._G.b[s:e] = 0.25; self._G.c[s:e] = -65; self._G.d[s:e] = 2

        # CeL PKCd+ 背景低減
        pkcd_s, pkcd_e = idx["cel_pkcd"]

        # シナプス
        G = self._G
        self._synapses = []
        _cid = [0]  # mutable counter for unique names

        def _conn(src, tgt, p, w, inh=False, stdp=False):
            ss, se = idx[src]; ts, te = idx[tgt]
            sgn = -1.0 if inh else 1.0
            uid = _cid[0]; _cid[0] += 1
            if stdp and not inh:
                syn = Synapses(G, G, model="""
                    w : 1
                    dA_ltp/dt = -A_ltp / (100*ms) : 1 (event-driven)
                    dA_ltd/dt = -A_ltd / (100*ms) : 1 (event-driven)
                """,
                on_pre="v_post += w; A_ltp += 0.5; w = clip(w + A_ltd, 0, 15)",
                on_post="A_ltd -= 0.3; w = clip(w + A_ltp, 0, 15)",
                name=f"pfs{uid}")
            else:
                syn = Synapses(G, G, "w : 1", on_pre=f"v_post += {sgn} * w",
                               name=f"pfs{uid}")
            ci, cj = [], []
            conn_rng = np.random.default_rng(12345)
            for i in range(ss, se):
                for j in range(ts, te):
                    if conn_rng.random() < p:
                        ci.append(i); cj.append(j)
            if ci:
                syn.connect(i=ci, j=cj)
                n_src = se - ss
                if n_src >= 20:
                    w_s = w / np.sqrt(max(1, n_src * p))
                else:
                    w_s = w
                syn.w = conn_rng.uniform(0, w_s, len(syn))
            self._synapses.append(syn)
            return syn

        # 恐怖回路結合
        _conn("la_exc", "la_pv", 0.3, 3.0)
        _conn("la_pv", "la_exc", 0.4, 4.0, inh=True)
        _conn("la_vip", "la_pv", 0.5, 5.0, inh=True)
        self._syn_la_ba = _conn("la_exc", "ba_exc", 0.3, 3.0, stdp=True)
        self._syn_la_cel = _conn("la_exc", "cel_som", 0.25, 2.0, stdp=True)
        _conn("ba_exc", "cel_som", 0.15, 1.5)
        _conn("cel_som", "cel_pkcd", 0.7, 8.0, inh=True)
        _conn("cel_pkcd", "cel_som", 0.3, 3.0, inh=True)
        _conn("cel_pkcd", "cem", 0.3, 1.5, inh=True)
        _conn("cel_som", "cem", 0.6, 8.0)
        _conn("ba_exc", "cem", 0.25, 4.0)
        _conn("pl", "la_exc", 0.2, 2.0)
        self._syn_il_itc = _conn("il", "itc", 0.3, 2.5, stdp=True)
        _conn("itc", "cem", 0.5, 5.0, inh=True)
        _conn("ba_exc", "bnst", 0.2, 2.0)
        _conn("cel_som", "bnst", 0.2, 1.5)

        # モニター
        self._mon = SpikeMonitor(G, name="pf_mon")

        # ネットワーク
        self._net = Network(G, self._mon, *self._synapses)

    def run_trial(self, cs=True, us=False, sustained_threat=False,
                  phase="test", trial_num=0) -> FearV2TrialResult:
        """1試行。Networkを再利用するためSTDP重みは自然に蓄積。"""
        c = self.cfg
        idx = self._idx
        n_steps = int(c.duration_ms / c.dt_ms)
        noise_rng = np.random.default_rng(trial_num * 7 + 42)

        # ドライブ再構築
        drive = c.bg_noise + noise_rng.normal(0, c.bg_noise * 0.3, (n_steps, self._total_n))

        cs_start = int(c.cs_onset_ms / c.dt_ms)
        cs_end = int((c.cs_onset_ms + c.cs_dur_ms) / c.dt_ms)
        us_start = int(c.us_onset_ms / c.dt_ms)
        us_end = int((c.us_onset_ms + c.us_dur_ms) / c.dt_ms)

        # PKCd背景低減
        pkcd_s, pkcd_e = idx["cel_pkcd"]
        drive[:, pkcd_s:pkcd_e] *= 0.4

        # BA tonic背景（消去時は段階的に低減→STDP LTD誘導）
        ba_s, ba_e = idx["ba_exc"]
        ba_tonic = 4.0
        if phase == "extinction":
            # 消去が進むとBA駆動が弱まり、post-before-pre方向のLTDが優勢に
            ba_tonic = max(1.0, 4.0 - self._extinction_count * 0.3)
        drive[:, ba_s:ba_e] += ba_tonic

        if cs:
            la_s, la_e = idx["la_exc"]
            drive[cs_start:cs_end, la_s:la_s + c.n_la_exc // 3] += c.cs_amp
            drive[cs_start:cs_end, ba_s:ba_s + c.n_ba_exc // 3] += c.cs_amp * 0.8
            pl_s, pl_e = idx["pl"]
            drive[cs_start:cs_end, pl_s:pl_s + c.n_pl // 4] += 4.0
            il_s, il_e = idx["il"]
            il_base = 4.0
            if phase == "extinction":
                self._extinction_count += 1
                il_base += self._extinction_count * 1.5
            drive[cs_start:cs_end, il_s:il_s + c.n_il // 4] += il_base

        if us:
            la_s, la_e = idx["la_exc"]
            drive[us_start:us_end, la_s:la_e] += c.us_amp
            vip_s, vip_e = idx["la_vip"]
            drive[us_start:us_end, vip_s:vip_e] += 10.0
            som_s, som_e = idx["cel_som"]
            drive[us_start:us_end, som_s:som_e] += c.us_amp * 0.5

        if sustained_threat:
            bnst_s, bnst_e = idx["bnst"]
            drive[:, bnst_s:bnst_e] += c.sustained_threat_amp

        # 新しいTimedArrayを作成（Brian2はvalues書き換えをキャッシュに反映しない場合がある）
        self._I_drive = TimedArray(drive, dt=c.dt_ms * ms)

        # [F-01修正] 膜電位+回復変数をリセット（uドリフト防止）
        self._G.v = -65 + noise_rng.normal(0, 2, self._total_n)
        self._G.u = 0.2 * self._G.v[:]

        # スパイクモニターをリセット
        self._mon.resize(0)

        # 実行（TimedArrayをnamespaceで渡す）
        self._net.run(c.duration_ms * ms, namespace={"I_drive": self._I_drive})

        # 集計
        spk_i = np.array(self._mon.i[:])
        dur_s = c.duration_ms / 1000.0

        def _rate(name):
            s, e = idx[name]; n = e - s
            return int(np.sum((spk_i >= s) & (spk_i < e))) / n / dur_s if n > 0 else 0

        cem_r = _rate("cem")
        bnst_r = _rate("bnst")
        self._trial_count += 1

        # [LTD修正] 消去時の脱増強（depotentiation）
        # 生物学的根拠: USなしのCS提示は、BLA→CeAシナプスの脱増強を誘導する。
        # event-driven STDP LTDだけではBA tonic発火のため減衰が不十分。
        # 試行間の緩やかなシナプス弱体化（depotentiation）を明示的にモデル化。
        if phase == "extinction" and self._decay_rate > 0:
            for syn in [self._syn_la_ba, self._syn_la_cel]:
                if len(syn) > 0:
                    current_w = syn.w[:]
                    # 重みが高いほど減衰量が大きい（乗法的減衰）
                    syn.w = np.clip(current_w * (1.0 - self._decay_rate), 0, 15)

        # STDP重みの追跡
        la_ba_w = float(np.mean(self._syn_la_ba.w[:])) if len(self._syn_la_ba) > 0 else 0

        result = FearV2TrialResult(
            trial_num=trial_num, phase=phase,
            cs_presented=cs, us_presented=us,
            la_rate=_rate("la_exc"), ba_rate=_rate("ba_exc"),
            cel_som_rate=_rate("cel_som"), cel_pkcd_rate=_rate("cel_pkcd"),
            cem_rate=cem_r, pl_rate=_rate("pl"), il_rate=_rate("il"),
            bnst_rate=bnst_r,
            freeze_response=min(1.0, cem_r / 40.0),
            anxiety_level=min(1.0, bnst_r / 30.0),
            cs_la_weight=la_ba_w,
        )
        self._results.append(result)
        return result

    def run_conditioning(self, n=5):
        return [self.run_trial(cs=True, us=True, phase="conditioning", trial_num=i) for i in range(n)]

    def run_extinction(self, n=10):
        offset = self._trial_count
        return [self.run_trial(cs=True, us=False, phase="extinction", trial_num=offset+i) for i in range(n)]

    def run_test(self, n=3):
        offset = self._trial_count
        return [self.run_trial(cs=True, us=False, phase="test", trial_num=offset+i) for i in range(n)]

    @property
    def decay_rate(self) -> float:
        """消去時の重み減衰率。"""
        return self._decay_rate

    @decay_rate.setter
    def decay_rate(self, value: float) -> None:
        """消去時の重み減衰率を設定する。

        Args:
            value: 0.0（減衰なし）〜 1.0（即時消去）。
                   推奨: 0.03-0.10 （消去10-30試行で有意な減衰）。
        """
        self._decay_rate = max(0.0, min(1.0, value))

    @property
    def la_ba_weight(self) -> float:
        return float(np.mean(self._syn_la_ba.w[:])) if len(self._syn_la_ba) > 0 else 0

    @property
    def all_results(self):
        return list(self._results)
