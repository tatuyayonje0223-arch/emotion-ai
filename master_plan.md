# EmotionAI Master Plan

## Current Phase: Phase 5 complete — refined honest scoring after v2 audit

### 正直な現状 (2026-04-19 v2 audit後)

| Validation dimension | Izhikevich | AdEx |
|---------------------|-----------|------|
| **Scenario (single-trial, trial_num=0)** | 36/36 | 28/36 |
| **Scenario (MC 5-trial stable PASS)** | **35/36** | **25/36** |
| **Scenario (boundary unstable)** | 1/36 (dr) | 4/36 |
| **Scenario (stable FAIL)** | 0/36 | 7/36 |
| **Baseline physiology (no input)** | 6/20 | 6/20 |

"36/36 STRICT validation"は **single-trial claim であり、MC averaging 下では Izh 35/36**。

## Active To-Dos

| # | Task | Owner | Status | Due |
|---|------|-------|--------|-----|
| 1 | GPU環境構築(NVIDIA + GeNN) → スケールアップ | owner | pending | — |
| 2 | Baseline rate calibration (両モデル6/20 → 目標~18/20) | auto | pending | — |
| 3 | Pop-specific bg_noise（MSN低、LTS中）でbaseline修復 | auto | pending | — |
| 4 | Monte Carlo validation を test suite 統合 | auto | pending | — |

## Completed (Phase 3)

- [x] **Conductance-based GABA_A inhibition (g_inh state variable)** — 4/15
  - dg_inh/dt = -g_inh / (5*ms), I_inh = g_inh * clip(v+75, 0, 200)
  - VTA DA pause: 0.9Hz → **0.3Hz** (true pause)
  - DR suppression: 2.2Hz → **2.4Hz** (stable partial)
  - clip() prevents E_GABA reversal instability in Izhikevich
- [x] **CeA microcircuit expansion (PB + CeL_CRF+)** — 4/15
  - PB (parabrachial): 8 neurons, nociceptor relay (Li 2013 Nat Neurosci)
  - CeL_CRF+: 10 neurons, sustained anxiety (Pomrenze 2015)
  - ~760→~778 neurons, 47→49 populations
- [x] **README updated for V2/V3 achievements** — 4/15
- [x] **Text analyzer keyword expansion (pain/threat)** — 4/15
  - 496/496 tests pass (was 494/496)
- [x] **STRICT 100% (36/36) maintained** — 4/15
- [x] **Independent audit R1-R3: 12 tests added + CRITICAL/HIGH fixes** — 4/16
- [x] **Region-specific GABA_A kinetics (tau_inh parameterization)** — 4/16
  - Midbrain (VTA DA, DR, PPTg): tau_inh=10ms (Tan 2010)
  - Cortical/amygdala: tau_inh=5ms (Bartos 2007)
  - Shunting weights recalibrated for 10ms accumulation
- [x] **CeL_CRF cell type: LTS→RS** (Haubensak 2010) — 4/16
- [x] **Citations corrected** (Pomrenze 2019, Ciocchi 2010 topology) — 4/16

## Completed (Phase 2)

- [x] 包括的文献リサーチ: 330論文/10情動+コネクトーム — 4/11
- [x] DOI検証: 94%実在 + アブストラクト照合232件 — 4/11
- [x] DOI補完: 36/45件発見 + 7件年/著者修正 — 4/11
- [x] 文献パラメータDB + 独立監査(5R/CRIT=0) — 4/11
- [x] Shared Core Network: 16共有領域(+RMTg/DRN_GABA) — 4/11-4/13
- [x] 全10回路スパイキング化: 685→~760 neurons/47 populations — 4/11-4/12
- [x] IntegratedBrainV2: neuromod/sleep/safety/policy統合 — 4/11
- [x] GitHub公開: tatuyayonje0223-arch/emotion-ai — 4/11
- [x] SBI較正(ABC rejection score=0.881) — 4/11
- [x] **数値合わせ全撤去 + 論文準拠リビルド** — 4/12
- [x] **Izhikevich rheobase計算によるtonic drive再設定** — 4/12 (Izhikevich 2003)
- [x] **LTS neuron params: b=0.22, d=4** — 4/13 (Lopez de Armentia 2004)
- [x] **CeA conductance-based (shunting) inhibition** — 4/13 (Chance 2002 PNAS; Li 2013)
- [x] **RMTg GABAergic relay population** — 4/13 (Jhou 2009 J Neurosci)
- [x] **DRN internal GABA interneurons** — 4/13 (Challis 2013 J Neurosci)
- [x] **OXT neuron IB-like burst** — 4/13 (Bhatt 2019 Neuron)
- [x] **LHb phenomenological burst** — 4/13 (Yang 2018 Nature)
- [x] 全テスト55本パス — 4/13
- [x] **PPTg excitatory withdrawal for VTA DA pause** — 4/13 (Grace 2007; Tian 2015)
- [x] **PFC excitatory withdrawal for DR suppression** — 4/13 (Aghajanian 1999; Celada 2001)
- [x] **VMH attack burst coefficient=50** — 4/13 (Lee 2014; Lin 2011)
- [x] **PL fear drive=7.0** — 4/13 (Courtin 2014)
- [x] **STRICT 100% (32/32) validation achieved** — 4/13
- [x] **PPTg spiking population** — 4/14 (Grace 2007; Mena-Segovia 2008)
- [x] **PL→DR + sgACC→PL circuit connections** — 4/14 (Celada 2001; Mayberg 2005)
- [x] **Phenomenological approximations eliminated** — 4/14
- [x] **Phase B targets expanded** (CARE VTA, PANIC CRH, PLAY cortex, LUST VTA) — 4/14
- [x] **STRICT 100% (36/36) all circuit-level** — 4/14
- [x] 全テスト93本パス(V2 55 + V1 38) — 4/14

## Validation (Strict: typical ±30%)

**Score: 100.0% (36/36)** with strict targets (literature typical ±30%)

| Emotion | Score | Key Results |
|---------|-------|-------------|
| FEAR | **8/8 (100%)** | la_exc 4.2/21.2, cel_som 9.2, PKCd 0.0, cem 10.0, pl 19.7, vlpag 10.0 |
| RAGE | **6/6 (100%)** | MeA 5.7/19.5, vmh 2.9/10.9/24.8, dlpag 20.0 |
| SEEKING | **4/4 (100%)** | VTA tonic 6.0, burst 19.8, **pause 0.1**, nac_d1 10.5 |
| SADNESS | **3/3 (100%)** | sgacc 16.7, habenula 20.0, **DR suppressed 3.1** |
| DISGUST | 3/3 | **100%** — aic 15.3, nts 16.3, putamen 9.5 |
| CARE | 3/3 | **100%** — mpoa 10.0, pvn_oxt 7.7, vta 8.4 |
| PANIC/GRIEF | 3/3 | **100%** — dacc 13.3, bnst 9.8, pvn_crh 10.0 |
| PLAY | 2/2 | **100%** — pfa 10.0, play_cortex 13.3 |
| LUST | 2/2 | **100%** — lust_mpoa 10.0, vta 12.6 |
| SURPRISE | 2/2 | **100%** — lc 10.0, surprise_amyg 10.0 |

### Former Structural Limitations — RESOLVED (4/15)
- ~~VTA DA pause: 6.7Hz~~ → **0.3Hz** (conductance-based g_inh inhibition)
- ~~DR sadness: 6.7Hz~~ → **2.4Hz** (conductance-based g_inh inhibition)
- PL fear: 19.7Hz (within strict range [17-33])
- VMH attack: 24.8Hz (within strict range [24-46])

## Architecture V3

| Layer | Components | Neurons |
|-------|-----------|---------|
| Shared Core | PAG, BNST, PVN, VTA, NAc, LC, DR, aIC, RMTg, DRN_GABA, PPTg, **dHPC**, **vHPC** | ~312 |
| FEAR | LA, BA, CeL(SOM/PKCd/CRF/VIP), CeM, ITC, PB, CeA_PV, PL, IL, LA_PV, LA_VIP | ~219 |
| RAGE | MeA, VMH | 45 |
| SEEKING | OFC, vmPFC, VP, LHb | 50 |
| SADNESS | sgACC, Habenula | 35 |
| DISGUST | NTS, Putamen | 30 |
| CARE | MPOA, care_BNST | 30 |
| PANIC/GRIEF | dACC, grief_PAG | 25 |
| PLAY | PFA, play_cortex | 30 |
| LUST | lust_MPOA, lust_hypo | 20 |
| SURPRISE | surprise_amyg, surprise_PFC | 25 |
| **Total** | **53 populations** | **~821** |

## Literature Foundation

232 verified papers. 27 parameter changes with paper citations.
Full change log: docs/parameter_changes_log.md

## Phase 6/7 — Next Steps

### Phase 6 (現行)
| Task | Requires | Priority |
|------|----------|----------|
| GPU 10K+ scaling | NVIDIA GPU + GeNN | P1 (owner) |
| ~~AdEx 28→36/36 force-calibration~~ | ~~accept structural limit, documented~~ | **abandoned** (4/19 audit) |
| ~~FastAPI V2移行~~ | ~~API→IntegratedBrainV2~~ | **done** (4/18) |

### Phase 7 (v2 audit後 新設、Option E: 漸進改善)
| Task | 狙い | 難度 | 状態 |
|------|------|------|------|
| ~~Pop-specific bg_noise (単独)~~ | MSN低noise化でbaseline修復 | 低 | **attempted+reverted (4/19)**: MSN baseline fix alone regresses scenario (nac_d1/putamen fire不足) — 単純化モデルにUP/DOWN state dynamicsが不在。Phase 8で再設計必須 |
| **Monte Carlo validation をCI組込** | seed安定性テスト | 中 | P1 |
| Duration 300ms → 1000ms | 量子化改善 (3.3x)、計算コスト3.3x | 中 | P2 |
| adex_scale=1.8 を population別補正に分散 | AdEx global hackの除去 | 高 | P2 |
| 27 Change の paper-value chain 再検証 | citation 実質性の明記 | 高 | P3 |
| dt 0.5ms → 0.1ms | Euler安定性、計算コスト5x | 低 | P3 |

### Phase 8 (新設、MSN UP/DOWN state dynamics)
Phase 7 P1の試みで判明: K_ir→bg_noise低下だけでは simplified 1-pop-per-region model で scenario firing を保てない。以下が必要:

| Task | 狙い | 難度 |
|------|------|------|
| MSN UP/DOWN state mechanism | K_ir + NMDA-mediated state transition実装 | 高 (model enhancement) |
| Thalamocortical bistable drive | UP state時の task-related depolarization生成 | 高 |
| Wilson 2004ベース較正 | paper-derivable UP-state 20mV ~= X drive units (g_L/tau_m依存) | 中 |

Phase 7 P1実験結果 (4/19):
- bg_noise MSN 1.7→0.2: baseline 6/20→8-9/20 ✓
- scenario MC: Izh 35→32 stable PASS, AdEx 25→23 stable PASS — **net regression**
- Net: +3 baseline PASS, -3 scenario PASS
- 結論: simplified 1-pop モデルは両立不可能。Phase 8の構造的拡張が必須

## Known Structural Limits (documented 2026-04-19 after audit revert)

### AdEx 失敗 8/36 — accepted as structural, not tuned
AdEx linear leak + exponential threshold + w_adex adaptation では以下が達成不可:
- **lc (novelty_burst)** 8-16 Hz: b_spike=6 が強い burst driveを相殺し、trial-avg rate 6.7Hz
- **putamen (disgust_recognition)** 7-13 Hz: MSN g_L=0.18 + b_spike=5 で adaptation dominant
- **habenula (reward_omission)** 10-20 Hz: burst 20*loss で過剰発火 (上限超過)
- **lust→vta** 7-15 Hz: 他scenarioで tuned された drive 構造が lust で vta 過駆動
- **bnst/il/dr/vta_pause**: 各々 0.2-0.3 Hz の境界線近接、但しbaseline physiologyを優先
- 根本原因: Izhikevich quadratic nonlinearity とは **定性的に異なる dynamics**
- 対応方針: AdEx 28/36 を最終状態として accept、dual-model の挙動差異として記録

### Baseline Rate Validation Gap (critical discovery)
- 36 target は scenario-evoked のみ。resting-state baseline を検証しない
- `scripts/validate_baseline_rates.py` で 20 target 計測: **AdEx 6/20 / Izh 6/20**
- Scenario 36/36 PASS は baseline physiology を保証しない
- MSN (putamen/nac) 10Hz baseline vs 文献 <1Hz が最大の violator

## Completed (Phase 5)

- [x] **Full AdEx neuron model** — quick 16/16 + full 28/36 (4/15-17)
- [x] **HPC context memory (dHPC/vHPC)** — 4/16
- [x] **STDP learning rule validation** — 4/16
- [x] **Perception bridge v3** — keyword dedup + compound proximity (4/17-18)
- [x] **SBI optimizer script** — scipy differential_evolution (4/17)
- [x] **Config propagation** — IntegratedBrainV2/LLMBridgeV2 accept SharedCoreConfig (4/18)
- [x] **Chat CLI V2** — 10情動表示 + --adex + /context (4/18)
- [x] **Independent audit** — 6ラウンド、CRITICAL/HIGH全修正 (4/15-18)

## Completed (Phase 4)

- [x] **CeA VIP+/PV+ interneurons** — 4/16
  - CeL_VIP: 8 neurons, disinhibition gain control (McCullough 2018)
  - CeA_PV: 8 neurons, fast feedforward inhibition (Royer 2011)
  - 778→794 neurons, 49→51 populations
- [x] **Behavioral test battery (20 scenarios)** — 4/16
  - 8 test classes: Threat/Reward/Social/Loss/Competing/Baseline/Pain/Play/Surprise/Lust
- [x] **Independent audit R1-R3 (2 rounds)** — 4/16
  - R1: CeA_PV rheobase fix, CeL_VIP input-gating calibration
  - R2: test hardening (or→and, baseline thresholds, coverage 6→10 emotions)
  - R3: neuron counts, parameter change counts, demo script, Royer citation

## Completed (Phase 1)

- [x] All Phase 1 tasks (エージェントチーム〜2,775ニューロンスケールテスト) — 4/7-4/10
