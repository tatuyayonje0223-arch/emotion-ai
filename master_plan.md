# EmotionAI Master Plan

## Current Phase: Phase 3 — Conductance-based inhibition + CeA expansion (100% strict validation)

## Active To-Dos

| # | Task | Owner | Status | Due |
|---|------|-------|--------|-----|
| 1 | GPU環境構築(NVIDIA + GeNN) → スケールアップ | owner | pending | — |

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
| SEEKING | **4/4 (100%)** | VTA tonic 6.0, burst 19.8, **pause 0.0**, nac_d1 10.5 |
| SADNESS | **3/3 (100%)** | sgacc 16.7, habenula 20.0, **DR suppressed 3.8** |
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
| Shared Core | PAG, BNST, PVN, VTA, NAc, LC, DR, aIC, **RMTg**, **DRN_GABA**, **PPTg** | ~265 |
| FEAR | LA, BA, CeL(SOM/PKCd/**CRF**), CeM, ITC, **PB**, PL, IL, LA_PV, LA_VIP | ~218 |
| RAGE | MeA, VMH | 45 |
| SEEKING | OFC, vmPFC, VP, LHb | 50 |
| SADNESS | sgACC, Habenula | 35 |
| DISGUST | NTS, Putamen | 30 |
| CARE | MPOA, care_BNST | 30 |
| PANIC/GRIEF | dACC, grief_PAG | 25 |
| PLAY | PFA, play_cortex | 30 |
| LUST | lust_MPOA, lust_hypo | 20 |
| SURPRISE | surprise_amyg, surprise_PFC | 25 |
| **Total** | **49 populations** | **~778** |

## Literature Foundation

232 verified papers. 24 parameter changes with paper citations.
Full change log: docs/parameter_changes_log.md

## Phase 4 — Next Steps

| Task | Requires | Priority |
|------|----------|----------|
| GPU 10K+ scaling | NVIDIA GPU + GeNN | P1 (owner) |
| Full AdEx neuron model migration | Replace Izhikevich entirely | P2 |
| Complete CeA: VIP+/PV+ interneurons | Additional cell types | P3 |
| Behavioral test battery | Scenario definitions | P3 |

## Completed (Phase 1)

- [x] All Phase 1 tasks (エージェントチーム〜2,775ニューロンスケールテスト) — 4/7-4/10
