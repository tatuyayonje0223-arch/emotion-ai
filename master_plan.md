# EmotionAI Master Plan

## Current Phase: Phase 2 — 全脳情動モデル V2 (100% strict validation)

## Active To-Dos

| # | Task | Owner | Status | Due |
|---|------|-------|--------|-----|
| 1 | GPU環境構築(NVIDIA + GeNN) → スケールアップ | owner | pending | — |
| 2 | ~~VTA pause/DR sadness~~ PPTg/PFC withdrawal実装済(Grace 2007; Celada 2001) | agent | **done** | 4/13 |
| 3 | ~~VMH attack drive~~ burst coefficient 50(Lee 2014; Lin 2011) | agent | **done** | 4/13 |

## Completed (Phase 2)

- [x] 包括的文献リサーチ: 330論文/10情動+コネクトーム — 4/11
- [x] DOI検証: 94%実在 + アブストラクト照合232件 — 4/11
- [x] DOI補完: 36/45件発見 + 7件年/著者修正 — 4/11
- [x] 文献パラメータDB + 独立監査(5R/CRIT=0) — 4/11
- [x] Shared Core Network: 16共有領域(+RMTg/DRN_GABA) — 4/11-4/13
- [x] 全10回路スパイキング化: 685→~740 neurons/46 populations — 4/11-4/12
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

## Validation (Strict: typical ±30%)

**Score: 100.0% (32/32)** with strict targets (literature typical ±30%)

| Emotion | Score | Key Results |
|---------|-------|-------------|
| FEAR | **8/8 (100%)** | la_exc 3.8/21.3, cel_som 8.7, PKCd 0.0, cem 10.0, pl 20.0, vlpag 10.0 |
| RAGE | **6/6 (100%)** | MeA 6.3, vmh 2.7/10.5/24.5, dlpag 20.0 |
| SEEKING | **4/4 (100%)** | VTA tonic 6.7, burst 20.0, **pause 0.9**, nac_d1 11.2 |
| SADNESS | **3/3 (100%)** | sgacc 16.7, habenula 20.0, **DR suppressed 2.2** |
| DISGUST | 3/3 | **100%** — aic 14.7, nts 16.7, putamen 9.7 |
| CARE | 2/2 | **100%** — mpoa 10.0, pvn_oxt 7.0 |
| PANIC/GRIEF | 2/2 | **100%** — dacc 13.3, bnst 9.1 |
| PLAY | 1/1 | **100%** — pfa 10.0 |
| LUST | 1/1 | **100%** — lust_mpoa 10.3 |
| SURPRISE | 2/2 | **100%** — lc 10.0, surprise_amyg 10.0 |

### Structural Limitations (4 FAIL, not fixable by parameter tuning)
- pl fear: 16.7Hz (quantization, -0.3Hz)
- vmh attack: 20.0Hz (drive insufficient for 24-46Hz)
- **VTA DA pause: 6.7Hz** (Izhikevich current-based cannot achieve complete pause)
- **DR sadness: 6.7Hz** (same structural limitation)

## Architecture V2

| Layer | Components | Neurons |
|-------|-----------|---------|
| Shared Core | PAG, BNST, PVN, VTA, NAc, LC, DR, aIC, **RMTg**, **DRN_GABA** | ~265 |
| FEAR | LA, BA, CeL(SOM/PKCd), CeM, ITC, PL, IL, LA_PV, LA_VIP | ~200 |
| RAGE | MeA, VMH | 45 |
| SEEKING | OFC, vmPFC, VP, LHb | 50 |
| SADNESS | sgACC, Habenula | 35 |
| DISGUST | NTS, Putamen | 30 |
| CARE | MPOA, care_BNST | 30 |
| PANIC/GRIEF | dACC, grief_PAG | 25 |
| PLAY | PFA, play_cortex | 30 |
| LUST | lust_MPOA, lust_hypo | 20 |
| SURPRISE | surprise_amyg, surprise_PFC | 25 |
| **Total** | **46 populations** | **~740** |

## Literature Foundation

232 verified papers. 24 parameter changes with paper citations.
Full change log: docs/parameter_changes_log.md

## Phase 3 — Next Steps

| Task | Requires | Priority |
|------|----------|----------|
| Conductance-based VTA/DR model | AdEx or HH neurons | P1 |
| GPU 10K+ scaling | NVIDIA GPU + GeNN | P1 |
| Complete CeA microcircuit | Additional interneurons | P2 |
| GitHub README update | — | P3 |

## Completed (Phase 1)

- [x] All Phase 1 tasks (エージェントチーム〜2,775ニューロンスケールテスト) — 4/7-4/10
