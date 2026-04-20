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

### Phase 9 (新設、Behavioral validation framework) — 初回 null result (4/19)

v3 audit で rate-matching validation の limitations を確定。behavioral prediction へ pivot:
設計doc: `docs/behavioral_validation_framework.md`

| Phase | Task | Status |
|-------|------|--------|
| 9.1 | GoEmotions 27→10 mapping + embedded sample | **done** (4/19) |
| 9.2 | Baseline 実装 (random/keyword/model_rates) | **done** (4/19) |
| 9.3 | Metric 実装 (accuracy/F1/McNemar) | **done** (4/19) |
| 9.4 | Pilot eval on embedded 38 instances | **done** (4/19) |
| 9.4-full | Full GoEmotions n=500 validation subsample | **done** (4/19) — model 19.2% vs keyword 28.0%, McNemar **p=0.0003** |
| 9.5 | Writeup + README update | done for pilot (4/19) |

**Pilot 結果 (embedded n=38)**:
- Random: 15.8% accuracy
- Keyword (text_analyzer_v3): **55.3%**
- Model_rates (IntegratedBrainV2): **36.8%**
- McNemar p=0.070 (small n), keyword beats model by 18.4%

**Full 結果 (GoEmotions validation n=500)** — pilot を統計的に confirm:
- Random: 8.4%, Keyword: **28.0%**, Model_rates: **19.2%**
- McNemar **p=0.0003** (chi2=13.40) — null result 有意
- Model 47 correct / keyword wrong, Model 91 wrong / keyword correct
- Systematic SEEKING over-prediction: 355/500 predictions SEEKING vs 90/500 true
- **Pre-registered hypothesis rejected with high statistical significance**

**戦略影響**:
- Path 3c (B2B interpretable emotion AI) の value proposition 要再考
  "interpretable" は残るが "accurate" は keyword/LLM に劣る
- Path 3d (education/demo) / 3e (portfolio) が主軸

### Phase 9.6 Lesion specificity (2026-04-19)
Overall null result でも circuit-level specificity は部分的に retained:
- FEAR/RAGE/SADNESS: 自 circuit input 削除で accuracy → 0% (specificity ✅)
- SEEKING: lesion でも predict 継続 (readout bias)
- 5 emotions (CARE/LUST/PLAY/PANIC/SURPRISE): baseline 0%、測定不可

Path 3c 再 pivot: "accurate classifier" → "**mechanistic diagnostic model for 3 emotions**"

### Phase 9.7 Readout fix — SEEKING gate + argmax fallback (2026-04-19/20)
- **SEEKING readout は唯一非 gated だった bug** 発見
- Gate 追加 (parity) + argmax 初期値 `best_val=0.0` (tie-break 公平化)
- Scenario MC: **両モデル unchanged** (fix は behavioral layer のみ影響)
- Behavioral: 19.2% → **22.2%** (accuracy 改善)
- BUT: **"model correct ∩ keyword wrong = 0 instances"** — unique predictive value zero
- 結論: 元 19.2% は SEEKING-bias の偶然一致、真の null signal 露出

詳細: `docs/phase9_results_initial.md`, `docs/phase9_results_full.md`, `docs/phase9_lesion_specificity.md`, `docs/phase9_readout_fix.md`

### Phase 9.8 Dimensional V/A regression (2026-04-20)
Classification null の後、連続 V/A での model 性能を検証 (n=500):

| Metric | Model | Keyword | Δ |
|--------|------:|--------:|---:|
| Valence Pearson | +0.319 | +0.311 | +0.008 (非有意) |
| Valence MAE | 0.513 | 0.591 | **-13%** |
| Arousal Pearson | -0.019 | -0.080 | +0.060 |
| Arousal MAE | 0.450 | 0.503 | **-10%** |
| Joint R² | **-0.445** | -0.695 | model least bad |

**初の partial positive**: model が MAE で 10-13% 改善。ただし全 baseline で
R² 負 (mean predictor に劣る)、Pearson Δ は非有意。advantage は hand-coded
emotion→V/A weight table 由来の可能性 → Phase 9.9 で control experiment 必要

詳細: `docs/phase9_dimensional_va.md`

### Phase 9.9 Hybrid control — decisive null (2026-04-20)
Phase 9.8 の "model wins V/A" が weight-table 由来か simulation 由来かの control test。
Hybrid (keyword hits + V/A weights, simulation 無し) を baseline 追加:

| Metric | Hybrid | Model | Gap |
|--------|-------:|------:|----:|
| Arousal Pearson | **+0.272** | -0.019 | hybrid **14×** |
| Arousal MAE | **0.219** | 0.450 | hybrid **-51%** |
| Valence Pearson | +0.303 | +0.319 | ~tie |
| Joint R² | **-0.035** | -0.445 | hybrid 12× |

**Simulation は arousal signal を破壊** → Path 3c classifier/dimensional 用途
**完全に dead**。残存価値は Phase 9.6 lesion specificity のみ。

詳細: `docs/phase9_hybrid_control.md`

### Phase 9.10 Population-level lesion — circuit 直接 silence (2026-04-20)
Phase 9.6 input lesion の caveat「input でなく neuron firing 直接 silence」を実施。
`tonic_overrides = {pop: -10.0}` で readout 全 contributor を hyperpolarize:

| Emotion | Input lesion (9.6) | Pop lesion (9.10) |
|---------|:------------------:|:-----------------:|
| FEAR | ✅ | ❌ 多経路冗長 (LeDoux 2000 と整合) |
| RAGE | ✅ | ✅ |
| **SEEKING** | ❌ readout bias | ✅ **new** |
| SADNESS | ✅ | ✅ |
| 他 5 emotions | untestable (baseline 0%) | untestable |

**4/10 emotion で circuit-level specificity 確立** (RAGE/SEEKING/SADNESS 両 lesion 手法、FEAR input のみ)。
FEAR の多経路冗長性は 生物学的 fear circuit (evolutionary robust) と qualitative 整合。

Path 3c 最終位置: **mechanistic diagnostic for 3-4 specific emotions**

詳細: `docs/phase9_pop_lesion.md`

### Phase 9.11 Reference ceiling — LLM + trained ML (2026-04-20)

| Baseline | n | Accuracy | Macro-F1 |
|----------|--:|---------:|---------:|
| Random | 200 | 10.5% | 0.075 |
| **Gemini 2.5-flash (LLM ceiling)** | 100 | **24.0%** | 0.195 |
| **Keyword argmax** | 200 | **23.0%** | 0.163 |
| Model_rates | 200 | 17.5% | 0.133 |
| LR trained (5000 examples) | 200 | 12.0% | 0.128 |

**発見**:
- LLM ceiling 24% ≈ keyword 23% → **task 自体が inherently hard**
- LR trained が keyword argmax に敗北 → 10 keyword hit features では effective
  decision boundary 学習不可能
- Model (17.5%) は LR と keyword の中間 → simulation は LR より signal 多いが
  keyword 超えず

**Revised narrative**: null finding 維持だが "catastrophically bad" でなく
"inherently hard task" の文脈で model は reasonable middle position

詳細: `docs/phase9_ceiling.md`

### Phase 9.12 Coarse-grained Ekman 6-way (2026-04-20) — positive finding

10-way null が 6-way で消失。Ekman classical 6 emotions に collapse:

| Baseline | Accuracy | Macro-F1 |
|----------|---------:|---------:|
| Random | 24.8% | 0.137 |
| **Keyword** | **36.4%** | 0.355 |
| **Model_rates** | **36.4%** | 0.349 |

**Δ = 0.000** (tied)。10-way gap (-5.5%) は SEEKING/CARE/PLAY/LUST が全て
"joy" に collapse することで cancel out。

Phase 9 final finding triad:
1. Fine-grained 10-way: **null** (model < keyword, p<0.0001)
2. **Coarse-grained 6-way Ekman: tied** (model = keyword)
3. Circuit specificity: 4/10 emotions (lesion 根拠)

Viable positioning: "interpretable coarse emotion classifier" — 6 basic
emotions を keyword 相当精度 + circuit-level interpretability value

詳細: `docs/phase9_coarse_grained.md`

### Phase 9.13 Top-k accuracy (2026-04-20)

| k | Keyword | Model | Δ |
|---|--------:|------:|----:|
| top-1 | 16.6% | 10.8% | -5.8% |
| top-2 | 32.2% | 30.0% | -2.2% |
| top-3 | 44.0% | 41.2% | -2.8% |
| top-5 | 52.8% | 55.2% | -2.4% |

Gap narrows as k 増加 → **ranking quality 近似、argmax commitment のみ弱い**
(注: top-1 numbers が Phase 9.4 より低いのは tie-break fallback 差)

詳細: `docs/phase9_topk.md`

### Phase 9.14 Multi-label evaluation (2026-04-20)

GoEmotions 500 validation の **全 mappable instances** (1-label 419 / 2-label 76 / 3-label 5):

| Metric | Keyword | Model | Δ |
|--------|--------:|------:|----:|
| F1@1 | 0.166 | 0.109 | -0.057 |
| F1@3 | 0.233 | 0.218 | -0.015 |
| F1@5 | 0.208 | 0.201 | **-0.007** |
| hit@5 | 60.6% | 58.6% | -2.0% |

Gap が k 増加で rapidly narrow → **ranking quality ほぼ identical、argmax
のみ弱い**。Multi-label でも null 反転せず。

詳細: `docs/phase9_multilabel.md`

### Parallel: positioning 統合 (3c/3d/3e path)
v3 audit 後の 4/19 セッションで追加:

| Path | 成果物 | Status |
|------|-------|--------|
| 3e Portfolio | `docs/portfolio_article_draft.md` | draft 完成 |
| 3d Education demo | `demo/index.html` (FastAPI + vanilla JS web UI) | 実装完了 |
| 3c B2B interpretable | Phase 9 behavioral validation が前提 | 設計 phase |
| P0 public claim cleanup | README rewrite with honest scoring | 完了 |

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
