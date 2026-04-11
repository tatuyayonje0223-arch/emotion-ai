# EmotionAI Master Plan

## Current Phase: Phase 2 — 全脳情動モデル V2 (30% complete)

## Active To-Dos

| # | Task | Owner | Status | Due |
|---|------|-------|--------|-----|
| 1 | GPU環境構築(NVIDIA + GeNN) | owner | pending | — |
| 2 | GitHub公開判断 | owner | pending | — |
| 3 | Phase B 5回路のスパイキング化 (GPU後) | agent | todo | GPU後 |
| 4 | SBI較正: RAGE/SADNESS/DISGUST回路 | agent | todo | — |
| 5 | 文献DOI未取得45件の補完 | agent | todo | — |

## Completed (Phase 2)

- [x] 包括的文献リサーチ: 330論文/10情動+コネクトーム — 4/11
- [x] DOI検証: 170 DOI, 94%実在(CrossRef API) — 4/11
- [x] アブストラクト照合: 232件, 70% content MATCH(PubMed) — 4/11
- [x] 文献パラメータDB: literature_circuit_params.yaml(232論文準拠) — 4/11
- [x] パラメータDB独立監査: 5R/18件→CRIT=0/HIGH=0に修正 — 4/11
- [x] Shared Core Network: 14共有領域, 245ニューロン(11テストパス) — 4/11
- [x] Phase A 5スパイキング回路: FEAR/RAGE/SEEKING/SADNESS/DISGUST — 4/11
- [x] Phase B 5 mean-field回路: CARE/PANIC_GRIEF/PLAY/LUST/SURPRISE — 4/11
- [x] 情動間相互作用: 5対の文献準拠クロストーク — 4/11
- [x] 統合readout: valence/arousal/dominance + 10情動活性度 — 4/11
- [x] IntegratedBrainV2: neuromod/sleep/safety/policy統合(12テストパス) — 4/11
- [x] 全テスト86本パス(V2: 48本 + 既存: 38本, 133秒) — 4/11

## Completed (Phase 1)

- [x] エージェントチーム構築(9体) — 4/7
- [x] 抽象パイプライン(MVP) — 4/7
- [x] Wilson-Cowan神経質量モデル — 4/7
- [x] Brian2スパイキング恐怖/報酬/ストレス回路 — 4/7-4/8
- [x] CeA脱抑制(SOM+/PKCd+/CeM) — 4/8
- [x] BNST持続不安 + PL/IL分割 — 4/8
- [x] VTA分割(DA_lat/DA_med/GABA) + NAc Shell/Core — 4/8
- [x] AdEx mean-fieldハイブリッド脳 — 4/8
- [x] 神経修飾(eCB/ACh/シータ/構造的可塑性) — 4/8
- [x] 睡眠リプレイ + EmotionBrain統合 — 4/8
- [x] Brain-LLMブリッジ — 4/8
- [x] Allen Brain Atlas API統合 — 4/8
- [x] 独立監査13ラウンド収束 — 4/8-4/10
- [x] SBI(ABC rejection)パラメータ推定 score=0.928 — 4/9
- [x] DA neuronチューニング burst=25Hz完全一致 — 4/10
- [x] STDP真のLTP動作(PersistentFearCircuit) — 4/10
- [x] 自発的回復シナリオ — 4/10
- [x] 2,775ニューロンスケールテスト — 4/10

## V2 Architecture

| Layer | Components | Neurons |
|-------|-----------|---------|
| Shared Core | PAG(vl/dl), BNST, PVN(CRH/OXT), VTA(DA_lat/DA_med/GABA), NAc(shell_D1/D2, core_D1), LC, DR, aIC | 245 |
| FEAR | LA, BA, CeL(SOM/PKCd), CeM, ITC, PL, IL, LA_PV, LA_VIP | 195 |
| RAGE | MeA, VMH | 45 |
| SEEKING | OFC, vmPFC, VP, LHb | 50 |
| SADNESS | sgACC, Habenula | 35 |
| DISGUST | NTS, Putamen | 25 |
| **Spiking Total** | **34 populations** | **570** |
| Mean-field | CARE(2), PANIC_GRIEF(2), PLAY(2), LUST(2), SURPRISE(2) | 10 regions |

## Validation Scores (V1, preserved)

| Circuit | Score | Key Metric |
|---------|-------|-----------|
| Fear | 0.805 | BLA 9.1Hz, SOM+/PKCd+ 3.28 |
| Reward | 0.864 | DA burst 25Hz (literature match) |
| Stress | 1.000 | All 5 checks PASS |
| Extinction | PASS | 41% reduction |
| **Average** | **0.890** | |

## V2 Test Results (4/11)

| Test Suite | Tests | Status |
|-----------|-------|--------|
| Shared Core Network | 11 | 11/11 PASS |
| EmotionBrainV2 (10 emotions) | 25 | 25/25 PASS |
| IntegratedBrainV2 (E2E) | 12 | 12/12 PASS |
| Existing (fear_v2, emotion_systems, persistent_fear) | 38 | 38/38 PASS |
| **Total** | **86** | **86/86 PASS** |

## Literature Foundation

| Emotion | Verified Papers | Top Paper (score) |
|---------|----------------|-------------------|
| FEAR | 30 | Duvarci & Pare 2014 Neuron (1.00) |
| RAGE | 25 | Golden 2016 Nature (0.93) |
| SEEKING | 23 | Nestler & Carlezon 2006 Biol Psych (1.00) |
| SADNESS | 19 | Hamilton 2015 Biol Psych (0.92) |
| DISGUST | 18 | Small 2003 Neuron (1.00) |
| CARE | 20 | Kirsch 2005 J Neurosci (0.91) |
| PANIC/GRIEF | 21 | Gundel 2003 Am J Psych (0.88) |
| PLAY | 15 | Siviy & Panksepp 2011 Neurosci BB Rev (0.84) |
| LUST | 15 | Dominguez & Hull 2005 Physiol Behav (0.85) |
| SURPRISE | 18 | Sara & Bouret 2012 Neuron (0.88) |
| Connectome | 28 | Kober 2008 NeuroImage (0.95) |
| **Total** | **232** | |

## Phase 3 — スケールアップと検証深化

| Task | Requires | Priority |
|------|----------|----------|
| 10K spiking via Brian2GeNN GPU | NVIDIA GPU | P1 |
| Phase B→スパイキング昇格 | GPU | P1 |
| SBI較正: 新回路(RAGE/SADNESS/DISGUST) | Phase A完了 | P2 |
| Ciocchi 2010完全再現 | パラメータフィッティング | P2 |
| 論文レベルのベンチマーク比較 | Duggins 2024との照合 | P3 |
| GitHub公開 + 技術レポート | owner判断 | P3 |
