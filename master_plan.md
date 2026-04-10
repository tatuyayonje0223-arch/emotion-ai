# EmotionAI Master Plan

## Current Phase: Phase 1 — トイモデル探索 (95% complete)

## Active To-Dos

| # | Task | Owner | Status | Due |
|---|------|-------|--------|-----|
| 1 | GPU環境構築(NVIDIA + GeNN) | owner | pending | — |
| 2 | GitHub公開判断 | owner | pending | — |
| 3 | 10Kニューロンスケールアップ | agent(GPU後) | todo | GPU後 |
| 4 | PersistentFearCircuitのLTD実装 | agent | todo | — |

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

## Validation Scores

| Circuit | Score | Key Metric |
|---------|-------|-----------|
| Fear | 0.805 | BLA 9.1Hz, SOM+/PKCd+ 3.28 |
| Reward | 0.864 | DA burst 25Hz (literature match) |
| Stress | 1.000 | All 5 checks PASS |
| Extinction | PASS | 41% reduction |
| **Average** | **0.890** | |

## Phase 2 — スケールアップと検証深化

| Task | Requires | Priority |
|------|----------|----------|
| 10K spiking via Brian2GeNN GPU | NVIDIA GPU | P1 |
| Ciocchi 2010完全再現 | パラメータフィッティング | P2 |
| 論文レベルのベンチマーク比較 | Duggins 2024との照合 | P3 |
| GitHub公開 + 技術レポート | owner判断 | P3 |
