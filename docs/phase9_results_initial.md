# Phase 9 初回 Behavioral Validation 結果 (2026-04-19)

## 概要

`docs/behavioral_validation_framework.md` 設計の framework で、最初の behavioral validation を実施。
Pre-registered 仮説: **model_rates_classify が keyword baseline を+10% accuracy 以上上回る**。

結果: **仮説棄却**。Model は Keyword より 18.4% 劣った。

## Dataset

- Embedded sample: 40 instances, 38 mappable single-label
- GoEmotions full (58k) はまだ未実行 (`datasets` lib 未インストール、Phase 9.4 の本番)
- 本結果は **小 sample (n=38) pilot**。本番は full GoEmotions で再実行予定

## Class distribution (n=38)

| Label | Count |
|-------|------:|
| FEAR | 4 |
| RAGE | 4 |
| SEEKING | 6 |
| SADNESS | 4 |
| DISGUST | 2 |
| CARE | 4 |
| PANIC_GRIEF | 4 |
| PLAY | 2 |
| LUST | 2 |
| SURPRISE | 6 |

## Results

| Baseline | Accuracy | Macro-F1 |
|----------|---------:|---------:|
| Random (uniform 10-class) | 15.8% | 0.137 |
| Keyword (existing text_analyzer_v3) | **55.3%** | **0.533** |
| Model_rates (IntegratedBrainV2 argmax) | **36.8%** | **0.364** |

### McNemar paired test

Model vs Keyword:
- Model correct, Keyword wrong: 2 instances
- Model wrong, Keyword correct: 9 instances
- chi2 = 3.27, p-value = 0.070

Not significant at α=0.05 but the trend is clear. Full GoEmotions (n~5800 validation split)
would likely push p well below threshold.

### Per-class breakdown (model_rates)

| Label | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| FEAR | 1.00 | 0.75 | 0.86 | 4 |
| RAGE | 1.00 | 0.25 | 0.40 | 4 |
| SEEKING | 0.20 | 0.67 | 0.31 | 6 |
| SADNESS | 0.67 | 0.50 | 0.57 | 4 |
| DISGUST | 1.00 | 1.00 | 1.00 | 2 |
| **CARE** | **0.00** | **0.00** | **0.00** | 4 |
| **PANIC_GRIEF** | **0.00** | **0.00** | **0.00** | 4 |
| **PLAY** | **0.00** | **0.00** | **0.00** | 2 |
| **LUST** | **0.00** | **0.00** | **0.00** | 2 |
| SURPRISE | 1.00 | 0.33 | 0.50 | 6 |

### Confusion matrix (model_rates)

```
        FEAR RAGE SEEK SADN DISG CARE PANI PLAY LUST SURP
FEAR       3    0    1    0    0    0    0    0    0    0
RAGE       0    1    2    0    0    0    0    0    1    0
SEEKIN     0    0    4    0    0    0    0    0    2    0
SADNES     0    0    2    2    0    0    0    0    0    0
DISGUS     0    0    0    0    2    0    0    0    0    0
CARE       0    0    1    0    0    0    0    0    3    0  ← all mispredicted
PANIC_     0    0    3    1    0    0    0    0    0    0  ← 3/4 → SEEKING
PLAY       0    0    1    0    0    0    0    0    1    0
LUST       0    0    2    0    0    0    0    0    0    0
SURPRI     0    0    4    0    0    0    0    0    0    2  ← 4/6 → SEEKING
```

## 観察された系統的バイアス

1. **SEEKING over-prediction**: 18/38 predictions は SEEKING。実際は 6/38 が true SEEKING
   - 原因推定: VTA DA activation は多くの scenario で cascade (CARE で vta_da_lat=7.1Hz,
     LUST で 12.6Hz, PLAY で reward drive, ...)
   - readout function `seeking_act = VTA DA * 0.4 + NAc * 0.3` が他 emotion でも fire する

2. **CARE/LUST/PLAY の完全失敗**: social/reward scenario は全て SEEKING に分類
   - 10 emotion が readout level で十分に differentiate されていない

3. **高 precision populations (FEAR/DISGUST/RAGE)**: 構造が distinct (扁桃体/島皮質/VMH)
   で他 emotion と neural footprint が異なるため keyword baseline と同等

## Pre-registered success criteria 照合

1. ❌ `model_rates_classify >= keyword + 10%` absolute accuracy → **FAILED by -18.4%**
2. △ McNemar p < 0.05 after Bonferroni → p=0.070 (small n=38、本番で検証)
3. — Lesioned model ±5% specificity → 未実施

## 解釈

**Null hypothesis accepted**: The 821-neuron spiking model does NOT add behavioral
prediction value over the text_analyzer_v3 keyword baseline on the 10-emotion
classification task.

Possible reasons:
1. **Text→drive translation is lossy**: keyword hits × 0.4 linear mapping
   is a crude "perception" layer
2. **Readout functions poorly differentiate**: 10 emotions collapse to a
   single winning activation through argmax, ignoring multi-circuit interactions
3. **Rate-matching validation was overfitting**: The 27 parameter changes
   that made scenario targets PASS did not generalize to predicting text-based
   emotion labels
4. **Scale problem**: 821 neurons cannot capture the nuanced emotional distinctions
   in colloquial English Reddit text

## 次ステップ

### 即実施 (現セッション済)
- README に Phase 9 null result 記載
- master_plan.md Phase 9 status 更新

### Phase 9.4 本番 (next session)
- `pip install datasets` → GoEmotions 58k full load
- validation split ~5800 instances で再実行
- n 大きいため McNemar p が significant になる予測 → null result confirm
- Results file: `docs/phase9_results_full.md`

### Phase 9.5 writeup
- このファイル + full result を統合
- Portfolio article (`docs/portfolio_article_draft.md`) に Phase 9 section 追加
- README の validation claim を "rate-matching PASS but no behavioral predictive value"
  と正直に記載

### 戦略方針

**Phase 9 null result は Option 3c (B2B interpretable AI) の path を大きく制約**:
- 「neural simulation で interpretable emotion AI」の value proposition は
  classification task では keyword より劣る
- "Interpretability" は残るが "accuracy" の spread は keyword + LLM より低い
- **商用化 path は Education (3d) / Portfolio (3e) が主軸、3c は再考必要**

### 反省 (自己批判)

当初の Phase 7-8 の「simplified model が baseline と task の両立ができない」構造的発見は、
実はこの Phase 9 null result の前兆だった:
- Rate-matching validation で「27 change で 36/36 PASS」しても、behavior prediction には
  転移しない
- Validation framework として rate-matching を選んだこと自体が**科学的 mistake**だった
  可能性が高い

v1/v2/v3 audit で繰り返し flag した「数値合わせは overfit」の警告が、ここで empirical に
confirm された。
