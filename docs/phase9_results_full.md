# Phase 9 Full GoEmotions Validation Result (2026-04-19)

## Summary

Pilot null result (n=38, p=0.07) **statistically confirmed** at larger scale
(n=500, p=0.0003). 821-neuron spiking model provides **no behavioral prediction
value** over keyword matching on 10-emotion classification.

## Method

- Dataset: GoEmotions validation split (full 5426 instances), 500 taken (limit for
  time-bounded first run). Mappable single-label instances.
- Baselines: random / keyword (text_analyzer_v3) / model_rates (IntegratedBrainV2 argmax)
- Metric: accuracy, macro-F1, per-class P/R/F1, McNemar paired test

## Results (n=500)

| Baseline | Accuracy | Macro-F1 |
|----------|---------:|---------:|
| Random (uniform 10-class) | 8.4% | 0.067 |
| **Keyword** | **28.0%** | **0.231** |
| **Model_rates** | **19.2%** | **0.165** |

Note: Both keyword and model accuracy are low (vs 50%+ on curated pilot) due to:
- Class imbalance (CARE 138/500, PANIC_GRIEF 3/500)
- Reddit comment noise (short, slang, sarcasm)
- 10-class task is inherently hard on informal text

## McNemar paired test: model_rates vs keyword

- Model correct / Keyword wrong: **47**
- Model wrong / Keyword correct: **91**
- chi-square = **13.40**
- **p-value = 0.0003** (two-sided)

**Interpretation**: Keyword baseline outperforms model_rates at high statistical
significance. Rejection of `H0: both classifiers have equal error rates` is
robust (would survive Bonferroni correction for up to ~150 comparisons).

## Per-class breakdown (model_rates)

| Label | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| FEAR | 0.33 | 0.50 | 0.40 | 14 |
| RAGE | 0.82 | 0.11 | 0.19 | 82 |
| SEEKING | 0.18 | 0.72 | 0.29 | 90 |
| SADNESS | 0.47 | 0.25 | 0.33 | 36 |
| DISGUST | 0.75 | 0.25 | 0.38 | 12 |
| **CARE** | **0.00** | **0.00** | **0.00** | 138 |
| **PANIC_GRIEF** | **0.00** | **0.00** | **0.00** | 3 |
| **PLAY** | **0.00** | **0.00** | **0.00** | 30 |
| **LUST** | **0.00** | **0.00** | **0.00** | 8 |
| SURPRISE | 0.30 | 0.03 | 0.06 | 87 |

Model gets 0/0 on **4 of 10 classes** (CARE, PANIC_GRIEF, PLAY, LUST).
All 138 true CARE instances misclassified (96 as SEEKING, 31 as LUST).

## Confusion matrix (model_rates)

```
        FEAR RAGE SEEK SADN DISG CARE PANI PLAY LUST SURP
FEAR       7    0    5    0    0    0    0    0    2    0
RAGE       4    9   63    0    0    0    0    0    5    1
SEEKIN     2    1   65    1    0    0    2    0   16    3
SADNES     1    0   20    9    0    0    2    0    4    0
DISGUS     0    0    7    1    3    0    0    0    1    0
CARE       2    0   96    7    0    0    0    0   31    2  ← all 138 misclassified
PANIC_     0    0    3    0    0    0    0    0    0    0  ← all 3 → SEEKING
PLAY       1    0   23    0    0    0    0    0    6    0  ← 23/30 → SEEKING
LUST       0    0    7    0    0    0    0    0    0    1
SURPRI     4    1   66    1    1    0    0    0   11    3  ← 66/87 → SEEKING
```

### Systematic SEEKING over-prediction (confirmed at scale)

Model predicted SEEKING on 355 of 500 instances (71%) despite true SEEKING being
only 90/500 (18%). All social/reward-adjacent scenarios (CARE / LUST / PLAY /
PANIC_GRIEF) collapse into SEEKING via shared VTA DA activation.

This is **not a parameter issue** — it's a **readout function design** issue.
The `seeking_act = VTA DA * 0.4 + NAc * 0.3` readout fires for any emotion
involving reward-like circuit activation, which in biology includes affiliation,
nurturing, and play.

## Pre-registered success criteria check

| Criterion | Target | Result | Pass? |
|-----------|--------|--------|------|
| Model ≥ keyword + 10% | ≥ 38% accuracy | 19.2% accuracy | ❌ |
| McNemar p < 0.05 after Bonferroni | p < 0.001 (α/50) | p = 0.0003 | ✅ significant for null direction |
| Lesioned specificity ±5% | (not run) | — | — |

**Conclusion**: pre-registered hypothesis **rejected with high confidence**.
Neural simulation does not add behavioral predictive value for 10-emotion
classification over keyword matching.

## Caveats

1. **500 / 5426** validation instances (9.2% subsample). Remaining 4926 would
   likely strengthen significance but unlikely to flip direction
2. **Single random seed** (trial_num=0). MC averaging deferred since directional
   result is clear
3. **Readout function** may be suboptimal — could be rewritten to improve but
   that's downstream engineering, not a conceptual fix
4. **Text → drive** translation is crude (keyword hits × 0.4). A trained text
   encoder might do better but defeats the "simple neural interpretability" pitch
5. **No LLM reference ceiling**: GPT-4 zero-shot would give a ceiling estimate
   (likely 50-70%). Deferred pending Anthropic API setup

## Strategic impact

### Path 3c (B2B interpretable AI): **value proposition collapses**

Pre-audit hypothesis: "21 neuron model offers interpretability + accuracy trade-off
over LLM".

Post-audit reality:
- Interpretability: ✅ retained (can trace neural activations)
- Accuracy: ❌ **keyword matching outperforms the model with p=0.0003**

"Interpretable but less accurate than keyword matching" is not a product.

### Path 3d (Education/Demo): **unchanged, still valid**

Demo showing how emotion circuits fire in different scenarios is educational
regardless of downstream classification accuracy. The visualizations showcase
neuroscience literature at appropriate scale.

### Path 3e (Portfolio): **strengthened by null result**

The project now has a clear publishable finding: "Here is the bottom-up
brain-inspired model that sounds promising but empirically underperforms
keyword matching on realistic text." Portfolio value from the **journey** and
**honest reporting**, not from claimed accuracy.

## Next actions

1. ~~Full GoEmotions 500-sample eval~~ **done** (this report)
2. Full 5426 validation eval (overnight run) — optional, result direction unchanged
3. LLM reference ceiling (Claude / GPT-4 zero-shot) — when API configured
4. Lesioned baseline: test circuit specificity (even null result on overall
   accuracy doesn't preclude specific emotions being decodable)
5. Rewrite readout function to reduce SEEKING bias — may marginally improve
   but will not close 9% gap without fundamentally different model
