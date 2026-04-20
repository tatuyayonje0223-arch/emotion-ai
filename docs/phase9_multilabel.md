# Phase 9.14 Multi-label Classification (2026-04-20)

## Motivation

GoEmotions instances can have 1-5 labels per text (e.g., "I'm happy but
nervous" = joy + nervousness). Previous Phase 9 restricted to single-label
mappable instances. Multi-label captures realistic emotional ambiguity
and tests model's full ranking capability.

## Method

GoEmotions validation n=500 — all mappable instances (1-label: 419,
2-label: 76, 3-label: 5).

For each baseline:
- `keyword`: rank 10 EmotionAI labels by keyword hit counts
- `model`: rank by IntegratedBrainV2 activations

Metrics per instance (using the instance's EA label **set**):
- **P@k**: |top-k predicted ∩ true set| / k
- **R@k**: |top-k predicted ∩ true set| / |true set|
- **F1@k**: harmonic mean of P@k and R@k
- **hit@k**: 1 if any top-k prediction is in true set else 0

## Results

| Metric | Keyword | Model | Δ (model - keyword) |
|--------|--------:|------:|--------------------:|
| P@1 | 0.180 | 0.118 | -0.062 |
| R@1 | 0.159 | 0.105 | -0.055 |
| F1@1 | 0.166 | 0.109 | -0.057 |
| hit@1 | 0.180 | 0.118 | -0.062 |
| P@2 | 0.174 | 0.166 | -0.008 |
| R@2 | 0.309 | 0.294 | -0.015 |
| F1@2 | 0.219 | 0.208 | -0.011 |
| hit@2 | 0.346 | 0.330 | -0.016 |
| P@3 | 0.165 | 0.153 | -0.011 |
| R@3 | 0.428 | 0.401 | -0.027 |
| F1@3 | 0.233 | 0.218 | -0.015 |
| hit@3 | 0.476 | 0.452 | -0.024 |
| P@5 | 0.131 | 0.126 | -0.005 |
| R@5 | 0.555 | 0.534 | -0.021 |
| F1@5 | 0.208 | 0.201 | -0.007 |
| hit@5 | 0.606 | 0.586 | -0.020 |

## Findings

### 1. Gap narrows dramatically with k

| k | F1 gap | hit gap |
|---|-------:|--------:|
| 1 | -0.057 | -0.062 |
| 2 | -0.011 | -0.016 |
| 3 | -0.015 | -0.024 |
| 5 | -0.007 | -0.020 |

At top-5, F1 gap is 0.007 (3% relative). Model's ranking over 10
emotions carries similar information to keyword's, **except at argmax
commitment**.

### 2. Multi-label doesn't flip the null

The hope was "maybe model captures multi-emotion instances better."
It doesn't — keyword still wins at every k. But the gap is much smaller
at top-3+ than strict top-1.

### 3. hit@5 near 60% for both

At 5-out-of-10 top predictions, both methods capture true emotions for
~60% of instances. Model is 2% lower in absolute terms. This is the
least-bad framing for model performance on this task.

## Integrated Phase 9 picture

| Evaluation | Model | Best reference | Outcome |
|-----------|------:|---------------:|---------|
| 10-way top-1 strict | 17-22% | keyword 23-28% | null (-5.5%) |
| **10-way top-k ranking** | **F1@5 0.201** | **keyword 0.208** | **~tied** |
| **10-way multi-label F1@5** | **0.201** | **0.208** | **~tied (-0.007)** |
| **10-way multi-label hit@5** | **58.6%** | **60.6%** | **~tied (-2.0%)** |
| **6-way coarse Ekman** | **36.4%** | **36.4%** | **tied (0%)** |
| V/A arousal Pearson | -0.019 | hybrid +0.272 | 14× worse |
| V/A valence Pearson | +0.319 | hybrid +0.303 | ≈tied |
| Circuit specificity | 4/10 | — | unique ✅ |

## Final synthesis

**Where model strictly loses**: 10-way fine-grained argmax only
(SEEKING over-prediction bias; readout collapses to single emotion)

**Where model matches keyword**:
- Ranking at top-k (k≥2)
- Multi-label top-k
- Coarse 6-way Ekman

**Where model provides unique value**:
- Circuit-level lesion specificity for FEAR/RAGE/SEEKING/SADNESS

**Where model actively hurts**:
- Arousal prediction (vs hybrid weight table alone, Phase 9.9)

## Caveats

- n=500 single seed
- Multi-label instances are only 16% of subset (1-label dominates)
- Precision@k decreases with k (normal) while recall@k increases —
  F1@k peaks around k=3 for both baselines

## Implications (final)

The neural model's ranking information is nearly equivalent to keyword's
across all granularities except single-prediction argmax. Combined with
circuit-level interpretability, this positions the model as:

**"A 10-dimensional emotion activation layer useful for ranking
(nearly-equivalent to keyword), with bonus causal interpretability for
4 emotion circuits, but not as a strict single-label classifier."**

The architectural commitment to argmax readout is specifically where the
model underperforms; the underlying circuit activations are as
informative as keyword hit counts.
