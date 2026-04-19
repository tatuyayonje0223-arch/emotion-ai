# Phase 9.8 Dimensional Affect Regression (V/A) — first positive angle (2026-04-20)

## Motivation

After Phase 9.7 confirmed **zero unique classification value** (model correct ∩
keyword wrong = 0 instances), dimensional affect offers a different question:

> Does the neural simulation capture **continuous valence and arousal**
> (Russell circumplex) better than keyword matching?

This is scientifically motivated:
- Panksepp/Ekman 10-category model is a discrete abstraction
- Biological affect is better modeled continuously (Russell 1980, Posner 2005)
- If VTA/LC/BNST dynamics correlate with V/A, neural simulation may contribute
- Keyword matching only computes crude positive/negative sentiment + raw arousal

## Method

Ground truth V/A per label: mapping from 10 EmotionAI labels to Russell-style
V/A coordinates (stored in `phase9/dimensional.py` EA_TO_VA). E.g., FEAR =
(-0.8, 0.8); SADNESS = (-0.7, 0.2); PLAY = (0.8, 0.8).

Three baselines:
1. `random_va`: uniform in V:[-1,1], A:[0,1]
2. `keyword_va`: text_analyzer's sentiment_score (V), arousal_estimate (A)
3. `model_va`: IntegratedBrainV2 `EmotionStateV2.valence` and `.arousal`

Metrics: Pearson r, MAE (per dimension), joint R² (combined SSE/SST).

Dataset: GoEmotions validation n=500 single-label mappable (same as 9.4-full).

## Results

| baseline | V Pearson | V MAE | A Pearson | A MAE | Joint R² |
|----------|----------:|------:|----------:|------:|---------:|
| random_va | -0.049 | 0.743 | +0.028 | 0.313 | -0.939 |
| keyword_va | +0.311 | 0.591 | -0.080 | 0.503 | -0.695 |
| **model_va** | **+0.319** | **0.513** | **-0.019** | **0.450** | **-0.445** |

### Model vs Keyword

| Metric | Model | Keyword | Δ (model - keyword) |
|--------|------:|--------:|--------------------:|
| Valence Pearson | +0.319 | +0.311 | +0.008 |
| Valence MAE | 0.513 | 0.591 | **-0.078 (13% better)** |
| Arousal Pearson | -0.019 | -0.080 | +0.060 |
| Arousal MAE | 0.450 | 0.503 | **-0.053 (10% better)** |

**Model beats keyword on all 4 metrics**.

## Interpretation (honest caveats)

### What the result is

- On MAE, model delivers a consistent 10-13% improvement over keyword.
  This is the first clear-cut advantage across any Phase 9 comparison.
- On Pearson, model nominally leads but valence Δ=+0.008 is not statistically
  significant at n=500 (Fisher z-transform SE ≈ 0.064; required Δ ≈ 0.13 for p<0.05).

### What the result is not

- **Joint R² is still negative for all three baselines** (-0.445, -0.695, -0.939).
  This means **all methods, including model, are worse than predicting the mean**.
  Model is "least bad" but not "good".
- **Arousal correlation near zero** (r=-0.019). Model doesn't meaningfully
  capture arousal variance.
- **The advantage is marginal**. Model is ~10% better MAE-wise, but all methods
  produce fundamentally poor dimensional predictions.

### Why this partial success

- Model's `valence` and `arousal` are computed in
  `emotion_circuits_v2.py` as weighted sums of 10 emotion activations using
  Russell-style weights (valence_weights, arousal_weights dicts).
- When a specific emotion circuit fires (e.g., fear_act from threat input), it
  contributes negative valence +0.9 arousal to the integrated readout.
- This mapping is **explicitly designed** into the codebase, not emergent from
  circuit dynamics alone. Keyword matching has no equivalent emotion→V/A layer.
- The model's advantage likely comes from **this post-processing layer**, not
  from the neural simulation per se.

### Why the dimensional finding doesn't flip the overall conclusion

- Phase 9.4: classification null (model 19-22% vs keyword 28%)
- Phase 9.6: lesion specificity for 3 emotions
- Phase 9.7: unique correct = 0 after bias fix
- Phase 9.8: dimensional improvement 10-13% MAE but still poor absolute

Combined: model offers **small improvement on dimensional reporting** but
**no advantage on classification**. The 10 emotion → V/A weights are hand-
designed; extracting V/A directly from keyword analysis + a simple lookup
table would likely match the model without any simulation.

## Implications for positioning

- **Path 3c (B2B interpretable AI)**: dimensional advantage is too small (13%
  MAE) for a product claim. Still not a viable classifier; still not usable
  as a dimensional estimator at this absolute accuracy.
- **Path 3d (Education)**: V/A output is a nice interactive visualization
  alongside 10 emotion activations. The demo UI can now show "valence =
  +0.3, arousal = 0.6" in addition to per-emotion bars.
- **Path 3e (Portfolio)**: "dimensional reporting shows 10-13% improvement
  over keyword matching — but still below mean-prediction baseline. Honest
  null with a footnote."

## Caveats for future work

1. **Ground truth is per-label, not per-instance**. True Warriner 2013 V/A
   norms applied per-word would give a different (more accurate) ground truth.
2. **n=500** — full 5426 validation would strengthen MAE estimates but
   direction unlikely to flip at +0.008 Pearson Δ.
3. **Model's V/A is post-processed** from 10 emotion activations using
   hand-coded weights. A control would be: apply the same weights to
   **keyword emotion activations** (bypass the neural simulation). If that
   control matches model, the simulation adds no value.
4. **Arousal almost random** (r=-0.019). Worth investigating whether model
   arousal correlates at all with any observable text feature.

## Control experiment (future P1)

To disambiguate "does the neural simulation help, or just the V/A weight table":
```python
# Baseline: hybrid_va
hits = text_analyzer.analyze_text(text).features  # 10 emotion keyword hits
emotions = {name.replace('_hits', '').upper(): hits[name] for name in ...}
# Apply same weights as model
valence = sum(emotions[e] * valence_weights[e] for e in ...) / sum(emotions.values())
arousal = sum(emotions[e] * arousal_weights[e] for e in ...) / sum(emotions.values())
```

If hybrid_va ≈ model_va: **neural simulation provides no unique value** even
dimensionally; all advantage came from the weight table on top of keyword.
If hybrid_va < model_va: circuit dynamics genuinely contribute.

This is Phase 9.9 if project continues.
