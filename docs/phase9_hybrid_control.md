# Phase 9.9 Hybrid Control Experiment — decisive null (2026-04-20)

## Pre-registered hypothesis

Phase 9.8 found model_va beats keyword_va on V/A MAE (13%/10%). Two possible
explanations:

A. **Neural simulation contributes**: circuit dynamics produce V/A signal
   beyond what keyword alone reveals
B. **Hand-coded weight table does all the work**: the `valence_weights` and
   `arousal_weights` dicts applied to the 10 emotion activations explain the
   advantage; simulation adds nothing

The control: apply the SAME weights to keyword hit counts (bypass simulation
entirely). If hybrid ≈ model → explanation B is correct.

## Method

Added `hybrid_va_baseline()` in `phase9/dimensional.py`:
- Reads keyword hits from `text_analyzer_v3.analyze_text()`
- Treats each `{emotion}_hits` count as emotion activation
- Applies the exact same `_VALENCE_WEIGHTS` / `_AROUSAL_WEIGHTS` (synced with
  model's `emotion_circuits_v2.py` lines 907/916)
- NO neural simulation called

Ran against GoEmotions validation n=500 (same subset as Phase 9.4-full and 9.8).

## Results

| Baseline | V Pearson | V MAE | A Pearson | A MAE | Joint R² |
|----------|----------:|------:|----------:|------:|---------:|
| random_va | -0.049 | 0.743 | +0.028 | 0.313 | -0.939 |
| keyword_va | +0.311 | 0.591 | -0.080 | 0.503 | -0.695 |
| **hybrid_va** | **+0.303** | **0.531** | **+0.272** | **0.219** | **-0.035** |
| model_va | +0.319 | 0.513 | -0.019 | 0.450 | -0.445 |

## Findings

### 1. Arousal: hybrid massively beats model

| Metric | Hybrid | Model | Gap |
|--------|-------:|------:|----:|
| Arousal Pearson | **+0.272** | -0.019 | **hybrid 14× higher** |
| Arousal MAE | **0.219** | 0.450 | **hybrid 51% better** |

The neural simulation is not just failing to help — it is **actively destroying
the arousal signal** that the weight table extracts cleanly from keyword counts.

### 2. Valence: hybrid and model tied

| Metric | Hybrid | Model | Gap |
|--------|-------:|------:|----:|
| Valence Pearson | +0.303 | +0.319 | model +0.016 |
| Valence MAE | 0.531 | 0.513 | model -0.018 |

Model's valence advantage vanishes once the fair control is applied. The
remaining +0.016 Pearson Δ is within random variation.

### 3. Joint R²: hybrid is the only model-like baseline competitive with mean

| Baseline | Joint R² |
|----------|---------:|
| random_va | -0.939 |
| keyword_va | -0.695 |
| **hybrid_va** | **-0.035** (nearly as good as always predicting the mean) |
| model_va | -0.445 |

Hybrid is the first method to come within noise of the trivial mean-prediction
baseline. The neural simulation is notably worse.

## Why does the simulation hurt?

Reverse-engineering the result:

1. Model's 10 emotion activations are input-gated: `if threat>0.1: fear_act = ...
   else: fear_act = 0`
2. For typical GoEmotions text, 0-2 emotions trigger gates; the other 8+ are 0
3. V/A = weighted sum over activations / total activation → dominated by the
   single triggered emotion
4. When `total_act` is small or only one emotion fires, V/A collapses to that
   emotion's fixed weight (noisy binary signal)

Hybrid:
1. Keyword hits are continuous counts across 10 emotions (often multi-hit)
2. Sum over all 10 with weights → smoother, higher-variance V/A
3. Result: smoother V/A = better fit to continuous ground truth

The neural simulation's gated readout **coarsens** the information that keyword
counts already carry.

## Overall Phase 9 narrative

| Sub-phase | Finding |
|-----------|---------|
| 9.1-9.3 | Framework + baselines |
| 9.4 pilot | Model 37% vs keyword 55% (n=38, p=0.07) |
| 9.4 full | Model 19% vs keyword 28% (n=500, p=0.0003) |
| 9.6 lesion | FEAR/RAGE/SADNESS circuits causally necessary (specificity) |
| 9.7 readout fix | After bias correction: model unique correct = 0 |
| 9.8 V/A | Model wins V/A MAE by 10-13% (seemed positive) |
| **9.9 control** | **Hybrid beats model dramatically on arousal; valence tied.** |

**Final**: The 821-neuron spiking simulation provides:
- ❌ No classification value (strictly subset of keyword correct)
- ❌ No unique dimensional value (hybrid weight table alone beats it on arousal)
- ✅ Circuit specificity for FEAR/RAGE/SADNESS only
- ✅ Visualization / educational value independent of accuracy

## Strategic positioning — final

| Path | Status |
|------|--------|
| 3a (biometric emotion recognition) | Not feasible |
| 3b (chatbot) | LLM dominates |
| **3c (B2B interpretable AI)** | **Dead as classifier / dimensional estimator**. Only viable as "mechanistic diagnostic model for 3 emotions" + "educational explainability layer on top of LLM" |
| 3d (education/demo) | ✅ retained — visualize circuits + V/A + lesion effects |
| 3e (portfolio) | ✅ strengthened — decisive null result + clean control experiment is publishable contribution |

## Scientific contribution this project offers (honestly)

1. **Demonstration that rate-matching validation doesn't predict behavior**
   (27 parameter changes → 36/36 scenario PASS → 0 unique behavioral correct)
2. **Circuit specificity evidence** for FEAR/RAGE/SADNESS drive lesions
3. **Clean control experiment showing weight-table dominates**: neural simulation
   coarsens dimensional affect signal vs direct keyword→weight application
4. **Chain of independent self-audits** documenting overfit → null → control
   pipeline as a case study in scientific integrity

## Caveats

- Ground truth V/A is per-label, not per-instance (Warriner norms would be more
  granular but require per-word processing)
- n=500 subset of 5426 validation split
- Single random seed
- The "model actively hurts" finding is specific to this readout design;
  different readouts (softmax, trained head) might capture more simulation
  signal — but that would move away from the "pure interpretable circuit" pitch
