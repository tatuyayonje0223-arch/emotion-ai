# Behavioral Validation Framework (Phase 9+ design doc)

## Motivation

Current validation (`scripts/optimize_adex.py` + `quantitative_targets_v2.py`) is **rate-matching**:
"Under condition X, does population Y fire at Z Hz?"

v2/v3 audit concluded this has **no predictive power**. Passing 36/36 tells us the model reproduces
descriptive statistics of published studies for populations we chose. It does NOT tell us:
- Does the model predict emotion-laden behavior?
- Does the model generalize to unseen scenarios?
- Does it match human behavioral data beyond neural rates?

Behavioral validation tests **functional correctness** of the circuit output rather than neural activity.

## Design principles

1. **Prediction-first**: evaluate model against **held-out** behavioral data, not training/tuning data
2. **Latent → observable**: neural rates are latent; behavior is observable. Only observable matters for correctness
3. **Compare against baselines**: no-model random + LLM-only + lesion variants (remove specific circuits)
4. **Dimension-aware**: report accuracy + confusion matrix + precision-recall per emotion, not single-number score
5. **External datasets**: no internally-generated scenarios; use published emotion elicitation datasets

## Candidate dataset sources

### A. IAPS (International Affective Picture System)
- ~1200 emotional images + valence/arousal/dominance ratings from large N humans
- Widely used in affective neuroscience
- **Use**: input text description of image, model outputs valence/arousal → compare to human mean
- **Bottleneck**: IAPS requires credentialed access

### B. GoEmotions (Google, 2020)
- 58k Reddit comments labeled with 27 emotion categories + neutral
- Creative commons license
- Subset of 27 emotions maps loosely to Panksepp+Ekman 10
- **Use**: input text → model produces 10-dim emotion activation → classify → compare to labels
- **Easy**: already tokenized, labeled, open

### C. SEMAINE (Sensitive Artificial Listener)
- Audio-video dialogue corpus with continuous valence/arousal annotation
- Used for emotion recognition benchmarks
- **Use**: dialogue → model state → predict valence/arousal trajectory
- **Harder**: requires multimodal but audio can be transcribed first

### D. OpenNeuro emotion task fMRI data
- fMRI BOLD signal during emotion elicitation tasks
- Can be mapped to rate model outputs via biophysical transfer function
- **Hardest**: requires BOLD simulation layer, but most scientifically rigorous

## Proposed first test: GoEmotions classification

**Task**: Given text, classify into 10-emotion Panksepp+Ekman taxonomy.

**Baseline comparisons**:
1. `random`: chance accuracy ~10%
2. `text_analyzer_v3`: existing keyword-based perception bridge (in repo)
3. `LLM_zero_shot`: prompt GPT-4 to classify into 10 emotions (expected: high accuracy, reference ceiling)
4. `model_rates_classify`: run IntegratedBrainV2, classify by max-activation emotion
5. `model_lesion_{emotion}`: same as (4) but with that emotion's circuit set to zero drive

**Expected findings**:
- LLM_zero_shot ~60-80% (from literature)
- Current text_analyzer_v3 ~40-60% (keyword overlap)
- model_rates_classify: **unknown** — this is the real test of "does the circuit buy us anything"
- If model ≤ text_analyzer: neural simulation is overhead, not value-add
- If model >> text_analyzer: shows circuit dynamics contribute

**Success criterion**: model_rates_classify ≥ text_analyzer_v3 + 10% accuracy, with
statistically-significant difference (McNemar's test, p<0.05 after Bonferroni).

## Implementation plan

```
Phase 9.1: Dataset preparation
  - Download GoEmotions subset
  - Map 27 → 10 emotion taxonomy (documented in phase9_emotion_mapping.md)
  - Split 80/10/10 (train-for-calibration-never / dev / test)
  - Hold out test split immutable until final eval

Phase 9.2: Baseline implementations
  - baseline_random.py: random per-instance class
  - baseline_keyword.py: existing text_analyzer_v3
  - baseline_llm.py: GPT-4 zero-shot classification via Anthropic API
  - baseline_lesioned.py: model with specific circuit disabled

Phase 9.3: Metric implementation
  - accuracy / macro-F1 / per-class precision/recall
  - confusion matrix
  - McNemar test for baseline comparison

Phase 9.4: Evaluation run
  - All baselines + model on dev split
  - Iterate if model significantly underperforms
  - Final test-split evaluation only ONCE (no tuning on test)

Phase 9.5: Writeup
  - Results in docs/behavioral_validation_results.md
  - Update README claims with actual predictive accuracy
  - Include in portfolio article / preprint
```

## Guardrails against overfit (learned from v1/v2 audits)

1. **Never tune on test split** — single-use for final reporting only
2. **Multiple seeds** — report mean ± std over N≥20 runs
3. **Public audit trail** — each iteration committed separately with diff
4. **Null hypothesis taken seriously** — if model = random, accept and report
5. **No mid-evaluation tweaking** — if dev result is disappointing, write it up anyway

## Timeline estimate

- Phase 9.1-9.2: 1-2 weeks (dataset + 3 baselines)
- Phase 9.3: 2-3 days (metrics)
- Phase 9.4: 1 week (eval + iteration within dev split)
- Phase 9.5: 1 week (writeup)
- **Total: 3-5 weeks of focused work**

## Dependencies

- `datasets` library (Hugging Face) for GoEmotions
- Existing Anthropic / OpenAI API access for LLM baseline
- `scikit-learn` for metrics and McNemar
- Current IntegratedBrainV2 for model inference

## Exit criteria (success or honest failure)

- **Success**: model_rates_classify > text_analyzer_v3 with significant margin + justification of why
- **Partial**: model ≈ keyword baseline but provides interpretability (circuit tracing) — may still have
  value for 3c B2B path
- **Failure**: model < keyword baseline → acknowledge, update README, pivot or archive

All outcomes are acceptable **IF honestly reported**. The point is to test the hypothesis,
not to make the model look good.
