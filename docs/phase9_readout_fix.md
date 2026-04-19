# Phase 9 Readout Function Fix — SEEKING gate + argmax fallback (2026-04-19/20)

## Background

Phase 9.4-full revealed **systematic SEEKING over-prediction** (355/500 predictions vs
90/500 true). Root cause investigation:

- 9/10 emotion readouts were input-gated: `if input > threshold: compute; else: 0.0`
- **Only SEEKING was ungated**: computed directly from VTA DA + NAc rates
- VTA DA baseline tonic ~3 Hz + `seeking_act = VTA×0.4 + NAc×0.3` = floor-level
  seeking activation always present
- With 9 other emotions forced to 0 by gates (when inputs absent), SEEKING
  default-won argmax

## Applied fixes

### Fix 1: SEEKING input gate (principled parity)

Added `if reward > 0.1:` gate to SEEKING readout in
`src/brian2_circuits/emotion_circuits_v2.py` line ~805.

Motivation: **restore parity with the other 9 emotions**. No tuning. No target-
hit iteration. This is a structural correction, not a number adjustment.

### Fix 2: argmax fallback (strict inequality)

Changed initial `best_val = -1.0` → `best_val = 0.0` and `>` → `>` (was already `>`
but behavior was different with -1.0 initialization).

When all 10 activations are 0.0 (fully silent), `0.0 > 0.0` is False so no
emotion can win — best_label stays at default "SURPRISE". This removes the
iteration-order bias where FEAR (first in dict) won in tie situations.

Alternative considered: return None / "UNKNOWN" when silent. Rejected because
classification task requires a prediction. Using SURPRISE as consistent
fallback is at least documented and stable across runs.

## Impact evaluation

### Scenario MC regression (Izh / AdEx)

No regression. Both models unchanged on scenario validation:

| Model | Before fix | After fix |
|-------|-----------|-----------|
| Izh | 35/36 stable | 35/36 stable |
| AdEx | 25/36 stable | 25/36 stable |

SEEKING gate firing is scenario-dependent. In the paper-validated scenarios,
reward is explicitly specified for SEEKING scenarios (phasic_burst / tonic /
pause / reward all have reward ≥ 0.1 or explicit loss drives for pause).
So scenario behavior unchanged.

### Behavioral classification (GoEmotions n=500)

Three configurations compared:

| Config | Model acc | vs keyword | p-value | Model ∩ keyword-wrong |
|--------|----------:|-----------:|--------:|----------------------:|
| Original (no fixes) | 19.2% | -8.8% | 0.0003 | 47 |
| Gate fix only | 10.8% | -17.2% | <0.0001 | 5 |
| **Gate + argmax fix** | **22.2%** | **-5.8%** | <0.0001 | **0** |

"Model correct / keyword wrong" count tells the real story:
- Original: 47 instances where neural model uniquely correct (mostly SEEKING bias
  coincidences — model wrongly said SEEKING on 355 instances and keyword said
  wrong-emotion on many)
- After both fixes: **0 instances**. Model's correct answers are a strict subset
  of keyword's correct answers.

### Interpretation

Gate + argmax fix is **scientifically correct** (parity + unbiased tie-breaking)
but the honest final result is **worse in some sense**:
- Raw accuracy 19.2% → 22.2% (+3%, looks better)
- But "unique correct predictions" 47 → 0 (neural model provides ZERO unique value)

Before fixes, the 19.2% accuracy was partly a coincidence of the SEEKING bias
overlapping with class imbalance (SEEKING 90/500 was a large true class). The
fixes removed that coincidence, revealing the underlying null signal.

## Final strength of null result

- Model accuracy (22.2%) < Keyword accuracy (28.0%) with p<0.0001
- Model correct ∩ keyword wrong: **0 instances**
- Neural simulation provides **zero unique behavioral prediction value**

This is the **strongest form of null result**: not just "model is a bit worse"
but "everything model does, keyword does better or equally". The 821-neuron
spiking simulation adds no signal beyond what keyword matching already captures.

## What this means for Path 3c

Path 3c (B2B interpretable AI) was already in serious trouble after Phase 9.4.
With the argmax fix, even "partial value" claim (model catches some different
instances than keyword) collapses. Path 3c is essentially dead as a classifier
product.

**Remaining value** (from Phase 9.6 lesion specificity):
- FEAR/RAGE/SADNESS **circuit-level specificity** preserved
- Not a classifier, but a **mechanistic educational/diagnostic tool**
- Value limited to explainability and teaching

## Engineering notes

The SURPRISE fallback is itself arbitrary. A rigorous classifier would:
1. Report uncertainty when all activations are 0
2. Use class priors for tie-breaking
3. Or return "abstain" and benchmark with abstention metrics

These are downstream engineering, not scientific fixes. For the current null
result, fixing the fallback further wouldn't flip the conclusion.

## Files modified

- `src/brian2_circuits/emotion_circuits_v2.py`: SEEKING input gate added
- `phase9/baselines.py`: `best_val = 0.0` initialization

## Next work (if project continues)

1. Probabilistic readout (softmax over 10 emotions, not argmax) — would smooth
   tie-breaking but won't fix zero-unique-value problem
2. Train a small classifier head on top of circuit activations — this moves
   away from "pure circuit interpretability" toward "circuit features + ML"
3. Fundamentally different task: instead of 10-way classification, test
   **valence × arousal** continuous regression where neural simulation might
   contribute (e.g., VTA DA maps to positive valence + moderate arousal)
