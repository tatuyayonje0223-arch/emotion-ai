# Phase 9: GoEmotions 27 → Panksepp+Ekman 10 Emotion Mapping

## Source taxonomies

### GoEmotions (27 labels + neutral)
Demszky et al. 2020 "GoEmotions: A Dataset of Fine-Grained Emotions" (ACL).
Single or multi-label emotion tags on 58k Reddit comments.

```
admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity,
desire, disappointment, disapproval, disgust, embarrassment, excitement, fear,
gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief,
remorse, sadness, surprise, neutral
```

### EmotionAI 10 (Panksepp 7 + Ekman 3)
From `src/brian2_circuits/emotion_circuits_v2.py`:
```
FEAR, RAGE, SEEKING, SADNESS, DISGUST, CARE, PANIC_GRIEF, PLAY, LUST, SURPRISE
```

## Mapping rationale

Mapping is **lossy by design**. 27 fine-grained tags compress to 10 circuit outputs.
Some GoEmotions labels have no clean EmotionAI equivalent (e.g., "realization" is
epistemic, not in Panksepp/Ekman).

| GoEmotions | EmotionAI | Rationale |
|-----------|-----------|-----------|
| fear | FEAR | direct |
| nervousness | FEAR | threat anticipation |
| anger | RAGE | direct |
| annoyance | RAGE | low-intensity rage |
| disapproval | RAGE | social rage component |
| joy | SEEKING | reward/appetitive (Panksepp SEEKING) |
| excitement | SEEKING | approach motivation |
| optimism | SEEKING | expected reward |
| gratitude | SEEKING+CARE | reward acknowledgement (ambiguous; primary: SEEKING) |
| pride | SEEKING | self-reward |
| admiration | CARE | social appreciation (primary) |
| approval | CARE | social reinforcement (primary) |
| caring | CARE | direct |
| love | CARE+LUST | social bonding (primary: CARE) |
| desire | LUST | direct |
| sadness | SADNESS | direct |
| disappointment | SADNESS | reward omission |
| grief | PANIC_GRIEF | direct (Panksepp separation-distress) |
| remorse | SADNESS+PANIC_GRIEF | self-directed negative |
| disgust | DISGUST | direct |
| embarrassment | PANIC_GRIEF | social distress |
| surprise | SURPRISE | direct |
| curiosity | SURPRISE | epistemic surprise |
| confusion | SURPRISE | uncertainty arousal |
| realization | SURPRISE | insight/shift |
| amusement | PLAY | direct (play-related affect) |
| relief | — | no direct mapping; drop |
| neutral | — | no emotion; drop |

## Unmappable labels (excluded from evaluation)

- `relief`: valence-shift signal, not a discrete emotion in Panksepp/Ekman
- `neutral`: baseline state, not a target emotion

## Multi-label handling

GoEmotions has multi-label instances (e.g., "joy + surprise"). Handling:
- If any ground-truth label maps to EmotionAI X, treat X as positive
- Primary mapping (CARE for `love`) used when instance is single-label

## Evaluation variants

### Variant A: Single-label strict
Only use GoEmotions instances with exactly one label. Model predicts argmax. Direct accuracy.

### Variant B: Multi-label top-k
Use all instances. Model outputs 10-dim softmax. Report top-1 and top-3 accuracy.

### Variant C: Coarse-grained (Ekman basic 6)
Collapse further to 6 (anger/disgust/fear/happiness/sadness/surprise) for comparability
with classical emotion recognition benchmarks.

## Ground truth limitations

- GoEmotions labels are **Reddit annotator consensus**, not ground-truth physiological state
- Only 3 annotators per instance (low agreement threshold)
- Text-only modality (lacks behavioral/physiological context)
- Cultural bias: predominantly English-speaking Reddit users

These limitations are **real-world** — any deployed emotion AI faces similar signal ambiguity.

## Baseline expectations (pre-registered)

Before running:
- `random` (uniform 10-class): accuracy ~10%
- `keyword` (existing text_analyzer_v3): **expected 30-45%** (simple keyword matching)
- `llm_zero_shot` (GPT-4): **expected 55-70%** (reference ceiling for text→emotion)
- `model_rates_classify` (IntegratedBrainV2 argmax): **UNKNOWN** — the actual test
- `model_lesioned_{X}`: expected drop of 5-10% on X-labeled instances

## Success criterion (pre-registered to prevent p-hacking)

Model is "useful" if:
1. `model_rates_classify >= keyword + 10%` absolute accuracy
2. McNemar test p < 0.05 after Bonferroni correction (5 baselines × 10 classes = 50 comparisons → α=0.001)
3. `model_lesioned_{X}` shows ≥5% drop on X-labeled instances (functional specificity)

If model fails (1), accept null result and update README to reflect that rate-level
fitting does not translate to behavioral prediction.
