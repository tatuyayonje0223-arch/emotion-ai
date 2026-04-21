# A 821-Neuron Brain-Inspired Emotion Simulation Doesn't Beat "Always Predict the Most Common Class" on Fine-Grained Tasks — And I Didn't Notice for 14 Sub-Phases. A Case Study in Validation Framework Traps.

**Author**: shukaku (@mapshukaku)
**Date**: 2026-04 (v4 — incorporates 4 rounds of self-audit)
**Repository**: https://github.com/tatuyayonje0223-arch/emotion-ai

---

## Abstract

I built a 821-neuron spiking simulation of 10 emotion circuits
(Panksepp 7 + Ekman 3) using Brian2, referencing 232 neuroscience papers,
and achieved "36/36 STRICT rate-matching validation." Then I ran
**4 rounds of independent self-audit** and **15 sub-phases of behavioral
validation experiments**. The definitive finding: the neural simulation
performs **below the majority-class baseline** on 10-way and 6-way
classification (by 8.4% and 16.8% respectively), and only matches
keyword matching on 2-way binary valence where both exceed majority by
~3%. All "circuit specificity" claims based on lesions have **Fisher
exact p ≥ 0.1** at the tested sample sizes — directional but not
statistically supported. The **hybrid control** (apply the model's
post-processing weights to raw keyword hits, bypassing simulation)
beats the model by 14× on arousal regression. The most valuable
outputs are the decisive null with clean controls, the documentation
of how rate-matching validation induces overfit, and a cautionary tale
about missing the majority-class baseline for 14 sub-phases.

---

## 1. Why this project

Panksepp's affective neuroscience argues emotions emerge from specific
brain circuits (fear → amygdala + PAG, reward → VTA DA → NAc, etc.).
LLMs classify emotions by gradient-descent pattern matching without
such circuits. I wanted to test: does a **minimal spiking simulation**
of these circuits contribute something that LLM / keyword approaches
cannot?

I built what a solo developer can reasonably build: 821 neurons across
53 populations implementing the core architecture from 232 cited papers.

## 2. What I built

```
IntegratedBrainV2
├── SharedCoreNetwork (19 regions, ~312 neurons)
│   PAG, BNST, PVN, VTA, NAc, LC, DR, aIC, RMTg, DRN_GABA,
│   PPTg, dHPC, vHPC
├── 10 emotion circuits (53 populations, ~821 neurons)
│   FEAR / RAGE / SEEKING / SADNESS / DISGUST / CARE /
│   PANIC_GRIEF / PLAY / LUST / SURPRISE
├── Conductance-based GABA_A inhibition (g_inh state variable)
├── Neuromodulation: DA / 5-HT / NE / ACh / OXT / CORT
├── STDP + reward-modulated plasticity
└── Dual neuron model (Izhikevich + AdEx)
```

Brian2 implementation, 49 test files, 572+ passing tests.
Initial validation: 36/36 STRICT rate targets on Izhikevich.

## 3. Audit v1 & v2: the rate-matching validation itself was the trap

**Audit v1** (post AdEx 36/36 push): caught me reverse-engineering tonic
values from targets under "paper-justified" framing. Reverted.

**Audit v2** (more comprehensive): realized **the original Izh 36/36**
was built on 27 parameter changes with the same pattern. `PKCd tonic =
-0.5` (negative, non-physiological), `VTA DA b_spike=9 tau_w=100ms`
(values chosen to hit `tonic ~3Hz, burst ~32Hz, pause ~1Hz` target
trio), AdEx-specific synapse multipliers — all tuned to match 36 rate
targets without principled derivation.

Adding Monte Carlo averaging (n=5) revealed:
- Izh: 36/36 STRICT → **35/36 MC-stable**
- AdEx: 28/36 → **25/36 MC-stable**

Baseline physiology probe revealed both models fire MSN at 10 Hz when
literature says <1 Hz at rest. The 36 scenario targets never checked
baseline. Fixing baseline (Phase 7 P1: pop-specific bg_noise for MSN)
broke scenario firing — simplified model can't have both without
UP/DOWN state dynamics.

## 4. Audit v3: pivoting from "classifier" to "interpretable research tool"

With rate-matching validation itself under question, I designed a
**behavioral prediction validation framework** (GoEmotions text →
10-emotion classification). Plus control experiments and reference
ceilings. This was Phase 9.

## 5. Phase 9 behavioral validation — the 15 sub-phases

### 5.1 Fine-grained 10-way: null (9.4)

GoEmotions validation n=500, single-label mappable:

| Baseline | Accuracy |
|----------|---------:|
| Random | 8.4% |
| Keyword argmax | 28.0% |
| Neural model | 19.2% |

McNemar test model vs keyword: chi² = 13.40, **p = 0.0003**.
Systematic SEEKING over-prediction: 355/500 predictions vs 90/500 true.

### 5.2 Readout fix (9.7)

Found the SEEKING readout was the only ungated one — VTA baseline tonic
caused floor-level activation. Added `if reward > 0.1:` gate. Also
fixed argmax fallback bias (`best_val = -1.0 → 0.0`).

After fixes:
- Model: 22.2% (was 19.2%)
- Keyword: 28.0% (unchanged)
- **Model correct AND keyword wrong = 0 instances** — not "slightly
  worse" but "strict subset."

### 5.3 Dimensional V/A: apparent positive → decisive null via control (9.8 + 9.9)

Model seemed to beat keyword on V/A MAE (13% / 10% better). Pre-registered
control: apply model's hand-coded V/A weight table to keyword hits
directly, bypassing simulation:

| Metric | Model (with sim) | Hybrid (no sim) |
|--------|-----------------:|----------------:|
| Valence Pearson | +0.319 | +0.303 |
| Arousal Pearson | **-0.019** | **+0.272** |
| Arousal MAE | **0.450** | **0.219** |
| Joint R² | -0.445 | **-0.035** |

Hybrid beats model by **14× on arousal** and 51% better MAE. The
simulation was **coarsening** the signal that keyword counts already
carry. All apparent V/A advantage was the weight table, not the neural
simulation.

### 5.4 Lesion specificity (9.6 + 9.10)

Classic neuroscience logic: if circuit X is causally necessary for
emotion X, silencing X should specifically impair X predictions.

Input-level lesion (zero the drive): FEAR/RAGE/SADNESS show accuracy
drops from baseline to 0%. SEEKING doesn't (readout bias).

Population-level lesion (tonic = -10 on readout populations):
RAGE/SEEKING/SADNESS show 0% accuracy. FEAR's redundant multi-pathway
architecture (la_exc → ba_exc → cem) means silencing readout alone
doesn't collapse FEAR; extended lesion (6 populations) collapses FEAR
in smoke tests.

**v4 statistical reality check** (Fisher exact test on all 7 lesions):

| Test | Baseline | Lesion | p-value | Significant? |
|------|---------:|-------:|--------:|:------------:|
| input FEAR n=3 | 66.7% | 0.0% | 0.2000 | ✗ |
| input RAGE n=11 | 27.3% | 0.0% | 0.1071 | ✗ |
| input SADNESS n=7 | 28.6% | 0.0% | 0.2308 | ✗ |
| pop RAGE n=11 | 27.3% | 0.0% | 0.1071 | ✗ |
| pop SEEKING n=8 | 12.5% | 0.0% | 0.5000 | ✗ |
| pop SADNESS n=7 | 28.6% | 0.0% | 0.2308 | ✗ |
| pop FEAR n=3 | 66.7% | 66.7% | 0.8000 | ✗ |

**Zero tests reach p<0.05.** Smallest p=0.107. "4/10 emotions with
specificity" must be demoted to "descriptive directional drops at
statistically underpowered sample sizes." Per-class n in GoEmotions
validation is 3-11 — nowhere near enough for significance on a
50% → 0% drop.

### 5.5 Reference ceilings (9.11)

Added Gemini 2.5-flash and trained Logistic Regression:

| Baseline | n | Accuracy |
|----------|--:|---------:|
| Random | 200 | 10.5% |
| Gemini 2.5-flash zero-shot | 100 | 24.0% |
| Keyword argmax | 200 | 23.0% |
| Neural model | 200 | 17.5% |
| LR trained on 5000 | 200 | 12.0% |

LLM ceiling only 1% above keyword. Task is inherently hard.

### 5.6 Coarse-grained Ekman 6 (9.12)

Collapsing 10 EA → 6 Ekman basic emotions, then testing: model 36.4%
**tied** with keyword 36.4%. The 10-way null disappears.

**v4 correction**: the majority-class baseline at 6-way is
**always-joy = 53.2%**. Both keyword and model are **16.8% BELOW
majority**. "Tied" is tied at catastrophic failure vs trivial prior,
not a positive finding.

### 5.7 Top-k and multi-label (9.13 + 9.14)

Relaxing from strict argmax:

| Metric | Keyword | Model | Δ |
|--------|--------:|------:|----:|
| top-1 | 16.6% | 10.8% | -5.8% |
| top-5 | 52.8% | 55.2% | -2.4% |
| multi-label F1@5 | 0.208 | 0.201 | -0.007 |

Model's ranking quality near-equivalent to keyword; argmax commitment
is specifically where it loses. But **v4 majority baseline check**:
always-predict-CARE would give ≈28% at top-1, higher than either
method's top-1.

### 5.8 Binary valence (9.15, v4-mandated)

v4 audit required testing the simplest task — 2-way positive vs negative
valence:

| Method | Accuracy |
|--------|---------:|
| Random | 48.5% |
| Majority (dominant valence) | 63.5% |
| **Keyword** | **66.8%** |
| **Model** | **66.6%** |

**First Phase 9 metric where both methods exceed majority.** Both gain
~3% over trivial prior. Model and keyword tied (Δ=0.2%).

## 6. Audit v4: the biggest finding was the control I missed

After 14 sub-phases of experiments and commits, I realized I had never
computed the **majority-class baseline**. That's the simplest possible
control: always predict the most common class.

| Granularity | Majority | Keyword | Model | Model vs majority |
|-------------|---------:|--------:|------:|------------------:|
| 10-way fine | 27.6% | 28.0% | 19.2% | **-8.4%** ❌ |
| 6-way Ekman | 53.2% | 36.4% | 36.4% | **-16.8%** ❌ |
| **Binary valence** | **63.5%** | **66.8%** | **66.6%** | **+3.1%** ✅ |

**The neural simulation performs below the majority-class baseline on
fine-grained and coarse-grained tasks.** It only exceeds majority on
binary valence — the coarsest possible formulation — and there it's
tied with keyword matching.

## 7. Honest synthesis (post-v4)

| Where model fails | Relative to |
|-------------------|-------------|
| 10-way argmax | -8.4% below majority, -8.8% below keyword |
| 6-way Ekman | -16.8% below majority |
| Arousal regression | 14× worse than hybrid weight-table control |
| Circuit specificity | Fisher p ≥ 0.1 for all 7 tested lesions (underpowered) |

| Where model matches | Relative comparison |
|---------------------|---------------------|
| Binary valence | +3.1% above majority (tied with keyword +3.3%) |

| Where model has unique value | |
|------------------------------|---|
| None, at any Phase 9 metric with adequate statistical support | |

## 8. What I learned (7 lessons, updated post-v4)

### 8.1 The majority-class baseline is the simplest and most important control
I ran 14 sub-phases of experiments without it. My "keyword > model" null
looked meaningful until I computed `n_majority_class / n_total`. Every
comparison to random + keyword should include majority baseline. It's
a 5-line check that saves you from 14 sub-phases of misdirected narrative.

### 8.2 Rate-matching validation doesn't predict behavior
27 parameter changes produced 36/36 STRICT PASS, but this didn't
translate to behavioral prediction. Validation framework choice matters
more than validation score at this model scale.

### 8.3 "Paper-cited" ≠ "paper-derived"
Every parameter change has a citation. But citations identify
populations and rate ranges, not specific tonic values. Values
reverse-engineered from targets, even with citations, are numerical
fitting.

### 8.4 Control experiments matter more than novel positive results
Phase 9.8 V/A "partial positive" was demolished by Phase 9.9 hybrid
control. Pre-register control experiments before claiming novel findings.

### 8.5 Null results require statistical rigor too
I reported "4/10 emotions show circuit specificity" across Phase 9.6 +
9.10 based on descriptive accuracy drops. Fisher exact test revealed
zero of seven tests reach p<0.05. Statistical tests for directional
claims at low-n are just as essential as for positive claims.

### 8.6 Single-seed stochastic validation is fragile
My v2 audit established MC averaging as necessary. Yet Phase 9.4-9.14
used single seed throughout. Consistent methodology across all phases
is harder than it sounds.

### 8.7 Confirmation bias survives 3 audit rounds
Audits v1-v3 caught different issues each time. v4 caught that I was
framing results to fit "interpretable coarse classifier" positioning
even when data showed "fails below majority baseline." The audit
process is valuable, but each round has blind spots the previous
didn't see.

## 9. What's left of the project

### Honestly defensible
- **Educational demo** (`demo/index.html`): visualization of circuit
  activations across scenarios, independent of classification accuracy
- **Portfolio case study**: this document — a concrete example of
  validation traps and 4-round audit process
- **Bug discoveries** (Phase 9.7): SEEKING gate, argmax fallback —
  structural findings in the codebase, not metric claims
- **Hybrid V/A null** (Phase 9.9): clean control with decisive result
- **Cautionary tale**: 36/36 STRICT validation on scenarios does not
  predict ability to classify text-based emotion labels

### No longer defensible
- "Interpretable classifier" — below majority baseline on 2 of 3 tested
  granularities
- "4/10 circuit specificity" — not statistically significant
- "Ranked predictions nearly equivalent to keyword" — keyword isn't
  much better than majority either

### For other builders
1. **Run the majority-class baseline first**. It's 5 lines.
2. **Report Fisher tests for small-n accuracy drops**, not just descriptive %.
3. **Pre-register control experiments** before claiming novel positives.
4. **MC average stochastic simulations** every time.
5. **Every audit round has blind spots the previous didn't see.** Keep
   auditing. Don't assume the last one caught everything.
6. **Rate-matching validation induces overfit pressure** at any model
   scale, but especially at 10^4× reduction from biological reality.
7. **Be honest when the data says "below trivial baseline."** The
   interesting part is usually in the audit chain, not the positive claim.

## 10. Repository

- https://github.com/tatuyayonje0223-arch/emotion-ai
- 20 commits across 3 weeks
- 49 test files (572 passing)
- 15 Phase 9 sub-phases + 4 audit rounds
- Full reproducibility via scripts in `phase9/`
- Audit chain: `docs/audit_v4.md`, `parameter_changes_log.md`,
  `docs/phase9_*.md` (10 documents)

---

*Feedback: @mapshukaku. The portfolio value is the journey: the
failed 36/36 claim, the 3 revert commits, the 14 sub-phases chasing
a null, the moment in v4 when I computed `n_majority / n_total` and
realized half the narrative was wrong.*

---

## Acknowledgements

232 papers' authors. The Panksepp/Ekman taxonomies.
LeDoux 2000 + Duvarci & Pare 2014 for the fear redundancy insight that
predicted Phase 9.10's FEAR lesion resilience.
Myself-from-3-weeks-ago for the overconfident commits I had to revert
4 times.

The majority-class baseline I forgot to include.
