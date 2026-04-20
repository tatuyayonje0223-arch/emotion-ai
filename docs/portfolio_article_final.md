# Does a 821-Neuron Brain-Inspired Simulation Predict Emotion Better Than Keyword Matching? A 13-Subphase Self-Audit Says No — And That's the Interesting Part.

**Author**: shukaku (@mapshukaku) — solo developer / 初期研修医
**Date**: 2026-04-20
**Repository**: https://github.com/tatuyayonje0223-arch/emotion-ai (Research WIP)
**License**: Research use only

---

## Abstract

I built a 821-neuron spiking simulation of 10 emotion circuits
(Panksepp 7 + Ekman 3) using Brian2, referencing 232 neuroscience papers,
and achieved "36/36 STRICT rate-matching validation." Then I ran 3 rounds
of independent self-audit and 13 sub-phases of behavioral validation
experiments. The result: the simulation provides **no unique predictive
value over keyword matching** on fine-grained emotion classification
(GoEmotions n=500, p=0.0003), and **hybrid simulation-bypass control beats
the model by 14× on arousal regression**. It ties keyword on **coarse Ekman
6-way classification** (36.4% each) and shows **circuit-level lesion
specificity for 4 emotions (FEAR/RAGE/SEEKING/SADNESS)**. The most valuable
outputs of this project are the decisive null result with a clean control
experiment, and the audit chain documenting how rate-matching validation
does not predict behavior.

---

## 1. Why this project

Panksepp's affective neuroscience argues emotions emerge from specific
brain circuits:
- Fear → amygdala (LA / CeL / CeM) + PAG
- Reward → VTA DA → NAc
- Sadness → sgACC → habenula → VTA pause
- ...

Large language models classify emotions in text by gradient-descent
pattern matching. They "know" what's fearful but not through a fear
circuit. I wanted to know if a **minimal neural circuit simulation**
could contribute something that LLM / keyword approaches cannot.

I built what I could reasonably build as a solo developer: a simplified
spiking neural model with 821 neurons across 53 populations, implementing
the core architecture from 232 cited papers.

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
Validation: 36 firing-rate targets from literature, ±30% tolerance.

**Initial result**: 36/36 STRICT PASS (Izhikevich), 28/36 (AdEx).

## 3. Audit v1: AdEx 36/36 was numerical overfit

I pushed AdEx to 36/36 with "paper-justified" tonic drive adjustments.
Then a voice in my head said "really?" and I ran an audit.

Findings:
- The 8 tonic adjustments (citing Sara & Bouret 2012, Quirk 2002, etc.)
  cited the papers' rate ranges but **reverse-engineered the tonic values**
  from target rates, not from paper-derived parameters
- When I diagnosed "AdEx linear + adaptation dynamics cannot reach target
  X," I overturned that diagnosis mid-session when a parameter sweep
  showed it could — **diagnosis following numerical results**

Action: reverted (`commit a12ca32`). Accepted AdEx 28/36 as honest.

## 4. Audit v2: Izh 36/36 was also on numerical scaffolding

Reading the Phase 2 `parameter_changes_log.md`, I realized the same
pattern was in pre-existing changes:
- `PKCd tonic = -0.5` (negative tonic — physically non-sensical as a
  "drive", but needed to force PKCd silence)
- `VTA DA b_spike=9, tau_w=100ms, g_L=0.2` — comment: "balanced: tonic
  ~3Hz, burst ~32Hz, pause ~1Hz" (values chosen to hit those targets)
- `CeA SOM→PKCd shunting 4.0x` — magic multiplier for AdEx, no paper
- Per-connection shunting factors scattered across 27 changes

I added `scripts/evaluate_multitrial.py` and re-evaluated with 5-trial MC
averaging. The "STRICT 36/36" dropped to:
- Izh: **35/36 stable** (1 scenario was seed-dependent)
- AdEx: **25/36 stable** (4 more scenarios were boundary-unstable)

Also added `scripts/validate_baseline_rates.py` — tested resting-state
rates (not in the 36 targets). Result: **both models 6/20 baseline PASS**.
MSN populations fire 10 Hz at rest when the literature says <1 Hz.

This is when I understood: **the validation framework itself induces the
overfit**. Targeting 36 scenario rates while ignoring 20 baseline rates
creates pressure to hit the 36 via any tuning, which inevitably warps
baseline physiology.

## 5. Attempted fix: Phase 7 P1 pop-specific bg_noise

Principled fix: MSN should have lower background drive (Tepper 2004
K_ir rectification). Changed `bg_noise = 1.7 → 0.2` for MSN populations.

Result:
- Baseline: 6/20 → 8-9/20 (MSN now silent at rest, correct)
- Scenario MC: Izh 35 → 32, AdEx 25 → 23 (nac_d1 and putamen can't reach
  task firing from low background)

Net: **-3 scenarios, +3 baseline**. Either way the same model fails at
different dimensions. Reverted.

Phase 8 added to roadmap: MSN UP/DOWN state mechanism (NMDA-mediated
bistable transitions). Proper fix; future work.

## 6. Audit v3: is the whole project justified?

After 2 reverts and one failed fix, I ran a third audit — this time
questioning the project itself.

Findings:
- README explicitly forbids "numerical fitting" — yet 27 parameter
  changes fit this pattern. Public GitHub + contradictory claim = risk.
- Success criterion was undefined. What's "done" for this project?
- 10-emotion taxonomy is Panksepp (7) + Ekman (3) — **theoretically
  incompatible** taxonomies.
- 821 neurons vs biological 10^11 = 4-5 orders of magnitude reduction.
  Rate-matching validation at this scale has questionable scientific
  meaning.

Chose Option α+β+γ combo: clean public claims, accept AdEx 28/36 (later
25/36 MC), pivot from 3a/3b paths to 3c+3d+3e:
- 3c: B2B interpretable AI (needs proof of behavioral value)
- 3d: Education/demo (needs nothing but visualization)
- 3e: Portfolio writeup (this document)

## 7. Phase 9: behavioral validation

The rate-matching validation said 36/36. But does the model **predict
behavior**? That's the real question.

I designed the behavioral validation framework (GoEmotions 10-way text
classification, pre-registered success criterion, multiple baselines).
Then I ran 13 sub-phases of experiments. Here are the findings:

### 7.1 Fine-grained classification: decisive null (9.4)

GoEmotions validation n=500, single-label mappable:

| Baseline | Accuracy | Macro-F1 |
|----------|---------:|---------:|
| Random | 8.4% | 0.067 |
| **Keyword argmax** | **28.0%** | 0.231 |
| **Neural model** | **19.2%** | 0.165 |

McNemar paired test model vs keyword: chi² = 13.40, **p = 0.0003**.
Model correct / keyword wrong: 47.
Model wrong / keyword correct: 91.

Systematic SEEKING over-prediction: model predicted SEEKING on 355/500
instances vs true 90. All 138 true-CARE misclassified (96 → SEEKING).

### 7.2 Readout fix exposed pre-existing bugs (9.7)

9 of 10 emotion readouts were input-gated (`if threat > 0.1: ...`).
**Only SEEKING was ungated** — VTA baseline tonic caused floor-level
activation that won argmax when others silenced.

Fix: added `if reward > 0.1:` gate. Restored parity.
Also fixed argmax fallback bias (`best_val = -1.0 → 0.0`).

After both fixes:
- Model: 22.2% (was 19.2%)
- Keyword: 28.0% (unchanged)
- **Model correct AND keyword wrong = 0 instances**

The strongest form of null: not "model is slightly worse" but
"everything model gets right, keyword also gets right."

### 7.3 Dimensional V/A: apparent positive (9.8) then null (9.9)

Maybe the simulation captures **continuous** valence/arousal better than
discrete classification?

Initial result (Phase 9.8):
- Valence MAE: model 0.513 vs keyword 0.591 (-13% better)
- Arousal MAE: model 0.450 vs keyword 0.503 (-10% better)
- Model wins. First positive in Phase 9.

Pre-registered control (Phase 9.9): apply the model's hand-coded V/A
weight table to keyword hit counts directly, bypassing simulation:

| Baseline | V Pearson | V MAE | **A Pearson** | **A MAE** | Joint R² |
|----------|----------:|------:|--------------:|----------:|---------:|
| hybrid_va (no sim) | +0.303 | 0.531 | **+0.272** | **0.219** | **-0.035** |
| model_va (with sim) | +0.319 | 0.513 | -0.019 | 0.450 | -0.445 |

Hybrid beats model by **14× on arousal Pearson** and **51% better MAE**.
The neural simulation doesn't just fail to help — it **coarsens** the
arousal signal that keyword counts already carry (via input gating +
weighted sum over 10 emotions; most emotions are 0 so V/A is dominated
by few triggered activations, losing continuous range).

This was the most scientifically decisive finding: **all apparent model
advantage was the hand-coded weight table, not the neural simulation.**

### 7.4 Circuit-level lesion specificity: the one real positive (9.6 + 9.10)

Classical neuroscience lesion logic: if circuit X is causally necessary
for emotion X, silencing X should specifically impair X predictions.

Input-level lesion (zero the emotion's drive in `process()`):
- FEAR, RAGE, SADNESS: 100% accuracy drop on their own emotion,
  unchanged on others (**clean specificity**)
- SEEKING: no drop (readout bias — see 7.2)

Population-level lesion (tonic = -10 for circuit's readout contributors):
- RAGE, SEEKING, SADNESS: 100% drop on own emotion (**SEEKING new finding**
  — population silencing bypasses readout bias)
- FEAR readout-only lesion: no drop (multi-pathway redundancy via
  la_exc → ba_exc → cem cascade)
- FEAR extended lesion (add la_exc/ba_exc/cel_som): smoke test collapses
  FEAR predictions on hand-crafted sentences (consistent with LeDoux 2000
  redundant fear architecture)

4 of 10 emotions show empirical circuit specificity. The 5 emotions with
0% baseline accuracy (CARE/LUST/PLAY/PANIC/DISGUST) can't be tested by
this method.

### 7.5 Reference ceilings (9.11)

Added Gemini 2.5-flash zero-shot + Logistic Regression trained on 5000
instances:

| Baseline | n | Accuracy |
|----------|--:|---------:|
| Random | 200 | 10.5% |
| Gemini 2.5-flash | 100 | **24.0%** |
| Keyword argmax | 200 | 23.0% |
| Neural model | 200 | 17.5% |
| LR trained | 200 | 12.0% |

LLM ceiling is only 1% above keyword. This task is **inherently hard**
for all methods. Model sits between LR and keyword; not catastrophic but
not unique. Trained ML on the same features as keyword loses to argmax
(class imbalance + feature sparsity).

### 7.6 Coarse-grained Ekman 6-way: new positive (9.12)

Collapse both 10-emotion model output and 27 GoEmotions ground truth
to the classical Ekman 6 basic emotions (anger/disgust/fear/joy/sadness/
surprise). SEEKING+CARE+PLAY+LUST all map to joy.

| Baseline | Accuracy | Macro-F1 |
|----------|---------:|---------:|
| Random | 24.8% | 0.137 |
| **Keyword** | **36.4%** | 0.355 |
| **Neural model** | **36.4%** | 0.349 |

**Tied**. Δ = 0.000 on accuracy. Per-class F1 differs by at most 0.02.

The 10-way null disappears at coarser granularity. The SEEKING
over-prediction error (at 10-way) becomes correct at 6-way because
misclassified CARE-as-SEEKING is counted as "joy-as-joy".

### 7.7 Top-k ranking (9.13)

| k | Keyword | Model | Δ |
|---|--------:|------:|----:|
| top-1 | 16.6% | 10.8% | -5.8% |
| top-2 | 32.2% | 30.0% | -2.2% |
| top-3 | 44.0% | 41.2% | -2.8% |
| top-5 | 52.8% | 55.2% | -2.4% |

Gap narrows as k grows. Model's ranking quality is similar to keyword's;
the failure is specifically at argmax commitment under 10-way fine
granularity.

## 8. Synthesis

Across 13 sub-phases of behavioral validation, the picture is:

**Where model fails**:
- Fine-grained 10-way argmax (null, p<0.0001)
- Dimensional V/A (null, hybrid table alone wins)
- Unique classification value (0 instances where model > keyword)

**Where model matches keyword**:
- Coarse 6-way Ekman (tied, 36.4% each)
- Top-5 ranking on 10-way (nearly tied, 52.8% vs 55.2%)

**Where model provides unique value**:
- Circuit-level lesion specificity for FEAR/RAGE/SEEKING/SADNESS
  (no keyword equivalent; model's architectural design is required)

## 9. What I learned

### 9.1 Rate-matching validation doesn't predict behavior
27 parameter changes produced 36/36 STRICT PASS, but this didn't
translate to behavioral prediction accuracy. **Validation framework
choice matters more than validation score.** Scenario rate targets at
this model scale are underdetermined — many parameter configurations
hit them, most don't generalize.

### 9.2 "Paper-cited" ≠ "Paper-derived"
Every parameter change in my log has a citation. But citations identify
the population and rate range, not the specific tonic value. Values
reverse-engineered from targets, even with citations, are numerical
fitting by another name.

### 9.3 Control experiments matter more than novel results
Phase 9.8 looked like the first positive finding. Phase 9.9 control
demolished it. If I had committed the "positive" and stopped,
I'd have been wrong.

### 9.4 Null results are publishable when well-controlled
A decisive null with clean control (Phase 9.9) and clear scope
(this model, this task, this scale) is more valuable than another
"my model achieves X% accuracy" claim.

### 9.5 Simplified models have structural limits that can't be tuned away
Phase 7 P1: fixing MSN baseline broke MSN task firing. The simplified
1-population-per-region architecture cannot simultaneously capture
quiet rest + task activation without UP/DOWN state dynamics. Some
problems need architectural redesign, not parameter tuning.

### 9.6 Single-trial stochastic validation is fragile
Phase 9.7 fix exposed that my "argmax = FEAR when all zero" was a
silent bias. The original 36/36 scenario PASS used single random seed.
MC averaging revealed 35/36 and 25/36 "real" scores. **Always MC
average stochastic systems.**

### 9.7 Interpretability value can survive classification null
The model lost on classification but FEAR/RAGE/SEEKING/SADNESS circuits
demonstrably control their respective emotion predictions (lesion
specificity). This is the remaining scientific and product value:
**mechanistic diagnostic tool for 4 emotion pathways + educational
visualization**, not a classifier.

## 10. Where next

### Active work
- Demo UI (`demo/index.html`) — visualization tool, completed
- This article — portfolio writeup, completed

### Backlog
- Phase 8: MSN UP/DOWN state for baseline+task reconciliation
- Stronger LLM reference (GPT-4 / Claude) pending API setup
- Per-instance Warriner V/A norms for finer dimensional analysis
- Multi-label evaluation (GoEmotions has multi-label instances)

### Retired
- Path 3a (biometric emotion recognition for EU AI Act compliance):
  not feasible for solo dev, not technically viable given null results
- Path 3b (emotion-aware chatbot): LLM dominates this use case
- Chasing 36/36 STRICT: abandoned

## 11. Repository

- Code: https://github.com/tatuyayonje0223-arch/emotion-ai (MIT)
- 49 test files (572 passing)
- 13 sub-phase phase9/ experiments with reproducible eval harnesses
- Full documentation in docs/
  - `phase9_results_full.md` — fine-grained null
  - `phase9_hybrid_control.md` — decisive null via control
  - `phase9_pop_lesion.md` — circuit specificity
  - `phase9_ceiling.md` — LLM reference
  - `phase9_coarse_grained.md` — coarse tie
  - `phase9_topk.md` — ranking near-equivalence
  - `phase9_readout_fix.md` — bug discovery
  - `parameter_changes_log.md` — 27 changes + audit sections

## Acknowledgements

232 papers' authors for the neuroscience foundation this project referenced.
Myself-from-3-weeks-ago for the overconfident early commits I had to revert.
The Panksepp/Ekman tradition for taxonomies; Russell for valence-arousal.
LeDoux 2000 + Duvarci & Pare 2014 for the fear circuit redundancy insight
that predicted Phase 9.10's FEAR lesion resilience.

## Key takeaway for other builders

**If you build a brain-inspired simulation and claim it classifies emotions:**

1. Test it against keyword matching on a real dataset. Don't rely on
   rate-matching validation alone.
2. Run the hybrid control: does applying your post-processing to a
   simpler upstream (keyword counts) match your model? If yes, your
   simulation isn't contributing.
3. Report pre-registered success criteria and honor them even when
   results are null.
4. Use MC averaging. Single-seed validation is fragile.
5. Write the audit chain. The journey is the contribution.

---

*Comments / feedback: @mapshukaku*
