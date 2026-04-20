# Audit v4: Phase 9 Behavioral Validation Self-Audit (2026-04-20)

Previous audits:
- v1 (commit a12ca32): AdEx 36/36 overfit revert
- v2 (commit ec3732c): Izh 36/36 also on 数値合わせ pattern (Monte Carlo revealed)
- v3 (commit d071447): project-level integrity pivot to 3c+3d+3e positioning
- **v4 (this doc)**: Phase 9 behavioral validation findings critically reviewed

## Critical findings

### CR-01 Majority-class baseline was missing throughout Phase 9

All 14 sub-phases compared model to random + keyword, but never to the
trivial majority-class baseline. Recomputing with real class distributions:

**10-way EmotionAI classification (n=500)**:
| Method | Accuracy |
|--------|---------:|
| Random (uniform 10) | 10% |
| **Majority (always-CARE)** | **27.6%** |
| Keyword argmax | 28.0% |
| Model_rates | 19.2% |

Model is **8.4% BELOW** the majority baseline. Keyword barely exceeds it
(+0.4%). The Phase 9.4 "p=0.0003 null" is about keyword > model, but the
**bigger story is both are at or below trivial priors**.

**6-way Ekman classification (n=500)**:
| Method | Accuracy |
|--------|---------:|
| Random | 24.8% |
| Keyword | 36.4% |
| Model_rates | 36.4% |
| **Majority (always-joy)** | **53.2%** |

**Both are 16.8% BELOW always-joy**. The "tied at 36.4%" celebrated as
"new positive finding" in Phase 9.12 is a tie at **catastrophic failure
vs trivial baseline**.

### CR-02 Lesion specificity claims have no statistical support

Phase 9.6 + 9.10 claimed "FEAR/RAGE/SEEKING/SADNESS circuits show
specificity" based on per-class accuracy drops (e.g., RAGE 27.3% → 0%).

Fisher exact test on 2×2 contingency (baseline-correct/wrong vs
lesion-correct/wrong):

| Test | Baseline acc | Lesion acc | p-value | p<0.05? |
|------|-------------:|-----------:|--------:|:-------:|
| input_FEAR_n3 | 66.7% | 0.0% | 0.2000 | ✗ |
| input_RAGE_n11 | 27.3% | 0.0% | 0.1071 | ✗ |
| input_SADNESS_n7 | 28.6% | 0.0% | 0.2308 | ✗ |
| pop_RAGE_n11 | 27.3% | 0.0% | 0.1071 | ✗ |
| pop_SEEKING_n8 | 12.5% | 0.0% | 0.5000 | ✗ |
| pop_SADNESS_n7 | 28.6% | 0.0% | 0.2308 | ✗ |
| pop_FEAR_n3 | 66.7% | 66.7% | 0.8000 | ✗ |

**Zero of seven lesion tests reach p<0.05.** Smallest p=0.1071
(RAGE n=11). "Specificity confirmed" claim must be demoted to
"descriptive directional drop, not statistically supported at n used."

### CR-03 Lesion definition drift = cherry-picking

Phase 9.10 evolved POP_LESION_TARGETS over 3+ iterations, expanding the
silenced population set each time until "specificity" appeared:
- v1: 1 node (e.g., la_exc for FEAR) → no drop on FEAR predictions
- v2: 2 readout contributors → partial drop
- v3: all readout contributors → RAGE/SEEKING/SADNESS show clean drop
- v4 (extended FEAR): +3 more nodes → smoke-test FEAR collapse

Each iteration refined definition until a positive emerged. This is
**definition search / cherry-picking**. A pre-registered lesion target
set chosen before eval would avoid the issue.

### CR-04 Single-seed validation contradicts v2 audit's lesson

v2 audit (Phase 9.5) explicitly established Monte Carlo averaging as
required for stochastic validation. Yet Phase 9.4-9.14 all used
single-seed (trial_num=0) evaluation. The methodology conclusion from
v2 was not applied to subsequent phases.

Variability impact: seed change could produce ±3-5% accuracy shift.
"Model 19.2% vs keyword 28.0%" at single-seed could be "model 15-23% vs
keyword 24-32%" at different seed — gap could narrow or widen.

### CR-05 Confirmation-bias-oriented framing

Portfolio article and commit messages frame every result to fit the
"interpretable coarse emotion classifier" narrative. Alternative
interpretations (model is simply worse below majority baseline) not
explored. Every sub-phase finding was labeled "tied" or "partial
positive" where the data also supports "failed by comparable/more
margin than reported."

## Demoted claims (revisions required)

| Claim | Old framing | Corrected framing |
|-------|------------|-------------------|
| Phase 9.4 | "model fails 10-way (p=0.0003)" | "both keyword and model at/below majority baseline; model worse by 8.4% of majority" |
| Phase 9.12 | "tied at 6-way (new positive)" | "both 17 points below always-joy majority baseline; tied at catastrophic failure" |
| Phase 9.6 | "3/10 show input-level specificity" | "3/10 show descriptive accuracy drop; Fisher p ≥ 0.1 (not significant)" |
| Phase 9.10 | "SEEKING specificity newly demonstrated" | "SEEKING descriptive drop after extended pop lesion; Fisher p = 0.5 (definitely not significant)" |
| Phase 9.11 | "LLM ceiling 24% → task inherently hard" | "Gemini zero-shot 24% is weak ceiling; majority 27.6% exceeds keyword 28% — task difficulty is partly trivial-prior inflation" |

## Preserved claims (survived v4)

- **Hybrid V/A control (Phase 9.9)**: decisive null with control. Hybrid
  beats model by 14× Pearson on arousal. This finding does not depend
  on majority baseline since both methods are being compared to each
  other and to ground truth correlation — the comparison is direct.
- **Bug discovery (Phase 9.7)**: SEEKING gate missing, argmax fallback
  bias. These are real structural findings in the codebase.
- **Pop-specific bg_noise causes MSN UP-state regression (Phase 7 P1)**:
  principled model behavior, not a metric claim.

## Recommendations

### Immediate (this session)
1. Add majority baseline to all Phase 9 result tables (CR-01)
2. Report Fisher p-values for all lesion claims (CR-02)
3. Update README "positive findings" sections to honest framing
4. Update portfolio article narrative

### Short-term
5. Re-run key evaluations with MC 10-seed averaging (CR-04)
6. Pre-register lesion targets BEFORE eval for Phase 9.15+ (CR-03)
7. Include 2-way valence test (simpler task, may reveal different picture)

### Long-term
8. Phase 8 MSN UP/DOWN state (still relevant)
9. Stronger LLM ceiling (GPT-4/Claude with few-shot) if feasible
10. Larger-n lesion tests (n≥30 per class for adequate power)

## Phase 9.15 Binary valence test (v4 empirical verification)

v4 recommendation R5-01 fulfilled: tested 2-way positive vs negative
valence (simpler task). Result:

| Method | Accuracy |
|--------|---------:|
| Random | 48.5% |
| Majority (dominant valence) | 63.5% |
| **Keyword** | **66.8%** |
| **Model_rates** | **66.6%** |

Both methods **exceed majority baseline by ~3.1%** (vs keyword 3.3%).
First Phase 9 metric where both methods provide value over trivial prior.

Model and keyword are tied at binary (Δ = 0.2%). At this coarsest
granularity, both methods genuinely contribute information but neither
has unique value over the other.

Granularity pattern:
| Granularity | Model vs majority |
|-------------|------------------:|
| 10-way fine | -8.4% (below) |
| 6-way Ekman | -16.8% (far below) |
| **2-way binary valence** | **+3.1% (above)** |

Interpretation: model's ability to distinguish emotions degrades rapidly
with granularity, starting from coarse binary where it works.

## Final honest Phase 9 narrative (after v4)

The 821-neuron brain-inspired simulation:
- Performs **below majority-class baseline** on 10-way (19.2% vs 27.6%)
- Performs **far below majority baseline** on 6-way coarse (36.4% vs 53.2%)
- Lesion specificity claims are directional but **not statistically significant** at tested n
- Hybrid control proves the V/A "advantage" was the hand-coded weight table
- Only unambiguous findings are the bug discoveries (SEEKING gate,
  argmax fallback) and the hybrid V/A null with control

The "interpretable coarse classifier" positioning from v3 is not supported
by v4 data. The coarse tie is below trivial baseline; the circuit
specificity is statistically unsupported at tested n.

**Most defensible remaining positioning**:
- Educational demo (path 3d) — visualization value independent of accuracy
- Portfolio of negative-result experimentation (path 3e, updated narrative)
- Research into validation methodology failures (what validation frameworks
  induce overfit)

Path 3c (B2B interpretable AI) is functionally dead at quantitative claims.
