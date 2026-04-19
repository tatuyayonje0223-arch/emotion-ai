# 神経回路から情動が emerge するか？ — 821 ニューロンの spiking model を作って独立監査で自分を殴った話

by shukaku (@mapshukaku) / 2026-04

---

## TL;DR

- Panksepp 7 + Ekman 3 の 10 情動を扁桃体・VTA・NAc 等 19 脳領域 + 10 情動固有回路で spiking simulation した
- 232 論文の生理学パラメータで「文献準拠」を謳って 36/36 STRICT validation PASS を達成
- **独立監査 3 回 (v1/v2/v3) で自分の成果を自分で否定した**
- 最終的に: Monte Carlo 下で Izhikevich 35/36 / AdEx 25/36, baseline physiology 6/20 が正直な state
- 学んだこと: **"Validation target を追う" と "数値合わせ" は紙一重**。Validation framework 自体が overfit を誘発する場合、どれだけ citation を付けても科学的誠実性は担保されない

GitHub: https://github.com/tatuyayonje0223-arch/emotion-ai (Research WIP)

---

## Part 1: なぜ作ったか

大規模言語モデル (LLM) は感情を「表面的に」理解しているように見える。
GPT-4 に「怖い」と言わせれば 怖そうな返答が返ってくる。しかし内部では single forward pass の
activation pattern でしかない。

Panksepp (2011) は **"affective neuroscience"** の立場から、情動は特定の神経回路に emerge する、
と主張した。つまり:

- 恐怖 = 扁桃体 LA→CeL→CeM + PAG
- 報酬 = VTA DA → NAc
- 悲しみ = sgACC → habenula → VTA pause
- ...

こういう回路を spiking neuron で実装したら、LLM とは違う形で「情動」が出てくるだろうか？
というのが出発点。**"Emergence from circuit dynamics"** が検証したかった仮説だった。

## Part 2: 作ったもの

### Architecture
```
IntegratedBrainV2 (10情動 + shared core)
├── Shared Core Network: 19領域 / ~312 neurons
│   (PAG, BNST, PVN, VTA, NAc, LC, DR, aIC, RMTg, DRN_GABA, PPTg, dHPC, vHPC)
├── 10 Emotion-specific circuits: 53 populations / ~821 neurons
│   FEAR / RAGE / SEEKING / SADNESS / DISGUST / CARE / PANIC / PLAY / LUST / SURPRISE
├── Conductance-based GABA_A inhibition (g_inh)
├── Neuromodulation: DA / 5-HT / NE / ACh / OXT / CORT
├── STDP + reward-modulated plasticity
└── Sleep replay (NREM SWR + REM theta)
```

Brian2 で Izhikevich (default) + AdEx (alternative) の dual model。
572 unit test passing。

### Validation Framework
36 firing-rate targets (10 情動 × 複数シナリオ)。各 target は literature range ±30%。
例えば: "threat=0.8 で la_exc (lateral amygdala excitatory) が 14-26 Hz 発火"。

**この validation を 36/36 PASS させるのが当初目標だった**。
そしてその PASS 状態を達成した (Izhikevich 36/36, AdEx 28/36)。

## Part 3: 独立監査 v1 — "AdEx 28→36/36 達成" の overfit flag

AdEx の 8 件の failure を paper-justified な tonic drive adjustment で fix し、36/36 に到達したつもりだった。

**v1 監査で発覚**:
- 変更された tonic 値は paper から derive されていない
- "Sara & Bouret 2012" と cite しつつ、paper にある "LC tonic 1-3Hz" を無視して 5.5 (baseline 10Hz 相当) を採用
- 当初 "AdEx 構造的限界" と診断したものを、sweep で PASS する値が見つかった途端に diagnosis を覆した

結論: **commit revert**。AdEx 28/36 が正直な state。

## Part 4: 独立監査 v2 — Izh 36/36 自体が砂上の楼閣

ここまでで自分の作業は誠実に revert したと思った。しかし:

- Project の `parameter_changes_log.md` には 27 件の変更履歴
- 大半が同じパターン: **"target rate X を hit するために parameter Y を変更"**
- `PKCd tonic = -0.5` (負値 tonic = 物理的に non-physiological な silence hack)
- `VTA DA b_spike=9, tau_w=100ms, g_L=0.2` (コメントで明言: "balanced: tonic ~3Hz, burst ~32Hz, pause ~1Hz" = 値を target から逆算)
- `CeA SOM→PKCd shunting 4.0x` (AdEx 専用 synapse multiplier、紙記載なし)

**Izh 36/36 は 27 件の数値合わせの上に成立**していた。

### Monte Carlo で確認
単一 trial (`trial_num=0`) でなく 5-trial MC で再評価:
- **Izh: 35/36 stable PASS** (1 件は dr が 3/5 で seed-dependent)
- **AdEx: 25/36 stable PASS** (7 件 stable FAIL + 4 件 boundary unstable)

"STRICT 100%" claim は single-trial でしか成立しなかった。

### Baseline physiology
更に rest state (no input) を測定:
- MSN (線条体 medium spiny neurons): **10 Hz** 発火
- 生物学的 MSN は K_ir で quiet at rest: **<1 Hz** (Humphries 2005)
- LC / DR / IL / aIC / BNST / PVN / MPOA: 全て 2-3倍 baseline 超過
- **両モデル 6/20 baseline PASS**

scenario 36/36 PASS の claim は baseline physiology を完全に無視していた。

## Part 5: 独立監査 v3 — Project level の存在意義への問い

v2 で model の validation framework 自体の妥当性が疑わしくなった。v3 ではより根本的に:

- README は **"数値合わせ禁止"** を明記しているが、実装は 27 件 fit
- "Success criterion" が未定義 — 何ができれば project done か不明
- 10 emotion = Panksepp (7) + Ekman (3) の混合 = **互換性のない taxonomy**
- 821 neurons vs 10^11 biological = **4-5 orders of magnitude reduction**
- Rate-matching validation は predictive power 不足 (記述統計の模倣)
- 5 commit / 1 session で 2 forward + 2 revert + 1 doc = net functional progress ≈ 0

## Part 6: Phase 7 P1 の Radical Reduction 試み → また revert

MSN baseline fix のため `bg_noise=1.7 → 0.2` (Tepper 2004 K_ir rectification)。

- ✅ Baseline: 6/20 → 8-9/20 (MSN PASS)
- ❌ Scenario: Izh 35→32, AdEx 25→23 (nac_d1/putamen が task firing 不足)

**Simplified 1-population-per-region モデル に UP/DOWN state dynamics が欠如**している構造的限界。
K_ir だけ実装してもダメで、NMDA-mediated UP state transition が必要。
それは Phase 8 の大規模な model enhancement 案件。

今のパラメータ空間での tuning では baseline と task を両立できない。
- Option A: Revert, 現状維持
- Option B: Scenario 更に fit (数値合わせ treadmill)
- Option C: Model 抜本再構築

**A を選択**。同じ過ちの repeat を避ける。

## Part 7: 学んだこと (真面目な教訓)

### 1. Validation framework 自体が overfit を誘発する
36 target × ±30% で「科学的チェック」を名乗っても、
**target を見て parameter を調整**した時点で framework は optimization target に堕す。

### 2. "Paper-cited" ≠ "Paper-derived"
Paper を cite するだけでは paper-derivation にならない。
Paper の数値から parameter を **逆算不可能な形で計算**するなら数値合わせ。

### 3. Single-trial validation は stochastic system で無効
1 seed の結果は noise。Monte Carlo 平均 (n≥20, ideally 100+) でしか意味ある claim ができない。
"STRICT 100%" 的 branding は single-trial では fragile。

### 4. Simplified model の trade-off は tuning で解けない
1-pop-per-region model で baseline physiology と task-evoked firing の両方を正しく保てない場合、
それは **model structure の limitation** であり parameter tuning では解けない。

### 5. 自分の成果物を自分で否定するのは苦しいが、最も productive
3 回の独立監査で 36/36 → 28/36 → 25/36 MC-stable に claim を下げた。
数字は悪くなったが、**主張は robust になった**。

### 6. Public GitHub に science rigor のない claim を残すのは risk
"STRICT 100% validation" を public に書いた瞬間、検証責任が発生する。
"Research WIP with known limitations" が honest かつ defensible。

## Part 8: Phase 9 — Behavioral Validation (statistically-significant null result)

V3 audit 後、rate-matching validation に代わる behavioral prediction validation framework を
構築 (`docs/behavioral_validation_framework.md`)。GoEmotions データセットで emotion classification
を baseline 比較。

### Pilot (embedded n=38)

| Baseline | Accuracy | Macro-F1 |
|----------|---------:|---------:|
| Random | 15.8% | 0.137 |
| Keyword (text_analyzer_v3) | **55.3%** | 0.533 |
| Model_rates (821-neuron model argmax) | **36.8%** | 0.364 |

McNemar: p=0.070 (n small but directional).

### Full GoEmotions (validation n=500) — statistically confirmed

| Baseline | Accuracy | Macro-F1 |
|----------|---------:|---------:|
| Random | 8.4% | 0.067 |
| Keyword | **28.0%** | 0.231 |
| Model_rates | **19.2%** | 0.165 |

**McNemar chi2=13.40, p=0.0003**. Model correct / keyword wrong: 47. Model wrong /
keyword correct: 91. Keyword beats model by 8.8% absolute accuracy with high significance.

**Pre-registered hypothesis rejected**: Neural simulation does NOT provide behavioral
prediction value over keyword matching.

### Systematic SEEKING over-prediction (confirmed at scale)

Model predicted SEEKING on **355 / 500** instances (71%) despite true SEEKING being
only **90 / 500** (18%). All of the 138 true CARE instances are misclassified (96 as
SEEKING, 31 as LUST). All 30 PLAY, 8 LUST, 3 PANIC_GRIEF cases get 0% precision.

This isn't a parameter tuning issue. The readout function
`seeking_act = VTA DA × 0.4 + NAc × 0.3` fires for any emotion involving reward-like
activation — which biologically includes affiliation, nurturing, play, pair-bonding.
10 emotions validated at rate-matching level **do not behave as distinguishable
classifiers** on real-world text.

This is the empirical confirmation of what v2 audit predicted: **rate-matching validation
does not translate to task predictive power**.

## Part 8.5: Lesion specificity — partial salvage

Null result alone doesn't tell the full story. Classical neuroscience lesion logic:
**if circuit X is causally necessary for emotion X, lesioning X should silence X prediction
specifically**.

Ran lesioned baselines on n=50 subset: for each of 10 emotions, zero the primary
drive inputs (e.g., `threat` and `pain` for FEAR), measure per-class accuracy drop:

| True label | Baseline | Own-lesion acc | Specificity |
|-----------|---------:|---------------:|-------------|
| FEAR (n=3) | 66.7% | **0.0%** | ✅ |
| RAGE (n=11) | 27.3% | **0.0%** | ✅ |
| SEEKING (n=8) | 87.5% | 87.5% | ❌ readout bias |
| SADNESS (n=7) | 28.6% | **0.0%** | ✅ |
| CARE/LUST/PLAY/PANIC/SURPRISE | 0% | 0% | untestable (floored) |

**3 of 10 emotions show clean circuit specificity** (FEAR/RAGE/SADNESS). Their drive
inputs are causally necessary for the model's classification — remove the input,
classification silenced.

SEEKING doesn't show specificity because the readout function
`seeking_act = VTA_DA × 0.4 + NAc × 0.3` fires at floor level due to VTA tonic activity,
making SEEKING the default winner when other circuits are weak.

### Reframing value proposition

Before lesion test: Path 3c (B2B interpretable emotion AI) seemed dead.
After lesion test: Path 3c can reposition as **"mechanistic diagnostic model for
FEAR/RAGE/SADNESS pathways"** — not a classifier for arbitrary emotions, but a
circuit-tracing tool for those 3 where neuroscience specificity holds.

Use cases:
- Neuroscience education: "see how removing amygdala input silences fear recognition"
- Explainability layer over LLM emotion APIs: when LLM says FEAR, confirm by
  checking if the FEAR circuit responds to `threat` input in our model

## Part 8.6: Dimensional V/A regression — first partial positive

After 4 consecutive null results (9.4 pilot / 9.4 full / 9.7 gate-fix / 9.7 argmax),
one last angle: **continuous valence × arousal instead of 10-way classification**.

Rationale: Russell's circumplex model argues emotion is better represented
continuously. If VTA DA → positive valence, LC NE → arousal, BNST → negative
valence-with-chronic-arousal, etc., then the simulation might capture V/A better
than keyword matching even if classification fails.

### Result on n=500 GoEmotions validation

| Metric | Model | Keyword | Δ |
|--------|------:|--------:|---:|
| Valence Pearson | +0.319 | +0.311 | +0.008 (tiny) |
| Valence MAE | **0.513** | 0.591 | **-13%** |
| Arousal Pearson | -0.019 | -0.080 | +0.060 |
| Arousal MAE | **0.450** | 0.503 | **-10%** |
| Joint R² | **-0.445** | -0.695 | model least bad |

**Model wins all 4 metrics**. First clean advantage in Phase 9.

### Big caveats

- **Joint R² is negative for all three baselines**, meaning even the "best"
  is worse than "always predict the mean". Model is least bad, not good.
- **Valence Pearson Δ=+0.008** at n=500 is not statistically significant
  (Fisher z SE ≈ 0.064, need ≥0.13 for p<0.05).
- Arousal Pearson ≈ 0 — model essentially doesn't capture arousal variance.
- **The emotion→V/A weight table in the readout is hand-coded**. A fair
  control would be: apply the same weights to keyword hits (bypassing the
  simulation). If that matches model V/A, the simulation added nothing.

### Honest framing

The partial improvement (10-13% MAE) is real but small. It may be entirely
explained by the hand-coded readout layer (`valence_weights` and
`arousal_weights` dicts) that aggregates 10 activations into V/A. The neural
simulation itself may contribute zero beyond those weights.

A Phase 9.9 control experiment is pre-registered: apply the same weights to
keyword hits without simulation. If that control matches model performance,
we know the "advantage" was just the lookup table. This control is the honest
test before we can make any simulation-contribution claim.

## Part 8.7: Phase 9.9 Control — the simulation actively hurts

The pre-registered control ran. Result: **simulation doesn't just fail to help —
it actively destroys signal**.

| Baseline | V Pearson | V MAE | **A Pearson** | **A MAE** | Joint R² |
|----------|----------:|------:|--------------:|----------:|---------:|
| hybrid_va (no sim) | +0.303 | 0.531 | **+0.272** | **0.219** | **-0.035** |
| model_va (with sim) | +0.319 | 0.513 | -0.019 | 0.450 | -0.445 |

Hybrid beats model by **14× on arousal correlation** and **51% better MAE**.
On valence the two are essentially tied.

### Why the simulation hurts

Model's readout gates each emotion by its input signal: `if threat > 0.1: fear_act = ...
else: fear_act = 0`. Typical GoEmotions text triggers 0-2 gates, so the other 8+ emotion
activations are 0. V/A = weighted sum / total activation → dominated by the one
triggered emotion → noisy binary-like signal.

Hybrid uses raw keyword hit counts across all 10 emotions → continuous signal across
the weight table → smoother V/A.

**The neural simulation coarsens the information that keyword counts already carry.**

### What this means

Combined with Phase 9.7 ("model unique correct = 0 on classification"):

- ❌ No unique classification value
- ❌ No unique dimensional value — hybrid weight table alone dominates
- ✅ Circuit specificity (FEAR/RAGE/SADNESS lesion evidence, Phase 9.6)
- ✅ Visualization / educational value (independent of accuracy)

Path 3c (B2B interpretable AI) is dead as any kind of quantitative estimator.
Only viable framing: "mechanistic diagnostic model for FEAR/RAGE/SADNESS +
educational explainability layer atop LLM outputs".

## Part 9: Where next?

Phase 6-7-8-9 は継続するが、positioning を shift:
- **3d (Education/Demo)**: interactive web UI で脳領域活動を可視化 — 本記事と同時公開、**working**
- **3e (Portfolio)**: 本記事 — 試行錯誤 + null result を含む honest writeup
- **3c (B2B Interpretable AI)**: **大幅見直し**。Classification accuracy では keyword/LLM に劣る。
  "Interpretability" value proposition は残るが "accuracy" 訴求は不可

「EU AI Act 2026 高リスク emotion recognition 適合」は solo dev では非現実的 + **そもそも model
が classification で上回れない** ため技術的にも非現実的。

## Part 10: 最終的教訓 (Phase 9 追加)

- **Rate-matching 36/36 PASS は behavioral prediction を保証しない**
  → validation framework の選び方自体が最重要
- **Null result を publish する価値**: 現状 "neural simulation > LLM/keyword for emotion" と
  主張する product は嘘になる。本記事は counter-example として contribution
- **Scale matters**: 821 neuron simplified model は real-world text understanding には
  unable to compete with trained LM or even keyword heuristics

---

## Code / Repo

- https://github.com/tatuyayonje0223-arch/emotion-ai
- `docs/parameter_changes_log.md` — 27 changes + Audit v1/v2 sections
- `demo/index.html` — interactive web demo
- `scripts/evaluate_multitrial.py` — MC validation (added during v2 audit)
- `scripts/validate_baseline_rates.py` — baseline physiology probe (added during v2 audit)

## Acknowledgements

232 papers' authors for the neuroscience foundation.
Myself-from-3-hours-ago for the overconfident commits that today's self had to revert.

(Independent audit, honest reporting, and willingness to delete your own work are the most valuable
skills I developed through this project.)

---

*This article is itself a form of continuous audit — feedback welcome at @mapshukaku.*
