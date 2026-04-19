# Emotion-Capable Brain-Inspired AI

10情動回路を ~821 スパイキングニューロンでモデル化した**研究用探索プロジェクト**です。
232論文の解剖学・生理学パラメータを参照して構築していますが、**忠実な再現**ではなく、
定性的模倣を通じた「情動が神経回路からどう emerge しうるか」の探索を目的とします。

> ⚠️ **Current status**: Research-stage exploration model. Not a production emotion-recognition system.
> Validation framework is rate-matching (not behavioral prediction). Known limitations documented below.

## プロジェクト方針

- **神経科学文献に整合**しつつ、**実装は簡略化モデル** (1-population-per-region, 821 neurons vs 10^11 biological)
- **Validation はチェック目的** であり、最適化ターゲットとしての使用は可能な限り避ける
- 全パラメータ変更は citation 付きで `docs/parameter_changes_log.md` に記録
- ただし、**27件の change のうち一部は empirical calibration** (target rate から逆算) であることを認める。詳細は同ログ参照

## V2 アーキテクチャ (2026-04-15)

```
IntegratedBrainV2 (テキスト → 10情動 → readout → ポリシー)
├── SharedCoreNetwork (19共有領域, ~312 neurons)
│   PAG(vl/dl) / BNST / PVN(CRH/OXT) / VTA(DA_lat/DA_med/GABA)
│   NAc(shell_D1/D2, core_D1) / LC / DR / aIC
│   RMTg (Jhou 2009) / DRN_GABA (Challis 2013) / PPTg (Grace 2007)
├── 10 Spiking Emotion Circuits (53 populations, ~821 neurons)
│   ├── FEAR: LA→BA→CeL(SOM+/PKCd+/CRF+/VIP+)→CeM + PB/PV + dHPC/vHPC + PL/IL
│   ├── RAGE: MeA(LTS)→VMH→dlPAG + 5-HT(DR) inhibition
│   ├── SEEKING: VTA DA RPE → NAc + OFC/vmPFC/VP/LHb→RMTg
│   ├── SADNESS: sgACC → habenula → RMTg/DRN_GABA (disynaptic)
│   ├── DISGUST: NTS → aIC → putamen
│   ├── CARE: MPOA → VTA + PVN OXT burst
│   ├── PANIC/GRIEF: dACC → BNST → PAG
│   ├── PLAY: PFA thalamus → cortex
│   ├── LUST: MPOA → VTA + hypothalamus
│   └── SURPRISE: LC NE burst → amygdala
├── Conductance-based GABA_A inhibition (g_inh state variable)
├── Neuromodulation: eCB / ACh / theta / structural plasticity
├── Sleep Replay: NREM(SWR) + REM(theta consolidation)
└── Safety + LLM Bridge + Session API
```

### 技術的見どころ
- **Conductance-based (g_inh) shunting inhibition** (Chance 2002 PNAS; Bartos 2007)
- **CeA expanded microcircuit**: SOM+/PKCd+/CRF+/VIP+/PV+ + PB nociceptor relay
- **Disynaptic VTA DA pause**: LHb→RMTg→VTA (Jhou 2009)
- **Dual neuron model**: Izhikevich (default) + AdEx (optional)
- **STDP + reward-modulated plasticity**

## Validation — 現在のスコア

### Scenario targets (scenario-evoked firing rate vs literature range, ±30% tolerance)

単一 trial (trial_num=0) と Monte Carlo 5-trial の両方で報告:

| Model | Single-trial | **MC 5-trial stable PASS** | Unstable (boundary) | Stable FAIL |
|-------|-------------:|---------------------------:|--------------------:|------------:|
| Izhikevich | 36/36 | **35/36** | 1/36 | 0/36 |
| AdEx | 28/36 | **25/36** | 4/36 | 7/36 |

MC平均化下での stable PASS が信頼できる指標。Single-trial PASS は境界線付近で seed-dependent。

### Baseline physiology (no-input resting state)

| Model | Baseline PASS |
|-------|--------------:|
| Izhikevich | **6/20** |
| AdEx | **6/20** |

主な逸脱: MSN (putamen/nac_shell) 10 Hz vs 文献 <1 Hz (Humphries 2005), LC 8-10 Hz vs 1-3 Hz (Sara & Bouret 2012)。
根本原因: グローバル `bg_noise=1.7` + 簡略化 1-pop-per-region モデルに UP/DOWN state dynamics 不在。

## Known Limitations (2026-04-19 独立監査で確定)

1. **Rate-matching validation は predictive power 不足** — behavioral prediction
   **empirically confirmed as null at statistical significance** (Phase 9):
   - Pilot (n=38 embedded): model 36.8% vs keyword 55.3%, p=0.070
   - **Full GoEmotions (n=500 validation): model 19.2% vs keyword 28.0%, McNemar p=0.0003**
   - Systematic SEEKING over-prediction (355/500 predictions vs 90/500 true)
   - Details: `docs/phase9_results_initial.md`, `docs/phase9_results_full.md`
   - **Circuit specificity test**: lesioning primary inputs silences FEAR/RAGE/SADNESS
     classification (baseline → 0%), but not SEEKING (readout bias). Partial interpretability
     value retained. Details: `docs/phase9_lesion_specificity.md`
   - **Readout fix (SEEKING gate + argmax)**: parity restored (22.2% model vs 28.0% keyword,
     p<0.0001). **"Model correct ∩ keyword wrong = 0 instances"**: neural simulation provides
     zero unique behavioral prediction value after bias is removed.
     Details: `docs/phase9_readout_fix.md`
   - **Dimensional affect regression**: first partial positive — model beats keyword on
     V/A MAE by 10-13% (n=500). But joint R² is negative for all baselines; likely driven
     by hand-coded emotion→V/A weight table, not circuit dynamics. Control experiment
     (Phase 9.9) needed to confirm. Details: `docs/phase9_dimensional_va.md`
2. **21+ parameter changes は target rate から逆算した empirical calibration** — 紙から直接 derive した値ではない
3. **Baseline rate violation** — scenario PASS と baseline PASS が同時に成立しない (simplified model の構造的限界)
4. **Monte Carlo n=5 は統計的に不足** — neuroscience 標準は n≥20 bootstrap CI
5. **Simulation duration 300ms** — 1 spike/10 neurons = 0.33 Hz quantization
6. **dt=0.5ms with Euler method** — Izhikevich 2003 は dt≤0.1ms 推奨
7. **10 emotion taxonomy は Panksepp(7) + Ekman(3) の混合** — 理論的に競合する2 model の折衷
8. **SEEKING over-prediction bias**: readout function が CARE/LUST/PLAY/PANIC の多くを
   SEEKING と判定 (Phase 9 confusion matrix 参照)

詳細: `docs/parameter_changes_log.md` (Audit v2 section) + `docs/phase9_results_initial.md`

## Roadmap

- **Phase 6**: GPU scaling (owner) + MC validation CI 組込
- **Phase 7**: 方法論改善 (duration 拡張、adex_scale 撤去、citation 再検証、dt 改善)
- **Phase 8**: MSN UP/DOWN state dynamics 実装（baseline と task 両立のため）
- **long-term**: Behavioral prediction validation 枠組み、OpenNeuro/Allen dataset 比較、preprint 執筆

## 文献基盤

232 verified papers (CrossRef API 94% valid, PubMed abstract match 70%).

| Emotion | Papers | Top Paper |
|---------|--------|-----------|
| FEAR | 30 | Duvarci & Pare 2014 Neuron |
| RAGE | 25 | Golden 2016 Nature |
| SEEKING | 23 | Nestler & Carlezon 2006 |
| SADNESS | 19 | Hamilton 2015 Biol Psychiatry |
| DISGUST | 18 | Small 2003 Neuron |
| CARE | 20 | Kirsch 2005 J Neurosci |
| PANIC/GRIEF | 21 | Gundel 2003 Am J Psychiatry |
| PLAY | 15 | Siviy & Panksepp 2011 |
| LUST | 15 | Dominguez & Hull 2005 |
| SURPRISE | 18 | Sara & Bouret 2012 Neuron |
| Connectome | 48 | Kober 2008 NeuroImage |

## クイックスタート

```bash
pip install brian2 pydantic fastapi uvicorn pyyaml numpy scipy

# V2 10情動デモ
PYTHONPATH=. python scripts/emotion_brain_v2_demo.py
PYTHONPATH=. python scripts/emotion_brain_v2_demo.py --text "I'm terrified"

# 対話チャット (MockProvider: API不要)
PYTHONPATH=. python scripts/emotion_chat.py --mock

# バリデーション (single-trial)
PYTHONPATH=. python scripts/run_v2_validation.py           # Izhikevich
PYTHONPATH=. python scripts/run_v2_validation.py --adex    # AdEx

# Monte Carlo validation (seed 安定性)
PYTHONPATH=. python scripts/evaluate_multitrial.py --n-trials 5 --model both

# Baseline physiology check
PYTHONPATH=. python scripts/validate_baseline_rates.py

# テスト
python -m pytest tests/ -v
```

## 技術スタック

- Python 3.14 + Brian2 2.10.1
- Dual neuron model: Izhikevich + AdEx (custom tau_m=1ms to match Izhikevich time scale)
- Conductance-based GABA_A (g_inh)
- SBI: ABC rejection parameter estimation
- 232 papers verified via CrossRef + PubMed
- Pydantic / FastAPI / numpy / scipy / PyYAML

## ライセンス

Research use only. Not intended for production emotion recognition or clinical application.

## Independent Audits (self-conducted, 2026-04)

- v1 audit (commit 0fa1ade): AdEx 36/36 over-calibration detected, reverted
- v2 audit (commit ec3732c): Izh 36/36 claim も 27 changes の overfit pattern 上に成立を発覚、
  Monte Carlo honest scoring 確立
- v3 audit: Production fitness 評価、rate-matching paradigm の限界特定、
  behavioral validation へ pivot 計画

監査レポートは `docs/parameter_changes_log.md` 参照。
