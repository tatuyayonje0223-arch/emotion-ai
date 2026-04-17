# Emotion-Capable Brain-Inspired AI

232検証済み論文に基づき、ヒトの10情動回路を~821スパイキングニューロンでモデル化するプロジェクト。

**注意**: これは「忠実な再現」ではなく「定性的な模倣の研究用モデル」です。

## 科学第一原則

全パラメータ変更に論文出典を付与。数値合わせ（overfitting to metrics）は禁止。
バリデーションは「チェック」であり「最適化ターゲット」ではない。
変更ログ: [docs/parameter_changes_log.md](docs/parameter_changes_log.md)

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

### Key innovations
- **Conductance-based (g_inh) shunting inhibition** (Chance 2002 PNAS; Bartos 2007): continuous GABA_A conductance state with 5ms decay — replaces instantaneous voltage kicks. Achieves true VTA DA pause (0.3 Hz) and DR suppression (2.4 Hz)
- **CeA expanded microcircuit**: SOM+/PKCd+/CRF+/VIP+/PV+ populations + PB nociceptor relay (Li 2013; McCullough 2018; Royer 2016; Pomrenze 2019)
- **RMTg GABAergic relay** (Jhou 2009 J Neurosci): disynaptic LHb→RMTg→VTA DA pause pathway
- **DRN internal GABA** (Challis 2013 J Neurosci): LHb→DRN_GABA→DR 5-HT suppression
- **PPTg tonic excitation** (Grace 2007; Mena-Segovia 2008): explicit spiking population provides tonic drive to VTA DA. Loss → RMTg inhibition of PPTg → excitatory withdrawal
- **PL→DR excitatory pathway** (Celada 2001; Aghajanian 1999): PFC provides tonic Glu to DR. Loss → sgACC→PL inhibition → DR suppression emerges from circuit dynamics
- **LTS adapted params** (Lopez de Armentia 2004): b=0.22, d=4 for CeL/BNST/MeA neurons
- **LHb burst firing** (Yang 2018 Nature): phenomenological burst for DA pause

## バリデーション

**Score: 100.0% (36/36)** with strict targets (literature typical ±30%)

| Emotion | Score | Key Results |
|---------|-------|-------------|
| FEAR | **8/8 (100%)** | la_exc 4.2/21.2Hz, CeA shunting PKCd=0Hz, CeM 10Hz, PL 19.7Hz |
| RAGE | **6/6 (100%)** | MeA 5.7/19.5Hz, VMH 2.9/10.9/24.8Hz, dlPAG 20Hz |
| SEEKING | **4/4 (100%)** | VTA DA tonic 6.0, burst 19.8, **pause 0.3Hz**, NAc 10.5Hz |
| SADNESS | **3/3 (100%)** | sgACC 16.7Hz, habenula 20Hz, **DR suppressed 2.4Hz** |
| DISGUST | **3/3 (100%)** | aIC 15.3, NTS 16.3, putamen 9.5Hz |
| CARE | **3/3 (100%)** | MPOA 10Hz, PVN OXT 7.7Hz, VTA 8.4Hz |
| PANIC/GRIEF | **3/3 (100%)** | dACC 13.3, BNST 9.8, PVN CRH 10Hz |
| PLAY | **2/2 (100%)** | PFA 10Hz, play cortex 13.3Hz |
| LUST | **2/2 (100%)** | MPOA 10Hz, VTA 12.6Hz |
| SURPRISE | **2/2 (100%)** | LC 10Hz, amygdala 10Hz |

### Conductance-based inhibition model
Previous structural limitations (VTA DA pause at 6.7Hz, DR at 6.7Hz) resolved by replacing instantaneous voltage-kick shunting with continuous GABA_A conductance dynamics:
- `dg_inh/dt = -g_inh / tau_g` (tau_g = 5ms, GABA_A)
- `I_inh = g_inh * clip(v + 75, 0, 200)` (E_GABA = -75mV, clamped for Izhikevich stability)
- VTA DA pause: 6.7Hz → **0.3Hz** (true pause via RMTg sustained inhibition)
- DR suppression: 6.7Hz → **2.4Hz** (partial suppression via DRN_GABA)

## 文献基盤

232 verified papers across 10 emotions + brain connectome.
DOI verification: 94% valid (CrossRef API). Content match: 70% (PubMed abstracts).
26 parameter changes with paper citations logged.

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

# テキスト入力
PYTHONPATH=. python scripts/emotion_brain_v2_demo.py --text "I'm terrified"

# 定量バリデーション (strict ±30%)
PYTHONPATH=. python scripts/run_v2_validation.py

# テスト
python -m pytest tests/ -v
```

## 技術スタック

- Python 3.14 + Brian2 2.10.1 (dual model: Izhikevich 36/36 + AdEx 16/16, both 100%)
- Conductance-based (g_inh) GABA_A shunting inhibition
- SBI: ABC rejection parameter estimation
- 232 papers verified via CrossRef API + PubMed abstracts
- Pydantic, FastAPI, numpy, scipy, PyYAML

## ライセンス

Research use only.
