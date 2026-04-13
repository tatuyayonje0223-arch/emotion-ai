# Emotion-Capable Brain-Inspired AI

232検証済み論文に基づき、ヒトの10情動回路を~740スパイキングニューロンでモデル化するプロジェクト。

**注意**: これは「忠実な再現」ではなく「定性的な模倣の研究用モデル」です。

## 科学第一原則

全パラメータ変更に論文出典を付与。数値合わせ（overfitting to metrics）は禁止。
バリデーションは「チェック」であり「最適化ターゲット」ではない。
変更ログ: [docs/parameter_changes_log.md](docs/parameter_changes_log.md)

## V2 アーキテクチャ (2026-04-13)

```
IntegratedBrainV2 (テキスト → 10情動 → readout → ポリシー)
├── SharedCoreNetwork (16共有領域, ~265 neurons)
│   PAG(vl/dl) / BNST / PVN(CRH/OXT) / VTA(DA_lat/DA_med/GABA)
│   NAc(shell_D1/D2, core_D1) / LC / DR / aIC
│   RMTg (Jhou 2009) / DRN_GABA (Challis 2013)
├── 10 Spiking Emotion Circuits (46 populations, ~740 neurons)
│   ├── FEAR: LA→BA→CeL(SOM+/PKCd+ shunting inhibition)→CeM + PL/IL
│   ├── RAGE: MeA(LTS)→VMH→dlPAG + 5-HT(DR) inhibition
│   ├── SEEKING: VTA DA RPE → NAc + OFC/vmPFC/VP/LHb→RMTg
│   ├── SADNESS: sgACC → habenula → RMTg/DRN_GABA (disynaptic)
│   ├── DISGUST: NTS → aIC → putamen (100% validation)
│   ├── CARE: MPOA → VTA + PVN OXT burst (100% validation)
│   ├── PANIC/GRIEF: dACC → BNST → PAG (100% validation)
│   ├── PLAY: PFA thalamus → cortex (100% validation)
│   ├── LUST: MPOA → VTA + hypothalamus (100% validation)
│   └── SURPRISE: LC NE burst → amygdala (100% validation)
├── Neuromodulation: eCB / ACh / theta / structural plasticity
├── Sleep Replay: NREM(SWR) + REM(theta consolidation)
└── Safety + LLM Bridge + Session API
```

### Key innovations
- **CeA shunting inhibition** (Chance 2002 PNAS; Li 2013 Nat Neurosci): conductance-based SOM+→PKCd+ → PKCd+=0Hz during CS
- **RMTg GABAergic relay** (Jhou 2009 J Neurosci): disynaptic LHb→RMTg→VTA DA pause pathway
- **DRN internal GABA** (Challis 2013 J Neurosci): LHb→DRN_GABA→DR 5-HT suppression
- **LTS adapted params** (Lopez de Armentia 2004): b=0.22, d=4 for CeL/BNST/MeA neurons
- **LHb burst firing** (Yang 2018 Nature): phenomenological burst for DA pause

## バリデーション

**Score: 87.5% (28/32)** with strict targets (literature typical ±30%)

| Emotion | Score | Key Results |
|---------|-------|-------------|
| FEAR | 6/8 | la_exc 3.8/21Hz, CeA disinhibition working, shunting PKCd=0Hz |
| RAGE | 4/6 | MeA 6.3Hz, VMH scalable 2.7/10.5/26.7Hz |
| SEEKING | 3/4 | VTA DA tonic 6.7Hz, burst 20Hz (Schultz 1997 match) |
| SADNESS | 2/3 | sgACC 16.7Hz, habenula 20Hz |
| DISGUST | **3/3 (100%)** | aIC 14.7, NTS 16.7, putamen 9.7Hz |
| CARE | **2/2 (100%)** | MPOA 10Hz, PVN OXT 7Hz |
| PANIC/GRIEF | **2/2 (100%)** | dACC 13.3, BNST 9.1Hz |
| PLAY | **1/1 (100%)** | PFA 10Hz |
| LUST | **1/1 (100%)** | MPOA 10.3Hz |
| SURPRISE | **2/2 (100%)** | LC 10Hz, amygdala 10Hz |

### Structural limitations (4 FAIL)
- PL fear: 16.7Hz (quantization, -0.3Hz)
- VMH attack: 20Hz (drive insufficient for 24-46Hz target)
- **VTA DA pause: 6.7Hz** (Izhikevich current-based limitation; needs conductance-based model)
- **DR sadness: 6.7Hz** (same structural limitation)

## 文献基盤

232 verified papers across 10 emotions + brain connectome.
DOI verification: 94% valid (CrossRef API). Content match: 70% (PubMed abstracts).
24 parameter changes with paper citations logged.

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
python -m pytest tests/test_v2_emotion_brain.py tests/test_v2_shared_core.py tests/test_v2_integrated.py -v
```

## 技術スタック

- Python 3.14 + Brian2 2.10.1 (Izhikevich spiking neurons)
- Conductance-based (shunting) inhibition for CeA disinhibition
- SBI: ABC rejection parameter estimation
- 232 papers verified via CrossRef API + PubMed abstracts
- Pydantic, FastAPI, numpy, scipy, PyYAML

## ライセンス

Research use only.
