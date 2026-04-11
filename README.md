# Emotion-Capable Brain-Inspired AI

232検証済み論文に基づき、ヒトの10情動回路を685スパイキングニューロンでモデル化するプロジェクト。

**注意**: これは「忠実な再現」ではなく「定性的な模倣の研究用モデル」です。

## V2 アーキテクチャ (2026-04-12)

```
IntegratedBrainV2 (テキスト → 10情動 → readout → ポリシー)
├── SharedCoreNetwork (14共有領域, 245 neurons)
│   PAG(vl/dl) / BNST / PVN(CRH/OXT) / VTA(DA_lat/DA_med/GABA)
│   NAc(shell_D1/D2, core_D1) / LC / DR / aIC
├── 10 Spiking Emotion Circuits (44 populations, 685 neurons)
│   ├── FEAR: LA→BA→CeL(SOM+/PKCd+)→CeM + PL/IL + STDP
│   ├── RAGE: MeA→VMH→dlPAG + 5-HT inhibition
│   ├── SEEKING: VTA DA RPE → NAc + OFC/vmPFC/VP/LHb
│   ├── SADNESS: sgACC → habenula → VTA/DR inhibition
│   ├── DISGUST: NTS → aIC → putamen
│   ├── CARE: MPOA → VTA + PVN OXT
│   ├── PANIC/GRIEF: dACC → BNST → PAG + opioid
│   ├── PLAY: PFA thalamus → cortex + eCB/DA
│   ├── LUST: MPOA → VTA + hypothalamus
│   └── SURPRISE: LC NE burst → amygdala → PFC
├── Neuromodulation: eCB / ACh / theta / structural plasticity
├── Sleep Replay: NREM(SWR) + REM(theta consolidation)
└── Safety + LLM Bridge + Session API
```

## 文献基盤

| 情動 | 検証済み論文 | トップ論文 |
|------|-------------|-----------|
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
| **Total** | **232** | DOI 94% verified (CrossRef) |

## クイックスタート

```bash
pip install brian2 pydantic fastapi uvicorn pyyaml numpy scipy

# V2 10情動デモ
PYTHONPATH=. python scripts/emotion_brain_v2_demo.py

# テキスト入力で情動処理
PYTHONPATH=. python scripts/emotion_brain_v2_demo.py --text "I'm terrified"

# 定量バリデーション
PYTHONPATH=. python scripts/run_v2_validation.py

# テスト実行
python -m pytest tests/test_v2_emotion_brain.py tests/test_v2_shared_core.py tests/test_v2_integrated.py -v

# 論文DOI検証
PYTHONPATH=. python scripts/verify_literature_full.py --phase doi-search --limit 10
```

## バリデーション

V2スコア: **50.0% (12/24 targets PASS)** — SBI較正済み (ABC rejection, score=0.881)

| 回路 | スコア | 主要結果 |
|------|--------|---------|
| FEAR | 2/8 | LA baseline 3.9Hz(PASS), IL extinction 13.3Hz(PASS) |
| RAGE | 5/6 (83%) | MeA/VMH/dlPAG全てターゲット範囲内 |
| SEEKING | 2/4 | VTA DA tonic 10Hz(PASS), burst 20Hz(PASS) |
| SADNESS | 2/3 | sgACC 29.8Hz(PASS), DR suppressed 0Hz(PASS) |
| DISGUST | 1/3 | aIC 25.3Hz(PASS) |

V1スコア (preserved): 恐怖0.805 / 報酬0.864 / ストレス1.000 / 平均0.890

## 技術スタック

- Python 3.14 + Brian2 2.10.1 (Izhikevich spiking neurons)
- SBI: ABC rejection + differential evolution
- 232 papers verified via CrossRef API + PubMed abstracts
- Pydantic, FastAPI, numpy, scipy, PyYAML

## ライセンス

Research use only.
