# Emotion-Capable Brain-Inspired AI

情動神経回路の簡略モデルを探索するプロジェクト。

## 概要

547スパイキングニューロン（Izhikevich）+ 4 mean-field領域で、ヒトの情動回路を定性的に模倣する。
恐怖条件付け/消去、報酬学習、ストレス応答の3回路をBrian2上に実装。

**注意**: これは「忠実な再現」ではなく「定性的な模倣のトイモデル」です。

## クイックスタート

```bash
pip install brian2 pydantic fastapi uvicorn pyyaml numpy scipy

# 脳+LLM統合対話（API不要）
python scripts/emotion_chat.py --mock

# E2Eデモ（恐怖→睡眠→消去）
python scripts/emotion_brain_demo.py

# 定量検証レポート
python scripts/run_full_validation.py

# API起動
python scripts/run_api.py
```

## アーキテクチャ

```
EmotionBrain (正式最終システム)
├── ハイブリッド脳
│   ├── Brian2スパイキング (547 neurons)
│   │   ├── 恐怖: LA→BA→CeL(SOM+/PKCd+脱抑制)→CeM + PL/IL + BNST
│   │   ├── 報酬: VTA(DA tuned a=0.01,d=10)→NAc(Shell/Core×D1/D2)
│   │   └── ストレス: BLA→PVN(MR/GR) + LC
│   └── AdEx mean-field (4 regions)
│       └── 島皮質 / ACC / dlPFC / 海馬
├── 神経修飾: eCB / ACh / シータ / 構造的可塑性
├── 睡眠リプレイ: NREM(SWR) + REM(シータ固定化)
└── 安全チェック + LLMブリッジ + セッションAPI
```

## 検証スコア

| 回路 | スコア | 主要結果 |
|------|--------|---------|
| 恐怖 | 0.805 | BLA 9.1Hz(目標8), SOM+/PKCd+ 3.28(目標3.0) |
| 消去 | PASS | 40.9%低下(目標30%+) |
| 報酬 | — | DA tonic 3.5Hz(目標5), burst 29.5Hz(目標25) |
| ストレス | 1.000 | 全5項目PASS |

## 監査

9ラウンドの独立監査で完全収束（CRITICAL/HIGH/MEDIUM/LOW = 全て0）。

## 技術スタック

- Python 3.14 + Brian2 2.10.1
- Pydantic, FastAPI, numpy, scipy
- Allen Brain Atlas API integration
- SBI (ABC rejection) parameter estimation

## ライセンス

Research use only.
