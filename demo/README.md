# EmotionAI Interactive Demo

教育用 web デモ — 10 情動回路の発火パターンをブラウザで可視化。

## 起動方法

### 1. FastAPI server を起動
```bash
cd /path/to/EmotionAI
PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. CORS を有効化 (demo ブラウザからの fetch 許可)
`src/api/main.py` に以下を追加 (既に実装済みの場合は不要):
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; production では限定
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3. demo/index.html をブラウザで開く
- 直接 file:// で開く、または
- 簡易 HTTP server 経由: `cd demo && python -m http.server 8080` → http://localhost:8080

## 機能

- **9 入力 slider** (threat / reward / social / novelty / pain / loss / frustration / contamination / attachment_need)
- **Preset** buttons: Fear / Reward / Social / Novelty
- **出力可視化**:
  - 10 情動活性度 (emotion readout) — horizontal bar chart
  - 主要 19 population 発火率 — color-coded (赤 >15Hz, 青 5-15, 灰 <5)
  - 全 53 population 発火率 — collapsible detail

## 設計理念

- Pure static HTML + vanilla JS (ビルド不要)
- 既存 `/brain/v2/` API を使用
- 神経科学教育向け: 「情動がどの脳領域から emerge するか」を直感的に体験
- Disclaimer 組込: モデル制限を常に明示

## Known limitations (as displayed in UI)

- 821 neuron 簡略化モデル (biological ~10^11 の 4-5 桁縮小)
- Validation は firing rate matching のみ — behavioral prediction 未検証
- Izh MC 35/36 / AdEx MC 25/36 / baseline 6/20 (既知の制限)
- 27 件の parameter change の一部は empirical calibration
