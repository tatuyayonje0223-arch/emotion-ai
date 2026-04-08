# パラメータ根拠テーブル

[監査P1対応] 各パラメータの値・出典・感度を文書化する。

## 凡例
- **出典**: 論文参照 / 推定 / 手動チューニング
- **感度**: HIGH=値を変えると質的に結果が変わる / MED=量的に変わる / LOW=ほぼ影響なし
- **検証**: 文献値と比較済み(✓) / 未検証(✗)

## Izhikevichニューロンパラメータ

| パラメータ | 値 | 出典 | 感度 | 検証 |
|-----------|-----|------|------|------|
| RS: a | 0.02 | Izhikevich 2003 Table 1 | LOW | ✓ |
| RS: b | 0.2 | Izhikevich 2003 | LOW | ✓ |
| RS: c | -65 | Izhikevich 2003 | LOW | ✓ |
| RS: d | 8 | Izhikevich 2003 | LOW | ✓ |
| FS/PV: a | 0.1 | Izhikevich 2003 | LOW | ✓ |
| IB/DA: c | -55 | Izhikevich 2003 | MED | ✓ |
| ノイズ強度 | 1.0-3.0 | 手動チューニング | MED | ✗ |

## シナプスパラメータ

| パラメータ | 値 | 出典 | 感度 | 検証 |
|-----------|-----|------|------|------|
| STDP A_plus | 0.005-0.008 | Bi & Poo 2001 ���範囲内 | HIGH | 部分的 |
| STDP A_minus | 0.003-0.005 | Bi & Poo 2001 | HIGH | 部分的 |
| STDP tau_plus | 20 ms | Bi & Poo 2001 | MED | ✓ |
| STDP tau_minus | 20 ms | Bi & Poo 2001 | MED | ✓ |
| 適格性トレース tau | 1000 ms | 推定（文献値~1秒に合わせた） | HIGH | ✗ |
| w_max | 8-12 | 手動チューニング | MED | ✗ |

## 回路結合パラメータ（恐怖回路v2）

| 結合 | p(接続確率) | w(重み) | 出典 | 感度 | 検証 |
|------|------------|---------|------|------|------|
| LA_exc→LA_pv | 0.3 | 3.0 | 推定(皮質E-I回路の一般値) | MED | ✗ |
| LA_pv→LA_exc | 0.4 | 4.0 | 推定 | HIGH | ✗ |
| VIP→PV(脱抑制) | 0.5 | 5.0 | Krabbe2019に基づく推定 | HIGH | ✗ |
| LA→BA | 0.3 | 3.0 | Pitkänen2000+推定 | MED | ✗ |
| LA/BA→CeL_SOM | 0.3 | 3.0 | 文献推定 | HIGH | ✗ |
| CeL_SOM→CeL_PKCd | 0.5 | 5.0 | Ciocchi2010 | HIGH | 部分的 |
| CeL_PKCd→CeM | 0.5 | 6.0 | Ciocchi2010 | HIGH | 部分的 |
| IL→ITC | 0.3 | 2.5 | Quirk2003+推定 | MED | ✗ |
| ITC→CeM | 0.5 | 5.0 | Likhtik2008+推定 | MED | ✗ |

## 文献ベース結合マトリクス（allen_connectivity.py）

| 投射 | vol | dens | 出典 | 信頼度 |
|------|-----|------|------|--------|
| LA→CeA | 0.45 | 0.35 | Pitkänen2000, LeDoux2007 | 0.78 |
| BLA→NAc | 0.35 | 0.25 | Stuber2011, Namburi2015 | 0.90 |
| VTA→NAc | 0.60 | 0.50 | Ikemoto2007 | 1.00 |
| CeA→PAG | 0.55 | 0.45 | LeDoux2007 | 0.70 |
| PL→BLA | 0.35 | 0.25 | Vertes2004 | 0.66 |

注意: vol/densの値は**著者による推定値**であり、Allen Atlas APIの実測値ではない。

## Wilson-Cowan readout関数（brain.py）

| 変�� | 式 | 係数根拠 | 感度 | 検証 |
|------|-----|---------|------|------|
| valence | DA×0.3+END×0.2+NAc×0.2-CORT×0.3-amygdala×0.3+5HT×0.1 | **手動設定。理論的根拠なし** | HIGH | ✗ |
| arousal | NE×0.4+Glu×0.2+sympathetic×0.3+LC×0.1 | 手動設定 | HIGH | ✗ |
| threat | amygdala×0.5+CORT×0.2+NE×0.15+PAG×0.15 | 手動設定 | HIGH | ✗ |

**重要**: これらの係数は研究者が任意に設定した値であり、データ駆動ではない。
将来的にはPCA/クラスタリングベースの自動読み出��に移行すべき。

## 正規化定数

| 定数 | 値 | 用途 | 根拠 | 問題 |
|------|-----|------|------|------|
| freeze_response正規化 | 40.0 Hz | CeM発火率→凍結反応(0-1) | 手動設定 | スケール依存 |
| anxiety_level正規化 | 30.0 Hz | BNST発火率→不安(0-1) | 手動設定 | スケール依存 |
| approach_tendency | (D1-D2)/(D1+D2) | D1/D2バランス→接近傾向 | 概念的に妥当 | 線形仮定 |

## 未検証パラメータの対処方針

1. **短期**: 感度HIGH のパラメータを±50%変動させるテストを追加
2. **中期**: Allen Brain Atlas APIの実測データで vol/dens を上書き
3. **長期**: ベイズ推定で���ラメータの事後分布を推定
