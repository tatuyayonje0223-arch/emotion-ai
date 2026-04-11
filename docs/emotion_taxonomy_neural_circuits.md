# 全情動の分類体系と神経回路基盤 — 包括的研究レポート

**作成日**: 2026-04-07
**目的**: EmotionAIプロジェクトで全情動をモデル化するための神経科学的基盤の整理
**現状**: 恐怖(Fear)・報酬(Reward)・ストレス(Stress)の3回路のみ実装済み

---

## 1. 主要な情動分類フレームワーク

### 1.1 Ekmanの基本情動（6つ）

| 情動 | 英語 | 快/不快 | 覚醒度 | 進化的機能 |
|------|------|---------|--------|-----------|
| 喜び | Happiness | 快 | 中〜高 | 社会的結合・接近行動 |
| 怒り | Anger | 不快 | 高 | 障害排除・資源防衛 |
| 恐怖 | Fear | 不快 | 高 | 脅威回避・生存 |
| 悲しみ | Sadness | 不快 | 低 | 喪失への適応・社会的支援要請 |
| 嫌悪 | Disgust | 不快 | 中 | 汚染回避・病原体防御 |
| 驚き | Surprise | 中性 | 高 | 注意の再配分・新規事象処理 |

※ Ekmanは後に「軽蔑(Contempt)」を7番目として追加。

### 1.2 Plutchikの情動の輪（8基本 + 複合情動）

#### 8つの基本情動（4つの対極ペア）

| 基本情動 | 対極 | 強度高 | 強度低 |
|---------|------|--------|--------|
| 喜び (Joy) | 悲しみ (Sadness) | 恍惚 (Ecstasy) | 平穏 (Serenity) |
| 信頼 (Trust) | 嫌悪 (Disgust) | 崇拝 (Admiration) | 受容 (Acceptance) |
| 恐怖 (Fear) | 怒り (Anger) | 恐慌 (Terror) | 不安 (Apprehension) |
| 驚き (Surprise) | 期待 (Anticipation) | 驚愕 (Amazement) | 注意散漫 (Distraction) |

#### 主要な複合情動（隣接基本情動の混合）

| 複合情動 | 構成 | 日本語 |
|---------|------|--------|
| Love | Joy + Trust | 愛 |
| Submission | Trust + Fear | 服従 |
| Awe | Fear + Surprise | 畏怖 |
| Disapproval | Surprise + Sadness | 不承認 |
| Remorse | Sadness + Disgust | 自責 |
| Contempt | Disgust + Anger | 軽蔑 |
| Aggressiveness | Anger + Anticipation | 攻撃性 |
| Optimism | Anticipation + Joy | 楽観 |

### 1.3 Pankseppの情動神経科学（7つの情動システム）

| システム | 日本語 | 快/不快 | 主要回路 | 主要神経伝達物質 |
|---------|--------|---------|---------|----------------|
| SEEKING | 探索/欲求 | 快 | VTA→NAc→PFC (中脳辺縁系DA経路) | ドーパミン, グルタミン酸 |
| RAGE | 怒り/激怒 | 不快 | MeA→VMH→PAG | サブスタンスP, グルタミン酸, ACh |
| FEAR | 恐怖 | 不快 | BLA/CeA→PAG→視床下部 | グルタミン酸, CRH, NE |
| LUST | 性的欲求 | 快 | 視床下部→VTA→扁桃体 | テストステロン, エストロゲン, OXT, AVP |
| CARE | 養育/慈愛 | 快 | BNST→VTA→PAG→視床下部 | オキシトシン, プロラクチン, DA |
| PANIC/GRIEF | 分離苦痛/悲嘆 | 不快 | ACC→BNST→PAG→視床下部 | オピオイド(↓), OXT(↓), CRH(↑) |
| PLAY | 遊び/社会的喜び | 快 | 視床(後内側) →皮質 | オピオイド, eCB, DA |

### 1.4 Barrettの構成的情動理論 (Theory of Constructed Emotion)

Barrett(2017, 2025更新)の理論では、情動は脳の予測処理から構成される:

- **情動は専用回路を持たない**: 特定の情動に1対1対応する脳領域はない
- **予測とカテゴリ化**: 脳が身体状態(内受容)を予測→入力とのミスマッチ→情動カテゴリとして解釈
- **核心情動 (Core Affect)**: 快不快(valence) + 覚醒度(arousal) の2次元連続空間
- **概念知識**: 過去の経験・文化・言語が情動カテゴリを構成
- **身体予算 (Body Budget)**: アロスタシス（身体資源の予測的調節）が情動の基盤

**EmotionAIへの含意**: 離散的な「情動回路」の実装は有用だが、最終的には予測処理フレームワークと内受容信号からの動的カテゴリ形成を目指すべき。

### 1.5 Damasioのソマティック・マーカー仮説

- **身体状態が意思決定を導く**: 情動は身体反応（心拍、発汗、内臓感覚）の脳内表象
- **一次誘導**: 扁桃体が身体反応を起動
- **二次誘導**: vmPFCが過去の身体状態の記憶を再活性化（「あたかも身体」ループ）
- **主要回路**: 扁桃体 → 島皮質(身体表象) → vmPFC(意思決定) → ACC(コンフリクト検出)

**EmotionAIへの含意**: 既存の内受容+島皮質+vmPFCモデルはソマティック・マーカーの基盤を既に含む。

### 1.6 日本の喜怒哀楽フレームワーク

| 情動 | 読み | 対応するPanksepp系 | 特徴 |
|------|------|-------------------|------|
| 喜 (Joy) | き | SEEKING + PLAY | 達成・社会的喜び |
| 怒 (Anger) | ど | RAGE | 不正・障害への反応 |
| 哀 (Sorrow) | あい | PANIC/GRIEF | 喪失・別離の悲しみ |
| 楽 (Pleasure) | らく | SEEKING(報酬消費) | 安らぎ・満足・快楽 |

#### 日本文化固有の情動概念

| 概念 | 読み | 意味 | 神経基盤(推定) |
|------|------|------|---------------|
| 甘え | あまえ | 他者の好意への依存・甘え | OXT + 報酬系(NAc) + 愛着回路 |
| 侘び寂び | わびさび | 不完全さ・無常の美学的受容 | DMN + 内受容 + ACC |
| もののあはれ | — | 万物の儚さへの共感的哀惜 | ACC + mPFC + 内受容(島皮質) |
| 恥 | はじ | 社会的逸脱への自己評価的苦痛 | dACC + 前部島皮質 + mPFC |
| 義理 | ぎり | 社会的義務感 | vmPFC + TPJ + dACC |

---

## 2. 全情動の神経回路マップ

### 2.1 基本情動と一次神経回路

#### 恐怖 (Fear) — 実装済み
- **主要回路**: BLA → CeA → PAG (凍結/逃走) / 視床下部 (自律神経)
- **神経伝達物質**: グルタミン酸(興奮), GABA(抑制), NE(覚醒増強), CRH(ストレス)
- **快不快**: 不快 (valence < 0)
- **覚醒度**: 高 (arousal > 0.7)
- **行動出力**: 凍結(freeze)→逃走(flight)→闘争(fight)の段階的反応
- **実装状態**: BLA/CeA脱抑制, BNST持続不安, PAG防御反応 全て実装済み

#### 報酬/喜び (Reward/Joy) — 部分実装
- **主要回路**: VTA → NAc Shell/Core → vmPFC/OFC
- **神経伝達物質**: ドーパミン(報酬予測誤差), オピオイド(快感/消費), GABA(VTA内抑制)
- **快不快**: 快 (valence > 0)
- **覚醒度**: 中〜高 (arousal 0.4-0.8)
- **行動出力**: 接近行動, 消費行動, 学習強化
- **実装状態**: VTA分割(DA_lat/DA_med/GABA), NAc Shell/Core 実装済み。「喜び」としての明示的readoutは未実装

#### ストレス (Stress) — 実装済み
- **主要回路**: 扁桃体 → 視床下部(CRH) → 下垂体(ACTH) → 副腎(コルチゾール)
- **神経伝達物質**: CRH, ACTH, コルチゾール, NE
- **快不快**: 不快 (valence < 0)
- **覚醒度**: 中〜高 (arousal 0.5-0.9)
- **行動出力**: 覚醒増大, 資源動員, 免疫調整
- **実装状態**: HPA軸カスケード, 負のフィードバック, 自律神経連動 実装済み

#### 怒り (Anger/Rage) — 未実装
- **主要回路**: 内側扁桃体(MeA) → 腹内側視床下部(VMH) → 背側PAG
- **神経伝達物質**: サブスタンスP(MeA→VMH投射), グルタミン酸(VMH→PAG), アセチルコリン, NE
- **快不快**: 不快 (valence < 0)
- **覚醒度**: 非常に高 (arousal > 0.8)
- **行動出力**: 威嚇表示, 攻撃行動, 障害排除
- **新規必要領域**: **内側扁桃体(MeA)**, **腹内側視床下部(VMH)**
- **新規必要神経伝達物質**: **サブスタンスP (SP)**

#### 嫌悪 (Disgust) — 未実装
- **主要回路**: 前部島皮質(anterior insula) → 基底核(被殻/淡蒼球) → OFC
- **神経伝達物質**: セロトニン(5-HT), GABA, グルタミン酸
- **快不快**: 不快 (valence < 0)
- **覚醒度**: 中 (arousal 0.4-0.6)
- **行動出力**: 回避, 嘔吐反射, 道徳的嫌悪(拡張)
- **新規必要**: 前部島皮質の**嫌悪特化サブモジュール**, **被殻(putamen)**, 味覚-内臓連合学習
- **既存活用**: 島皮質(InsulaState)は存在するが嫌悪特化処理がない

#### 悲しみ (Sadness/Grief) — 未実装
- **主要回路**: 膝下前帯状皮質(sgACC) → BNST → PAG → 視床下部
- **神経伝達物質**: オピオイド(↓低下が引き金), OXT(↓), セロトニン(↓), CRH(↑)
- **快不快**: 不快 (valence < 0)
- **覚醒度**: 低 (arousal < 0.3)
- **行動出力**: 社会的引きこもり, 泣き, 支援要請行動
- **新規必要領域**: **膝下ACC(sgACC)** — 現在のACCには含まれていない
- **既存活用**: BNST, PAG, HPA軸は存在。オピオイド/OXTも実装済み

#### 驚き (Surprise) — 未実装
- **主要回路**: 青斑核(LC) → 広域NE投射 + VTA → NAc(正の驚き) or 扁桃体(負の驚き)
- **神経伝達物質**: NE(位相的バースト), DA(予測誤差), ACh(注意切替)
- **快不快**: 中性(文脈依存)
- **覚醒度**: 非常に高・一過性 (arousal spike)
- **行動出力**: 行動中断(startle), 定位反射, 注意再配分
- **新規必要**: **驚愕反応(startle circuit)** の明示的実装, LC位相的バーストモード
- **既存活用**: LC, VTA, 扁桃体は存在。NE/DAシステムも実装済み

### 2.2 社会的・自己意識的情動

#### 恥 (Shame) — 未実装
- **主要回路**: dACC(社会的痛み) + 前部島皮質(情動的覚醒) + mPFC(自己参照) + 視床(行動抑制)
- **神経伝達物質**: NE(覚醒), コルチゾール(ストレス), セロトニン(社会的階層)
- **快不快**: 不快 (valence < 0)
- **覚醒度**: 中〜高 (arousal 0.5-0.7)
- **行動出力**: 視線回避, 身体の縮小, 社会的引きこもり
- **新規必要**: dACC**社会的痛みモジュール**, **mPFC自己参照処理**
- **既存活用**: ACC, 島皮質, コルチゾール系は存在

#### 罪悪感 (Guilt) — 未実装
- **主要回路**: TPJ(他者理解) + vmPFC(道徳判断) + 前部島皮質(情動的覚醒)
- **神経伝達物質**: セロトニン(道徳感), OXT(共感), コルチゾール(苦痛)
- **快不快**: 不快 (valence < 0)
- **覚醒度**: 中 (arousal 0.4-0.6)
- **行動出力**: 補償行動, 謝罪, 行動修正
- **新規必要**: **側頭頭頂接合部(TPJ)** — 完全に新規の領域
- **既存活用**: vmPFC, 島皮質は存在

#### 誇り (Pride) — 未実装
- **主要回路**: mPFC(自己参照) + 後部帯状皮質(PCC/自伝的記憶) + 報酬系(NAc)
- **神経伝達物質**: DA(報酬), セロトニン(社会的地位), テストステロン(支配性)
- **快不快**: 快 (valence > 0)
- **覚醒度**: 中〜高 (arousal 0.5-0.7)
- **行動出力**: 姿勢の拡張, 社会的誇示, 達成動機
- **新規必要**: **PCC(後部帯状皮質)**, mPFC自己参照モジュール
- **既存活用**: NAc報酬系, DA系は存在

#### 嫉妬 (Jealousy) — 未実装
- **主要回路**: 前頭-線条体-視床ループ + vmPFC + 島皮質(内受容/顕著性)
- **神経伝達物質**: DA(報酬への脅威), NE(覚醒), コルチゾール(ストレス)
- **快不快**: 不快 (valence < 0)
- **覚醒度**: 高 (arousal > 0.7)
- **行動出力**: 監視行動, 所有的行動, 攻撃(極端な場合)
- **既存活用**: 線条体, vmPFC, 島皮質は存在。DA/NE/コルチゾール系も実装済み

#### 羨望 (Envy) — 未実装
- **主要回路**: dACC(社会比較) + mPFC(他者評価) + 腹側線条体(↓報酬低下)
- **神経伝達物質**: DA(報酬系の相対的低下), セロトニン(社会的順位)
- **快不快**: 不快 (valence < 0)
- **覚醒度**: 中 (arousal 0.4-0.6)
- **行動出力**: 競争行動, 社会的比較, (シャーデンフロイデにつながる)
- **既存活用**: ACC, NAc, mPFC的機能は部分的に存在

### 2.3 複合情動・高次情動

#### 愛/愛着 (Love/Attachment) — 未実装
- **主要回路**: VTA → 尾状核(caudate) + 腹側淡蒼球(VP) + NAc
- **神経伝達物質**: OXT(結合), AVP(パートナー選好), DA(報酬/動機)
- **快不快**: 快 (valence > 0)
- **覚醒度**: 中 (arousal 0.4-0.6)
- **行動出力**: 接近, 維持, 保護, パートナー選好
- **新規必要**: **腹側淡蒼球(VP)**, **尾状核(caudate)**, **AVP(バソプレシン)システム**
- **既存活用**: VTA, NAc, OXTは実装済み

#### 好奇心/探索 (Curiosity/SEEKING) — 部分実装
- **主要回路**: VTA → NAc → PFC + 外側視床下部(LH) + 内側前脳束(MFB)
- **神経伝達物質**: DA(探索的動機), NE(新奇性反応), グルタミン酸
- **快不快**: 快 (valence > 0)
- **覚醒度**: 中〜高 (arousal 0.5-0.7)
- **行動出力**: 探索, 調査, 情報希求
- **新規必要**: **外側視床下部(LH)** サブ領域, 新奇性検出の明示的実装
- **既存活用**: VTA→NAc経路は実装済み。DA系も存在

#### 期待 (Anticipation) — 未実装
- **主要回路**: VTA → NAc(報酬期待) + OFC(期待値計算) + dlPFC(計画)
- **神経伝達物質**: DA(期待的位相バースト), グルタミン酸
- **快不快**: 快(報酬期待) / 不快(脅威期待)
- **覚醒度**: 中〜高 (arousal 0.5-0.7)
- **行動出力**: 準備行動, 計画, 注意集中
- **既存活用**: VTA, NAc, OFC, dlPFCは全て存在。DA系も実装済み

#### 退屈 (Boredom) — 未実装
- **主要回路**: DMN(mPFC + PCC) 活性↑ + dACC / 前部島皮質 活性↓
- **神経伝達物質**: DA(↓低下), NE(↓低下)
- **快不快**: 不快 (valence < 0, 軽度)
- **覚醒度**: 低 (arousal < 0.3)
- **行動出力**: 注意散漫, 探索動機(SEEKING系の再活性化)
- **新規必要**: **デフォルトモードネットワーク(DMN)** — PCC + mPFCの特殊モード
- **既存活用**: ACC, 島皮質は存在

#### 畏怖/崇高 (Awe) — 未実装
- **主要回路**: DMN + 前部島皮質 + LC(NE位相バースト) + 背側ACC
- **神経伝達物質**: NE(覚醒), DA(新奇性報酬), セロトニン(自己境界の変容)
- **快不快**: 快(正の畏怖) / 混合(脅威的畏怖)
- **覚醒度**: 高 (arousal > 0.7)
- **行動出力**: 行動停止, 開放感, 自己の相対化
- **新規必要**: DMN, セロトニンの**自己境界モジュール**
- **既存活用**: LC, 島皮質, ACC, セロトニン系は存在

---

## 3. 既存アーキテクチャでモデル可能な情動

### 3.1 既存回路で直接モデル可能（パラメータ調整+readout追加のみ）

| 情動 | 既存回路の活用方法 | 必要な追加 |
|------|-------------------|-----------|
| **不安 (Anxiety)** | BNST(実装済み) + 扁桃体低レベル持続活性 | readout関数のみ |
| **期待 (Anticipation)** | VTA→NAc DA位相バースト + OFC期待値 | readout関数 |
| **好奇心 (Curiosity)** | VTA→NAc SEEKING + 新奇性入力 | readout関数 + 新奇性検出強化 |
| **嫉妬 (Jealousy)** | 報酬系(↓) + 扁桃体(↑) + 島皮質(↑) | readout関数 + 社会的入力拡張 |
| **驚き (Surprise)** | LC位相バースト + 扁桃体/VTA(文脈依存) | startle反応モジュール + readout |
| **安心 (Relief)** | 扁桃体活性↓ + DA放出 + オピオイド放出 | readout関数(脅威→安全遷移の検出) |

### 3.2 小規模拡張でモデル可能（1-2領域の追加）

| 情動 | 必要な追加領域 | 実装の複雑度 |
|------|-------------|------------|
| **怒り (Anger/Rage)** | 内側扁桃体(MeA) + 腹内側視床下部(VMH) | 中 |
| **悲しみ (Sadness)** | 膝下ACC(sgACC) | 低〜中 |
| **嫌悪 (Disgust)** | 前部島皮質の嫌悪サブモジュール + 被殻 | 中 |
| **愛/愛着 (Love)** | 腹側淡蒼球(VP) + AVP系 | 中 |
| **養育 (Care)** | MPOA(内側視索前野) | 中 |

### 3.3 大規模拡張が必要（新しいネットワーク構造）

| 情動 | 必要な新構造 | 実装の複雑度 |
|------|-----------|------------|
| **恥/罪悪感** | TPJ + mPFC自己参照 + 社会的痛みネットワーク | 高 |
| **誇り** | PCC + mPFC自己参照 + 報酬系連携 | 高 |
| **畏怖 (Awe)** | DMN (PCC + mPFC特殊モード) | 高 |
| **退屈 (Boredom)** | DMN + 顕著性ネットワーク相互作用 | 高 |
| **甘え (Amae)** | 愛着回路 + 報酬系 + 社会的認知ネットワーク | 高 |

---

## 4. Pankseppの7情動システム — 詳細な実装仕様

### 4.1 SEEKING（探索/欲求）

#### 核心回路
```
VTA(DA neurons) → NAc Shell/Core → mPFC
    ↑                    ↑
外側視床下部(LH) ← 内側前脳束(MFB)
    ↑
扁桃体(BLA) → 期待的価値信号
```

#### 主要神経伝達物質
| 物質 | 役割 | 産生源 |
|------|------|--------|
| ドーパミン | 誘因顕著性(incentive salience)、報酬予測誤差 | VTA |
| グルタミン酸 | 興奮性伝達、LTP | 皮質→NAc投射 |
| オピオイド | 快感(liking)、消費報酬 | NAc内エンケファリン |
| eCB | 逆行性シグナル、DA放出調節 | 後シナプス |

#### 既存回路との関係
- **VTA分割(DA_lat/DA_med/GABA)**: 実装済み — SEEKINGの核心
- **NAc Shell/Core**: 実装済み — Shell=新奇報酬、Core=学習済み報酬
- **PFC**: vmPFC/OFC実装済み — 価値表象と期待
- **未実装**: 外側視床下部(LH)、内側前脳束(MFB)の明示的モデル

#### 実装可能性: **高（拡張容易）**
既存のVTA→NAc→PFC経路をSEEKINGシステムとしてreadout関数を追加。LHはSensoryInputの新規チャネル(appetitive_signal)として追加可能。

---

### 4.2 RAGE（怒り/激怒）

#### 核心回路
```
感覚入力(frustration/pain) → 内側扁桃体(MeA)
                                    ↓ (分界条床→サブスタンスP)
                            腹内側視床下部(VMH)
                                    ↓ (グルタミン酸→NMDA)
                            背側PAG(dPAG)
                                    ↓
                            攻撃行動出力
                            
PFC(vmPFC/OFC) ─┤ トップダウン抑制 → MeA/VMH
```

#### 主要神経伝達物質
| 物質 | 役割 | 産生源 |
|------|------|--------|
| サブスタンスP (SP) | MeA→VMH興奮性投射(NK1受容体) | 内側扁桃体 |
| グルタミン酸 | VMH→PAG興奮性投射(NMDA受容体) | VMH |
| セロトニン(5-HT) | 攻撃性の抑制(↓で攻撃↑) | 縫線核 |
| テストステロン | 攻撃閾値の低下 | 全身(脳内受容体) |
| NE | 覚醒増大 | LC |

#### 既存回路との関係
- **扁桃体**: 実装済みだが**外側核/基底外側核(BLA)のみ**。**内側核(MeA)は未実装**
- **PAG**: 実装済み(fight_output変数あり)だが、RAGE専用経路との区別がない
- **PFC抑制**: vmPFC→扁桃体抑制は実装済み
- **未実装**: MeA, VMH, サブスタンスP, テストステロン

#### 実装仕様
```python
# 新規領域
class MedialAmygdalaState(RegionState):
    """内側扁桃体。攻撃行動と社会的情報処理。"""
    name: str = "medial_amygdala"
    substance_p_output: float = 0.0  # SP産生
    
class VMHState(RegionState):
    """腹内側視床下部。攻撃行動の統合中枢。"""
    name: str = "vmh"
    rage_drive: float = 0.0  # 攻撃駆動力

# 新規神経伝達物質
class SubstancePState(NeurotransmitterState):
    """サブスタンスP。RAGE回路の主要伝達物質。"""
    tonic: float = 0.2
    reuptake_rate: float = 0.12

# 新規結合
Connection(source="medial_amygdala", target="vmh", weight=0.8,
           conn_type="excitatory", neuromodulator="substance_p")
Connection(source="vmh", target="pag", weight=0.7,
           conn_type="excitatory")  # rage→攻撃行動
Connection(source="vmPFC", target="medial_amygdala", weight=0.5,
           conn_type="inhibitory")  # トップダウン抑制
```

#### 実装可能性: **中（新規領域2つ + 新規伝達物質1つ）**

---

### 4.3 FEAR（恐怖）— 実装済み

#### 核心回路
```
感覚入力 → 視床 → BLA(外側核LA→基底核BA)
                        ↓
                    CeA(SOM+→PKCd+脱抑制→CeM)
                        ↓
         ┌─────────────┼──────────────┐
         ↓             ↓              ↓
    PAG(凍結/逃走)  視床下部(HPA軸)  LC(NE→覚醒)
```

#### 実装状態: **完了**
- BLA/CeA脱抑制: SOM+/PKCd+/CeM サブタイプ実装済み
- BNST持続不安: 実装済み
- PAG防御反応: freeze/flight/fight 実装済み
- HPA軸: CRH→ACTH→コルチゾール カスケード実装済み
- 恐怖条件付け/消去: STDP真LTP + 自発的回復実装済み
- 検証スコア: 0.805 (BLA 9.1Hz, SOM+/PKCd+ ratio 3.28)

---

### 4.4 LUST（性的欲求）

#### 核心回路
```
感覚入力(性的刺激) → 内側扁桃体(MeA) → MPOA(内側視索前野)
                                              ↓
性ステロイド(テストステロン/エストロゲン) → 視床下部(LH/VMN)
                                              ↓
                                          VTA → NAc(報酬)
                                              ↓
                                          PAG(交尾行動)
```

#### 主要神経伝達物質
| 物質 | 役割 | 特記 |
|------|------|------|
| テストステロン | 性的動機の調節 | 視床下部の受容体を介して作用 |
| エストロゲン | 性的受容性 | MPOA/VMNに作用 |
| オキシトシン(OXT) | 性的結合、オーガズム | 視床下部→全身放出 |
| バソプレシン(AVP) | パートナー選好(♂) | 腹側淡蒼球に作用 |
| ドーパミン | 性的報酬、動機づけ | VTA→NAc |
| NO(一酸化窒素) | MPOA内の局所伝達 | 性的覚醒 |

#### 既存回路との関係
- **VTA→NAc報酬系**: 実装済み
- **OXT**: 実装済み
- **PAG**: 実装済み（ただし性的行動出力は未定義）
- **未実装**: MeA(共通), MPOA, テストステロン/エストロゲン, AVP, NO

#### 実装可能性: **中〜高（MeAはRAGEと共有、MPOA + 性ステロイドが新規）**
RAGE実装と合わせてMeAを共通基盤として構築可能。MPOA + 性ステロイドホルモンは新規だが単純なモジュール。

---

### 4.5 CARE（養育/慈愛）

#### 核心回路
```
乳児の刺激(泣き声/匂い) → 扁桃体(BLA) → MPOA(内側視索前野)
                                              ↓
              OXT(視床下部室傍核から放出) → BNST → VTA
                                              ↓
                                          NAc(養育報酬)
                                              ↓
                                          PAG(養育行動)
```

#### 主要神経伝達物質
| 物質 | 役割 | 特記 |
|------|------|------|
| オキシトシン(OXT) | 母性行動の起動、結合 | 室傍核(PVN)から放出 |
| プロラクチン | 養育動機 | 下垂体 |
| ドーパミン | 養育報酬 | VTA→NAc |
| オピオイド | 接触による快感 | NAc内エンケファリン |
| エストロゲン | OXT受容体発現↑ | 出産前に上昇 |

#### 既存回路との関係
- **OXT**: 実装済み（tonic/phasic）
- **BNST**: 実装済み（持続不安用だが、CARE系にも関与）
- **VTA→NAc**: 実装済み
- **社会的入力**: SensoryInput.social_signal として存在
- **未実装**: MPOA, プロラクチン, CARE特化readout

#### 実装可能性: **高（MPOA追加 + readout関数で大部分カバー可能）**
BNSTをCARE/PANICの共通ノードとして再利用。OXT + DA + 社会的入力の組み合わせでCARE readoutを構築可能。

---

### 4.6 PANIC/GRIEF（分離苦痛/悲嘆）

#### 核心回路
```
分離/喪失刺激 → 前帯状皮質(dACC/sgACC)
                        ↓
                    BNST
                        ↓
                 ┌──────┼──────┐
                 ↓      ↓      ↓
              PAG    視床下部   LC
          (泣き声)  (CRH/ストレス) (覚醒)
          
オピオイド(↓低下) → PANIC増強
OXT(↓低下) → 分離苦痛増強
社会的再結合 → オピオイド↑ + OXT↑ → PANIC減弱
```

#### 主要神経伝達物質
| 物質 | 役割 | 方向 |
|------|------|------|
| 内因性オピオイド | 社会的温かさ、分離時に低下 | ↓で苦痛 |
| オキシトシン(OXT) | 社会的結合、分離時に低下 | ↓で苦痛 |
| CRH | ストレス応答増強 | ↑で苦痛 |
| グルタミン酸 | ACC→BNST興奮性伝達 | ↑で苦痛 |
| プロラクチン | 社会的慰め | ↑で苦痛緩和 |

#### 既存回路との関係
- **ACC**: 実装済み（ただしdACC/sgACCのサブ領域区分なし）
- **BNST**: 実装済み（CARE系と共有）
- **PAG**: 実装済み（泣き声出力の定義が必要）
- **オピオイド/OXT/CRH**: 全て実装済み
- **HPA軸**: 実装済み
- **未実装**: ACC内のサブ領域区分(dACC/sgACC)、泣き声行動出力、分離検出メカニズム

#### 実装仕様
```python
# ACCのサブ領域拡張
class ACCState(BaseModel):
    """前帯状皮質。サブ領域を持つ。"""
    name: str = "acc"
    dACC: RegionState  # 背側ACC（コンフリクト/社会的痛み）
    sgACC: RegionState  # 膝下ACC（悲しみ/悲嘆）
    conflict_signal: float = 0.0

# PANIC readout関数
def compute_panic_grief(brain: BrainState) -> float:
    """PANIC/GRIEFレベルの計算。
    オピオイドとOXTの低下 + ACCの活性化が主要指標。
    """
    opioid_deficit = max(0, 0.5 - brain.neurotransmitters.endorphin.effective_level)
    oxt_deficit = max(0, 0.5 - brain.neurotransmitters.oxytocin.effective_level)
    acc_activation = brain.acc.sgACC.output
    bnst_activation = brain.bnst.output  # BNSTへの参照
    return min(1.0, opioid_deficit * 0.3 + oxt_deficit * 0.3 + 
               acc_activation * 0.25 + bnst_activation * 0.15)
```

#### 実装可能性: **高（ACCサブ領域分割 + readout関数で対応可能）**
既存のBNST + オピオイド + OXT + HPA軸がそのまま使える。ACCの分割とPANIC特化readoutが主な作業。

---

### 4.7 PLAY（遊び/社会的喜び）

#### 核心回路
```
社会的遊び刺激 → 視床(後内側核) → 体性感覚皮質
                        ↓
                    背側PAG
                        ↓
                    運動出力(遊び行動)
                    
調節: オピオイド(↑で促進) + eCB(↑で促進) + DA(報酬)
抑制: 5-HT(↑で抑制) + NE(↑で抑制)
```

#### 主要神経伝達物質
| 物質 | 役割 | 方向 |
|------|------|------|
| 内因性オピオイド | 遊びの快感、社会的報酬 | ↑で促進 |
| 内因性カンナビノイド(eCB) | 遊びの促進 | ↑で促進 |
| ドーパミン | 遊びの動機づけ | ↑で促進 |
| セロトニン(5-HT) | 遊び行動の抑制 | ↑で抑制 |
| NE | 遊び行動の抑制(高覚醒は遊びと非互換) | ↑で抑制 |

#### 既存回路との関係
- **PAG**: 実装済み（ただし遊び行動出力は未定義）
- **オピオイド/eCB/DA**: 実装済み
- **5-HT/NE**: 実装済み
- **未実装**: 視床後内側核(遊び起点)、体性感覚皮質、遊び行動出力

#### 実装可能性: **中（視床サブ領域 + PLAY readout）**
PLAYは「脅威が低い + オピオイド/eCB/DAが高い + 5-HT/NEが中程度 + 社会的入力が高い」の条件で発現するreadoutとして実装可能。視床後内側核は新規だが、SensoryInputの拡張で代替可能。

---

## 5. 新規に必要な脳領域・システムの一覧

### 5.1 必要な新規脳領域

| 領域 | 略称 | 対応する情動 | 優先度 |
|------|------|-------------|--------|
| 内側扁桃体 | MeA | RAGE, LUST | P1 |
| 腹内側視床下部 | VMH | RAGE | P1 |
| 膝下前帯状皮質 | sgACC | 悲しみ/GRIEF | P1 |
| 背側前帯状皮質 | dACC | 恥, 社会的痛み, PANIC | P1 |
| 内側視索前野 | MPOA | CARE, LUST | P2 |
| 側頭頭頂接合部 | TPJ | 罪悪感, 共感 | P2 |
| 後部帯状皮質 | PCC | 誇り, DMN, 自伝的記憶 | P2 |
| 腹側淡蒼球 | VP | 愛/愛着, 快感(liking) | P2 |
| 尾状核 | Caudate | 愛, 習慣学習 | P3 |
| 被殻 | Putamen | 嫌悪, 運動制御 | P3 |
| 視床後内側核 | PMT | PLAY | P3 |
| 外側視床下部 | LH | SEEKING, 食欲, 覚醒 | P3 |

### 5.2 必要な新規神経伝達物質/神経ペプチド

| 物質 | 略称 | 対応する情動 | 優先度 |
|------|------|-------------|--------|
| サブスタンスP | SP | RAGE | P1 |
| バソプレシン | AVP | 愛着, LUST(♂) | P2 |
| プロラクチン | PRL | CARE, GRIEF緩和 | P2 |
| テストステロン | T | RAGE閾値, LUST | P3 |
| エストロゲン | E2 | CARE, LUST | P3 |
| 一酸化窒素 | NO | LUST(MPOA局所) | P3 |
| アセチルコリン | ACh | RAGE, 注意, 記憶 | P2 |

### 5.3 必要な新規SensoryInputチャネル

| チャネル | 対応する情動 | 説明 |
|---------|-------------|------|
| frustration_signal | RAGE | 目標阻害・不正検出 |
| separation_signal | PANIC/GRIEF | 愛着対象との分離検出 |
| sexual_signal | LUST | 性的刺激 |
| play_signal | PLAY | 遊び的社会刺激 |
| nurturing_signal | CARE | 養育対象(乳児)からの刺激 |
| moral_violation_signal | 恥/罪悪感 | 社会規範逸脱の検出 |
| self_evaluation_signal | 誇り/恥 | 自己評価フィードバック |

---

## 6. 実装ロードマップ（推奨優先順位）

### Phase 1: Pankseppの7システム完成（最優先）

Pankseppの7情動システムは皮質下回路に基づき、最も直接的にスパイキングモデルで実装可能。

| 順序 | システム | 必要な追加 | 工数(推定) |
|------|---------|-----------|-----------|
| 1 | **RAGE** | MeA + VMH + SP | 中(2-3日) |
| 2 | **PANIC/GRIEF** | ACC分割(dACC/sgACC) + readout | 低〜中(1-2日) |
| 3 | **SEEKING** | readout関数 + LH(optional) | 低(1日) |
| 4 | **CARE** | MPOA + PRL + readout | 中(2日) |
| 5 | **PLAY** | readout関数 + 遊び行動出力 | 低(1日) |
| 6 | **LUST** | MPOA(CARE共有) + 性ステロイド + AVP | 中(2日) |

### Phase 2: 複合情動のreadout拡張

既存回路の組み合わせから複合的な情動を読み出す関数を追加。

| 情動 | readout定義 |
|------|------------|
| 喜び (Joy) | DA↑ + オピオイド↑ + NAc↑ + 脅威↓ |
| 悲しみ (Sadness) | オピオイド↓ + OXT↓ + sgACC↑ + DA↓ |
| 嫌悪 (Disgust) | 前部島皮質↑ + 5-HT↑ + 回避行動出力 |
| 驚き (Surprise) | NE位相バースト + LC↑ + 行動中断 |
| 期待 (Anticipation) | DA位相バースト + OFC↑ + NAc↑ |
| 安心 (Relief) | 脅威→安全遷移 + DA放出 + オピオイド↑ |
| 退屈 (Boredom) | DA↓ + NE↓ + SEEKING↓ |

### Phase 3: 社会的認知ネットワーク

TPJ, PCC, mPFC自己参照モジュールの追加による高次社会的情動。

| 情動 | 必要な追加 |
|------|-----------|
| 恥 (Shame) | dACC社会的痛み + mPFC自己参照 |
| 罪悪感 (Guilt) | TPJ + vmPFC道徳判断 |
| 誇り (Pride) | PCC + mPFC自己参照 + 報酬系 |
| 共感 (Empathy) | TPJ + 島皮質(ミラー的) + ACC |
| 愛/愛着 (Love) | VP + AVP + 既存OXT/DA |

### Phase 4: 構成的情動フレームワーク

Barrett理論に基づく、カテゴリ化メカニズムの統合。

- 内受容予測モデル（予測符号化）
- 概念知識による情動カテゴリ構成
- 文脈依存的な情動生成
- 文化特有の情動概念（甘え等）の学習

---

## 7. 既存コードベースとの対応表

| 既存ファイル | 含む機能 | 拡張箇所 |
|-------------|---------|---------|
| `regions.py` | AmygdalaState, PAGState, VentralStriatumState等 | MeA, VMH, sgACC, MPOA, TPJ, PCC, VP追加 |
| `neurotransmitters.py` | 8種のNT系 | SP, AVP, PRL, ACh, 性ステロイド追加 |
| `connectivity.py` | 解剖学的結合28本 | RAGE/CARE/LUST/PLAY/GRIEF結合追加 |
| `brain.py` | BrainState, step_brain, compute_readout | 新領域の統合更新、emotion_readout拡張 |
| `homeostasis.py` | HPA軸, 自律神経, 内受容 | 性ステロイドホメオスタシス追加(optional) |
| `plasticity.py` | STDP, 情動タグ | RAGE/CARE学習則追加 |
| `brian2_circuits/` | スパイキングモデル | 新回路のBrian2実装 |

---

## 参考文献・情報源

### Pankseppの情動神経科学
- [Selected Principles of Pankseppian Affective Neuroscience (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6344464/)
- [Affective Neuroscience Theory and Personality: An Update (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7219919/)
- [The Neuropsychology of the SEEKING System (UPenn)](https://web.english.upenn.edu/~cavitch/pdf-library/Wright_and_Panksepp_Neuropsychology_of_the_SEEKING_System.pdf)
- [Affective neuroscience of the emotional BrainMind (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3181986/)

### Ekmanの基本情動と神経基盤
- [Basic Emotions in Human Neuroscience: Neuroimaging and Beyond (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5573709/)
- [Basic Emotions and Brain Circuitry (CUNY Pressbooks)](https://pressbooks.cuny.edu/psy320/chapter/basic-emotions/)
- [An Integrative Way for Studying Neural Basis of Basic Emotions (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6593191/)

### Barrettの構成的情動理論
- [The theory of constructed emotion: an active inference account (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5390700/)
- [Publications - Lisa Feldman Barrett Lab](https://affective-science.org/publications.shtml)

### Damasioのソマティック・マーカー仮説
- [Somatic Marker Hypothesis (ScienceDirect)](https://www.sciencedirect.com/topics/neuroscience/somatic-marker-hypothesis)

### RAGE/怒りの神経回路
- [Substance P in the medial amygdala regulates aggressive behaviors (Nature 2024)](https://www.nature.com/articles/s41386-024-01863-w)
- [Anger management: glutamate receptor mechanisms in aggression (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8959042/)
- [Amygdala and Hypothalamus: Focus on Aggression (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6565484/)

### 嫌悪の神経回路
- [Lateralized Deficits of Disgust Processing After Insula-Basal Ganglia Damage (Frontiers)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.01429/full)

### 社会的情動(恥/罪悪感)
- [Neural Signatures of Shame, Embarrassment, and Guilt: Voxel-Based Meta-Analysis (PMC 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10136704/)

### 嫉妬/羨望
- [Neural mechanisms of different types of envy: meta-analysis (Frontiers 2024)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1335548/full)

### 愛/愛着
- [The Neurobiology of Love and Pair Bonding (PMC 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10295201/)
- [Convergence of oxytocin and dopamine signalling (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S0149763424001441)

### VTA回路アーキテクチャ
- [Circuit Architecture of VTA Dopamine Neurons (Cell 2015)](https://www.cell.com/cell/fulltext/S0092-8674(15)00852-1)
- [The Formation and Function of the VTA Dopamine System (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11011984/)

### 退屈/アンヘドニア
- [Reward network mechanism in anhedonia and depression (PLOS ONE 2024)](https://journals.plos.org/plosone/article/file?type=printable&id=10.1371/journal.pone.0332816)
- [Circuit Mechanisms of Reward, Anhedonia, and Depression (IJNP)](https://academic.oup.com/ijnp/article/22/2/105/5098511)

### LC-NE系と新奇性/驚き
- [Neural Circuit Connections and Functions of LC-NE System (MDPI 2024)](https://www.mdpi.com/1422-0067/26/22/11163)

### 日本文化固有の情動
- [Amae in Japan and the United States (PubMed)](https://pubmed.ncbi.nlm.nih.gov/16768560/)
- [Positive Emotions in Japanese (Springer 2025)](https://link.springer.com/chapter/10.1007/978-981-96-2325-9_20)

### Plutchikの情動の輪
- [Plutchik's Wheel of Emotions (Six Seconds 2025)](https://www.6seconds.org/2025/02/06/plutchik-wheel-emotions/)
