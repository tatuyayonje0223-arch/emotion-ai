"""高度な神経修飾システム。

リサーチTier 1-2の未実装項目:
1. エンドカンナビノイド系(eCB) — 恐怖消去の主要メカニズム
2. アセチルコリン(ACh) — 恐怖記憶強度 + 報酬増幅
3. シータ振動基盤 — 扁桃体-海馬同期

エンドカンナビノイド (Marsicano et al. 2002, Lutz et al. 2015):
  - CB1Rは扁桃体BLAのグルタミン酸端末とGABA端末の両方に存在
  - 恐怖消去: GABAergic CB1Rダウンレギュレーション→抑制解除
  - 2-AG (位相的): 強い脱分極で放出→短期抑制(DSI/DSE)
  - AEA (持続的): 基底トーンでGABA端末のCB1R活性化

ACh (Jiang et al. 2022, Crouse et al. 2020):
  - 基底前脳NBM→BLA投射: 恐怖記憶の強度と持続性を制御
  - NAcのコリン作動性介在ニューロン(CIN): DA放出のゲーティング
  - ニコチン受容体: CCK+抑制性介在ニューロンを強力に活性化

シータ振動 (Seidenbecher et al. 2003, Likhtik et al. 2014):
  - BLA-海馬-mPFCの4-8Hzシータ同期→恐怖記憶の検索と固定化
  - 恐怖条件付け後: LA-CA1シータ同期↑
  - 消去学習: BLA→腹側海馬シータが必須
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class EndocannabinoidState:
    """エンドカンナビノイド系の状態。"""

    # 2-AG (位相的: 強い神経活動で放出)
    two_ag: float = 0.0       # 0-1

    # AEA/anandamide (持続的トーン)
    aea_tone: float = 0.3     # 0-1

    # CB1R活性（GABAとGluで別）
    cb1r_gaba: float = 0.5    # GABA端末のCB1R活性
    cb1r_glu: float = 0.3     # Glu端末のCB1R活性

    # 消去学習による変化
    extinction_signal: float = 0.0  # 消去進行度


@dataclass
class AcetylcholineState:
    """アセチルコリン系の状態。"""

    # NBM→BLA投射ACh
    nbm_ach: float = 0.3  # 0-1

    # NAc CIN
    nac_cin: float = 0.3  # 0-1

    # ニコチン受容体活性
    nicotinic_activation: float = 0.2  # 0-1

    # ムスカリン受容体活性
    muscarinic_activation: float = 0.3  # 0-1


@dataclass
class ThetaOscillator:
    """シータ振動生成器（4-8Hz）。

    内側中隔/対角帯からのペースメーカー入力を模倣。
    """

    frequency_hz: float = 6.0  # シータ周波数
    amplitude: float = 1.0     # 振幅
    phase: float = 0.0         # 現在位相 (radians)

    def step(self, dt_ms: float = 1.0) -> float:
        """1ステップ進めて現在のシータ値を返す。"""
        self.phase += 2 * math.pi * self.frequency_hz * dt_ms / 1000.0
        if self.phase > 2 * math.pi:
            self.phase -= 2 * math.pi
        return self.amplitude * math.sin(self.phase)

    @property
    def current_value(self) -> float:
        return self.amplitude * math.sin(self.phase)


def update_endocannabinoid(
    ecb: EndocannabinoidState,
    bla_activity: float,
    pfc_activity: float,
    is_extinction_trial: bool,
    dt_ms: float = 1.0,
) -> EndocannabinoidState:
    """エンドカンナビノイド系を更新する。

    恐怖消去のメカニズム:
    1. mPFC-IL活性化 → 2-AG放出↑
    2. 2-AG → GABA端末CB1R活性化 → GABA放出↓ → BLA主細胞の脱抑制
    3. 消去の進行 → AEAトーン↑ → 長期的な消去記憶の維持
    """
    new = EndocannabinoidState(
        two_ag=ecb.two_ag,
        aea_tone=ecb.aea_tone,
        cb1r_gaba=ecb.cb1r_gaba,
        cb1r_glu=ecb.cb1r_glu,
        extinction_signal=ecb.extinction_signal,
    )

    # 2-AG: 強い神経活動(BLA+PFC)で位相的に放出
    activity_drive = (bla_activity + pfc_activity) * 0.3
    new.two_ag += (activity_drive - new.two_ag * 0.2) * dt_ms * 0.01
    new.two_ag = max(0.0, min(1.0, new.two_ag))

    # 消去試行中: AEAトーン上昇 + CB1R_GABA活性↑
    if is_extinction_trial:
        new.extinction_signal = min(1.0, ecb.extinction_signal + 0.02 * dt_ms * 0.01)
        new.aea_tone = min(1.0, ecb.aea_tone + 0.01 * pfc_activity * dt_ms * 0.01)
        # GABA端末CB1R: 消去でダウンレギュレーション→抑制解除
        new.cb1r_gaba = max(0.1, ecb.cb1r_gaba - 0.005 * new.extinction_signal * dt_ms * 0.01)
    else:
        # 非消去時: 緩やかに基底レベルへ回帰
        new.extinction_signal *= (1.0 - 0.001 * dt_ms * 0.01)
        new.aea_tone += (0.3 - ecb.aea_tone) * 0.001 * dt_ms * 0.01

    # Glu端末CB1R: 2-AGで活性化
    new.cb1r_glu = min(1.0, 0.3 + new.two_ag * 0.5)

    return new


def update_acetylcholine(
    ach: AcetylcholineState,
    threat_signal: float,
    reward_signal: float,
    bla_activity: float,
    dt_ms: float = 1.0,
) -> AcetylcholineState:
    """ACh系を更新する。

    NBM→BLA: 脅威/新規刺激でACh放出↑ → 恐怖記憶強化
    NAc CIN: 報酬後の努力でACh↑ → DA放出増幅
    """
    new = AcetylcholineState(
        nbm_ach=ach.nbm_ach,
        nac_cin=ach.nac_cin,
        nicotinic_activation=ach.nicotinic_activation,
        muscarinic_activation=ach.muscarinic_activation,
    )

    # NBM→BLA ACh: 脅威で上昇
    nbm_drive = threat_signal * 0.4 + bla_activity * 0.2
    new.nbm_ach += (nbm_drive - ach.nbm_ach * 0.1) * dt_ms * 0.01
    new.nbm_ach = max(0.0, min(1.0, new.nbm_ach))

    # NAc CIN: 報酬で上昇
    cin_drive = reward_signal * 0.3
    new.nac_cin += (cin_drive - ach.nac_cin * 0.1) * dt_ms * 0.01
    new.nac_cin = max(0.0, min(1.0, new.nac_cin))

    # ニコチン受容体: ACh依存
    new.nicotinic_activation = 0.5 * (new.nbm_ach + new.nac_cin)

    # ムスカリン受容体: 緩やかに追従
    new.muscarinic_activation += (new.nbm_ach * 0.5 - ach.muscarinic_activation) * 0.05 * dt_ms * 0.01
    new.muscarinic_activation = max(0.0, min(1.0, new.muscarinic_activation))

    return new


def compute_theta_coherence(
    theta1: ThetaOscillator,
    theta2: ThetaOscillator,
) -> float:
    """2領域間のシータ同期度（位相結合度）を計算する。

    Phase Locking Value (PLV) の簡略版。
    """
    phase_diff = abs(theta1.phase - theta2.phase)
    # 位相差が小さいほど同期度が高い
    return max(0.0, math.cos(phase_diff))


@dataclass
class StructuralPlasticityState:
    """構造的可塑性（スパイン動態）の状態。

    恐怖条件付け: スパイン新生→記憶強化
    消去学習: スパイン除去→記憶弱化
    PNN(ペリニューロナルネット): 成熟度が高いほど消去耐性↑
    """

    spine_density: float = 1.0      # 相対的スパイン密度 (0-2, 1=ベースライン)
    pnn_maturity: float = 0.7       # PNN成熟度 (0=未成熟=消去容易, 1=成熟=消去困難)
    bdnf_level: float = 0.5         # BDNF (脳由来神経栄養因子)


def update_structural_plasticity(
    sp: StructuralPlasticityState,
    ltp_occurred: bool,
    ltd_occurred: bool,
    amygdala_activity: float,
    dt_ms: float = 1.0,
) -> StructuralPlasticityState:
    """構造的可塑性を更新する。"""
    new = StructuralPlasticityState(
        spine_density=sp.spine_density,
        pnn_maturity=sp.pnn_maturity,
        bdnf_level=sp.bdnf_level,
    )

    # LTP → スパイン新生 (BDNF依存)
    if ltp_occurred:
        growth = 0.01 * sp.bdnf_level * dt_ms * 0.01
        new.spine_density = min(2.0, sp.spine_density + growth)

    # LTD → スパイン除去 (PNN成熟度で制限)
    if ltd_occurred:
        pruning = 0.008 * (1.0 - sp.pnn_maturity) * dt_ms * 0.01
        new.spine_density = max(0.3, sp.spine_density - pruning)

    # BDNF: 扁桃体活性で上昇
    new.bdnf_level += (amygdala_activity * 0.3 - sp.bdnf_level * 0.05) * dt_ms * 0.01
    new.bdnf_level = max(0.0, min(1.0, new.bdnf_level))

    # PNN: ゆっくり成熟（発達的変化のモデル化）
    new.pnn_maturity = min(1.0, sp.pnn_maturity + 0.0001 * dt_ms * 0.01)

    return new
