"""定量検証ターゲット。文献データに基づく較正目標値。

[監査C2+H7修正] 「何をキャリブレーションの目標にするか」を明示定義する。

恐怖回路:
  Quirk et al. (1995) J Neurosci: LA neurons, CS-evoked firing
  Ciocchi et al. (2010) Nature: CeL SOM+/PKCdelta+ mutual inhibition
  Herry et al. (2008) Nature: BLA fear/extinction neurons
  Likhtik et al. (2014) Nature Neurosci: BLA-mPFC theta during safety

報酬回路:
  Schultz et al. (1997) Science: VTA DA RPE
  Cohen et al. (2012) Nature: VTA DA neurons during conditioning

ストレス回路:
  de Kloet et al. (2005): HPA axis cortisol dynamics
  Herman et al. (2016): Stress neurocircuitry
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FearCircuitTargets:
    """恐怖回路の文献ベース定量目標。"""

    # BLA baseline firing rate (Hz) — Pare & Bhatt 2011, Quirk 1995
    bla_baseline_hz: tuple[float, float] = (1.0, 15.0)

    # BLA CS-evoked during conditioning (Hz) — Quirk 1995
    bla_conditioned_hz: tuple[float, float] = (15.0, 50.0)

    # Conditioning increase ratio — at least 1.5x
    conditioning_ratio_min: float = 1.5

    # CeL SOM+ > CeL PKCd+ during fear expression — Ciocchi 2010
    # SOM+ should fire more than PKCd+ when fear is expressed
    cel_som_gt_pkcd: bool = True

    # CeM output during conditioning > baseline — essential for freeze
    cem_increase: bool = True

    # Extinction: 10-30 CS-alone trials for 50% reduction — Quirk 2003
    extinction_trials_for_50pct: tuple[int, int] = (5, 30)

    # IL > PL during extinction recall — Sierra-Mercado 2011
    il_gt_pl_during_extinction: bool = True

    # BNST > baseline during sustained threat — Davis 2010
    bnst_sustained_threat_increase: bool = True


@dataclass
class RewardCircuitTargets:
    """報酬回路の文献ベース定量目標。"""

    # VTA DA burst at unexpected reward (Hz) — Schultz 1997
    vta_da_burst_hz: tuple[float, float] = (10.0, 60.0)

    # VTA DA baseline tonic (Hz) — Grace 1991
    vta_da_tonic_hz: tuple[float, float] = (1.0, 10.0)

    # RPE > 0 for unexpected reward
    rpe_positive_for_unexpected: bool = True

    # RPE < 0 for reward omission (after training)
    rpe_negative_for_omission: bool = True

    # D1 rate > D2 rate during reward approach — Frank 2004
    d1_gt_d2_during_reward: bool = True

    # LHb activation during reward omission — Matsumoto & Hikosaka 2007
    lhb_increase_during_omission: bool = True


@dataclass
class StressCircuitTargets:
    """ストレス回路の文献ベース定量目標。"""

    # Cortisol peak at 15-30 min post-stress — de Kloet 2005
    # (simulation time units: 1 trial ≈ several minutes)
    cortisol_rises_with_stress: bool = True

    # Cortisol returns toward baseline during recovery
    cortisol_recovers: bool = True

    # Chronic stress: GR sensitivity decreases — Sapolsky 2000
    chronic_gr_decreases: bool = True

    # Chronic stress impairs recovery vs acute-only
    chronic_slower_recovery: bool = True

    # BLA activation with stressor
    bla_activation_with_stressor: bool = True

    # NE (LC) increases with stress
    ne_increases_with_stress: bool = True


# 全ターゲットのインスタンス
FEAR_TARGETS = FearCircuitTargets()
REWARD_TARGETS = RewardCircuitTargets()
STRESS_TARGETS = StressCircuitTargets()
