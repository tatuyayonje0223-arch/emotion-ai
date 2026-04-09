"""SBI推定によるキャリブレーション済みコンフィグ。

ABC rejection (n=50) の最良パラメータを固定。
手動設定ではなくベイズ推定の事後分布から導出。
"""

from src.brian2_circuits.fear_circuit_v2 import FearV2Config
from src.brian2_circuits.reward_circuit_v2 import RewardV2Config
from src.brian2_circuits.stress_circuit_v2 import StressV2Config


# SBI ABC rejection (n=50, score=0.928) の最良パラメータ
CALIBRATED_FEAR_CONFIG = FearV2Config(
    cs_amp=17.7,
    us_amp=14.7,
    bg_noise=1.7,
    sustained_threat_amp=5.0,
    duration_ms=250,
    cs_dur_ms=120,
    us_onset_ms=150,
    us_dur_ms=25,
)

# 報酬回路: tonic入力較正済み
CALIBRATED_REWARD_CONFIG = RewardV2Config(
    duration_ms=300,
    cs_dur_ms=80,
    reward_onset_ms=200,
    reward_dur_ms=40,
    reward_amp=15.0,
    bg_noise=3.0,
)

# ストレス回路: デフォルト（検証スコア~0.9で良好）
CALIBRATED_STRESS_CONFIG = StressV2Config(
    duration_ms=250,
)
