"""DEPRECATED: Brian2版 (src/brian2_circuits/) に置換済み。

移行マッピング:
  fear_circuit.py → brian2_circuits/fear_circuit_v2.py
  reward_circuit.py → brian2_circuits/reward_circuit_v2.py
  stress_circuit.py → brian2_circuits/stress_circuit_v2.py
"""

import warnings
warnings.warn(
    "src.circuits is deprecated. Use src.brian2_circuits instead.",
    DeprecationWarning,
    stacklevel=2,
)
