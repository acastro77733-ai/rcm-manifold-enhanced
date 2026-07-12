from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StateDynamicsConfig:
    input_gain: float = 1.0
    synchronization_gain: float = 0.15
    recall_gain: float = 0.20
    field_gain: float = 0.10
    decay: float = 0.03
    saturation_limit: float = 1.0
