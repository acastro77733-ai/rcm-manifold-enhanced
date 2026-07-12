from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TopologyConfig:
    repair_strength_floor: float = 0.26
    similarity_threshold: float = 0.55
    repair_decay_horizon: int = 8
    max_repairs: int = 2
