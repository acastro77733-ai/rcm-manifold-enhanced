from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TopologyConfig:
    repair_strength_floor: float = 0.26
    similarity_threshold: float = 0.55
    repair_decay_horizon: int = 8
    max_repairs: int = 2
    utility_threshold: float = 0.015
    compute_cost_weight: float = 0.01
    structural_cost_weight: float = 0.015
    surgery_cost_weight: float = 0.02
