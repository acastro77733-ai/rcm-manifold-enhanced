from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RepresentationConfig:
    overrepresentation_penalty: float = 0.05
    confidence_weight: float = 0.35
    structure_weight: float = 0.25
    memory_weight: float = 0.20
    min_dim: int = 2
    max_dim: int = 8
    trial_duration: int = 2
    minimum_improvement: float = 0.05
    switch_cost: float = 0.03
    cooldown_steps: int = 3
    collapse_threshold: float = 0.35
