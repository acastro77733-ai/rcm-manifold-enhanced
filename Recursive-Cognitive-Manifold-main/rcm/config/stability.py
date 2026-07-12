from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StabilityConfig:
    collapse_threshold: float = 0.75
    coherence_floor: float = 0.25
    resource_budget: float = 1.0
    stability_weight: float = 0.5
