from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FieldConfig:
    hrm_gain: float = 0.10
    diffusion_gain: float = 0.05
    stabilization_gain: float = 0.03
    repair_gain: float = 0.02
    region_update_gain: float = 0.08
    manifold_update_gain: float = 0.06
    child_feedback_gain: float = 0.05
    control_mode: str = "full"
    negligible_threshold: float = 0.01
    dominant_threshold: float = 1.25
    field_model: str = "legacy"
