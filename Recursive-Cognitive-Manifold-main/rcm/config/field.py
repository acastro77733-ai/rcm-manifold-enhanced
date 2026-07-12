from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FieldConfig:
    hrm_gain: float = 0.10
    diffusion_gain: float = 0.05
    stabilization_gain: float = 0.03
    repair_gain: float = 0.02
    field_model: str = "legacy"
