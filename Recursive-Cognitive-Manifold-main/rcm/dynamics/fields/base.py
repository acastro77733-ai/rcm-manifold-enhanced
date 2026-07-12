from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FieldModel:
    seed: int = 0
    state_dim: int = 4
    field_state: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    step_count: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def step(self, node_summary, mean_energy, mean_confidence, topology_stats):
        raise NotImplementedError

    def _resize_state(self, width: int):
        if width == self.field_state.shape[0]:
            return
        resized = np.zeros(width, dtype=float)
        overlap = min(width, self.field_state.shape[0])
        if overlap:
            resized[:overlap] = self.field_state[:overlap]
        self.field_state = resized

    def _fit_vector(self, vector, width: int) -> np.ndarray:
        fitted = np.zeros(width, dtype=float)
        overlap = min(width, len(vector))
        if overlap:
            fitted[:overlap] = np.asarray(vector, dtype=float)[:overlap]
        return fitted

    def _bounded(self, value: float) -> float:
        return float(np.clip(value, -2.0, 2.0))
