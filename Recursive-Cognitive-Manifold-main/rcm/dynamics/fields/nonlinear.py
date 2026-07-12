from __future__ import annotations

import numpy as np

from .base import FieldModel


class NonlinearFieldModel(FieldModel):
    def step(self, node_summary, mean_energy, mean_confidence, topology_stats):
        return self._evolve(node_summary, mean_energy, mean_confidence, topology_stats, nonlinear=True)
