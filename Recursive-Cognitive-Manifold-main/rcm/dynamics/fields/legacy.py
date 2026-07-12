from __future__ import annotations

import numpy as np

from .base import FieldModel


class LegacyFieldModel(FieldModel):
    def step(self, node_summary, mean_energy, mean_confidence, topology_stats):
        node_summary = np.asarray(node_summary, dtype=float)
        self._resize_state(node_summary.shape[0])

        topology_vector = np.array(
            [
                topology_stats.get("node_count", 0.0),
                topology_stats.get("edge_count", 0.0),
                topology_stats.get("density", 0.0),
                topology_stats.get("stability", 0.0),
            ],
            dtype=float,
        )
        topology_vector = self._fit_vector(topology_vector, self.field_state.shape[0])

        noise = self.rng.normal(0.0, 0.015, size=self.field_state.shape[0])
        drive = self._fit_vector(node_summary, self.field_state.shape[0])
        drive += 0.04 * topology_vector
        drive += 0.06 * mean_energy
        drive += 0.08 * mean_confidence
        self.field_state = 0.74 * self.field_state + 0.22 * drive + noise

        node_modulation = np.tanh(self.field_state)
        field_variance = float(np.var(self.field_state))
        synchronization_pressure = float(
            np.clip(0.45 * mean_confidence + 0.35 * topology_stats.get("stability", 0.0) + 0.2 * topology_stats.get("density", 0.0), 0.0, 1.0)
        )
        plasticity_pressure = float(
            np.clip(0.5 * field_variance + 0.3 * (1.0 - mean_confidence) + 0.2 * topology_stats.get("density", 0.0), 0.0, 1.0)
        )
        collapse_flag = bool(mean_energy < 0.25 or topology_stats.get("stability", 0.0) < 0.15)

        self.step_count += 1
        return {
            "metrics": {
                "field_norm": float(np.linalg.norm(self.field_state)),
                "field_mean": float(np.mean(self.field_state)) if self.field_state.size else 0.0,
                "field_variance": field_variance,
                "synchronization_pressure": synchronization_pressure,
                "plasticity_pressure": plasticity_pressure,
            },
            "collapse_flag": collapse_flag,
            "node_modulation": node_modulation,
            "synchronization_pressure": synchronization_pressure,
            "plasticity_pressure": plasticity_pressure,
            "field_variance": field_variance,
            "field_state": self.field_state,
        }
