import numpy as np


def summarize_region_nodes(region_nodes):
    if not region_nodes:
        return np.array([], dtype=float), 0.0, 0.0
    node_summary = np.mean([node.local_state for node in region_nodes], axis=0)
    mean_energy = float(np.mean([node.energy for node in region_nodes]))
    mean_confidence = float(np.mean([node.confidence for node in region_nodes]))
    return node_summary, mean_energy, mean_confidence


class RegionalHRMField:
    def __init__(self, seed: int = 0, state_dim: int = 4):
        self.seed = seed
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)
        self.field_state = np.zeros(state_dim, dtype=float)
        self.step_count = 0

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
        }

    def _resize_state(self, width: int):
        if width == self.field_state.shape[0]:
            return
        resized = np.zeros(width, dtype=float)
        overlap = min(width, self.field_state.shape[0])
        if overlap:
            resized[:overlap] = self.field_state[:overlap]
        self.field_state = resized

    def _fit_vector(self, vector, width: int):
        fitted = np.zeros(width, dtype=float)
        overlap = min(width, len(vector))
        if overlap:
            fitted[:overlap] = vector[:overlap]
        return fitted