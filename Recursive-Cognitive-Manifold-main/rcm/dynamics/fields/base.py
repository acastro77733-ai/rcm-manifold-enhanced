from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FieldModel:
    seed: int = 0
    state_dim: int = 4
    control_mode: str = "full"
    field_state: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    step_count: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self._frozen_parameters: dict[str, float] | None = None

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

    def _base_parameters(self, nonlinear: bool) -> dict[str, float]:
        return {
            "reaction_gain": 0.28 if nonlinear else 0.22,
            "diffusion_gain": 0.16,
            "transport_gain": 0.18,
            "topology_gain": 0.05,
            "energy_gain": 0.07,
            "confidence_gain": 0.09,
            "damping": 0.22,
        }

    def _resolve_parameters(self, nonlinear: bool) -> dict[str, float]:
        params = dict(self._base_parameters(nonlinear))
        mode = str(getattr(self, "control_mode", "full")).lower()
        if mode == "randomized_field":
            rng = np.random.default_rng(self.seed)
            for key in list(params.keys()):
                params[key] *= float(np.clip(1.0 + rng.normal(0.0, 0.18), 0.6, 1.4))
        elif mode in {"fixed_parameter", "frozen_parameter_field"}:
            if self._frozen_parameters is None:
                rng = np.random.default_rng(self.seed)
                self._frozen_parameters = {
                    key: value * float(np.clip(1.0 + rng.normal(0.0, 0.12), 0.75, 1.25))
                    for key, value in params.items()
                }
            params.update(self._frozen_parameters)
        elif mode == "linearized_field":
            params["reaction_gain"] = 0.20
        elif mode == "diffusion_only":
            params["reaction_gain"] = 0.0
            params["transport_gain"] = 0.0
        elif mode == "identity_field":
            params["reaction_gain"] = 1.0
            params["diffusion_gain"] = 0.0
            params["transport_gain"] = 0.0
            params["damping"] = 0.0
        elif mode == "no_field":
            params = {key: 0.0 for key in params}
        return params

    def _evolve(self, node_summary, mean_energy, mean_confidence, topology_stats, nonlinear: bool):
        node_summary = np.asarray(node_summary, dtype=float)
        self._resize_state(node_summary.shape[0])
        width = self.field_state.shape[0]
        mode = str(getattr(self, "control_mode", "full")).lower()

        topology_vector = np.array(
            [
                topology_stats.get("node_count", 0.0),
                topology_stats.get("edge_count", 0.0),
                topology_stats.get("density", 0.0),
                topology_stats.get("stability", 0.0),
            ],
            dtype=float,
        )
        topology_vector = self._fit_vector(topology_vector, width)

        params = self._resolve_parameters(nonlinear)
        drive = self._fit_vector(node_summary, width)
        drive += params["topology_gain"] * topology_vector
        drive += params["energy_gain"] * float(mean_energy)
        drive += params["confidence_gain"] * float(mean_confidence)

        if mode == "no_field":
            reaction = np.zeros(width, dtype=float)
            transport = np.zeros(width, dtype=float)
            diffusion = np.zeros(width, dtype=float)
            next_state = np.zeros(width, dtype=float)
        else:
            if mode == "identity_field":
                reaction = drive
            elif nonlinear and mode not in {"linearized_field", "diffusion_only"}:
                reaction = np.tanh(1.15 * drive) + 0.05 * np.sign(drive)
            else:
                reaction = drive
            reaction = params["reaction_gain"] * reaction
            transport = params["transport_gain"] * np.clip(drive - self.field_state, -1.0, 1.0)
            laplacian = np.roll(self.field_state, 1) + np.roll(self.field_state, -1) - (2.0 * self.field_state) if width > 1 else -self.field_state
            diffusion = params["diffusion_gain"] * np.clip(laplacian + (drive - self.field_state), -1.0, 1.0)
            next_state = (1.0 - params["damping"]) * self.field_state + reaction + transport + diffusion
        self.field_state = np.clip(next_state, -2.0, 2.0)

        node_modulation = np.tanh(self.field_state) if nonlinear and mode not in {"linearized_field", "identity_field"} else np.clip(self.field_state, -1.0, 1.0)
        field_variance = float(np.var(self.field_state)) if self.field_state.size else 0.0
        synchronization_pressure = float(
            np.clip(0.45 * mean_confidence + 0.30 * topology_stats.get("stability", 0.0) + 0.25 * topology_stats.get("density", 0.0), 0.0, 1.0)
        )
        plasticity_pressure = float(
            np.clip(0.45 * field_variance + 0.35 * (1.0 - mean_confidence) + 0.20 * topology_stats.get("density", 0.0), 0.0, 1.0)
        )
        state_influence = float(np.linalg.norm(node_modulation)) * synchronization_pressure
        clipped_fraction = float(np.mean(np.abs(self.field_state) >= 1.99)) if self.field_state.size else 0.0
        if state_influence <= 0.01:
            influence_status = "negligible"
        elif clipped_fraction > 0.5:
            influence_status = "clipped"
        elif state_influence >= 1.25:
            influence_status = "dominant"
        else:
            influence_status = "active"

        self.step_count += 1
        return {
            "metrics": {
                "field_norm": float(np.linalg.norm(self.field_state)),
                "field_energy": float(np.linalg.norm(self.field_state)),
                "field_mean": float(np.mean(self.field_state)) if self.field_state.size else 0.0,
                "field_variance": field_variance,
                "synchronization_pressure": synchronization_pressure,
                "plasticity_pressure": plasticity_pressure,
                "state_influence": state_influence,
                "influence_status": influence_status,
                "clipped_fraction": clipped_fraction,
                "mode": mode,
            },
            "collapse_flag": bool(mean_energy < 0.2 or topology_stats.get("stability", 0.0) < 0.1),
            "node_modulation": node_modulation,
            "node_influence": synchronization_pressure * node_modulation,
            "synchronization_pressure": synchronization_pressure,
            "plasticity_pressure": plasticity_pressure,
            "field_variance": field_variance,
            "field_state": self.field_state.copy(),
            "mode": mode,
        }
