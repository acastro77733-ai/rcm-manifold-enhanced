from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rcm.reproducibility import seed_everything


@dataclass
class UnifiedStateConfig:
    dt: float = 0.1
    clip_bounds: tuple[float, float] = (-10.0, 10.0)
    residual_scale: float = 0.01


@dataclass
class BoundedStateEngine:
    config: UnifiedStateConfig = field(default_factory=UnifiedStateConfig)
    last_contributions: list[dict[str, Any]] = field(default_factory=list)
    last_state_vector: np.ndarray | None = None

    def _state_blocks(self, manifold) -> dict[str, np.ndarray]:
        phi = float(np.mean([getattr(node, "confidence", 0.0) for node in manifold.nodes.values()]) if manifold.nodes else 0.0)
        geometry = float(np.mean([edge.strength for edge in manifold.edges.values()]) if manifold.edges else 0.0)
        topology = float(len(manifold.regions) + len(manifold.nodes) + len(manifold.edges))
        memory = float(np.mean([len(node.long_term_memory) for node in manifold.nodes.values()]) if manifold.nodes else 0.0)
        cognitive = float(np.mean([node.energy for node in manifold.nodes.values()]) if manifold.nodes else 0.0)
        hierarchy = float(len(manifold.child_manifolds))
        budget = float(max(1.0, manifold.state_dim))
        return {
            "phi": np.asarray([phi], dtype=float),
            "geometry": np.asarray([geometry], dtype=float),
            "topology": np.asarray([topology], dtype=float),
            "memory": np.asarray([memory], dtype=float),
            "cognitive": np.asarray([cognitive], dtype=float),
            "hierarchy": np.asarray([hierarchy], dtype=float),
            "budget": np.asarray([budget], dtype=float),
        }

    def _flatten_state(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(block, dtype=float).ravel() for block in blocks.values()])

    def _base_contribution(self, manifold, state_vector: np.ndarray) -> np.ndarray:
        return 0.01 * state_vector

    def _input_contribution(self, manifold, state_vector: np.ndarray, external_input: dict | None) -> np.ndarray:
        if not external_input:
            return np.zeros_like(state_vector)
        magnitude = float(np.mean([np.linalg.norm(np.asarray(signal, dtype=float)) for signal in external_input.values()]))
        return np.full_like(state_vector, fill_value=0.001 * magnitude)

    def _interaction_contribution(self, manifold, state_vector: np.ndarray) -> np.ndarray:
        if len(manifold.edges) == 0:
            return np.zeros_like(state_vector)
        return 0.0005 * np.ones_like(state_vector) * len(manifold.edges)

    def step(self, manifold, external_input: dict | None = None) -> dict[str, Any]:
        blocks = self._state_blocks(manifold)
        state_vector = self._flatten_state(blocks)
        base = self._base_contribution(manifold, state_vector)
        input_term = self._input_contribution(manifold, state_vector, external_input)
        interaction = self._interaction_contribution(manifold, state_vector)
        residual = self.config.residual_scale * np.random.default_rng(seed_everything(None)).standard_normal(size=state_vector.shape)
        update = self.config.dt * (base + input_term + interaction) + residual
        bounded = np.clip(state_vector + update, self.config.clip_bounds[0], self.config.clip_bounds[1])
        self.last_state_vector = bounded
        self.last_contributions = [
            {"term": "base", "value": base},
            {"term": "input", "value": input_term},
            {"term": "interaction", "value": interaction},
            {"term": "residual", "value": residual},
        ]
        return {
            "state_vector": bounded,
            "bounded": bool(np.all(np.isfinite(bounded))),
            "block_names": list(blocks.keys()),
            "contributions": self.last_contributions,
        }
