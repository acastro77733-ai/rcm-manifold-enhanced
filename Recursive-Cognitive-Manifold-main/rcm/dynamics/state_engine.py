from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rcm.reproducibility import seed_everything


@dataclass
class StateEngineConfig:
    dt: float = 0.12
    decay_rate: float = 0.08
    input_gain: float = 0.12
    topology_gain: float = 0.05
    recall_gain: float = 0.06
    activation_capacity: float = 2.0
    clip_bounds: tuple[float, float] = (-2.0, 2.0)


@dataclass
class StateEngine:
    config: StateEngineConfig = field(default_factory=StateEngineConfig)
    last_contributions: list[dict[str, Any]] = field(default_factory=list)
    last_state_vector: np.ndarray | None = None

    def _state_blocks(self, manifold) -> dict[str, np.ndarray]:
        if not manifold.nodes:
            return {"state": np.zeros(1, dtype=float)}

        node_states = np.stack([np.asarray(node.local_state, dtype=float) for node in manifold.nodes.values()], axis=0)
        mean_state = np.mean(node_states, axis=0) if node_states.size else np.zeros(manifold.state_dim, dtype=float)
        edge_strength = float(np.mean([edge.strength for edge in manifold.edges.values()]) if manifold.edges else 0.0)
        confidence = float(np.mean([getattr(node, "confidence", 0.0) for node in manifold.nodes.values()]))
        energy = float(np.mean([getattr(node, "energy", 0.0) for node in manifold.nodes.values()]))
        topology = float(len(manifold.regions) + len(manifold.nodes) + len(manifold.edges))
        return {
            "state": np.asarray(mean_state, dtype=float),
            "edge": np.asarray([edge_strength], dtype=float),
            "confidence": np.asarray([confidence], dtype=float),
            "energy": np.asarray([energy], dtype=float),
            "topology": np.asarray([topology], dtype=float),
        }

    def _flatten_state(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(block, dtype=float).ravel() for block in blocks.values()])

    def _base_contribution(self, manifold, state_vector: np.ndarray) -> np.ndarray:
        return -self.config.decay_rate * state_vector

    def _input_contribution(self, manifold, state_vector: np.ndarray, external_input: dict | None) -> np.ndarray:
        if not external_input:
            return np.zeros_like(state_vector)
        signals = []
        for signal in external_input.values():
            signals.append(np.asarray(signal, dtype=float).ravel())
        if not signals:
            return np.zeros_like(state_vector)
        magnitude = np.mean([np.linalg.norm(signal) for signal in signals])
        return np.full_like(state_vector, fill_value=self.config.input_gain * magnitude)

    def _topology_contribution(self, manifold, state_vector: np.ndarray) -> np.ndarray:
        if not manifold.edges:
            return np.zeros_like(state_vector)
        edge_count = float(len(manifold.edges))
        return np.full_like(state_vector, fill_value=self.config.topology_gain * edge_count)

    def _recall_contribution(self, manifold, state_vector: np.ndarray) -> np.ndarray:
        if not manifold.nodes:
            return np.zeros_like(state_vector)
        recalled = 0.0
        for node in manifold.nodes.values():
            if getattr(node, "long_term_memory", None):
                recalled += float(len(node.long_term_memory))
        recalled = max(recalled, 0.0)
        return np.full_like(state_vector, fill_value=self.config.recall_gain * recalled)

    def _apply_recall(self, manifold, node_states: np.ndarray, external_input: dict | None) -> np.ndarray:
        if not node_states.size:
            return node_states
        if not manifold.nodes:
            return node_states
        if external_input is None:
            return node_states
        if not external_input:
            return node_states

        recalled = np.zeros_like(node_states, dtype=float)
        for idx, (node_id, node) in enumerate(manifold.nodes.items()):
            if idx >= len(node_states):
                break
            signal = external_input.get(node_id)
            if signal is not None:
                recalled[idx] = np.asarray(signal, dtype=float).ravel()[: node_states.shape[1]]
            elif getattr(node, "long_term_memory", None):
                memory_trace = np.mean([np.asarray(entry[1], dtype=float) for entry in node.long_term_memory[-3:]], axis=0) if node.long_term_memory else np.zeros(node_states.shape[1], dtype=float)
                recalled[idx] = memory_trace[: node_states.shape[1]]
        if np.linalg.norm(recalled) > 0.0:
            node_states = node_states + 0.35 * recalled
        return node_states

    def _project_node_state(self, base_state: np.ndarray, influence: float, activation_capacity: float) -> np.ndarray:
        projected = np.asarray(base_state, dtype=float).ravel()
        projected = projected + influence * np.ones_like(projected)
        projected = np.clip(projected, -activation_capacity, activation_capacity)
        norm = float(np.linalg.norm(projected))
        if norm > activation_capacity:
            projected = projected * (activation_capacity / norm)
        return projected

    def step(self, manifold, external_input: dict | None = None, recall_enabled: bool = True, input_enabled: bool = True, topology_enabled: bool = True, decay_enabled: bool = True) -> dict[str, Any]:
        blocks = self._state_blocks(manifold)
        state_vector = self._flatten_state(blocks)
        base = self._base_contribution(manifold, state_vector) if decay_enabled else np.zeros_like(state_vector)
        input_term = self._input_contribution(manifold, state_vector, external_input) if input_enabled else np.zeros_like(state_vector)
        topology_term = self._topology_contribution(manifold, state_vector) if topology_enabled else np.zeros_like(state_vector)
        recall_term = self._recall_contribution(manifold, state_vector) if recall_enabled else np.zeros_like(state_vector)

        update = self.config.dt * (base + input_term + topology_term + recall_term)
        candidate = state_vector + update
        bounded = np.clip(candidate, self.config.clip_bounds[0], self.config.clip_bounds[1])
        bounded = np.clip(bounded, -self.config.activation_capacity, self.config.activation_capacity)

        self.last_state_vector = bounded
        self.last_contributions = [
            {"term": "decay", "value": base},
            {"term": "input", "value": input_term},
            {"term": "topology", "value": topology_term},
            {"term": "recall", "value": recall_term},
        ]

        node_states = []
        if manifold.nodes:
            state_dim = max(1, manifold.state_dim)
            node_count = len(manifold.nodes)
            for index, (node_id, node) in enumerate(manifold.nodes.items()):
                base_state = np.asarray(node.local_state, dtype=float).ravel()
                if base_state.size < state_dim:
                    padded = np.zeros(state_dim, dtype=float)
                    padded[: base_state.size] = base_state
                    base_state = padded
                else:
                    base_state = base_state[:state_dim]

                influence = float(bounded[index % max(1, len(bounded))]) if len(bounded) else 0.0
                if recall_enabled and external_input is not None and node_id in external_input:
                    influence += 0.1 * float(np.linalg.norm(np.asarray(external_input[node_id], dtype=float)))
                projected = self._project_node_state(base_state, influence, self.config.activation_capacity)
                node_states.append(projected)
            node_states = np.asarray(node_states, dtype=float)
            if node_states.shape[0] != node_count:
                node_states = np.asarray([np.zeros(state_dim, dtype=float) for _ in range(node_count)], dtype=float)
        else:
            node_states = np.asarray([], dtype=float)

        def _state_to_vector(state):
            return np.asarray(state, dtype=float).ravel()

        node_vectors = [_state_to_vector(state) for state in node_states]
        return {
            "state_vector": bounded,
            "node_states": node_vectors,
            "bounded": bool(np.all(np.isfinite(bounded))),
            "block_names": list(blocks.keys()),
            "contributions": self.last_contributions,
            "recall_applied": bool(recall_enabled and manifold.nodes and any(getattr(node, "long_term_memory", None) for node in manifold.nodes.values())),
        }
