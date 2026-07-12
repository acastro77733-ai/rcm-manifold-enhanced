from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StateEngineConfig:
    dt: float = 0.12
    external_gain: float = 0.28
    local_gain: float = 0.24
    field_gain: float = 0.18
    diffusion_gain: float = 0.16
    recall_gain: float = 0.18
    regional_gain: float = 0.14
    hierarchy_gain: float = 0.12
    topology_gain: float = 0.08
    damping_gain: float = 0.10
    residual_gain: float = 0.06
    forecast_gain: float = 0.22
    confidence_gain: float = 0.08
    energy_gain: float = 0.10
    activation_capacity: float = 2.0
    clip_bounds: tuple[float, float] = (-2.0, 2.0)
    negligible_threshold: float = 1e-4
    dominant_threshold: float = 1.5


@dataclass
class GovernedState:
    node_order: list[int]
    node_fields: dict[int, np.ndarray]
    regional_fields: dict[tuple[int, ...], np.ndarray]
    geometry_state: np.ndarray
    topology_state: np.ndarray
    memory_influence: dict[int, np.ndarray]
    hierarchical_state: dict[int, np.ndarray]
    resource_allocation: dict[str, float]
    stability_variables: dict[str, float]
    forecast_state: dict[str, Any]

    def flatten(self) -> tuple[np.ndarray, list[str]]:
        blocks = [
            ("nodes", np.concatenate([self.node_fields[node_id] for node_id in self.node_order], axis=0) if self.node_order else np.zeros(0, dtype=float)),
            (
                "regions",
                np.concatenate(
                    [np.asarray(self.regional_fields[signature], dtype=float).ravel() for signature in sorted(self.regional_fields)],
                    axis=0,
                )
                if self.regional_fields
                else np.zeros(0, dtype=float),
            ),
            ("geometry", np.asarray(self.geometry_state, dtype=float).ravel()),
            ("topology", np.asarray(self.topology_state, dtype=float).ravel()),
            (
                "memory",
                np.concatenate(
                    [self.memory_influence.get(node_id, np.zeros_like(self.node_fields[node_id])) for node_id in self.node_order],
                    axis=0,
                )
                if self.node_order
                else np.zeros(0, dtype=float),
            ),
            (
                "hierarchy",
                np.concatenate([self.hierarchical_state.get(node_id, np.zeros_like(self.node_fields[node_id])) for node_id in self.node_order], axis=0)
                if self.node_order
                else np.zeros(0, dtype=float),
            ),
            (
                "resources",
                np.asarray(
                    [
                        self.resource_allocation.get("representation_dim", 0.0),
                        self.resource_allocation.get("node_count", 0.0),
                        self.resource_allocation.get("region_count", 0.0),
                        self.resource_allocation.get("budget_pressure", 0.0),
                    ],
                    dtype=float,
                ),
            ),
            (
                "stability",
                np.asarray(
                    [
                        self.stability_variables.get("mean_confidence", 0.0),
                        self.stability_variables.get("mean_energy", 0.0),
                        self.stability_variables.get("mean_sync", 0.0),
                        self.stability_variables.get("collapse_energy", 0.0),
                        self.stability_variables.get("instability", 0.0),
                    ],
                    dtype=float,
                ),
            ),
            (
                "forecast",
                np.asarray(
                    [
                        self.forecast_state.get("prediction_error", 0.0),
                        self.forecast_state.get("uncertainty", 0.0),
                        self.forecast_state.get("coherence", 0.0),
                    ],
                    dtype=float,
                ),
            ),
        ]
        vectors = [block for _, block in blocks if np.asarray(block).size]
        return (np.concatenate(vectors, axis=0) if vectors else np.zeros(0, dtype=float), [name for name, _ in blocks])


@dataclass
class StateEngine:
    config: StateEngineConfig = field(default_factory=StateEngineConfig)
    last_contributions: list[dict[str, Any]] = field(default_factory=list)
    last_state_vector: np.ndarray | None = None
    last_governed_state: GovernedState | None = None

    def _node_order(self, manifold) -> list[int]:
        return sorted(int(node_id) for node_id in manifold.nodes)

    def _fit_vector(self, signal, width: int) -> np.ndarray:
        vector = np.zeros(width, dtype=float)
        if signal is None:
            return vector
        signal = np.asarray(signal, dtype=float).ravel()
        overlap = min(width, signal.size)
        if overlap:
            vector[:overlap] = signal[:overlap]
        return vector

    def _node_matrix(self, manifold, node_order: list[int]) -> np.ndarray:
        if not node_order:
            return np.zeros((0, max(1, getattr(manifold, "state_dim", 1))), dtype=float)
        return np.stack([self._fit_vector(manifold.nodes[node_id].local_state, manifold.state_dim) for node_id in node_order], axis=0)

    def _field_drive(self, manifold, node_id: int, width: int) -> np.ndarray:
        drive = np.zeros(width, dtype=float)
        field_state = getattr(getattr(manifold, "hrm_field", None), "field_state", None)
        if field_state is not None:
            drive += self._fit_vector(field_state, width)
        for signature, region in manifold.regions.items():
            if node_id not in signature:
                continue
            regional_field = getattr(getattr(region, "hrm_field", None), "field_state", None)
            if regional_field is not None:
                drive += self._fit_vector(regional_field, width)
        return np.tanh(drive)

    def _external_forcing(self, manifold, node_order: list[int], width: int, external_input: dict | None) -> np.ndarray:
        forcing = np.zeros((len(node_order), width), dtype=float)
        if not external_input:
            return forcing
        for index, node_id in enumerate(node_order):
            forcing[index] = self._fit_vector(external_input.get(node_id), width)
        return forcing

    def _memory_influence(self, manifold, node_order: list[int], node_states: np.ndarray, recall_enabled: bool) -> tuple[dict[int, np.ndarray], np.ndarray]:
        width = node_states.shape[1] if node_states.size else max(1, getattr(manifold, "state_dim", 1))
        memory_vectors: dict[int, np.ndarray] = {}
        memory_matrix = np.zeros((len(node_order), width), dtype=float)
        if not recall_enabled:
            return memory_vectors, memory_matrix
        for index, node_id in enumerate(node_order):
            node = manifold.nodes[node_id]
            recalled = manifold.episodic_memory.retrieve(node_states[index], node_id=node_id, top_k=3, min_similarity=0.0)
            if recalled is not None:
                confidence = float(recalled.get("outcome", {}).get("confidence", 0.0))
                similarity = float(recalled.get("similarity", 0.0))
                target = self._fit_vector(recalled.get("next_state"), width)
                memory_vector = similarity * max(confidence, 0.0) * target
            elif getattr(node, "long_term_memory", None):
                traces = [self._fit_vector(entry[1], width) for entry in list(node.long_term_memory)[-3:]]
                memory_vector = np.mean(traces, axis=0) if traces else np.zeros(width, dtype=float)
            else:
                memory_vector = np.zeros(width, dtype=float)
            memory_vectors[node_id] = memory_vector
            memory_matrix[index] = memory_vector
        return memory_vectors, memory_matrix

    def _regional_targets(self, manifold, node_order: list[int], node_states: np.ndarray) -> tuple[dict[tuple[int, ...], np.ndarray], np.ndarray]:
        width = node_states.shape[1] if node_states.size else max(1, getattr(manifold, "state_dim", 1))
        regional_fields: dict[tuple[int, ...], np.ndarray] = {}
        regional_matrix = np.zeros((len(node_order), width), dtype=float)
        index_lookup = {node_id: index for index, node_id in enumerate(node_order)}
        for signature, region in manifold.regions.items():
            if not signature:
                continue
            indices = [index_lookup[node_id] for node_id in signature if node_id in index_lookup]
            if not indices:
                continue
            summary = np.mean(node_states[indices], axis=0)
            field_state = getattr(getattr(region, "hrm_field", None), "field_state", None)
            summary = summary + 0.35 * self._fit_vector(field_state, width)
            regional_fields[signature] = summary
            for index in indices:
                regional_matrix[index] = summary - node_states[index]
        return regional_fields, regional_matrix

    def _hierarchy_feedback(self, manifold, node_order: list[int], node_states: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
        width = node_states.shape[1] if node_states.size else max(1, getattr(manifold, "state_dim", 1))
        hierarchy_vectors = {node_id: np.zeros(width, dtype=float) for node_id in node_order}
        hierarchy_matrix = np.zeros((len(node_order), width), dtype=float)
        index_lookup = {node_id: index for index, node_id in enumerate(node_order)}
        for child_manifold in getattr(manifold, "child_manifolds", []):
            parent_region_map = getattr(child_manifold, "parent_region_map", {})
            for meta_id, signature in parent_region_map.items():
                if meta_id not in child_manifold.nodes:
                    continue
                child_state = self._fit_vector(child_manifold.nodes[meta_id].local_state, width)
                for node_id in signature:
                    if node_id not in hierarchy_vectors:
                        continue
                    hierarchy_vectors[node_id] = hierarchy_vectors[node_id] + child_state
        for node_id, feedback in hierarchy_vectors.items():
            index = index_lookup[node_id]
            hierarchy_matrix[index] = feedback - node_states[index]
        return hierarchy_vectors, hierarchy_matrix

    def _diffusion(self, manifold, node_order: list[int], node_states: np.ndarray) -> np.ndarray:
        diffusion = np.zeros_like(node_states, dtype=float)
        if not node_order:
            return diffusion
        index_lookup = {node_id: index for index, node_id in enumerate(node_order)}
        for (source, target), edge in manifold.edges.items():
            if source not in index_lookup or target not in index_lookup:
                continue
            left = index_lookup[source]
            right = index_lookup[target]
            delta = node_states[right] - node_states[left]
            diffusion[left] += edge.strength * delta
        if manifold.edges:
            diffusion /= max(len(manifold.edges), 1)
        return diffusion

    def _topology_transport(self, manifold, node_order: list[int], node_states: np.ndarray, topology_enabled: bool) -> np.ndarray:
        transport = np.zeros_like(node_states, dtype=float)
        if not topology_enabled:
            return transport
        index_lookup = {node_id: index for index, node_id in enumerate(node_order)}
        geometry_weight = float(getattr(manifold, "last_geometry_feedback", {}).get("laplacian_energy", 0.0))
        geometry_weight = 1.0 / (1.0 + geometry_weight)
        for (source, target), edge in manifold.edges.items():
            if source not in index_lookup or target not in index_lookup:
                continue
            left = index_lookup[source]
            right = index_lookup[target]
            resonance = 1.0 + float(getattr(edge, "resonance", 0.0))
            latency = 1.0 + max(float(getattr(edge, "latency", 1.0)), 0.0)
            transport[left] += geometry_weight * (edge.strength * resonance / latency) * np.tanh(node_states[right] - node_states[left])
        return transport

    def _local_reaction(self, manifold, node_order: list[int], node_states: np.ndarray) -> np.ndarray:
        reactions = np.zeros_like(node_states, dtype=float)
        for index, node_id in enumerate(node_order):
            field_drive = self._field_drive(manifold, node_id, node_states.shape[1])
            reactions[index] = np.tanh(node_states[index] + self.config.field_gain * field_drive) - 0.18 * np.power(node_states[index], 3)
        return reactions

    def _stability_variables(self, manifold, node_states: np.ndarray) -> dict[str, float]:
        mean_confidence = float(np.mean([getattr(node, "confidence", 0.0) for node in manifold.nodes.values()])) if manifold.nodes else 0.0
        mean_energy = float(np.mean([getattr(node, "energy", 0.0) for node in manifold.nodes.values()])) if manifold.nodes else 0.0
        mean_sync = float(np.mean([getattr(node, "synchronization_state", 0.0) for node in manifold.nodes.values()])) if manifold.nodes else 0.0
        collapse_energy = float(getattr(manifold, "collapse_energy", 0.0))
        instability = float(np.max(np.linalg.norm(node_states, axis=1))) if node_states.size else 0.0
        return {
            "mean_confidence": mean_confidence,
            "mean_energy": mean_energy,
            "mean_sync": mean_sync,
            "collapse_energy": collapse_energy,
            "instability": instability,
        }

    def _forecast_state(self, node_states: np.ndarray, memory_matrix: np.ndarray, regional_matrix: np.ndarray, hierarchy_matrix: np.ndarray, external_matrix: np.ndarray) -> dict[str, Any]:
        predicted = node_states + self.config.forecast_gain * (0.45 * memory_matrix + 0.35 * regional_matrix + 0.20 * hierarchy_matrix)
        residual = external_matrix - predicted if external_matrix.size else -predicted
        return {
            "prediction": predicted,
            "prediction_error": float(np.mean(np.linalg.norm(residual, axis=1))) if residual.size else 0.0,
            "uncertainty": float(np.mean(np.std(predicted, axis=1))) if predicted.size else 0.0,
            "coherence": float(1.0 / (1.0 + np.mean(np.linalg.norm(regional_matrix, axis=1)))) if regional_matrix.size else 1.0,
        }

    def _resource_allocation(self, manifold) -> dict[str, float]:
        current_dim = float(getattr(getattr(manifold, "representation_controller", None), "current_dim", getattr(manifold, "state_dim", 0)))
        max_dim = float(getattr(getattr(manifold, "representation_controller", None), "max_dim", max(current_dim, 1.0)))
        return {
            "representation_dim": current_dim,
            "node_count": float(len(manifold.nodes)),
            "region_count": float(len(manifold.regions)),
            "budget_pressure": float(current_dim / max(max_dim, 1.0)),
        }

    def _contribution_status(self, value: np.ndarray) -> str:
        norm = float(np.linalg.norm(value))
        if norm <= self.config.negligible_threshold:
            return "negligible"
        if norm >= self.config.dominant_threshold:
            return "dominant"
        return "active"

    def step(self, manifold, external_input: dict | None = None, recall_enabled: bool = True, input_enabled: bool = True, topology_enabled: bool = True, decay_enabled: bool = True) -> dict[str, Any]:
        node_order = self._node_order(manifold)
        node_states = self._node_matrix(manifold, node_order)
        width = node_states.shape[1] if node_states.size else max(1, getattr(manifold, "state_dim", 1))

        external_matrix = self._external_forcing(manifold, node_order, width, external_input) if input_enabled else np.zeros((len(node_order), width), dtype=float)
        memory_influence, memory_matrix = self._memory_influence(manifold, node_order, node_states, recall_enabled)
        regional_fields, regional_matrix = self._regional_targets(manifold, node_order, node_states)
        hierarchical_state, hierarchy_matrix = self._hierarchy_feedback(manifold, node_order, node_states)
        diffusion = self._diffusion(manifold, node_order, node_states)
        topology_transport = self._topology_transport(manifold, node_order, node_states, topology_enabled)
        local_reaction = self._local_reaction(manifold, node_order, node_states)
        damping = -node_states if decay_enabled else np.zeros_like(node_states)
        forecast_state = self._forecast_state(node_states, memory_matrix, regional_matrix, hierarchy_matrix, external_matrix)
        residual = np.tanh(np.asarray(forecast_state["prediction"], dtype=float) - node_states)

        total_update = self.config.dt * (
            (self.config.external_gain * external_matrix)
            + (self.config.local_gain * local_reaction)
            + (self.config.diffusion_gain * diffusion)
            + (self.config.recall_gain * memory_matrix)
            + (self.config.regional_gain * regional_matrix)
            + (self.config.hierarchy_gain * hierarchy_matrix)
            + (self.config.topology_gain * topology_transport)
            + (self.config.damping_gain * damping)
            + (self.config.residual_gain * residual)
        )
        next_states = np.clip(node_states + total_update, self.config.clip_bounds[0], self.config.clip_bounds[1])

        norms = np.linalg.norm(next_states, axis=1, keepdims=True) if next_states.size else np.zeros((0, 1), dtype=float)
        safe_scale = np.maximum(norms / max(self.config.activation_capacity, 1e-12), 1.0)
        next_states = next_states / safe_scale

        geometry_state = np.asarray(
            [
                float(getattr(manifold, "last_geometry_feedback", {}).get("laplacian_energy", 0.0)),
                float(len(getattr(manifold, "last_geometry_feedback", {}).get("surgery_attempted", []))),
                float(len(getattr(manifold, "last_geometry_feedback", {}).get("surgery_applied", []))),
            ],
            dtype=float,
        )
        topology_state = np.asarray(
            [
                float(len(manifold.nodes)),
                float(len(manifold.edges)),
                float(np.mean([edge.strength for edge in manifold.edges.values()])) if manifold.edges else 0.0,
                float(np.mean([edge.resonance for edge in manifold.edges.values()])) if manifold.edges else 0.0,
            ],
            dtype=float,
        )
        governed_state = GovernedState(
            node_order=node_order,
            node_fields={node_id: next_states[index].copy() for index, node_id in enumerate(node_order)},
            regional_fields=regional_fields,
            geometry_state=geometry_state,
            topology_state=topology_state,
            memory_influence=memory_influence,
            hierarchical_state=hierarchical_state,
            resource_allocation=self._resource_allocation(manifold),
            stability_variables=self._stability_variables(manifold, next_states),
            forecast_state=forecast_state,
        )
        state_vector, block_names = governed_state.flatten()

        contribution_terms = [
            ("external", self.config.external_gain * external_matrix),
            ("local_reaction", self.config.local_gain * local_reaction),
            ("diffusion", self.config.diffusion_gain * diffusion),
            ("memory", self.config.recall_gain * memory_matrix),
            ("regional_coupling", self.config.regional_gain * regional_matrix),
            ("hierarchical_feedback", self.config.hierarchy_gain * hierarchy_matrix),
            ("topology_transport", self.config.topology_gain * topology_transport),
            ("damping", self.config.damping_gain * damping),
            ("residual", self.config.residual_gain * residual),
        ]
        self.last_contributions = [
            {
                "term": term,
                "value": value,
                "norm": float(np.linalg.norm(value)),
                "status": self._contribution_status(value),
            }
            for term, value in contribution_terms
        ]
        self.last_state_vector = state_vector
        self.last_governed_state = governed_state

        return {
            "state_vector": state_vector,
            "node_states": [governed_state.node_fields[node_id].copy() for node_id in node_order],
            "node_order": node_order,
            "bounded": bool(np.all(np.isfinite(state_vector))),
            "block_names": block_names,
            "contributions": self.last_contributions,
            "governed_state": governed_state,
            "recall_applied": bool(recall_enabled and any(np.linalg.norm(vector) > 0.0 for vector in memory_influence.values())),
            "forecast_state": {
                "prediction_error": governed_state.forecast_state["prediction_error"],
                "uncertainty": governed_state.forecast_state["uncertainty"],
                "coherence": governed_state.forecast_state["coherence"],
            },
        }
