from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from rcm.reproducibility import DEFAULT_SEED


class NoiseType(str, Enum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    SALT_AND_PEPPER = "salt_and_pepper"
    DROPOUT = "dropout"
    ADVERSARIAL = "adversarial"
    QUANTIZATION = "quantization"
    FIELD_PERTURBATION = "field_perturbation"


@dataclass
class FaultRecord:
    kind: str
    target: str
    details: dict = field(default_factory=dict)
    original_data: dict = field(default_factory=dict)


class RCMPerturbationLab:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._delay_queues: dict[int, list[dict]] = {}

    def perturb_node_states(self, manifold, node_ids, noise_type: NoiseType, intensity: float = 0.25):
        original_states = {node_id: manifold.nodes[node_id].local_state.copy() for node_id in node_ids if node_id in manifold.nodes}
        for node_id in list(original_states.keys()):
            node = manifold.nodes[node_id]
            node.local_state = self._apply_noise(node.local_state, noise_type, intensity)
        return FaultRecord(
            kind="perturb_node_states",
            target="nodes",
            details={"node_ids": list(original_states.keys()), "noise_type": noise_type.value, "intensity": intensity},
            original_data={"states": original_states},
        )

    def drop_nodes(self, manifold, node_ids):
        removed_nodes = {node_id: deepcopy(manifold.nodes[node_id]) for node_id in node_ids if node_id in manifold.nodes}
        removed_edges = {
            edge_key: deepcopy(edge)
            for edge_key, edge in manifold.edges.items()
            if edge_key[0] in removed_nodes or edge_key[1] in removed_nodes
        }
        removed_regions = {
            signature: deepcopy(region)
            for signature, region in manifold.regions.items()
            if any(node_id in removed_nodes for node_id in signature)
        }

        for edge_key in list(removed_edges.keys()):
            manifold.edges.pop(edge_key, None)
        for node_id in removed_nodes:
            manifold.nodes.pop(node_id, None)

        manifold._update_regions()
        if hasattr(manifold, "_apply_regional_fields"):
            manifold._apply_regional_fields()

        return FaultRecord(
            kind="drop_nodes",
            target="nodes",
            details={"node_ids": list(removed_nodes.keys())},
            original_data={
                "nodes": removed_nodes,
                "edges": removed_edges,
                "regions": removed_regions,
            },
        )

    def sever_edges(self, manifold, edges, bidirectional: bool = True):
        edge_keys = set()
        for source, target in edges:
            edge_keys.add((source, target))
            if bidirectional:
                edge_keys.add((target, source))
        removed_edges = {edge_key: deepcopy(manifold.edges[edge_key]) for edge_key in edge_keys if edge_key in manifold.edges}
        for edge_key in removed_edges:
            manifold.edges.pop(edge_key, None)
        manifold._update_regions()
        return FaultRecord(
            kind="sever_edges",
            target="edges",
            details={"edges": sorted(removed_edges.keys()), "bidirectional": bidirectional},
            original_data={"edges": removed_edges},
        )

    def delay_signal(self, manifold, node_id: int, signal, delay_steps: int):
        queue = self._delay_queues.setdefault(id(manifold), [])
        queue.append(
            {
                "node_id": node_id,
                "signal": np.asarray(signal, dtype=float).copy(),
                "delay_steps": int(delay_steps),
            }
        )
        return FaultRecord(
            kind="delay_signal",
            target="signal_queue",
            details={"node_id": node_id, "delay_steps": int(delay_steps)},
        )

    def step_with_delays(self, manifold, external_input=None):
        delivered = {}
        queue = self._delay_queues.setdefault(id(manifold), [])
        retained = []
        for item in queue:
            item["delay_steps"] -= 1
            if item["delay_steps"] <= 0:
                delivered[item["node_id"]] = item["signal"]
            else:
                retained.append(item)
        self._delay_queues[id(manifold)] = retained

        merged_input = {}
        if external_input:
            merged_input.update(external_input)
        merged_input.update(delivered)
        if not merged_input:
            merged_input = None
        return manifold.step(merged_input)

    def perturb_field(self, field, attribute: str = "field", intensity: float = 0.3):
        actual_attribute = attribute if hasattr(field, attribute) else "field_state"
        if not hasattr(field, actual_attribute):
            raise AttributeError(f"Field has neither '{attribute}' nor 'field_state'")
        original_value = np.asarray(getattr(field, actual_attribute), dtype=float).copy()
        perturbed = self._apply_noise(original_value, NoiseType.FIELD_PERTURBATION, intensity)
        setattr(field, actual_attribute, perturbed)
        return FaultRecord(
            kind="perturb_field",
            target=actual_attribute,
            details={"attribute": actual_attribute, "intensity": intensity},
            original_data={"value": original_value, "field": field},
        )

    def corrupt_episodes(self, manifold, fraction: float = 0.30, intensity: float = 0.25):
        event_count = len(manifold.episodic_memory.events)
        if event_count == 0:
            return FaultRecord(kind="corrupt_episodes", target="episodic_memory", details={"corrupted_indices": []})

        count = max(1, int(np.ceil(event_count * fraction)))
        indices = sorted(self.rng.choice(event_count, size=min(count, event_count), replace=False).tolist())
        original_events = {}
        for index in indices:
            episode = manifold.episodic_memory.events[index]
            original_events[index] = deepcopy(episode)
            for node_id, state in episode["activations"].items():
                episode["activations"][node_id] = self._apply_noise(state, NoiseType.GAUSSIAN, intensity)
            for key in ("energy", "confidence"):
                if key in episode and episode[key] is not None:
                    for node_id, value in episode[key].items():
                        episode[key][node_id] = float(value + self.rng.normal(0.0, intensity * 0.1))
            if episode.get("next_activations"):
                for node_id, state in episode["next_activations"].items():
                    episode["next_activations"][node_id] = self._apply_noise(state, NoiseType.UNIFORM, intensity)

        transition_count = len(manifold.episodic_memory.transitions)
        transition_indices = []
        original_transitions = {}
        if transition_count:
            transition_pick = max(1, int(np.ceil(transition_count * fraction)))
            transition_indices = sorted(self.rng.choice(transition_count, size=min(transition_pick, transition_count), replace=False).tolist())
            for index in transition_indices:
                transition = manifold.episodic_memory.transitions[index]
                original_transitions[index] = deepcopy(transition)
                transition["current_state"] = self._apply_noise(transition["current_state"], NoiseType.GAUSSIAN, intensity)
                transition["next_state"] = self._apply_noise(transition["next_state"], NoiseType.ADVERSARIAL, intensity)
                transition["outcome"]["energy"] = float(transition["outcome"]["energy"] + self.rng.normal(0.0, intensity * 0.1))
                transition["outcome"]["confidence"] = float(np.clip(transition["outcome"]["confidence"] + self.rng.normal(0.0, intensity * 0.1), 0.0, 1.0))

        return FaultRecord(
            kind="corrupt_episodes",
            target="episodic_memory",
            details={"corrupted_indices": indices, "transition_indices": transition_indices, "intensity": intensity},
            original_data={"events": original_events, "transitions": original_transitions},
        )

    def lesion_region(self, manifold, region_signature, loss_fraction: float = 0.50):
        normalized = tuple(sorted(int(node_id) for node_id in region_signature))
        region = None
        for signature, current_region in manifold.regions.items():
            if tuple(sorted(int(node_id) for node_id in signature)) == normalized:
                region = current_region
                break
        if region is None:
            raise KeyError(f"Unknown region signature: {region_signature}")

        node_ids = list(region.node_ids)
        count = max(1, int(np.ceil(len(node_ids) * loss_fraction)))
        selected = sorted(self.rng.choice(node_ids, size=min(count, len(node_ids)), replace=False).tolist())
        return self.drop_nodes(manifold, selected)

    def restore(self, manifold, fault: FaultRecord):
        if fault.kind == "perturb_node_states":
            for node_id, state in fault.original_data.get("states", {}).items():
                if node_id in manifold.nodes:
                    manifold.nodes[node_id].local_state = state.copy()
            return

        if fault.kind == "drop_nodes":
            manifold.nodes.update(deepcopy(fault.original_data.get("nodes", {})))
            manifold.edges.update(deepcopy(fault.original_data.get("edges", {})))
            manifold.regions.update(deepcopy(fault.original_data.get("regions", {})))
            manifold._update_regions()
            return

        if fault.kind == "sever_edges":
            manifold.edges.update(deepcopy(fault.original_data.get("edges", {})))
            manifold._update_regions()
            return

        if fault.kind == "perturb_field":
            field = fault.original_data["field"]
            setattr(field, fault.details["attribute"], fault.original_data["value"].copy())
            return

        if fault.kind == "corrupt_episodes":
            for index, episode in fault.original_data.get("events", {}).items():
                manifold.episodic_memory.events[index] = deepcopy(episode)
            for index, transition in fault.original_data.get("transitions", {}).items():
                manifold.episodic_memory.transitions[index] = deepcopy(transition)
            return

    def _apply_noise(self, state, noise_type: NoiseType, intensity: float):
        state = np.asarray(state, dtype=float)
        if noise_type == NoiseType.GAUSSIAN:
            return state + self.rng.normal(0.0, intensity, size=state.shape)
        if noise_type == NoiseType.UNIFORM:
            return state + self.rng.uniform(-intensity, intensity, size=state.shape)
        if noise_type == NoiseType.SALT_AND_PEPPER:
            perturbed = state.copy()
            mask = self.rng.random(state.shape) < min(max(intensity, 0.0), 1.0)
            values = self.rng.choice([-1.0, 1.0], size=state.shape)
            perturbed[mask] = values[mask] * max(np.max(np.abs(state)), 1.0)
            return perturbed
        if noise_type == NoiseType.DROPOUT:
            keep = self.rng.random(state.shape) >= min(max(intensity, 0.0), 1.0)
            return state * keep
        if noise_type == NoiseType.ADVERSARIAL:
            direction = np.sign(state)
            direction[direction == 0.0] = 1.0
            return state + intensity * direction
        if noise_type == NoiseType.QUANTIZATION:
            steps = max(2, int(round(1.0 / max(intensity, 1e-3))))
            min_value = float(np.min(state))
            max_value = float(np.max(state))
            if max_value - min_value < 1e-9:
                return state.copy()
            scaled = (state - min_value) / (max_value - min_value)
            quantized = np.round(scaled * steps) / steps
            return quantized * (max_value - min_value) + min_value
        if noise_type == NoiseType.FIELD_PERTURBATION:
            t = np.linspace(0.0, 1.0, state.size, endpoint=False)
            perturbation = np.zeros_like(t)
            for frequency in (1.0, 2.5, 5.0):
                phase = self.rng.uniform(0.0, 2.0 * np.pi)
                perturbation += np.sin(2.0 * np.pi * frequency * t + phase)
            perturbation /= 3.0
            return state + intensity * perturbation.reshape(state.shape)
        raise ValueError(f"Unsupported noise type: {noise_type}")


def capture_state(manifold):
    return {
        "node_states": {node_id: node.local_state.copy() for node_id, node in manifold.nodes.items()},
        "node_confidence": {node_id: node.confidence for node_id, node in manifold.nodes.items()},
        "edge_strengths": {edge_key: edge.strength for edge_key, edge in manifold.edges.items()},
        "region_signatures": [tuple(sorted(int(node_id) for node_id in signature)) for signature in manifold.regions.keys()],
        "node_count": len(manifold.nodes),
        "edge_count": len(manifold.edges),
        "region_count": len(manifold.regions),
    }


def compare_to_baseline(manifold, baseline):
    current = capture_state(manifold)
    baseline_node_ids = set(baseline["node_states"].keys())
    current_node_ids = set(current["node_states"].keys())
    shared_nodes = sorted(baseline_node_ids & current_node_ids)
    if shared_nodes:
        mean_state_error = float(
            np.mean([
                np.linalg.norm(current["node_states"][node_id] - baseline["node_states"][node_id])
                for node_id in shared_nodes
            ])
        )
        mean_confidence_error = float(
            np.mean([
                abs(current["node_confidence"][node_id] - baseline["node_confidence"][node_id])
                for node_id in shared_nodes
            ])
        )
    else:
        mean_state_error = float("inf")
        mean_confidence_error = float("inf")

    baseline_edges = set(baseline["edge_strengths"].keys())
    current_edges = set(current["edge_strengths"].keys())
    shared_edges = sorted(baseline_edges & current_edges)
    if shared_edges:
        mean_edge_strength_error = float(
            np.mean([
                abs(current["edge_strengths"][edge_key] - baseline["edge_strengths"][edge_key])
                for edge_key in shared_edges
            ])
        )
    else:
        mean_edge_strength_error = float("inf") if baseline_edges else 0.0

    node_survival = len(shared_nodes) / max(len(baseline_node_ids), 1)
    edge_survival = len(shared_edges) / max(len(baseline_edges), 1)
    region_count_ratio = current["region_count"] / max(baseline["region_count"], 1)

    return {
        "mean_state_error": mean_state_error,
        "mean_confidence_error": mean_confidence_error,
        "mean_edge_strength_error": mean_edge_strength_error,
        "node_survival": node_survival,
        "edge_survival": edge_survival,
        "region_count_ratio": region_count_ratio,
    }


def recovery_score(damaged_metrics, recovered_metrics):
    damaged_state_error = damaged_metrics["mean_state_error"]
    recovered_state_error = recovered_metrics["mean_state_error"]
    if not np.isfinite(damaged_state_error) or damaged_state_error <= 0.0:
        error_reduction = 1.0 if recovered_state_error == 0.0 else 0.0
    else:
        error_reduction = np.clip((damaged_state_error - recovered_state_error) / damaged_state_error, 0.0, 1.0)

    node_recovery = np.clip(recovered_metrics["node_survival"] - damaged_metrics["node_survival"], 0.0, 1.0)
    edge_recovery = np.clip(recovered_metrics["edge_survival"] - damaged_metrics["edge_survival"], 0.0, 1.0)
    return float((0.5 * error_reduction) + (0.25 * node_recovery) + (0.25 * edge_recovery))


lab = RCMPerturbationLab(seed=DEFAULT_SEED)