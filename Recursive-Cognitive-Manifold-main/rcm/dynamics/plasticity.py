from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EdgeUtility:
    coactivation: float = 0.0
    task_contribution: float = 0.0
    recall_contribution: float = 0.0
    stability_contribution: float = 0.0
    noise_contribution: float = 0.0


@dataclass
class TopologyBudget:
    max_degree: int = 4
    max_edge_count: int = 12
    repair_allowance: int = 2
    pruning_threshold: float = 0.05
    structural_cooldown: int = 3


@dataclass
class TopologyAdaptationState:
    last_change_step: int = 0
    utility: dict[tuple[int, int], EdgeUtility] = field(default_factory=dict)


def state_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def update_edge_dynamics(manifold, edge_usage, topology_budget: TopologyBudget | None = None, adaptation_state: TopologyAdaptationState | None = None, step: int = 0):
    if topology_budget is None:
        topology_budget = TopologyBudget()
    if adaptation_state is None:
        adaptation_state = TopologyAdaptationState()

    pruned_edges = []
    sampled_edge_keys = {edge_key for edge_key in manifold.edges.keys()}
    if len(sampled_edge_keys) > 8:
        sampled_edge_keys = set(list(sampled_edge_keys)[:: max(1, len(sampled_edge_keys) // 8)])

    for (source, target), edge in list(manifold.edges.items()):
        source_node = manifold.nodes[source]
        target_node = manifold.nodes[target]
        similarity = state_similarity(source_node.local_state, target_node.local_state)
        edge.resonance = 0.7 * edge.resonance + 0.3 * similarity

        utility = adaptation_state.utility.get((source, target))
        if utility is None:
            utility = EdgeUtility(coactivation=similarity, task_contribution=0.0, recall_contribution=0.0, stability_contribution=0.0, noise_contribution=0.0)
            adaptation_state.utility[(source, target)] = utility

        usage = float(edge_usage.get((source, target), 0.0))
        utility.coactivation = 0.7 * utility.coactivation + 0.3 * similarity
        utility.task_contribution = 0.8 * utility.task_contribution + 0.2 * usage
        utility.recall_contribution = 0.8 * utility.recall_contribution + 0.2 * max(0.0, similarity - 0.2)
        utility.stability_contribution = 0.8 * utility.stability_contribution + 0.2 * min(1.0, edge.strength)
        utility.noise_contribution = 0.9 * utility.noise_contribution + 0.1 * max(0.0, 0.5 - similarity)

        if (source, target) in sampled_edge_keys or (target, source) in sampled_edge_keys:
            delta_w = 0.04 * similarity + 0.03 * utility.task_contribution + 0.02 * utility.recall_contribution + 0.02 * utility.stability_contribution - 0.015 * utility.noise_contribution - 0.01 * edge.strength
            edge.strength = float(np.clip(edge.strength + delta_w, 0.0, 1.0))
        else:
            edge.strength = float(np.clip(edge.strength - 0.005, 0.0, 1.0))

        if edge.strength < topology_budget.pruning_threshold and edge.age > 3:
            pruned_edges.append((source, target))

    degree_counts = {}
    for source, target in manifold.edges:
        degree_counts[source] = degree_counts.get(source, 0) + 1
        degree_counts[target] = degree_counts.get(target, 0) + 1

    if step - adaptation_state.last_change_step >= topology_budget.structural_cooldown:
        for edge_key in list(manifold.edges):
            if len(manifold.edges) <= topology_budget.max_edge_count:
                break
            if degree_counts.get(edge_key[0], 0) > topology_budget.max_degree or degree_counts.get(edge_key[1], 0) > topology_budget.max_degree:
                pruned_edges.append(edge_key)
                break

    for edge_key in pruned_edges:
        if edge_key in manifold.edges:
            del manifold.edges[edge_key]
            manifold._record_topology_event(edge_key[0], f"prune:{edge_key[0]}->{edge_key[1]}")

    if not pruned_edges and step >= 0 and len(manifold.edges) >= topology_budget.max_edge_count:
        for edge_key in list(manifold.edges.keys())[:1]:
            del manifold.edges[edge_key]
            pruned_edges.append(edge_key)

    return pruned_edges


def apply_edge_masking(manifold, edge_key, edge_usage, topology_budget: TopologyBudget | None = None, adaptation_state: TopologyAdaptationState | None = None, step: int = 0):
    if topology_budget is None:
        topology_budget = TopologyBudget()
    if adaptation_state is None:
        adaptation_state = TopologyAdaptationState()

    source, target = edge_key
    edge = manifold.edges.get(edge_key)
    if edge is None:
        return {"utility_delta": 0.0, "masked": False}

    baseline = edge.strength
    edge.strength = max(0.0, baseline - 0.1)
    source_node = manifold.nodes[source]
    target_node = manifold.nodes[target]
    similarity = state_similarity(source_node.local_state, target_node.local_state)
    usage = float(edge_usage.get(edge_key, 0.0))
    objective_change = 0.05 * similarity + 0.03 * usage - 0.02 * edge.strength
    edge.strength = baseline

    utility = adaptation_state.utility.get(edge_key)
    if utility is None:
        utility = EdgeUtility()
        adaptation_state.utility[edge_key] = utility
    utility.task_contribution = 0.85 * utility.task_contribution + 0.15 * max(0.0, objective_change)
    utility.recall_contribution = 0.85 * utility.recall_contribution + 0.15 * max(0.0, similarity)
    utility.stability_contribution = 0.85 * utility.stability_contribution + 0.15 * min(1.0, baseline)
    utility.noise_contribution = 0.85 * utility.noise_contribution + 0.15 * max(0.0, 0.5 - similarity)
    return {"utility_delta": float(objective_change), "masked": True}


def request_simplicial_surgery(manifold, edge_usage, max_requests: int = 2):
    complex_ = getattr(manifold, "geometry_complex", None)
    if complex_ is None or not getattr(complex_, "half_edges", None):
        return []

    candidates = []
    seen = set()
    for (source, target), edge in manifold.edges.items():
        undirected = tuple(sorted((source, target)))
        if undirected in seen:
            continue
        seen.add(undirected)

        if (source, target) not in complex_.half_edges or (target, source) not in complex_.half_edges:
            continue
        half_edge = complex_.half_edges[(source, target)]
        if half_edge.twin is None:
            continue

        left = manifold.nodes.get(source)
        right = manifold.nodes.get(target)
        if left is None or right is None:
            continue
        similarity = state_similarity(left.local_state, right.local_state)
        usage = float(edge_usage.get((source, target), 0.0) + edge_usage.get((target, source), 0.0))
        dissimilarity = max(0.0, 1.0 - similarity)
        pressure = (0.55 * dissimilarity) + (0.30 * min(edge.strength, 1.0)) + (0.15 * np.tanh(usage))
        if pressure < 0.45:
            continue
        candidates.append(
            {
                "edge": (source, target),
                "priority": float(pressure),
                "similarity": float(similarity),
                "usage": float(usage),
            }
        )

    candidates.sort(key=lambda item: item["priority"], reverse=True)
    return candidates[: max(1, int(max_requests))]