import numpy as np


def state_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def update_edge_dynamics(manifold, edge_usage):
    pruned_edges = []
    for (source, target), edge in list(manifold.edges.items()):
        source_node = manifold.nodes[source]
        target_node = manifold.nodes[target]
        similarity = state_similarity(source_node.local_state, target_node.local_state)
        edge.resonance = 0.7 * edge.resonance + 0.3 * similarity
        reinforcement = 0.05 * similarity + 0.01 * edge_usage[(source, target)]
        decay = 0.02 if similarity < 0.15 else 0.0
        edge.strength = float(np.clip(edge.strength + reinforcement - decay, 0.0, 2.0))
        if edge.strength < 0.05 and edge.age > 3:
            pruned_edges.append((source, target))

    for edge_key in pruned_edges:
        del manifold.edges[edge_key]
        manifold._record_topology_event(edge_key[0], f"prune:{edge_key[0]}->{edge_key[1]}")
    return pruned_edges


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