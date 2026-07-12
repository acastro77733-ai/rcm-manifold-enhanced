from itertools import combinations

import numpy as np


def infer_specialization(manifold, component):
    if not component:
        return "undifferentiated"
    states = np.vstack([manifold.nodes[node_id].local_state for node_id in component])
    dominant_axis = int(np.argmax(np.mean(np.abs(states), axis=0)))
    labels = ["sensorimotor", "associative", "predictive", "integrative"]
    return labels[dominant_axis % len(labels)]


def region_activation(manifold, component):
    states = np.vstack([manifold.nodes[node_id].local_state for node_id in component])
    weights = np.asarray(
        [max(0.05, manifold.nodes[node_id].confidence) * max(0.1, manifold.nodes[node_id].energy) for node_id in component],
        dtype=float,
    )
    weights = weights / max(float(np.sum(weights)), 1e-12)
    return np.sum(states * weights[:, None], axis=0)


def cross_region_coupling(manifold, left, right):
    weights = []
    for source in left:
        for target in right:
            edge = manifold.edges.get((source, target))
            if edge is not None:
                weights.append(edge.strength * (1.0 + edge.resonance))
    if not weights:
        return 0.0
    return float(np.mean(weights))


def abstract_regions(manifold, stable_regions):
    from rcm.cognition.manifold import RecursiveCognitiveManifold

    meta = RecursiveCognitiveManifold(
        state_dim=manifold.state_dim,
        level=manifold.level + 1,
        label=f"meta-{manifold.label}",
        hrm_seed=2000 + manifold.level + 1,
    )
    region_to_meta = {}
    meta.parent_region_map = {}
    for meta_id, signature in enumerate(stable_regions):
        region = manifold.regions[signature]
        state = region_activation(manifold, list(signature))
        meta.add_node(meta_id, state)
        meta.nodes[meta_id].specialization = region.specialization
        meta.nodes[meta_id].confidence = region.stability
        meta.nodes[meta_id].energy = float(np.mean([manifold.nodes[node_id].energy for node_id in signature])) if signature else 0.0
        meta.nodes[meta_id].synchronization_state = float(region.stability)
        region_to_meta[signature] = meta_id
        meta.parent_region_map[meta_id] = signature

    for left, right in combinations(stable_regions, 2):
        coupling = cross_region_coupling(manifold, left, right)
        if coupling > 0.2:
            meta.connect(region_to_meta[left], region_to_meta[right], strength=coupling, latency=max(0.5, 1.5 - coupling))
    return meta