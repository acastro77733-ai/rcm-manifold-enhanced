from collections import defaultdict

import numpy as np


def stimulate_node(manifold, node_id: int, signal):
    if node_id not in manifold.nodes:
        raise KeyError(f"Unknown node_id: {node_id}")
    signal = np.asarray(signal, dtype=float)
    node = manifold.nodes[node_id]
    node.local_state = manifold._fit_state(signal)
    node.energy = min(1.5, node.energy + 0.2)
    node.confidence = min(1.0, node.confidence + 0.05)
    node.short_term_memory.append(node.local_state.copy())


def propagate_states(manifold):
    updated_states = {}
    edge_usage = defaultdict(int)
    for node_id in manifold.nodes:
        incoming = []
        for (source, target), edge in manifold.edges.items():
            if target != node_id:
                continue
            source_state = manifold.nodes[source].local_state
            signal = source_state * edge.strength * (1.0 + edge.resonance)
            incoming.append(signal / edge.latency)
            edge.traversal_frequency += 1
            edge.age += 1
            edge_usage[(source, target)] += 1
        if incoming:
            aggregate = np.mean(incoming, axis=0)
            updated_states[node_id] = 0.6 * manifold.nodes[node_id].local_state + 0.4 * aggregate
        else:
            updated_states[node_id] = manifold.nodes[node_id].local_state.copy()
    return updated_states, edge_usage