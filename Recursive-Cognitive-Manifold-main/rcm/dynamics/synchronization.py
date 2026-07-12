import numpy as np


def coherence(memory) -> float:
    if len(memory) < 2:
        return 0.5
    stack = np.vstack(memory)
    variance = float(np.mean(np.var(stack, axis=0)))
    return 1.0 / (1.0 + variance)


def apply_synchronization(manifold, updated_states):
    for node_id, new_state in updated_states.items():
        node = manifold.nodes[node_id]
        node.local_state = new_state
        node.short_term_memory.append(new_state.copy())
        node.energy = float(np.clip(node.energy * 0.97 + np.linalg.norm(new_state) * 0.03, 0.0, 2.0))
        node.confidence = float(np.clip(coherence(node.short_term_memory), 0.0, 1.0))
        node.synchronization_state = node.confidence