def recall_recent_episode(manifold, steps_back: int = 1):
    if not manifold.episodic_memory.events:
        return None
    index = max(len(manifold.episodic_memory.events) - steps_back, 0)
    return manifold.episodic_memory.events[index]


def combine_state_with_prediction(node_state, prediction, similarity: float, confidence: float):
    blend = max(0.0, min(0.5, 0.1 + 0.4 * max(similarity, 0.0) * max(confidence, 0.0)))
    return (1.0 - blend) * node_state + blend * prediction


def recall_node_history(manifold, node_id: int, limit: int = 5):
    node = manifold.nodes[node_id]
    return list(node.long_term_memory)[-limit:]


def recall_semantic_motifs(manifold, min_count: int = 1):
    return {
        motif: prototype
        for motif, prototype in manifold.semantic_memory.snapshot().items()
        if prototype["usage_count"] >= min_count
    }