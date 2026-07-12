from rcm.hierarchy.abstraction import abstract_regions


def _signature_to_meta_id(child_manifold):
    mapping = {}
    for meta_id, signature in getattr(child_manifold, "parent_region_map", {}).items():
        mapping[tuple(sorted(int(node_id) for node_id in signature))] = int(meta_id)
    return mapping


def _merge_child_state(previous_child, next_child):
    previous_index = _signature_to_meta_id(previous_child)
    next_index = _signature_to_meta_id(next_child)

    for signature, next_meta_id in next_index.items():
        previous_meta_id = previous_index.get(signature)
        if previous_meta_id is None:
            continue
        if previous_meta_id not in previous_child.nodes or next_meta_id not in next_child.nodes:
            continue

        prev_node = previous_child.nodes[previous_meta_id]
        next_node = next_child.nodes[next_meta_id]
        width = min(len(prev_node.local_state), len(next_node.local_state))
        if width:
            next_node.local_state[:width] = 0.65 * prev_node.local_state[:width] + 0.35 * next_node.local_state[:width]
        next_node.energy = 0.65 * prev_node.energy + 0.35 * next_node.energy
        next_node.confidence = 0.65 * prev_node.confidence + 0.35 * next_node.confidence

    for left_signature, left_meta_id in next_index.items():
        for right_signature, right_meta_id in next_index.items():
            if left_meta_id == right_meta_id:
                continue
            prev_left = previous_index.get(left_signature)
            prev_right = previous_index.get(right_signature)
            if prev_left is None or prev_right is None:
                continue
            previous_edge = previous_child.edges.get((prev_left, prev_right))
            next_edge = next_child.edges.get((left_meta_id, right_meta_id))
            if previous_edge is None or next_edge is None:
                continue
            next_edge.strength = 0.7 * previous_edge.strength + 0.3 * next_edge.strength
            next_edge.latency = 0.7 * previous_edge.latency + 0.3 * next_edge.latency
            next_edge.resonance = 0.7 * previous_edge.resonance + 0.3 * next_edge.resonance
            next_edge.age = max(previous_edge.age, next_edge.age)
            next_edge.traversal_frequency = max(previous_edge.traversal_frequency, next_edge.traversal_frequency)

    next_child.time_step = previous_child.time_step


def refresh_child_manifolds(manifold):
    stable_regions = [
        signature
        for signature, region in manifold.regions.items()
        if len(signature) > 1 and region.stability > 0.85
    ]
    stable_regions = sorted(stable_regions)
    if stable_regions:
        abstracted = abstract_regions(manifold, stable_regions)
        if manifold.child_manifolds:
            _merge_child_state(manifold.child_manifolds[0], abstracted)
            manifold.child_manifolds[0] = abstracted
        else:
            manifold.child_manifolds.append(abstracted)
    else:
        manifold.child_manifolds.clear()