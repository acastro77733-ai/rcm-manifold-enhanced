from collections import defaultdict, deque


def get_nonmanifold_edges(complex_):
    counts = defaultdict(int)
    for u, v in complex_.half_edges.keys():
        counts[tuple(sorted((u, v)))] += 1
    return [edge for edge, count in counts.items() if count > 2]


def get_nonmanifold_vertices(complex_):
    nonmanifold_vertices = []
    for vertex_index in range(complex_.n_vertices):
        outgoing = [half_edge for key, half_edge in complex_.half_edges.items() if key[0] == vertex_index]
        if not outgoing:
            continue
        rings = 0
        visited = set()
        for start_half_edge in outgoing:
            if start_half_edge in visited:
                continue
            rings += 1
            current = start_half_edge
            while current and current not in visited:
                visited.add(current)
                current = current.twin.next if (current.twin and current.twin.next) else None
                if current == start_half_edge:
                    break
        if rings > 1:
            nonmanifold_vertices.append(vertex_index)
    return nonmanifold_vertices


def is_orientable(complex_):
    face_orientations = {}
    remaining_half_edges = set(complex_.half_edges.values())
    while remaining_half_edges:
        start = remaining_half_edges.pop()
        if start.face is None:
            continue
        queue = deque([(start, True)])
        while queue:
            current, orientation = queue.popleft()
            if current.face is not None:
                if current.face in face_orientations:
                    if face_orientations[current.face] != orientation:
                        return False
                else:
                    face_orientations[current.face] = orientation
            for nxt in [current.next, current.next.next]:
                if nxt in remaining_half_edges:
                    remaining_half_edges.discard(nxt)
                    queue.append((nxt, orientation))
            if current.twin and current.twin in remaining_half_edges:
                remaining_half_edges.discard(current.twin)
                queue.append((current.twin, not orientation))
    return True


def compute_topology_invariants(complex_):
    vertex_count = complex_.n_vertices
    edge_count = len({tuple(sorted(edge)) for edge in complex_.half_edges.keys()})
    face_count = len(complex_.get_faces())
    chi = vertex_count - edge_count + face_count

    visited = set()
    components = 0
    for vertex_index in range(vertex_count):
        if vertex_index in visited:
            continue
        components += 1
        queue = deque([vertex_index])
        visited.add(vertex_index)
        while queue:
            current = queue.popleft()
            neighbors = {key[1] for key in complex_.half_edges.keys() if key[0] == current}
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return {
        "chi": chi,
        "V": vertex_count,
        "E": edge_count,
        "F": face_count,
        "num_nonmanifold_vertices": len(get_nonmanifold_vertices(complex_)),
        "num_nonmanifold_edges": len(get_nonmanifold_edges(complex_)),
        "is_orientable": is_orientable(complex_),
        "num_components": components,
    }