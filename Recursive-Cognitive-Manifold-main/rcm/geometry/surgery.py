from collections import deque


def repair_nonmanifold_vertex(complex_, vertex_index):
    outgoing = [half_edge for key, half_edge in complex_.half_edges.items() if key[0] == vertex_index]
    fans = []
    unvisited = set(outgoing)
    while unvisited:
        start = unvisited.pop()
        fan = []
        queue = deque([start])
        while queue:
            current = queue.popleft()
            fan.append(current)
            nxt = current.twin.next if (current.twin and current.twin.next and current.twin.next.origin == vertex_index) else None
            if nxt in unvisited:
                unvisited.discard(nxt)
                queue.append(nxt)
        fans.append(fan)

    if len(fans) <= 1:
        return False

    original_coordinates = complex_.vertices[vertex_index].copy()
    for fan_index in range(1, len(fans)):
        new_index = complex_.n_vertices
        complex_.vertices = complex_.np.vstack([complex_.vertices, original_coordinates])
        complex_.n_vertices += 1
        for half_edge in fans[fan_index]:
            face_index = half_edge.face
            face = list(complex_._current_faces[face_index])
            for offset in range(3):
                if face[offset] == vertex_index:
                    face[offset] = new_index
            complex_._current_faces[face_index] = tuple(face)
    complex_._build_half_edge_structure([list(face) for face in complex_._current_faces])
    return True


def flip_edge(complex_, u, v):
    if (u, v) not in complex_.half_edges:
        return False
    half_edge = complex_.half_edges[(u, v)]
    if not half_edge.twin:
        return False
    w = half_edge.next.next.origin
    z = half_edge.twin.next.next.origin
    neighbors_u = {key[1] for key in complex_.half_edges.keys() if key[0] == u}
    neighbors_v = {key[1] for key in complex_.half_edges.keys() if key[0] == v}
    if len(neighbors_u & neighbors_v) != 2:
        return False
    if (w, z) in complex_.half_edges or (z, w) in complex_.half_edges:
        return False

    twin = half_edge.twin
    half_edge_next = half_edge.next
    half_edge_prev = half_edge.next.next
    twin_next = twin.next
    twin_prev = twin.next.next
    half_edge.origin = z
    twin.origin = w
    half_edge.next, half_edge_prev.next, twin_next.next = half_edge_prev, twin_next, half_edge
    twin.next, twin_prev.next, half_edge_next.next = twin_prev, half_edge_next, twin
    twin_next.face, half_edge_next.face = half_edge.face, twin.face
    del complex_.half_edges[(u, v)]
    del complex_.half_edges[(v, u)]
    complex_.half_edges[(z, w)] = half_edge
    complex_.half_edges[(w, z)] = twin
    complex_._current_faces[half_edge.face] = (z, half_edge_prev.origin, half_edge_prev.next.origin)
    complex_._current_faces[twin.face] = (w, twin_prev.origin, twin_prev.next.origin)
    return True