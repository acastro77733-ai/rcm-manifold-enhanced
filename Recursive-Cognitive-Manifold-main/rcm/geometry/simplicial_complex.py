import numpy as np
from scipy.sparse import coo_matrix

from rcm.geometry.half_edge import HalfEdge
from rcm.geometry.invariants import compute_topology_invariants
from rcm.geometry.invariants import get_nonmanifold_edges
from rcm.geometry.invariants import get_nonmanifold_vertices
from rcm.geometry.invariants import is_orientable
from rcm.geometry.surgery import flip_edge
from rcm.geometry.surgery import repair_nonmanifold_vertex


class DynamicSimplicialComplex:
    np = np

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.pad(vertices, ((0, 0), (0, 1)), mode="constant") if vertices.shape[1] == 2 else vertices
        self.n_vertices = len(self.vertices)
        self.half_edges = {}
        self._current_faces = []
        self._build_half_edge_structure(faces)
        self.L = self.compute_cotangent_laplacian()

    def _build_half_edge_structure(self, faces):
        self.half_edges.clear()
        self._current_faces = [tuple(face) for face in faces]
        for face_index, face in enumerate(self._current_faces):
            edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
            half_edges = []
            for u, v in edges:
                half_edge = HalfEdge(u)
                half_edge.face = face_index
                self.half_edges[(u, v)] = half_edge
                half_edges.append(half_edge)
                if (v, u) in self.half_edges:
                    twin_half_edge = self.half_edges[(v, u)]
                    half_edge.twin, twin_half_edge.twin = twin_half_edge, half_edge
            half_edges[0].next, half_edges[1].next, half_edges[2].next = half_edges[1], half_edges[2], half_edges[0]

    def get_faces(self):
        unique_faces = {half_edge.face for half_edge in self.half_edges.values() if half_edge.face is not None}
        return np.array([self._current_faces[index] for index in unique_faces if index < len(self._current_faces)])

    def compute_topology_invariants(self):
        return compute_topology_invariants(self)

    def _get_nonmanifold_edges(self):
        return get_nonmanifold_edges(self)

    def _get_nonmanifold_vertices(self):
        return get_nonmanifold_vertices(self)

    def _is_orientable(self):
        return is_orientable(self)

    def repair_nonmanifold_vertex(self, vertex_index):
        return repair_nonmanifold_vertex(self, vertex_index)

    def compute_cotangent_laplacian(self):
        row_indices = []
        column_indices = []
        values = []
        for face in self.get_faces():
            for i_idx in range(3):
                i = face[i_idx]
                j = face[(i_idx + 1) % 3]
                k = face[(i_idx + 2) % 3]
                vi, vj, vk = self.vertices[i], self.vertices[j], self.vertices[k]
                u = vi - vk
                v = vj - vk
                cotangent = 0.5 * np.dot(u, v) / max(np.linalg.norm(np.cross(u, v)), 1e-12)
                row_indices.extend([i, j, i, j])
                column_indices.extend([j, i, i, j])
                values.extend([-cotangent, -cotangent, cotangent, cotangent])
        self.L = coo_matrix((values, (row_indices, column_indices)), shape=(self.n_vertices, self.n_vertices)).tocsr()
        return self.L

    def flip_edge(self, u, v):
        return flip_edge(self, u, v)