import unittest

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.hierarchy.child_manifold import refresh_child_manifolds


class ActiveHierarchyTests(unittest.TestCase):
    def _build_two_region_manifold(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, hrm_seed=42)
        manifold.add_node(0, [1.0, 0.0, 0.0, 0.0])
        manifold.add_node(1, [0.9, 0.1, 0.0, 0.0])
        manifold.add_node(2, [0.0, 0.0, 1.0, 0.0])
        manifold.add_node(3, [0.0, 0.0, 0.9, 0.1])

        manifold.connect(0, 1, strength=0.95, latency=1.0)
        manifold.connect(2, 3, strength=0.95, latency=1.0)
        manifold.connect(1, 2, strength=0.2, latency=1.0)

        manifold._update_regions()
        for region in manifold.regions.values():
            region.stability = 0.95
        refresh_child_manifolds(manifold)
        return manifold

    def test_refresh_child_manifold_preserves_computational_state(self):
        manifold = self._build_two_region_manifold()
        self.assertEqual(len(manifold.child_manifolds), 1)

        child = manifold.child_manifolds[0]
        self.assertGreaterEqual(len(child.nodes), 2)

        child.nodes[0].energy = 2.0
        child.nodes[0].confidence = 0.9
        child.nodes[0].local_state = np.asarray([0.5, 0.4, 0.3, 0.2], dtype=float)

        refresh_child_manifolds(manifold)
        refreshed_child = manifold.child_manifolds[0]

        self.assertGreater(refreshed_child.nodes[0].energy, 1.0)
        self.assertGreater(refreshed_child.nodes[0].confidence, 0.5)
        self.assertGreater(np.linalg.norm(refreshed_child.nodes[0].local_state), 0.0)

    def test_child_coupling_influences_parent_cross_region_edges(self):
        manifold = self._build_two_region_manifold()
        child = manifold.child_manifolds[0]

        parent_regions = sorted(manifold.regions.keys())
        self.assertEqual(len(parent_regions), 2)
        region_a, region_b = parent_regions

        inverse_map = {tuple(sorted(signature)): meta_id for meta_id, signature in child.parent_region_map.items()}
        meta_a = inverse_map[tuple(sorted(region_a))]
        meta_b = inverse_map[tuple(sorted(region_b))]

        if (meta_a, meta_b) not in child.edges:
            child.connect(meta_a, meta_b, strength=0.85, latency=1.0)
        child.edges[(meta_a, meta_b)].strength = 0.9
        child.edges[(meta_b, meta_a)].strength = 0.9

        before_strength = manifold.edges[(1, 2)].strength
        manifold._apply_child_manifold_fields()
        after_strength = manifold.edges[(1, 2)].strength

        self.assertGreater(after_strength, before_strength)
        self.assertTrue(manifold.last_child_field_metrics)
        self.assertIn("coupling_pressure", manifold.last_child_field_metrics[0])


if __name__ == "__main__":
    unittest.main()
