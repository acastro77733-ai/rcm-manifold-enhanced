import unittest

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold


class RuntimeInvariantTests(unittest.TestCase):
    def test_longer_rollout_preserves_finite_bounded_core_invariants(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="invariants", hrm_seed=23)
        for node_id in range(4):
            manifold.add_node(node_id, np.zeros(4, dtype=float))
        for node_id in range(3):
            manifold.connect(node_id, node_id + 1, strength=0.7, latency=1.0)

        for step in range(12):
            signal = {
                node_id: np.roll(np.array([1.0, 0.2, 0.0, 0.0], dtype=float), shift=(step + node_id) % 4)
                for node_id in manifold.nodes
            }
            manifold.step(signal)

        for node in manifold.nodes.values():
            self.assertTrue(np.isfinite(node.local_state).all())
            self.assertGreaterEqual(node.energy, 0.0)
            self.assertGreaterEqual(node.confidence, 0.0)
            self.assertLessEqual(node.confidence, 1.0)
        self.assertTrue(np.isfinite(manifold.last_field_metrics.get("field_energy", 0.0)))
        self.assertTrue(np.isfinite(manifold.last_geometry_feedback.get("laplacian_energy", 0.0)))


if __name__ == "__main__":
    unittest.main()
