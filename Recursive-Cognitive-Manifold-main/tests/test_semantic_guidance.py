import unittest

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.memory.semantic import MotifPrototype


class SemanticGuidanceTests(unittest.TestCase):
    def _make_two_node_region_manifold(self):
        manifold = RecursiveCognitiveManifold(state_dim=2, hrm_seed=42)
        manifold.add_node(0, [1.0, 0.0])
        manifold.add_node(1, [0.95, 0.05])
        manifold.connect(0, 1, strength=0.9, latency=1.0)
        manifold._update_regions()
        return manifold

    def test_semantic_motif_guides_region_state_update(self):
        manifold = self._make_two_node_region_manifold()
        signature = next(iter(manifold.regions.keys()))
        region = manifold.regions[signature]
        motif_signature = (region.specialization, len(signature))

        manifold.semantic_memory.prototypes[motif_signature] = MotifPrototype(
            signature=motif_signature,
            prototype_activation=np.asarray([0.5, 0.5], dtype=float),
            topological_pattern={
                "nodes": tuple(int(node_id) for node_id in signature),
                "node_count": len(signature),
                "mean_internal_strength": 0.9,
                "mean_external_strength": 0.0,
            },
            expected_transition=np.asarray([-0.8, 0.8], dtype=float),
            associated_outcome={"stability": 0.9, "mean_energy": 1.0, "mean_confidence": 0.9, "child_manifolds": 0.0},
            confidence=0.9,
            usage_count=4,
        )

        before_node0 = manifold.nodes[0].local_state.copy()
        before_node1 = manifold.nodes[1].local_state.copy()

        manifold._apply_semantic_guidance()

        after_node0 = manifold.nodes[0].local_state
        after_node1 = manifold.nodes[1].local_state

        self.assertNotEqual(float(np.linalg.norm(after_node0 - before_node0)), 0.0)
        self.assertNotEqual(float(np.linalg.norm(after_node1 - before_node1)), 0.0)
        self.assertTrue(manifold.last_semantic_guidance["region_guidance"])

    def test_snapshot_reports_semantic_guidance_metrics(self):
        manifold = self._make_two_node_region_manifold()
        signature = next(iter(manifold.regions.keys()))
        region = manifold.regions[signature]
        motif_signature = (region.specialization, len(signature))
        manifold.semantic_memory.prototypes[motif_signature] = MotifPrototype(
            signature=motif_signature,
            prototype_activation=np.asarray([0.6, 0.4], dtype=float),
            topological_pattern={
                "nodes": tuple(int(node_id) for node_id in signature),
                "node_count": len(signature),
                "mean_internal_strength": 0.85,
                "mean_external_strength": 0.0,
            },
            expected_transition=np.asarray([-0.3, 0.3], dtype=float),
            associated_outcome={"stability": 0.8, "mean_energy": 1.0, "mean_confidence": 0.8, "child_manifolds": 0.0},
            confidence=0.8,
            usage_count=3,
        )

        manifold.step({0: np.asarray([1.0, 0.0], dtype=float), 1: np.asarray([0.9, 0.1], dtype=float)})
        snapshot = manifold.snapshot()

        self.assertIn("semantic_guidance", snapshot)
        self.assertTrue(snapshot["semantic_guidance"]["enabled"])
        self.assertIn("region_guidance", snapshot["semantic_guidance"])


if __name__ == "__main__":
    unittest.main()
