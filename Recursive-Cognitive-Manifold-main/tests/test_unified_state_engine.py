import unittest

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.state_engine import BoundedStateEngine, UnifiedStateConfig


class UnifiedStateEngineTests(unittest.TestCase):
    def test_unified_state_engine_tracks_contributions_and_bounds(self):
        engine = BoundedStateEngine(config=UnifiedStateConfig(dt=0.1))
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="test")
        manifold.add_node(0, np.array([0.4, 0.1, 0.0, 0.0], dtype=float))
        manifold.add_node(1, np.array([0.0, 0.3, 0.2, 0.1], dtype=float))
        manifold.connect(0, 1, strength=0.8, latency=1.0)

        result = engine.step(manifold, external_input={0: np.array([1.0, 0.0, 0.0, 0.0], dtype=float)})

        self.assertEqual(result["state_vector"].shape, (7,))
        self.assertTrue(np.isfinite(result["state_vector"]).all())
        self.assertTrue(result["bounded"])
        self.assertGreaterEqual(len(engine.last_contributions), 3)
        self.assertTrue(any(entry["term"] == "base" for entry in engine.last_contributions))

    def test_manifold_step_records_unified_state_diagnostics(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="trace")
        manifold.add_node(0, np.array([0.4, 0.1, 0.0, 0.0], dtype=float))
        manifold.add_node(1, np.array([0.0, 0.3, 0.2, 0.1], dtype=float))
        manifold.connect(0, 1, strength=0.8, latency=1.0)

        snapshot = manifold.step({0: np.array([0.8, 0.1, 0.0, 0.0], dtype=float)})

        self.assertIn("unified_state", snapshot)
        self.assertIn("unified_metrics", snapshot)
        self.assertIn("unified_contributions", snapshot)
        self.assertIsNotNone(snapshot["unified_state"]["state_vector"])


if __name__ == "__main__":
    unittest.main()
