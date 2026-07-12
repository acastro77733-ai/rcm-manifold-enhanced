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

        self.assertGreater(result["state_vector"].shape[0], 7)
        self.assertTrue(np.isfinite(result["state_vector"]).all())
        self.assertTrue(result["bounded"])
        self.assertGreaterEqual(len(engine.last_contributions), 8)
        self.assertTrue(any(entry["term"] == "external" for entry in engine.last_contributions))
        self.assertIn("governed_state", result)
        self.assertEqual(sorted(result["governed_state"].node_fields.keys()), [0, 1])
        self.assertIn("forecast", result["block_names"])

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
        self.assertIn("forecast_state", snapshot["unified_state"])

    def test_manifold_trajectories_change_when_governed_engine_is_disabled(self):
        baseline = RecursiveCognitiveManifold(state_dim=4, level=0, label="baseline")
        baseline.add_node(0, np.array([0.4, 0.1, 0.0, 0.0], dtype=float))
        baseline.add_node(1, np.array([0.0, 0.3, 0.2, 0.1], dtype=float))
        baseline.connect(0, 1, strength=0.8, latency=1.0)

        ablated = RecursiveCognitiveManifold(state_dim=4, level=0, label="ablated")
        ablated.add_node(0, np.array([0.4, 0.1, 0.0, 0.0], dtype=float))
        ablated.add_node(1, np.array([0.0, 0.3, 0.2, 0.1], dtype=float))
        ablated.connect(0, 1, strength=0.8, latency=1.0)
        ablated.unified_state_engine.config = UnifiedStateConfig(
            dt=0.0,
            external_gain=0.0,
            local_gain=0.0,
            field_gain=0.0,
            diffusion_gain=0.0,
            recall_gain=0.0,
            regional_gain=0.0,
            hierarchy_gain=0.0,
            topology_gain=0.0,
            damping_gain=0.0,
            residual_gain=0.0,
            forecast_gain=0.0,
        )

        signal = {0: np.array([0.8, 0.1, 0.0, 0.0], dtype=float)}
        baseline.step(signal)
        ablated.step(signal)

        baseline_state = baseline.nodes[0].local_state.copy()
        ablated_state = ablated.nodes[0].local_state.copy()
        self.assertGreater(np.linalg.norm(baseline_state - ablated_state), 0.0)


if __name__ == "__main__":
    unittest.main()
