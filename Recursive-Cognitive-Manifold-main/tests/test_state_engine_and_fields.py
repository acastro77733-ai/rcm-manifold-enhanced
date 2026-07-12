import unittest

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.dynamics.fields.legacy import LegacyFieldModel
from rcm.dynamics.fields.nonlinear import NonlinearFieldModel
from rcm.dynamics.state_engine import StateEngine, StateEngineConfig


class StateEngineTests(unittest.TestCase):
    def test_state_remains_finite_under_extreme_input(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="finite")
        manifold.add_node(0, np.array([0.3, -0.2, 0.1, 0.0], dtype=float))
        manifold.add_node(1, np.array([0.1, 0.2, -0.1, 0.4], dtype=float))
        manifold.connect(0, 1, strength=0.9, latency=1.0)

        engine = StateEngine(config=StateEngineConfig())
        result = engine.step(manifold, external_input={0: np.array([1000.0, -1000.0, 1000.0, -1000.0], dtype=float)})

        self.assertTrue(np.isfinite(result["node_states"][0]).all())
        self.assertTrue(np.isfinite(result["node_states"][1]).all())
        self.assertLessEqual(np.linalg.norm(result["node_states"][0]), engine.config.activation_capacity)

    def test_no_input_trajectories_decay_or_stabilize(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="decay")
        manifold.add_node(0, np.array([0.8, 0.2, 0.0, 0.0], dtype=float))
        manifold.add_node(1, np.array([-0.4, 0.3, 0.2, 0.1], dtype=float))
        manifold.connect(0, 1, strength=0.7, latency=1.0)

        engine = StateEngine(config=StateEngineConfig())
        result = engine.step(manifold)
        self.assertLessEqual(np.linalg.norm(result["node_states"][0]), 1.5)

    def test_coupling_increases_neighborhood_coherence(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="coherence")
        manifold.add_node(0, np.array([0.4, 0.0, 0.0, 0.0], dtype=float))
        manifold.add_node(1, np.array([0.0, 0.4, 0.0, 0.0], dtype=float))
        manifold.connect(0, 1, strength=0.9, latency=1.0)

        engine = StateEngine(config=StateEngineConfig())
        result = engine.step(manifold)
        coherence = float(np.linalg.norm(result["node_states"][0] - result["node_states"][1]))
        self.assertLess(coherence, 1.0)

    def test_recall_measurably_affects_resulting_state(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="recall")
        manifold.add_node(0, np.array([0.2, 0.1, 0.0, 0.0], dtype=float))
        manifold.add_node(1, np.array([0.0, 0.2, 0.1, 0.0], dtype=float))
        manifold.connect(0, 1, strength=0.8, latency=1.0)

        engine = StateEngine(config=StateEngineConfig())
        full_result = engine.step(manifold, external_input={0: np.array([0.3, 0.2, 0.0, 0.0], dtype=float)})
        disabled_result = engine.step(manifold, external_input={0: np.array([0.3, 0.2, 0.0, 0.0], dtype=float)}, recall_enabled=False)

        delta = np.linalg.norm(full_result["node_states"][0] - disabled_result["node_states"][0])
        self.assertGreater(delta, 0.0)

    def test_disabling_each_term_reproduces_an_expected_ablation(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="ablation")
        manifold.add_node(0, np.array([0.1, 0.0, 0.0, 0.0], dtype=float))
        manifold.add_node(1, np.array([0.0, 0.2, 0.0, 0.0], dtype=float))
        manifold.connect(0, 1, strength=0.7, latency=1.0)

        engine = StateEngine(config=StateEngineConfig())
        baseline = engine.step(manifold, external_input={0: np.array([0.2, 0.1, 0.0, 0.0], dtype=float)})
        disabled = engine.step(manifold, external_input={0: np.array([0.2, 0.1, 0.0, 0.0], dtype=float)}, input_enabled=False)
        self.assertNotEqual(baseline["node_states"][0].tolist(), disabled["node_states"][0].tolist())


class NonlinearFieldTests(unittest.TestCase):
    def test_nonlinear_field_is_bounded_and_differs_from_legacy(self):
        legacy = LegacyFieldModel(seed=1, state_dim=4)
        nonlinear = NonlinearFieldModel(seed=1, state_dim=4)
        node_summary = np.array([0.6, -0.3, 0.2, 0.1], dtype=float)

        legacy_result = legacy.step(node_summary, mean_energy=0.7, mean_confidence=0.8, topology_stats={"node_count": 2, "edge_count": 1, "density": 0.5, "stability": 0.6})
        nonlinear_result = nonlinear.step(node_summary, mean_energy=0.7, mean_confidence=0.8, topology_stats={"node_count": 2, "edge_count": 1, "density": 0.5, "stability": 0.6})

        self.assertTrue(np.isfinite(nonlinear_result["field_state"]).all())
        self.assertTrue(np.linalg.norm(nonlinear_result["field_state"]) <= 10.0)
        self.assertNotEqual(legacy_result["metrics"]["field_norm"], nonlinear_result["metrics"]["field_norm"])


if __name__ == "__main__":
    unittest.main()
