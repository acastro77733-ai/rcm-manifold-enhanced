import unittest

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.dynamics.plasticity import TopologyAdaptationState, TopologyBudget, apply_edge_masking, update_edge_dynamics


class TopologyAdaptationTests(unittest.TestCase):
    def test_masking_updates_utility_and_prunes_under_budget(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="topology")
        manifold.add_node(0, np.array([0.2, -0.1, 0.0, 0.0], dtype=float))
        manifold.add_node(1, np.array([0.2, -0.1, 0.0, 0.0], dtype=float))
        manifold.connect(0, 1, strength=0.1, latency=1.0)

        adaptation_state = TopologyAdaptationState()
        budget = TopologyBudget(max_edge_count=1, pruning_threshold=0.05, structural_cooldown=0)
        result = apply_edge_masking(manifold, (0, 1), {(0, 1): 0.8}, topology_budget=budget, adaptation_state=adaptation_state, step=0)
        self.assertTrue(result["masked"])
        self.assertGreaterEqual(result["utility_delta"], 0.0)

        pruned = update_edge_dynamics(manifold, {(0, 1): 0.8}, topology_budget=budget, adaptation_state=adaptation_state, step=1)
        self.assertEqual(pruned, [(0, 1)])


if __name__ == "__main__":
    unittest.main()
