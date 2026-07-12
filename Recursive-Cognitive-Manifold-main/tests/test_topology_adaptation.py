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

    def test_topology_repair_records_utility_decisions(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="repair")
        manifold.add_node(0, np.array([0.8, 0.1, 0.0, 0.0], dtype=float))
        manifold.add_node(1, np.array([0.78, 0.12, 0.0, 0.0], dtype=float))
        manifold.add_node(2, np.array([0.0, 0.0, 0.5, 0.2], dtype=float))
        manifold.connect(0, 2, strength=0.6, latency=1.0)
        manifold.connect(2, 1, strength=0.6, latency=1.0)
        manifold.structural_memory.record(manifold.time_step, manifold.edges, [])
        manifold.edges = {key: edge for key, edge in manifold.edges.items() if key not in {(0, 2), (2, 0), (2, 1), (1, 2)}}
        manifold.nodes[0].topology_history.append((0, "link:0->1"))
        manifold.nodes[1].topology_history.append((0, "link:1->0"))

        repairs = manifold.attempt_topology_repair(max_repairs=1, similarity_threshold=0.5)

        self.assertLessEqual(len(repairs), 1)
        self.assertTrue(manifold.last_topology_decisions)
        self.assertIn("reason", manifold.last_topology_decisions[0])
        self.assertIn("utility_delta", manifold.last_topology_decisions[0])


if __name__ == "__main__":
    unittest.main()
