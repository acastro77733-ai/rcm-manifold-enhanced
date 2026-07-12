import unittest

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.memory.episodic import EpisodicMemory


class FunctionalRecallTests(unittest.TestCase):
    def test_retrieve_uses_top_k_weighted_prediction(self):
        memory = EpisodicMemory(maxlen=8)
        memory.transitions.append(
            {
                "node_id": 0,
                "time_step": 0,
                "current_state": np.asarray([1.0, 0.0], dtype=float),
                "next_state": np.asarray([1.0, 0.0], dtype=float),
                "outcome": {"energy": 1.0, "confidence": 1.0},
            }
        )
        memory.transitions.append(
            {
                "node_id": 0,
                "time_step": 1,
                "current_state": np.asarray([0.85, 0.15], dtype=float),
                "next_state": np.asarray([0.0, 1.0], dtype=float),
                "outcome": {"energy": 0.8, "confidence": 0.7},
            }
        )

        recalled = memory.retrieve(np.asarray([1.0, 0.0], dtype=float), node_id=0, top_k=2, min_similarity=0.0)

        self.assertIsNotNone(recalled)
        self.assertEqual(recalled["retrieval_count"], 2)
        self.assertIn("best_similarity", recalled)
        self.assertGreater(recalled["next_state"][0], 0.0)
        self.assertGreater(recalled["next_state"][1], 0.0)
        self.assertGreater(recalled["similarity"], 0.0)

    def test_manifold_recall_blends_predicted_future_activation(self):
        manifold = RecursiveCognitiveManifold(state_dim=2, hrm_seed=42)
        manifold.add_node(0, [0.0, 0.0])

        manifold.episodic_memory.transitions.append(
            {
                "node_id": 0,
                "time_step": 0,
                "current_state": np.asarray([1.0, 0.0], dtype=float),
                "next_state": np.asarray([0.0, 1.0], dtype=float),
                "outcome": {"energy": 1.0, "confidence": 1.0},
            }
        )
        manifold.episodic_memory.transitions.append(
            {
                "node_id": 0,
                "time_step": 1,
                "current_state": np.asarray([0.9, 0.1], dtype=float),
                "next_state": np.asarray([0.2, 0.8], dtype=float),
                "outcome": {"energy": 1.0, "confidence": 0.9},
            }
        )

        updated_states = {0: np.asarray([1.0, 0.0], dtype=float)}
        recalled_states = manifold._apply_episodic_recall(updated_states)

        self.assertGreater(recalled_states[0][1], 0.0)
        self.assertLess(recalled_states[0][0], 1.0)


if __name__ == "__main__":
    unittest.main()
