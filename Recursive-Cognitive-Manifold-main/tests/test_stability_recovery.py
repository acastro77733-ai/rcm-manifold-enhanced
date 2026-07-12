import unittest

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold


class StabilityRecoveryTests(unittest.TestCase):
    def test_instability_triggers_deterministic_recovery_and_rollback(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="stability")
        manifold.add_node(0, np.array([0.2, 0.0, 0.0, 0.0], dtype=float))
        manifold.add_node(1, np.array([0.1, 0.1, 0.0, 0.0], dtype=float))
        manifold.connect(0, 1, strength=0.8, latency=1.0)

        snapshot = manifold._snapshot_for_recovery()
        tx = manifold.stability_controller.begin_transaction(
            "representation",
            {"target_dim": 6},
            snapshot,
            trial_metrics={"score": 0.0},
            post_change_metrics={"score": 0.0},
            accepted=True,
        )
        manifold.representation_controller.current_dim = 6
        manifold.stability_controller.complete_transaction(tx, post_change_metrics={"score": 0.1}, accepted=True)

        manifold._previous_state_signature = np.array([0.2, 0.1, 0.0, 0.0], dtype=float)
        manifold._current_state_signature = np.array([1.2, 1.1, 0.8, 0.7], dtype=float)
        manifold.last_field_metrics = {"field_variance": 0.9}
        manifold.collapse_energy = 0.9

        result = manifold.stability_controller.respond_to_instability(manifold, level=3)
        self.assertTrue(result["rolled_back"])
        self.assertGreaterEqual(result["level"], 3)
        self.assertEqual(manifold.representation_controller.current_dim, 4)


if __name__ == "__main__":
    unittest.main()
