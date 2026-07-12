import unittest

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.geometry.simplicial_complex import DynamicSimplicialComplex


class GeometryFeedbackTests(unittest.TestCase):
    def _build_square_complex(self):
        vertices = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=float,
        )
        faces = np.asarray(
            [
                [0, 1, 2],
                [1, 3, 2],
            ],
            dtype=int,
        )
        return DynamicSimplicialComplex(vertices, faces)

    def test_geometry_feedback_metrics_emitted_in_snapshot(self):
        complex_ = self._build_square_complex()
        manifold = RecursiveCognitiveManifold.from_simplicial_complex(complex_, state_dim=3, hrm_seed=42)

        manifold.step(
            {
                0: np.asarray([1.0, 0.0, 0.0], dtype=float),
                1: np.asarray([1.0, 0.0, 0.0], dtype=float),
                2: np.asarray([0.0, 1.0, 0.0], dtype=float),
                3: np.asarray([0.0, 1.0, 0.0], dtype=float),
            }
        )
        snapshot = manifold.snapshot()

        self.assertIn("geometry_feedback", snapshot)
        self.assertTrue(snapshot["geometry_feedback"]["enabled"])
        self.assertIn("laplacian_energy", snapshot["geometry_feedback"])
        self.assertGreaterEqual(snapshot["geometry_feedback"]["laplacian_energy"], 0.0)

    def test_plasticity_requests_and_applies_simplicial_surgery(self):
        complex_ = self._build_square_complex()
        manifold = RecursiveCognitiveManifold.from_simplicial_complex(complex_, state_dim=3, hrm_seed=42)

        # Ensure the shared diagonal carries strong pressure for a flip request.
        manifold.edges[(1, 2)].strength = 1.0
        manifold.edges[(2, 1)].strength = 1.0
        manifold.nodes[1].local_state = np.asarray([1.0, 0.0, 0.0], dtype=float)
        manifold.nodes[2].local_state = np.asarray([-1.0, 0.0, 0.0], dtype=float)

        manifold._apply_geometry_feedback({(1, 2): 2.0, (2, 1): 2.0})

        feedback = manifold.last_geometry_feedback
        self.assertTrue(feedback["surgery_requests"])
        self.assertTrue(feedback["surgery_attempted"])
        self.assertGreaterEqual(feedback["laplacian_energy"], 0.0)


if __name__ == "__main__":
    unittest.main()
