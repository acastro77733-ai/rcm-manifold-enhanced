import unittest

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm_benchmarks import multi_scale_performance


class ScaleSweepTests(unittest.TestCase):
    def test_multi_scale_performance_reports_all_registered_scales(self):
        def factory(seed=None):
            manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="scale", hrm_seed=seed)
            for node_id in range(4):
                manifold.add_node(node_id, np.zeros(4, dtype=float))
            for node_id in range(3):
                manifold.connect(node_id, node_id + 1, strength=0.6, latency=1.0)
            return manifold

        report = multi_scale_performance(factory, seed=11)
        self.assertEqual(set(report.keys()), {"small", "medium", "large"})
        for entry in report.values():
            self.assertIn("steps", entry)
            self.assertIn("mean_step_seconds", entry)
            self.assertIn("steps_per_second", entry)


if __name__ == "__main__":
    unittest.main()
