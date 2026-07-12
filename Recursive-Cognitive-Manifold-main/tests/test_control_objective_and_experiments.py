import unittest

from rcm.experiments.comparative_suite import run_comparative_suite
from rcm.metrics import ManifoldMetrics
from rcm.experiments.comparative_suite import compute_control_objective


class ControlObjectiveTests(unittest.TestCase):
    def test_control_objective_is_scalar_and_prefers_stable_high_quality_runs(self):
        baseline = ManifoldMetrics(task_score=0.2, recall_score=0.1, coherence=0.2, collapse_energy=0.9, topology_cost=0.6, representation_cost=0.4, field_energy=0.8)
        improved = ManifoldMetrics(task_score=0.8, recall_score=0.7, coherence=0.8, collapse_energy=0.2, topology_cost=0.3, representation_cost=0.2, field_energy=0.3)
        self.assertTrue(compute_control_objective(baseline) < compute_control_objective(improved))
        self.assertTrue(isinstance(compute_control_objective(improved), float))

    def test_comparative_suite_returns_four_configurations_with_value_deltas(self):
        report = run_comparative_suite(seed=7, steps=3, repeat_count=1, seed_count=2, trajectory_count=2)
        self.assertEqual(len(report["configurations"]), 4)
        self.assertGreaterEqual(len(report["configurations"][0]["runs"]), 2)
        self.assertIn("value_delta", report["configurations"][0])
        self.assertIn("value_delta_ci95", report["configurations"][0])
        self.assertIn("control_objective", report["configurations"][0])
        self.assertIn("summary", report)
        self.assertIn("artifacts", report)


if __name__ == "__main__":
    unittest.main()
