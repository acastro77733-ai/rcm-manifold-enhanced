import unittest

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm_benchmarks import (
    _disable_episodic,
    _disable_geometry_feedback,
    _disable_semantic_hierarchy,
    _disable_structural,
    forecast_contract_benchmark,
    run_benchmark_suite,
)


class BenchmarkForecastIntegrationTests(unittest.TestCase):
    def test_forecast_contract_benchmark_reports_conditions_and_mechanisms(self):
        report = forecast_contract_benchmark(seed=5)
        self.assertEqual(report["task"], "temporal_forecasting")
        self.assertIn("full_rcm", report["conditions"])
        self.assertIn("no_hrm", report["conditions"])
        self.assertIn("mechanisms", report["conditions"]["full_rcm"])
        self.assertIn("test_mean_error", report["conditions"]["full_rcm"])

    def test_exact_benchmark_ablation_helpers_toggle_local_mechanism_flags(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="bench")
        manifold.add_node(0, np.array([0.0, 0.0, 0.0, 0.0], dtype=float))
        manifold.add_node(1, np.array([0.0, 0.0, 0.0, 0.0], dtype=float))
        manifold.connect(0, 1, strength=0.5, latency=1.0)

        _disable_semantic_hierarchy(manifold)
        self.assertFalse(manifold.semantic_guidance_enabled)
        self.assertFalse(manifold.hierarchy_enabled)

        _disable_structural(manifold)
        self.assertFalse(manifold.structural_plasticity_enabled)

        _disable_episodic(manifold)
        self.assertFalse(manifold.episodic_recall_enabled)

        _disable_geometry_feedback(manifold)
        self.assertFalse(manifold.geometry_feedback_enabled)

    def test_benchmark_suite_includes_forecast_and_paired_seed_sections(self):
        def factory(seed=None):
            manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="factory", hrm_seed=seed)
            for node_id in range(6):
                manifold.add_node(node_id, np.zeros(4, dtype=float))
            for node_id in range(5):
                manifold.connect(node_id, node_id + 1, strength=0.6, latency=1.0)
            return manifold

        report = run_benchmark_suite(factory, seed=3, include_timestamp=False)
        self.assertIn("forecast_contract", report["benchmarks"])
        self.assertIn("paired_seed_ablation", report["benchmarks"])
        self.assertIn("comparisons", report["benchmarks"]["paired_seed_ablation"])


if __name__ == "__main__":
    unittest.main()
