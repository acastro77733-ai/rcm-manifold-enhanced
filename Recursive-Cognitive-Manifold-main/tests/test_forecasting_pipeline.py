import unittest

import numpy as np

from rcm.experiments.forecasting import (
    MECHANISM_TASKS,
    build_dataset_split,
    build_forecast_model,
    describe_model_mechanisms,
    evaluate_forecast_model,
)


class ForecastingPipelineTests(unittest.TestCase):
    def test_dataset_split_uses_independent_deterministic_streams(self):
        split_a = build_dataset_split("temporal_forecasting", graph_size=4, seed=7, state_dim=4)
        split_b = build_dataset_split("temporal_forecasting", graph_size=4, seed=7, state_dim=4)

        self.assertEqual(split_a.seeds, split_b.seeds)
        self.assertNotEqual(split_a.seeds["train"], split_a.seeds["validation"])
        self.assertNotEqual(split_a.seeds["validation"], split_a.seeds["test"])
        self.assertEqual(len(split_a.train), len(split_b.train))
        np.testing.assert_allclose(split_a.train[0][0], split_b.train[0][0])
        self.assertIsNot(split_a.train[0][0], split_b.train[0][0])

    def test_all_forecast_models_expose_common_contract(self):
        model_names = [
            "full_rcm",
            "current_rcm",
            "no_guidance",
            "no_hrm",
            "fixed_topology",
            "no_hierarchy",
            "memory_only",
            "persistence",
            "linear_autoregression",
            "nearest_neighbor_transition",
            "graph_diffusion",
            "reservoir_recurrent",
            "fixed_graph_control",
        ]
        split = build_dataset_split("temporal_forecasting", graph_size=4, seed=11, state_dim=4)

        for name in model_names:
            model = build_forecast_model(name, seed=11, graph_size=4, state_dim=4)
            for method in ("fit", "observe", "predict", "update", "reset"):
                self.assertTrue(hasattr(model, method), msg=name)
            report = evaluate_forecast_model(model, split)
            self.assertIn("validation_mean_error", report, msg=name)
            self.assertIn("test_mean_error", report, msg=name)
            self.assertGreaterEqual(model.prediction_calls, len(split.validation) - 1, msg=name)
            self.assertEqual(model.prediction_calls, model.update_calls, msg=name)

    def test_ablation_descriptions_remove_only_intended_pathways(self):
        full = build_forecast_model("full_rcm", seed=5, graph_size=4, state_dim=4)
        no_hrm = build_forecast_model("no_hrm", seed=5, graph_size=4, state_dim=4)
        no_hierarchy = build_forecast_model("no_hierarchy", seed=5, graph_size=4, state_dim=4)
        fixed_topology = build_forecast_model("fixed_topology", seed=5, graph_size=4, state_dim=4)

        full_mechanisms = describe_model_mechanisms(full)
        no_hrm_mechanisms = describe_model_mechanisms(no_hrm)
        no_hierarchy_mechanisms = describe_model_mechanisms(no_hierarchy)
        fixed_topology_mechanisms = describe_model_mechanisms(fixed_topology)

        self.assertEqual(full_mechanisms["hrm_mode"], "full")
        self.assertEqual(no_hrm_mechanisms["hrm_mode"], "no_field")
        self.assertTrue(no_hrm_mechanisms["hierarchy"])
        self.assertFalse(no_hierarchy_mechanisms["hierarchy"])
        self.assertTrue(no_hierarchy_mechanisms["semantic_guidance"])
        self.assertFalse(fixed_topology_mechanisms["topology_adaptation"])
        self.assertEqual(full_mechanisms["semantic_guidance"], fixed_topology_mechanisms["semantic_guidance"])

    def test_task_registry_covers_required_mechanism_workloads(self):
        expected = {
            "temporal_forecasting",
            "corrupted_reconstruction",
            "topology_change_adaptation",
            "hierarchical_composition",
            "long_horizon_memory",
            "distribution_shift",
            "structural_lesion_recovery",
            "resource_constrained_computation",
        }
        self.assertEqual(expected, set(MECHANISM_TASKS.keys()))
        for task in MECHANISM_TASKS.values():
            self.assertTrue(task.rationale)


if __name__ == "__main__":
    unittest.main()
