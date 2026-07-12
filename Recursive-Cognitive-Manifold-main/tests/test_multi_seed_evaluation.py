import json
import tempfile
import unittest
from pathlib import Path

from rcm.experiments.multi_seed_evaluation import build_experiment_manifest, run_paired_multi_seed_evaluation, write_experiment_manifest, write_multi_seed_report


class MultiSeedEvaluationTests(unittest.TestCase):
    def test_multi_seed_evaluation_reports_paired_statistics(self):
        report = run_paired_multi_seed_evaluation(
            task_name="temporal_forecasting",
            seed_start=3,
            n_seeds=3,
            graph_size=4,
            state_dim=4,
            primary_condition="full_rcm",
            control_conditions=["no_hrm", "linear_autoregression"],
        )

        self.assertIn("raw_runs", report)
        self.assertIn("summaries", report)
        self.assertIn("comparisons", report)
        self.assertIn("full_rcm_vs_no_hrm", report["comparisons"])
        self.assertEqual(report["manifest"]["n_seeds"], 3)
        self.assertIn("bootstrap_ci", report["comparisons"]["full_rcm_vs_no_hrm"])
        self.assertIn("corrected_p_value", report["comparisons"]["full_rcm_vs_no_hrm"])
        self.assertIn("meets_primary_success_threshold", report["comparisons"]["full_rcm_vs_no_hrm"])
        self.assertIn("wilcoxon_signed_rank_p_value", report["comparisons"]["full_rcm_vs_no_hrm"])
        self.assertIn("sign_test_p_value", report["comparisons"]["full_rcm_vs_no_hrm"])
        self.assertIn("bonferroni_p_value", report["comparisons"]["full_rcm_vs_no_hrm"])
        self.assertIn("benjamini_hochberg_q_value", report["comparisons"]["full_rcm_vs_no_hrm"])
        self.assertIn("hodges_lehmann_estimate", report["comparisons"]["full_rcm_vs_no_hrm"])

    def test_manifest_writer_persists_machine_readable_assessment(self):
        manifest = build_experiment_manifest(
            task_name="temporal_forecasting",
            n_seeds=30,
            primary_condition="full_rcm",
            control_conditions=["no_hrm"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_experiment_manifest(manifest, Path(tmpdir) / "manifest.json")
            self.assertTrue(path.exists())
            content = path.read_text(encoding="utf-8")
            self.assertIn("predict_before_update", content)
            self.assertIn("unresolved_limitations", content)

    def test_report_writer_persists_multi_seed_output(self):
        report = run_paired_multi_seed_evaluation(
            task_name="temporal_forecasting",
            seed_start=1,
            n_seeds=2,
            graph_size=4,
            state_dim=4,
            primary_condition="full_rcm",
            control_conditions=["no_hrm"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_multi_seed_report(report, Path(tmpdir) / "report.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("comparisons", payload)
            self.assertIn("raw_runs", payload)


if __name__ == "__main__":
    unittest.main()
