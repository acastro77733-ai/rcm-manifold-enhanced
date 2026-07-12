import json
import tempfile
import unittest
from pathlib import Path

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.release_artifacts import build_module_map, capture_environment_report, write_release_artifacts


class ReleaseArtifactsTests(unittest.TestCase):
    def test_write_release_artifacts_creates_manifest_and_reference_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            manifest = write_release_artifacts(output_dir=output_dir, tag="v0.1.0-baseline")

            snapshot_path = output_dir / "reference_snapshots" / "baseline_snapshot.json"
            manifest_path = output_dir / "baseline_release.json"
            experiment_manifest_path = output_dir / "hrm_experiment_manifest.json"
            assessment_path = output_dir / "baseline_assessment.json"
            self.assertTrue(snapshot_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(experiment_manifest_path.exists())
            self.assertTrue(assessment_path.exists())

            payload = json.loads(manifest_path.read_text(encoding="ascii"))
            self.assertEqual(payload["release"]["tag"], "v0.1.0-baseline")
            self.assertIn("reference_snapshot", payload["artifacts"])
            self.assertIn("experiment_manifest", payload["artifacts"])
            self.assertIn("baseline_assessment", payload["artifacts"])

    def test_write_release_artifacts_indexes_primary_report_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            primary_report = output_dir / "primary_multi_seed_report.json"
            primary_report.write_text("{}\n", encoding="ascii")
            manifest = write_release_artifacts(output_dir=output_dir, tag="v0.1.0-baseline")
            self.assertIn("primary_multi_seed_report", manifest["artifacts"])

    def test_build_module_map_distinguishes_public_and_experimental(self):
        module_map = build_module_map()
        self.assertIn("rcm.memory", module_map["public"])
        self.assertIn("rcm.experiments", module_map["experimental"])
        self.assertIn("rcm.cognition", module_map["public"])

    def test_capture_environment_report_includes_platform_and_dependencies(self):
        report = capture_environment_report()
        self.assertIn("python_version", report)
        self.assertIn("platform", report)
        self.assertIn("dependencies", report)
        self.assertTrue(report["dependencies"])


if __name__ == "__main__":
    unittest.main()
