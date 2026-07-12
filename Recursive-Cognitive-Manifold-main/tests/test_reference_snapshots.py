import json
import tempfile
import unittest
from pathlib import Path

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.release_artifacts import write_release_artifacts


class ReferenceSnapshotTests(unittest.TestCase):
    def test_release_artifacts_write_reference_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            manifest = write_release_artifacts(output_dir=output_dir, tag="v0.1.0-baseline")
            snapshot_path = output_dir / "reference_snapshots" / "baseline_snapshot.json"
            self.assertTrue(snapshot_path.exists())
            payload = json.loads(snapshot_path.read_text(encoding="ascii"))
            self.assertEqual(payload["seed"], 42)
            self.assertIn("snapshot", payload)


if __name__ == "__main__":
    unittest.main()
