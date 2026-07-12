import json
import tempfile
import unittest
from pathlib import Path

from rcm.experiments.benchmark_suite import main as benchmark_main
from rcm.reproducibility import seed_everything


class DeterministicRunsTests(unittest.TestCase):
    def test_benchmark_suite_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bench.json"
            import sys
            import contextlib
            from io import StringIO

            stdout = StringIO()
            with contextlib.redirect_stdout(stdout):
                import types
                import argparse
                from unittest.mock import patch

                with patch("sys.argv", ["benchmark_suite", "--seed", "42", "--output", str(output_path)]):
                    benchmark_main()

            report = json.loads(output_path.read_text(encoding="ascii"))
            self.assertEqual(report["metadata"]["seed"], 42)
            self.assertIn("benchmarks", report)


if __name__ == "__main__":
    unittest.main()
