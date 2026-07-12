import argparse
from pathlib import Path

from rcm.experiments.specialization_demo import build_demo_manifold
from rcm.reproducibility import normalize_seed
from rcm.reproducibility import seed_everything
from rcm_benchmarks import run_benchmark_suite
from rcm_benchmarks import write_benchmark_report


def manifold_factory(seed=None):
    resolved_seed = normalize_seed(seed)
    return build_demo_manifold(seed=resolved_seed)[1]


def default_output_path(seed: int):
    return Path("artifacts") / "benchmarks" / f"benchmark_report_seed_{seed}.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Recursive Cognitive Manifold benchmark suite.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic benchmark setup.")
    parser.add_argument(
        "--include-timestamp",
        action="store_true",
        help="Include generation timestamp in report metadata (disabled by default for reproducibility).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the JSON benchmark report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    resolved_seed = seed_everything(args.seed)
    output_path = args.output or default_output_path(resolved_seed)
    report = run_benchmark_suite(manifold_factory, seed=resolved_seed, include_timestamp=args.include_timestamp)
    written_path = write_benchmark_report(report, output_path)
    print(f"Wrote benchmark report to {written_path}")
    print(f"Topology recovery restored edge ratio: {report['benchmarks']['topology_recovery']['restored_edge_ratio']:.3f}")


if __name__ == "__main__":
    main()