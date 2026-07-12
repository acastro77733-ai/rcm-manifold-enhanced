from __future__ import annotations

import argparse
from pathlib import Path

from rcm.experiments.multi_seed_evaluation import run_paired_multi_seed_evaluation, write_multi_seed_report


def parse_args():
    parser = argparse.ArgumentParser(description="Run the paired multi-seed primary HRM evaluation.")
    parser.add_argument("--task", default="temporal_forecasting", help="Registered mechanism task name.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed in the paired evaluation schedule.")
    parser.add_argument("--n-seeds", type=int, default=30, help="Number of paired seeds to evaluate.")
    parser.add_argument("--graph-size", type=int, default=4, help="Graph size used for the forecast evaluation split.")
    parser.add_argument("--state-dim", type=int, default=4, help="State dimensionality for the evaluated models.")
    parser.add_argument("--primary-condition", default="full_rcm", help="Primary condition to compare against controls.")
    parser.add_argument(
        "--controls",
        nargs="*",
        default=["no_hrm", "no_hierarchy", "fixed_topology", "memory_only", "linear_autoregression"],
        help="Control conditions to compare against the primary condition.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/releases/primary_multi_seed_report.json"),
        help="Output path for the serialized multi-seed evaluation report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_paired_multi_seed_evaluation(
        task_name=args.task,
        seed_start=args.seed_start,
        n_seeds=args.n_seeds,
        graph_size=args.graph_size,
        state_dim=args.state_dim,
        primary_condition=args.primary_condition,
        control_conditions=list(args.controls),
    )
    written = write_multi_seed_report(report, args.output)
    print(f"Wrote primary evaluation report to {written}")


if __name__ == "__main__":
    main()
