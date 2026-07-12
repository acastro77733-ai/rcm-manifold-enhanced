import json
from pathlib import Path
import sys


def dotted_get(payload, dotted_path):
    current = payload
    for part in dotted_path.split('.'):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing path component '{part}' in '{dotted_path}'")
        current = current[part]
    return current


def main():
    baseline_path = Path("benchmarks/ci_baseline_seed_42.json")
    report_path = Path("artifacts/benchmarks/ci_benchmark_report.json")

    baseline = json.loads(baseline_path.read_text(encoding="ascii"))
    report = json.loads(report_path.read_text(encoding="ascii"))

    expected_seed = int(baseline.get("seed", 42))
    actual_seed = int(report["metadata"]["seed"])
    if actual_seed != expected_seed:
        print(f"Seed mismatch: current={actual_seed} baseline={expected_seed}")
        return 1

    failures = []
    metrics = baseline.get("metrics", {})
    for label, spec in metrics.items():
        path = spec["path"]
        baseline_value = float(spec["baseline"])
        max_regression = float(spec.get("max_regression", 0.0))
        direction = spec.get("direction", "higher_is_better")
        current_value = float(dotted_get(report, path))
        difference = current_value - baseline_value

        print(label)
        print(f"Current\n{current_value:.6f}")
        print(f"Baseline\n{baseline_value:.6f}")
        print(f"Difference\n{difference:+.6f}")

        if direction == "higher_is_better":
            if current_value < baseline_value - max_regression:
                failures.append(
                    f"{label} regressed: current={current_value:.6f}, baseline={baseline_value:.6f}, allowed_drop={max_regression:.6f}"
                )
        elif direction == "lower_is_better":
            if current_value > baseline_value + max_regression:
                failures.append(
                    f"{label} regressed: current={current_value:.6f}, baseline={baseline_value:.6f}, allowed_increase={max_regression:.6f}"
                )
        else:
            failures.append(f"Unsupported direction '{direction}' for metric '{label}'")

    if failures:
        print("\nBaseline regression check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nBaseline regression check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
