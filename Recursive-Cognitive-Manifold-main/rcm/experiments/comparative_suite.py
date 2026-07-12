from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.config.dynamics import StateDynamicsConfig
from rcm.config.field import FieldConfig
from rcm.config.representation import RepresentationConfig
from rcm.config.stability import StabilityConfig
from rcm.config.topology import TopologyConfig
from rcm.metrics import ManifoldMetrics
from rcm.reproducibility import normalize_seed, seed_everything


def compute_control_objective(metrics: ManifoldMetrics) -> float:
    return float(
        0.45 * metrics.task_score
        + 0.25 * metrics.recall_score
        + 0.20 * metrics.coherence
        - 0.20 * metrics.collapse_energy
        - 0.05 * metrics.topology_cost
        - 0.05 * metrics.representation_cost
        - 0.05 * metrics.field_energy
    )


def _build_configuration(name: str, *, adaptive: bool, nonlinear_field: bool, collapse_recovery: bool, seed: int) -> RecursiveCognitiveManifold:
    dynamics_config = StateDynamicsConfig()
    topology_config = TopologyConfig()
    field_config = FieldConfig(field_model="nonlinear" if nonlinear_field else "legacy")
    stability_config = StabilityConfig() if collapse_recovery else StabilityConfig(collapse_threshold=10.0)
    representation_config = RepresentationConfig()
    if not adaptive:
        representation_config = RepresentationConfig(min_dim=2, max_dim=2, minimum_improvement=999.0, cooldown_steps=999)

    manifold = RecursiveCognitiveManifold(
        state_dim=4,
        level=0,
        label=name,
        hrm_seed=seed,
        dynamics_config=dynamics_config,
        topology_config=topology_config,
        field_config=field_config,
        stability_config=stability_config,
        representation_config=representation_config,
    )
    manifold.add_node(0, np.array([0.2, 0.0, 0.0, 0.0], dtype=float))
    manifold.add_node(1, np.array([0.1, 0.1, 0.0, 0.0], dtype=float))
    manifold.connect(0, 1, strength=0.8, latency=1.0)
    return manifold


def _run_configuration(manifold: RecursiveCognitiveManifold, steps: int) -> tuple[dict[str, Any], ManifoldMetrics]:
    for step in range(steps):
        manifold.step({0: np.array([0.1, 0.05, 0.0, 0.0], dtype=float), 1: np.array([0.05, 0.08, 0.0, 0.0], dtype=float)})
    return manifold.snapshot(), manifold.metrics


def bootstrap_confidence_interval(values: list[float] | np.ndarray, *, n_boot: int = 200, seed: int = 0) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True)
    means = samples.mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {"mean": float(arr.mean()), "lower": float(lower), "upper": float(upper)}


def run_comparative_suite(
    seed: int = 42,
    steps: int = 6,
    repeat_count: int = 3,
    seed_count: int = 4,
    trajectory_count: int = 4,
    output_dir: str | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    configs = [
        ("fixed_baseline", False, False, False),
        ("current_baseline", True, False, False),
        ("grounded_hrm_no_adaptive", True, True, False),
        ("full_grounded_hrm", True, True, True),
    ]
    results = []
    artifact_bundle = {"runs": []}

    for config_idx, (name, adaptive, nonlinear_field, collapse_recovery) in enumerate(configs):
        per_seed_runs = []
        for seed_idx in range(seed_count):
            seed_value = normalize_seed(seed + config_idx * 100 + seed_idx)
            objective_values = []
            delta_values = []
            for trajectory_idx in range(trajectory_count):
                manifold = _build_configuration(
                    name,
                    adaptive=adaptive,
                    nonlinear_field=nonlinear_field,
                    collapse_recovery=collapse_recovery,
                    seed=normalize_seed(seed_value + trajectory_idx),
                )
                _, metrics = _run_configuration(manifold, steps=steps)
                objective = compute_control_objective(metrics)
                objective_values.append(objective)
                baseline_seed = normalize_seed(seed + seed_idx)
                baseline_manifold = _build_configuration(
                    "baseline",
                    adaptive=False,
                    nonlinear_field=False,
                    collapse_recovery=False,
                    seed=baseline_seed,
                )
                _, baseline_metrics = _run_configuration(baseline_manifold, steps=steps)
                baseline_objective = compute_control_objective(baseline_metrics)
                delta_values.append(objective - baseline_objective)
                artifact_bundle["runs"].append(
                    {
                        "configuration": name,
                        "seed": int(seed_value),
                        "trajectory": trajectory_idx,
                        "control_objective": objective,
                        "value_delta": objective - baseline_objective,
                        "metrics": metrics.__dict__,
                    }
                )
            per_seed_runs.append(
                {
                    "seed": int(seed_value),
                    "objective_values": objective_values,
                    "delta_values": delta_values,
                    "mean_control_objective": float(np.mean(objective_values)) if objective_values else 0.0,
                    "mean_value_delta": float(np.mean(delta_values)) if delta_values else 0.0,
                    "ci95": bootstrap_confidence_interval(delta_values, seed=seed_value),
                }
            )

        aggregated_objectives = [entry["mean_control_objective"] for entry in per_seed_runs]
        aggregated_deltas = [entry["mean_value_delta"] for entry in per_seed_runs]
        results.append(
            {
                "name": name,
                "adaptive": adaptive,
                "nonlinear_field": nonlinear_field,
                "collapse_recovery": collapse_recovery,
                "control_objective": float(np.mean(aggregated_objectives)) if aggregated_objectives else 0.0,
                "value_delta": float(np.mean(aggregated_deltas)) if aggregated_deltas else 0.0,
                "value_delta_ci95": bootstrap_confidence_interval(aggregated_deltas, seed=seed + 1000),
                "repeat_count": repeat_count,
                "seed_count": seed_count,
                "trajectory_count": trajectory_count,
                "runs": per_seed_runs,
            }
        )

    report = {
        "seed": seed,
        "steps": steps,
        "configurations": results,
        "summary": {
            "best": max(results, key=lambda item: item["control_objective"]),
            "worst": min(results, key=lambda item: item["control_objective"]),
        },
        "artifacts": {"run_manifest": "comparative_suite.json", "run_details": "run_details.json"},
    }
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "comparative_suite.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (output_path / "run_details.json").write_text(json.dumps(artifact_bundle, indent=2), encoding="utf-8")
    return report
