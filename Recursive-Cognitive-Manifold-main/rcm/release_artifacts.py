from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.experiments.multi_seed_evaluation import build_experiment_manifest
from rcm.reproducibility import seed_everything


PUBLIC_MODULES = (
    "rcm.cognition",
    "rcm.dynamics",
    "rcm.geometry",
    "rcm.hierarchy",
    "rcm.memory",
    "rcm.reproducibility",
)

EXPERIMENTAL_MODULES = (
    "rcm.experiments",
    "rcm_benchmarks",
    "rcm_collapse_integration",
    "rcm_persistence",
    "rcm_perturbation_testing",
    "rcm.release_artifacts",
)


def build_module_map() -> dict[str, list[str]]:
    return {
        "public": list(PUBLIC_MODULES),
        "experimental": list(EXPERIMENTAL_MODULES),
    }


def _json_safe(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
        return _json_safe(value.__dict__)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _baseline_reference_snapshot(seed: int = 42) -> dict[str, Any]:
    seed_everything(seed)
    manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="baseline", hrm_seed=seed)
    manifold.add_node(0, np.array([0.8, 0.1, 0.0, 0.0], dtype=float))
    manifold.add_node(1, np.array([0.0, 0.9, 0.1, 0.0], dtype=float))
    manifold.connect(0, 1, strength=0.7, latency=1.0)
    for _ in range(3):
        manifold.step({0: np.array([0.9, 0.1, 0.0, 0.0], dtype=float)})
    snapshot = manifold.snapshot()
    return _json_safe({
        "seed": seed,
        "snapshot": snapshot,
        "node_states": {node_id: np.asarray(node.local_state, dtype=float).tolist() for node_id, node in manifold.nodes.items()},
        "edge_strengths": {edge_key: float(edge.strength) for edge_key, edge in manifold.edges.items()},
    })


def _write_dependency_lock(output_dir: Path, dependencies: dict[str, str | None]) -> Path:
    lock_path = output_dir / "requirements.lock.txt"
    lines = ["# Generated for reproducible release validation", ""]
    for package, version in dependencies.items():
        if version is None:
            continue
        lines.append(f"{package}=={version}")
    lock_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return lock_path


def capture_environment_report() -> dict[str, Any]:
    deps = {}
    for package in ("numpy", "scipy", "matplotlib"):
        try:
            deps[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            deps[package] = None
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "dependencies": deps,
        "dependency_reproduction": {
            "requirements_source": "requirements.txt",
            "resolved_versions": deps,
        },
    }


def write_release_artifacts(output_dir: str | Path | None = None, tag: str = "v0.1.0-baseline") -> dict[str, Any]:
    output_dir = Path(output_dir or "artifacts/releases")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_snapshot = _baseline_reference_snapshot()
    snapshot_dir = output_dir / "reference_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "baseline_snapshot.json"
    snapshot_path.write_text(json.dumps(reference_snapshot, indent=2, sort_keys=True) + "\n", encoding="ascii")

    module_map = build_module_map()
    environment_report = capture_environment_report()
    experiment_manifest = build_experiment_manifest(
        task_name="temporal_forecasting",
        n_seeds=30,
        primary_condition="full_rcm",
        control_conditions=["no_hrm", "no_hierarchy", "fixed_topology", "memory_only", "linear_autoregression"],
    )
    assessment = {
        "operational_implementation": "Governed state, field controls, hierarchy feedback, topology transactions, forecasting contract, and paired evaluation harness are implemented.",
        "causal_mechanism_activation": "RCM/HRM, no-guidance, no-HRM, fixed-topology, no-hierarchy, memory-only, and fixed-graph controls map to explicit runtime pathway toggles or field modes.",
        "task_benefit": "Benefit is measured through forecast errors, paired deltas, and compute-normalized summaries, but the full 30-seed primary study has not been executed in release generation.",
        "statistical_validation": "Bootstrap confidence intervals and paired effect sizes are emitted by the multi-seed harness.",
        "unresolved_limitations": "Numerical convergence and precision tests are present, but CI regression gates and a full production-scale experiment run still need to be wired into release automation.",
    }
    lock_path = _write_dependency_lock(output_dir, environment_report["dependencies"])
    manifest_path = output_dir / "hrm_experiment_manifest.json"
    manifest_path.write_text(json.dumps(experiment_manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    assessment_path = output_dir / "baseline_assessment.json"
    assessment_path.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n", encoding="ascii")
    manifest = {
        "release": {
            "tag": tag,
            "created_with": "rcm.release_artifacts.write_release_artifacts",
        },
        "artifacts": {
            "reference_snapshot": snapshot_path.name,
            "module_map": "module_map.json",
            "environment_report": "environment_report.json",
            "dependency_lock": lock_path.name,
            "experiment_manifest": manifest_path.name,
            "baseline_assessment": assessment_path.name,
        },
        "module_map": module_map,
        "environment": environment_report,
        "assessment": assessment,
    }

    primary_report_path = output_dir / "primary_multi_seed_report.json"
    if primary_report_path.exists():
        manifest["artifacts"]["primary_multi_seed_report"] = primary_report_path.name

    (output_dir / "module_map.json").write_text(json.dumps(module_map, indent=2, sort_keys=True) + "\n", encoding="ascii")
    (output_dir / "environment_report.json").write_text(json.dumps(environment_report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    (output_dir / "baseline_release.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return manifest
