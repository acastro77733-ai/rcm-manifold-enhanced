from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
import tracemalloc
from typing import Any

import numpy as np
from scipy import stats

from rcm.experiments.forecasting import build_dataset_split, build_forecast_model, evaluate_forecast_model


PRIMARY_CONDITIONS = ["full_rcm", "no_hrm", "no_hierarchy", "fixed_topology", "memory_only", "linear_autoregression"]


@dataclass(frozen=True)
class PrimaryMetricSpec:
    name: str
    direction: str
    success_threshold: float


PRIMARY_METRICS = {
    "test_mean_error": PrimaryMetricSpec("test_mean_error", "lower-better", 0.20),
}


def run_paired_multi_seed_evaluation(
    *,
    task_name: str,
    seed_start: int = 0,
    n_seeds: int = 30,
    graph_size: int = 4,
    state_dim: int = 4,
    primary_condition: str = "full_rcm",
    control_conditions: list[str] | None = None,
) -> dict[str, Any]:
    control_conditions = control_conditions or [condition for condition in PRIMARY_CONDITIONS if condition != primary_condition]
    raw_runs = []
    summaries = {}
    comparisons = {}

    for condition in [primary_condition] + list(control_conditions):
        condition_runs = []
        for offset in range(n_seeds):
            seed = seed_start + offset
            split = build_dataset_split(task_name, graph_size=graph_size, seed=seed, state_dim=state_dim)
            model = build_forecast_model(condition, seed=seed, graph_size=graph_size, state_dim=state_dim)
            tracemalloc.start()
            tick = time.perf_counter()
            report = evaluate_forecast_model(model, split)
            runtime = time.perf_counter() - tick
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            op_count = int(model.prediction_calls + model.update_calls)
            compute_normalized = (1.0 / max(report["test_mean_error"], 1e-12)) / max(op_count, 1)
            run_record = {
                "seed": seed,
                "condition": condition,
                "test_mean_error": float(report["test_mean_error"]),
                "validation_mean_error": float(report["validation_mean_error"]),
                "runtime_seconds": float(runtime),
                "peak_memory_bytes": int(peak),
                "op_count": op_count,
                "compute_normalized_performance": float(compute_normalized),
                "failed": bool(not np.isfinite(report["test_mean_error"])),
            }
            condition_runs.append(run_record)
            raw_runs.append(run_record)
        summaries[condition] = summarize_condition_runs(condition_runs)

    primary_values = np.asarray([run["test_mean_error"] for run in summaries[primary_condition]["raw_runs"]], dtype=float)
    raw_p_values = {}
    for control in control_conditions:
        control_values = np.asarray([run["test_mean_error"] for run in summaries[control]["raw_runs"]], dtype=float)
        diffs = primary_values - control_values
        comparison_key = f"{primary_condition}_vs_{control}"
        raw_p_value = paired_p_value(diffs)
        wilcoxon_p = wilcoxon_signed_rank_p_value(diffs)
        sign_p = sign_test_p_value(diffs)
        raw_p_values[comparison_key] = raw_p_value
        comparisons[comparison_key] = {
            "paired_differences": diffs.tolist(),
            "mean_difference": float(np.mean(diffs)) if diffs.size else 0.0,
            "median_difference": float(np.median(diffs)) if diffs.size else 0.0,
            "dispersion": float(np.std(diffs)) if diffs.size else 0.0,
            "effect_size": paired_effect_size(diffs),
            "raw_p_value": raw_p_value,
            "wilcoxon_signed_rank_p_value": wilcoxon_p,
            "sign_test_p_value": sign_p,
            "bootstrap_ci": bootstrap_confidence_interval(diffs),
            "trimmed_mean_difference": trimmed_mean(diffs, proportion_to_cut=0.1),
            "hodges_lehmann_estimate": hodges_lehmann_estimate(diffs),
            "failure_rate_difference": float(summaries[primary_condition]["failure_rate"] - summaries[control]["failure_rate"]),
        }

    corrected = holm_bonferroni(raw_p_values)
    bonferroni = bonferroni_correction(raw_p_values)
    benjamini_hochberg_values = benjamini_hochberg(raw_p_values)
    for comparison_key, corrected_p in corrected.items():
        comparisons[comparison_key]["corrected_p_value"] = corrected_p
        comparisons[comparison_key]["holm_bonferroni_p_value"] = corrected_p
        comparisons[comparison_key]["bonferroni_p_value"] = bonferroni[comparison_key]
        comparisons[comparison_key]["benjamini_hochberg_q_value"] = benjamini_hochberg_values[comparison_key]
        spec = PRIMARY_METRICS["test_mean_error"]
        comparisons[comparison_key]["meets_primary_success_threshold"] = bool(
            summaries[primary_condition]["mean"] <= spec.success_threshold
        )

    manifest = build_experiment_manifest(task_name=task_name, n_seeds=n_seeds, primary_condition=primary_condition, control_conditions=control_conditions)
    return {
        "manifest": manifest,
        "raw_runs": raw_runs,
        "summaries": summaries,
        "comparisons": comparisons,
    }


def summarize_condition_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([run["test_mean_error"] for run in runs], dtype=float)
    runtimes = np.asarray([run["runtime_seconds"] for run in runs], dtype=float)
    memories = np.asarray([run["peak_memory_bytes"] for run in runs], dtype=float)
    ops = np.asarray([run["op_count"] for run in runs], dtype=float)
    normalized = np.asarray([run["compute_normalized_performance"] for run in runs], dtype=float)
    failure_rate = float(np.mean([run["failed"] for run in runs])) if runs else 0.0
    return {
        "raw_runs": runs,
        "mean": float(np.mean(values)) if values.size else 0.0,
        "median": float(np.median(values)) if values.size else 0.0,
        "std": float(np.std(values)) if values.size else 0.0,
        "bootstrap_ci": bootstrap_confidence_interval(values),
        "failure_rate": failure_rate,
        "runtime_seconds": {
            "mean": float(np.mean(runtimes)) if runtimes.size else 0.0,
            "max": float(np.max(runtimes)) if runtimes.size else 0.0,
        },
        "peak_memory_bytes": {
            "mean": float(np.mean(memories)) if memories.size else 0.0,
            "max": float(np.max(memories)) if memories.size else 0.0,
        },
        "op_count": {
            "mean": float(np.mean(ops)) if ops.size else 0.0,
            "max": float(np.max(ops)) if ops.size else 0.0,
        },
        "compute_normalized_performance": {
            "mean": float(np.mean(normalized)) if normalized.size else 0.0,
            "median": float(np.median(normalized)) if normalized.size else 0.0,
        },
        "primary_success_threshold": float(PRIMARY_METRICS["test_mean_error"].success_threshold),
        "meets_primary_success_threshold": bool((float(np.mean(values)) if values.size else float("inf")) <= PRIMARY_METRICS["test_mean_error"].success_threshold),
    }


def bootstrap_confidence_interval(values: np.ndarray, n_boot: int = 400, seed: int = 0) -> list[float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_effect_size(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    if differences.size == 0:
        return 0.0
    denom = float(np.std(differences))
    if denom == 0.0:
        return 0.0
    return float(np.mean(differences) / denom)


def paired_p_value(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    if differences.size <= 1:
        return 1.0
    result = stats.ttest_1samp(differences, popmean=0.0, alternative="two-sided")
    p_value = getattr(result, "pvalue", 1.0)
    if not np.isfinite(p_value):
        return 1.0
    return float(p_value)


def wilcoxon_signed_rank_p_value(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    if differences.size <= 1 or np.allclose(differences, 0.0):
        return 1.0
    try:
        result = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox", correction=False)
    except ValueError:
        return 1.0
    p_value = getattr(result, "pvalue", 1.0)
    if not np.isfinite(p_value):
        return 1.0
    return float(p_value)


def sign_test_p_value(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    positives = int(np.sum(differences > 0.0))
    negatives = int(np.sum(differences < 0.0))
    trials = positives + negatives
    if trials == 0:
        return 1.0
    result = stats.binomtest(k=positives, n=trials, p=0.5, alternative="two-sided")
    p_value = getattr(result, "pvalue", 1.0)
    if not np.isfinite(p_value):
        return 1.0
    return float(p_value)


def holm_bonferroni(raw_p_values: dict[str, float]) -> dict[str, float]:
    if not raw_p_values:
        return {}
    ordered = sorted(raw_p_values.items(), key=lambda item: item[1])
    corrected = {}
    total = len(ordered)
    running_max = 0.0
    for index, (key, p_value) in enumerate(ordered):
        adjusted = min(1.0, (total - index) * float(p_value))
        running_max = max(running_max, adjusted)
        corrected[key] = running_max
    return corrected


def bonferroni_correction(raw_p_values: dict[str, float]) -> dict[str, float]:
    total = max(len(raw_p_values), 1)
    return {key: min(1.0, total * float(value)) for key, value in raw_p_values.items()}


def benjamini_hochberg(raw_p_values: dict[str, float]) -> dict[str, float]:
    if not raw_p_values:
        return {}
    ordered = sorted(raw_p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted = {}
    running_min = 1.0
    for index in range(total - 1, -1, -1):
        key, p_value = ordered[index]
        rank = index + 1
        candidate = min(1.0, (total / rank) * float(p_value))
        running_min = min(running_min, candidate)
        adjusted[key] = running_min
    return adjusted


def trimmed_mean(values: np.ndarray, proportion_to_cut: float = 0.1) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(stats.trim_mean(values, proportiontocut=proportion_to_cut))


def hodges_lehmann_estimate(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    pairwise = []
    for index, left in enumerate(values):
        for right in values[index:]:
            pairwise.append(0.5 * (float(left) + float(right)))
    return float(np.median(np.asarray(pairwise, dtype=float))) if pairwise else 0.0


def build_experiment_manifest(*, task_name: str, n_seeds: int, primary_condition: str, control_conditions: list[str]) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "primary_metric": PRIMARY_METRICS["test_mean_error"].__dict__,
        "n_seeds": int(n_seeds),
        "primary_condition": primary_condition,
        "control_conditions": list(control_conditions),
        "forecast_contract": ["fit", "observe", "predict", "update", "reset"],
        "evaluation_order": "predict_before_update",
        "notes": {
            "operational_implementation": "Implemented for the forecast contract and paired evaluation harness.",
            "causal_mechanism_activation": "Ablation conditions map to runtime pathway toggles or field modes.",
            "task_benefit": "Reported through paired test error deltas and compute-normalized performance.",
            "statistical_validation": "Bootstrap confidence intervals, paired t-tests, Wilcoxon signed-rank tests, sign tests, and multiple-comparison corrections are emitted.",
            "unresolved_limitations": "Nonparametric confidence intervals and more specialized robust paired estimators are not emitted automatically in unit tests.",
        },
    }


def write_experiment_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_multi_seed_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
