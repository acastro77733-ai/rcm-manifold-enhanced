from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np

import rcm.cognition.manifold as manifold_module
from rcm.reproducibility import seed_everything
from rcm_perturbation_testing import NoiseType
from rcm_perturbation_testing import RCMPerturbationLab
from rcm_perturbation_testing import capture_state
from rcm_perturbation_testing import compare_to_baseline
from rcm_perturbation_testing import recovery_score


def _state_similarity(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _instantiate_manifold(factory, seed: int):
    resolved_seed = seed_everything(seed)
    try:
        instance = factory(seed=resolved_seed)
    except TypeError:
        instance = factory()
    if isinstance(instance, tuple):
        return instance[-1]
    return instance


def _long_term_recall(manifold, node_id: int, cue):
    node = manifold.nodes[node_id]
    best_state = None
    best_similarity = -1.0
    for _, stored_state, _ in node.long_term_memory:
        similarity = _state_similarity(cue, stored_state)
        if similarity > best_similarity:
            best_similarity = similarity
            best_state = stored_state.copy()
    if best_state is None:
        return np.zeros_like(cue, dtype=float)
    return best_state


def memory_retention(factory, seed: int, stimulus, node_id: int, encode_steps: int, delay_steps: int):
    manifold = _instantiate_manifold(factory, seed)
    stimulus = np.asarray(stimulus, dtype=float)
    for _ in range(encode_steps):
        manifold.step({node_id: stimulus})
    encoded_state = manifold.nodes[node_id].local_state.copy()
    for _ in range(delay_steps):
        manifold.step()
    retained_state = manifold.nodes[node_id].local_state.copy()
    encoded_norm = np.linalg.norm(encoded_state)
    retained_norm = np.linalg.norm(retained_state)
    return {
        "encoded_state": encoded_state,
        "retained_state": retained_state,
        "state_similarity": _state_similarity(encoded_state, retained_state),
        "magnitude_retention": float(retained_norm / encoded_norm) if encoded_norm > 0.0 else 0.0,
    }


def recall_accuracy(factory, seed: int, cue, target, node_id: int = 0, encode_steps: int = 4, recall_fn=None):
    manifold = _instantiate_manifold(factory, seed)
    cue = np.asarray(cue, dtype=float)
    target = np.asarray(target, dtype=float)
    for _ in range(encode_steps):
        manifold.step({node_id: target})
    if recall_fn is None:
        recalled_state = _long_term_recall(manifold, node_id, cue)
    else:
        recalled_state = np.asarray(recall_fn(manifold, node_id, cue), dtype=float)
    return {
        "recalled_state": recalled_state,
        "state_similarity": _state_similarity(recalled_state, target),
    }


def adaptation_speed(factory, seed: int, target, threshold: float = 0.95, max_steps: int = 50, node_id: int = 0):
    manifold = _instantiate_manifold(factory, seed)
    target = np.asarray(target, dtype=float)
    steps_to_threshold = None
    final_similarity = 0.0
    for step_index in range(1, max_steps + 1):
        manifold.step({node_id: target})
        final_similarity = _state_similarity(manifold.nodes[node_id].local_state, target)
        if final_similarity >= threshold and steps_to_threshold is None:
            steps_to_threshold = step_index
            break
    if steps_to_threshold is None:
        steps_to_threshold = max_steps
    return {
        "steps_to_threshold": steps_to_threshold,
        "final_similarity": final_similarity,
    }


def catastrophic_interference(factory, seed: int, task_a, task_b, node_a: int = 0, node_b: int = 3, encode_steps: int = 4):
    manifold = _instantiate_manifold(factory, seed)
    task_a = np.asarray(task_a, dtype=float)
    task_b = np.asarray(task_b, dtype=float)
    for _ in range(encode_steps):
        manifold.step({node_a: task_a})
    task_a_reference = manifold.nodes[node_a].local_state.copy()
    for _ in range(encode_steps):
        manifold.step({node_b: task_b})
    return {
        "task_a_retention": _state_similarity(manifold.nodes[node_a].local_state, task_a_reference),
        "task_b_learning": _state_similarity(manifold.nodes[node_b].local_state, task_b),
    }


def _group_f1_score(expected_group, actual_group):
    expected = set(expected_group)
    actual = set(actual_group)
    if not expected and not actual:
        return 1.0
    overlap = len(expected & actual)
    precision = overlap / max(len(actual), 1)
    recall = overlap / max(len(expected), 1)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def region_formation(factory, seed: int, stimulation_groups, steps: int = 8):
    manifold = _instantiate_manifold(factory, seed)
    for _ in range(steps):
        external_input = {}
        for group, signal in stimulation_groups.items():
            for node_id in group:
                external_input[node_id] = np.asarray(signal, dtype=float)
        manifold.step(external_input)

    actual_groups = [tuple(sorted(int(node_id) for node_id in signature)) for signature in manifold.regions.keys()]
    f1_scores = []
    for expected_group in stimulation_groups.keys():
        expected_group = tuple(sorted(expected_group))
        best = 0.0
        for actual_group in actual_groups:
            best = max(best, _group_f1_score(expected_group, actual_group))
        f1_scores.append(best)

    stabilities = [region.stability for region in manifold.regions.values()]
    return {
        "partition_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "mean_region_stability": float(np.mean(stabilities)) if stabilities else 0.0,
        "region_count": len(manifold.regions),
    }


def hierarchy_utility(factory, seed: int, task_fn):
    hierarchical_manifold = _instantiate_manifold(factory, seed)
    hierarchical_task_score = float(task_fn(hierarchical_manifold))

    flat_manifold = _instantiate_manifold(factory, seed)
    original_refresh = manifold_module.refresh_child_manifolds
    manifold_module.refresh_child_manifolds = lambda manifold: manifold.child_manifolds.clear()
    try:
        flat_task_score = float(task_fn(flat_manifold))
    finally:
        manifold_module.refresh_child_manifolds = original_refresh

    return {
        "hierarchical_task_score": hierarchical_task_score,
        "flat_task_score": flat_task_score,
        "utility_delta": hierarchical_task_score - flat_task_score,
    }


def noise_recovery(factory, seed: int, noise_std: float = 0.4, settle_steps: int = 8, recovery_steps: int = 20):
    manifold = _instantiate_manifold(factory, seed)
    for _ in range(settle_steps):
        manifold.step()
    baseline = capture_state(manifold)
    perturbation_lab = RCMPerturbationLab(seed=seed)
    perturbation_lab.perturb_node_states(manifold, list(manifold.nodes.keys()), NoiseType.GAUSSIAN, noise_std)
    damaged = compare_to_baseline(manifold, baseline)
    for _ in range(recovery_steps):
        manifold.step()
    recovered = compare_to_baseline(manifold, baseline)
    return {
        "damaged_error": damaged["mean_state_error"],
        "recovered_error": recovered["mean_state_error"],
        "error_reduction": damaged["mean_state_error"] - recovered["mean_state_error"],
    }


def topology_recovery(factory, seed: int, remove_fraction: float = 0.25, recovery_steps: int = 20, repair_fn=None):
    manifold = _instantiate_manifold(factory, seed)
    baseline = capture_state(manifold)
    baseline_edges = list(manifold.edges.keys())
    if not baseline_edges:
        return {
            "damaged_edge_ratio": 1.0,
            "restored_edge_ratio": 1.0,
            "strength_similarity": 1.0,
        }

    rng = np.random.default_rng(seed)
    count = max(1, int(np.ceil(len(baseline_edges) * remove_fraction)))
    removed = rng.choice(len(baseline_edges), size=min(count, len(baseline_edges)), replace=False)
    removed_edges = [baseline_edges[index] for index in removed]
    perturbation_lab = RCMPerturbationLab(seed=seed)
    perturbation_lab.sever_edges(manifold, removed_edges, bidirectional=False)
    damaged_edge_ratio = len(manifold.edges) / max(len(baseline_edges), 1)

    for _ in range(recovery_steps):
        manifold.step()
        if repair_fn is not None:
            repair_fn(manifold, baseline_edges)
        elif hasattr(manifold, "attempt_topology_repair"):
            manifold.attempt_topology_repair()

    current_edges = set(manifold.edges.keys())
    baseline_edge_set = set(baseline_edges)
    restored_edge_ratio = len(current_edges & baseline_edge_set) / max(len(baseline_edge_set), 1)
    shared_edges = sorted(current_edges & baseline_edge_set)
    if shared_edges:
        mean_abs_error = float(np.mean([abs(manifold.edges[edge_key].strength - baseline["edge_strengths"][edge_key]) for edge_key in shared_edges]))
        strength_similarity = 1.0 / (1.0 + mean_abs_error)
    else:
        strength_similarity = 0.0

    return {
        "damaged_edge_ratio": damaged_edge_ratio,
        "restored_edge_ratio": restored_edge_ratio,
        "strength_similarity": strength_similarity,
    }


def _to_json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return value


def run_benchmark_suite(factory, seed: int, include_timestamp: bool = False):
    stimulus_a = np.array([1.0, 0.1, 0.0, 0.0], dtype=float)
    stimulus_b = np.array([0.0, 0.0, 1.0, 0.1], dtype=float)
    cue = np.array([0.8, 0.1, 0.0, 0.0], dtype=float)

    def hierarchy_task(manifold):
        for _ in range(4):
            manifold.step(
                {
                    0: stimulus_a,
                    1: np.array([0.96, 0.18, 0.0, 0.0], dtype=float),
                    3: stimulus_b,
                    4: np.array([0.0, 0.0, 0.92, 0.12], dtype=float),
                }
            )
        return len(manifold.child_manifolds) + np.mean([region.stability for region in manifold.regions.values()] or [0.0])

    def ablation_task(manifold):
        for _ in range(6):
            manifold.step({0: stimulus_a, 3: stimulus_b})
        region_stability = np.mean([region.stability for region in manifold.regions.values()] or [0.0])
        return region_stability + (0.5 * len(manifold.child_manifolds))

    def geometry_feedback_task(manifold):
        node_ids = sorted(manifold.nodes.keys())
        for _ in range(10):
            external_input = {}
            if node_ids:
                external_input[node_ids[0]] = stimulus_a
            if len(node_ids) > 1:
                external_input[node_ids[1]] = np.array([0.96, 0.18, 0.0, 0.0], dtype=float)
            if len(node_ids) > 2:
                external_input[node_ids[2]] = stimulus_b
            if len(node_ids) > 3:
                external_input[node_ids[3]] = np.array([0.0, 0.0, 0.92, 0.12], dtype=float)
            manifold.step(external_input)

        region_stability = float(np.mean([region.stability for region in manifold.regions.values()] or [0.0]))
        semantic_count = float(len(manifold.semantic_memory.snapshot()))
        child_count = float(len(manifold.child_manifolds))
        sync_pressure = float(getattr(manifold, "last_field_metrics", {}).get("synchronization_pressure", 0.0))
        return region_stability + (0.3 * semantic_count) + (0.5 * child_count) + (0.2 * sync_pressure)

    perturbation_lab = RCMPerturbationLab(seed=seed)

    phases = [
        AdversarialPhase(
            label="coherent",
            steps=3,
            input_fn=lambda _step, _manifold: {0: stimulus_a, 1: np.array([0.94, 0.16, 0.0, 0.0]), 3: stimulus_b},
        ),
        AdversarialPhase(
            label="noisy",
            steps=3,
            input_fn=lambda _step, _manifold: {0: cue, 4: np.array([0.0, 0.0, 0.8, 0.2])},
            perturb_fn=lambda _step, current_manifold: perturbation_lab.perturb_node_states(
                current_manifold,
                list(current_manifold.nodes.keys())[:2],
                NoiseType.GAUSSIAN,
                intensity=0.1,
            ),
        ),
    ]

    adversarial_runner = AdversarialRegionRunner(
        manifold=_instantiate_manifold(factory, seed),
        snapshot_fn=lambda manifold: {
            "region_count": len(manifold.regions),
            "child_manifolds": len(manifold.child_manifolds),
            "mean_region_stability": float(np.mean([region.stability for region in manifold.regions.values()])) if manifold.regions else 0.0,
        },
        phases=phases,
    )

    benchmarks = {
        "memory_retention": memory_retention(factory, seed, stimulus_a, 0, 4, 12),
        "recall_accuracy": recall_accuracy(factory, seed, cue, stimulus_a),
        "adaptation_speed": adaptation_speed(factory, seed, stimulus_a, threshold=0.95, max_steps=20),
        "catastrophic_interference": catastrophic_interference(factory, seed, stimulus_a, stimulus_b),
        "region_formation": region_formation(
            factory,
            seed,
            {
                (0, 1, 2): stimulus_a,
                (3, 4, 5): stimulus_b,
            },
            steps=8,
        ),
        "hierarchy_utility": hierarchy_utility(factory, seed, hierarchy_task),
        "noise_recovery": noise_recovery(factory, seed, noise_std=0.4, settle_steps=8, recovery_steps=20),
        "topology_recovery": topology_recovery(factory, seed, remove_fraction=0.25, recovery_steps=20),
        "computational_cost": computational_cost(factory, seed, steps=100),
        "ablation_sensitivity": ablation_sensitivity(factory, seed, ablation_task),
        "geometry_feedback_gain": geometry_feedback_gain(factory, seed, geometry_feedback_task),
        "adversarial_regions": adversarial_runner.run(),
    }

    metadata = {
        "seed": int(seed),
    }
    if include_timestamp:
        metadata["generated_at"] = datetime.now(timezone.utc).isoformat()

    return {
        "metadata": metadata,
        "benchmarks": benchmarks,
    }


def write_benchmark_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _to_json_safe(report)
    output_path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return output_path


def computational_cost(factory, seed: int, steps: int = 100):
    manifold = _instantiate_manifold(factory, seed)
    latencies = []
    tracemalloc.start()
    start = time.perf_counter()
    for _ in range(steps):
        tick = time.perf_counter()
        manifold.step()
        latencies.append(time.perf_counter() - tick)
    total = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    graph_elements = max(len(manifold.nodes) + len(manifold.edges), 1)
    return {
        "mean_step_seconds": float(np.mean(latencies)) if latencies else 0.0,
        "p95_step_seconds": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "steps_per_second": steps / total if total > 0.0 else 0.0,
        "peak_memory_bytes": int(peak),
        "seconds_per_graph_element": total / (steps * graph_elements) if steps > 0 else 0.0,
    }


def _disable_semantic_hierarchy(manifold):
    manifold.semantic_memory.observe = lambda current_manifold: None
    original_refresh = manifold_module.refresh_child_manifolds
    manifold_module.refresh_child_manifolds = lambda current_manifold: current_manifold.child_manifolds.clear()
    return lambda: setattr(manifold_module, "refresh_child_manifolds", original_refresh)


def _disable_structural(manifold):
    original_update = manifold_module.update_edge_dynamics
    manifold_module.update_edge_dynamics = lambda current_manifold, edge_usage: []
    return lambda: setattr(manifold_module, "update_edge_dynamics", original_update)


def _disable_episodic(manifold):
    manifold._apply_episodic_recall = lambda updated_states: updated_states
    manifold.episodic_memory.record = lambda nodes, time_step: None
    return lambda: None


def _disable_geometry_feedback(manifold):
    manifold._apply_geometry_feedback = lambda edge_usage: setattr(
        manifold,
        "last_geometry_feedback",
        {
            "enabled": False,
            "surgery_requests": [],
            "surgery_attempted": [],
            "surgery_applied": [],
            "laplacian_energy": 0.0,
        },
    )
    return lambda: None


def geometry_feedback_gain(factory, seed: int, score_fn):
    enabled_manifold = _instantiate_manifold(factory, seed)
    enabled_score = float(score_fn(enabled_manifold))

    disabled_manifold = _instantiate_manifold(factory, seed)
    restore_fn = _disable_geometry_feedback(disabled_manifold)
    try:
        disabled_score = float(score_fn(disabled_manifold))
    finally:
        restore_fn()

    delta = enabled_score - disabled_score
    ratio = enabled_score / disabled_score if disabled_score != 0.0 else 0.0
    return {
        "enabled_score": enabled_score,
        "disabled_score": disabled_score,
        "gain_delta": delta,
        "gain_ratio": ratio,
    }


def ablation_sensitivity(factory, seed: int, score_fn):
    baseline_manifold = _instantiate_manifold(factory, seed)
    baseline_score = float(score_fn(baseline_manifold))

    ablations = {
        "semantic_hierarchy": _disable_semantic_hierarchy,
        "structural_plasticity": _disable_structural,
        "episodic_memory": _disable_episodic,
    }
    results = {
        "baseline_score": baseline_score,
        "ablations": {},
    }

    for label, disable_fn in ablations.items():
        manifold = _instantiate_manifold(factory, seed)
        restore_fn = disable_fn(manifold)
        try:
            ablated_score = float(score_fn(manifold))
        finally:
            restore_fn()
        delta = ablated_score - baseline_score
        relative = ablated_score / baseline_score if baseline_score != 0.0 else 0.0
        results["ablations"][label] = {
            "ablated_score": ablated_score,
            "delta_from_baseline": delta,
            "relative_score_retention": relative,
        }
    return results


@dataclass
class AdversarialPhase:
    label: str
    steps: int
    input_fn: callable
    perturb_fn: callable | None = None


class AdversarialRegionRunner:
    def __init__(self, manifold, snapshot_fn, phases):
        self.manifold = manifold
        self.snapshot_fn = snapshot_fn
        self.phases = phases

    def run(self):
        records = []
        global_step = 0
        for phase in self.phases:
            for local_step in range(phase.steps):
                if phase.perturb_fn is not None:
                    phase.perturb_fn(local_step, self.manifold)
                external_input = phase.input_fn(local_step, self.manifold)
                tick = time.perf_counter()
                self.manifold.step(external_input)
                latency = time.perf_counter() - tick
                metrics = self.snapshot_fn(self.manifold)
                record = {
                    "global_step": global_step,
                    "local_phase_step": local_step,
                    "phase_label": phase.label,
                    "step_latency": latency,
                }
                if metrics:
                    record.update(metrics)
                records.append(record)
                global_step += 1
        return records


__all__ = [
    "AdversarialPhase",
    "AdversarialRegionRunner",
    "ablation_sensitivity",
    "adaptation_speed",
    "capture_state",
    "catastrophic_interference",
    "compare_to_baseline",
    "computational_cost",
    "geometry_feedback_gain",
    "hierarchy_utility",
    "memory_retention",
    "noise_recovery",
    "recall_accuracy",
    "recovery_score",
    "region_formation",
    "run_benchmark_suite",
    "topology_recovery",
    "write_benchmark_report",
]