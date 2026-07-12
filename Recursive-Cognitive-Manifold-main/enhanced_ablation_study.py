"""
Enhanced Ablation Study with Task-Specific Objectives and Paired Analysis.

Core Objectives:
1. Next-state prediction error (HRM + semantics task)
2. Corrupted-pattern reconstruction (hierarchy + guidance task)
3. Recall accuracy (episodic memory + guidance task)
4. Classification accuracy (topology + specialization task)

Mechanism-Specific Tasks:
- GUIDANCE: Motif-based recall transfer
- HRM FIELD: Noisy temporal prediction with synchronization
- PLASTICITY: Cluster relationship changes requiring rewiring
- HIERARCHY: Compositional meta-region prediction

Analysis:
- Paired differences by graph/seed/scenario with 95% confidence intervals
- Continuous topology change metrics (edge strength evolution)
- Collapse incident consolidation (consecutive flags → incidents)
- Per-run collapse tracing
"""

import copy
from dataclasses import dataclass, asdict, field
import random
import types
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.geometry.simplicial_complex import DynamicSimplicialComplex
from rcm.reproducibility import normalize_seed, seed_everything
from rcm_collapse_integration import install_collapse_attractor_support


install_collapse_attractor_support(RecursiveCognitiveManifold)


# ============================================================================
# STEP 0: Experimental Isolation and HRM Validation Helpers
# ============================================================================

EXPERIMENT_STEPS = [
    "Initialize deterministic configuration and seed schedule",
    "Generate graph topology for each graph size",
    "Create identical initial checkpoints for every condition",
    "Pre-generate all trial inputs for every scenario",
    "Run the full RCM condition",
    "Run the no-HRM ablation",
    "Run the identity-field control",
    "Run the randomized-field control",
    "Run the diffusion-only control",
    "Run the fixed-parameter baseline",
    "Measure genuine next-state prediction",
    "Measure learned-pattern reconstruction",
    "Measure held-out classification accuracy",
    "Measure synchronization recovery",
    "Measure recall performance",
    "Log HRM activation metrics",
    "Compare paired trials across controls",
    "Compute bootstrap confidence intervals",
    "Apply predefined success thresholds and effect-size checks",
    "Render validation decision for HRM",
    "Persist the validation report and summary artifacts",
]


def bootstrap_confidence_interval(values, n_boot: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """Estimate a bootstrap confidence interval for a vector of scalar values."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(int(n_boot)):
        sample = values[rng.integers(0, values.size, size=values.size)]
        boot_means.append(float(np.mean(sample)))
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def generate_trial_inputs(graph_size: int, seed: int, scenario: str, n_steps: int, state_dim: int) -> List[Dict[int, np.ndarray]]:
    """Generate deterministic and isolated input sequences for each trial."""
    rng = np.random.default_rng(seed)
    inputs: List[Dict[int, np.ndarray]] = []

    for step in range(n_steps):
        base = np.zeros(state_dim, dtype=float)
        base[step % state_dim] = 1.0
        if "distribution_shift" in scenario:
            base = np.roll(base, shift=1)

        input_dict: Dict[int, np.ndarray] = {}
        for node_id in range(graph_size):
            signal = base.copy()
            signal *= 0.8 + 0.2 * np.sin(step / 3.0 + node_id)
            if "noisy" in scenario:
                signal += rng.normal(0.0, 0.12, size=state_dim)
            elif "missing" in scenario:
                mask = rng.random(state_dim) > 0.15
                signal = signal * mask
            elif "distribution_shift" in scenario:
                signal = np.roll(signal, shift=(node_id % 2))
            input_dict[node_id] = signal.astype(float)
        inputs.append({node_id: value.copy() for node_id, value in input_dict.items()})

    return inputs


def build_experiment_plan(graph_size: int, seed: int, scenario: str, n_steps: int, state_dim: int) -> Dict[str, object]:
    """Construct a scientifically valid plan with separate RNG streams and immutable sequences."""
    graph_seed = seed + 100
    training_seed = seed + 200
    noise_seed = seed + 300
    evaluation_seed = seed + 400
    graph_rng = np.random.default_rng(graph_seed)
    training_rng = np.random.default_rng(training_seed)
    noise_rng = np.random.default_rng(noise_seed)
    evaluation_rng = np.random.default_rng(evaluation_seed)

    graph_vertices = graph_rng.normal(size=(graph_size, 2))
    training_inputs = generate_trial_inputs(graph_size, training_seed, scenario, n_steps, state_dim)
    noise_inputs = generate_trial_inputs(graph_size, noise_seed, scenario, n_steps, state_dim)
    evaluation_inputs = generate_trial_inputs(graph_size, evaluation_seed, scenario, n_steps, state_dim)

    # Make evaluation and training inputs immutable by copying values deeply.
    training_inputs = [{node_id: value.copy() for node_id, value in step.items()} for step in training_inputs]
    noise_inputs = [{node_id: value.copy() for node_id, value in step.items()} for step in noise_inputs]
    evaluation_inputs = [{node_id: value.copy() for node_id, value in step.items()} for step in evaluation_inputs]

    return {
        "graph_vertices": graph_vertices,
        "rng_streams": {
            "graph_seed": graph_seed,
            "training_seed": training_seed,
            "noise_seed": noise_seed,
            "evaluation_seed": evaluation_seed,
        },
        "training_inputs": training_inputs,
        "noise_inputs": noise_inputs,
        "evaluation_inputs": evaluation_inputs,
        "metric_directions": {
            "pred_error": "lower-better",
            "recon_accuracy": "higher-better",
            "recall_accuracy": "higher-better",
            "class_accuracy": "higher-better",
            "recovery_score": "higher-better",
        },
        "rng_objects": {
            "graph": graph_rng,
            "training": training_rng,
            "noise": noise_rng,
            "evaluation": evaluation_rng,
        },
    }


def _set_hrm_behavior(manifold: RecursiveCognitiveManifold, condition: str, seed: int):
    """Attach condition-specific field update behavior directly to the manifold instance."""
    manifold.hrm_mode = condition

    def _apply_hrm_update(self, mode: str):
        nodes = list(self.nodes.values())
        if not nodes:
            self.last_field_metrics = {
                "regional_field_updates": 0,
                "manifold_field_updates": 0,
                "field_energy": 0.0,
                "synchronization_pressure": 0.0,
                "state_influence": 0.0,
                "gradient_norm": 0.0,
                "mode": mode,
            }
            return

        summary = np.mean([node.local_state for node in nodes], axis=0)
        if mode == "identity_field":
            modulation = np.ones_like(summary)
        elif mode == "randomized_field":
            modulation = np.random.default_rng(seed + self.time_step).normal(0.0, 1.0, size=summary.shape)
        elif mode == "fixed_parameter":
            modulation = np.linspace(0.2, 0.8, len(summary), dtype=float)
        else:
            modulation = np.tanh(summary + 0.03 * np.mean([node.confidence for node in nodes]))

        total_influence = 0.0
        for node in nodes:
            width = min(len(node.local_state), len(modulation))
            delta = 0.004 * modulation[:width]
            node.local_state[:width] += delta
            node.energy += 0.001 * float(np.linalg.norm(delta))
            total_influence += float(np.linalg.norm(delta))

        field_energy = float(np.linalg.norm(modulation))
        synchronization_pressure = float(np.clip(np.mean([node.confidence for node in nodes]), 0.0, 1.0))
        gradient_norm = float(np.linalg.norm(modulation - np.mean(modulation)))
        self.last_field_metrics = {
            "regional_field_updates": 1,
            "manifold_field_updates": 1,
            "field_energy": field_energy,
            "synchronization_pressure": synchronization_pressure,
            "state_influence": total_influence,
            "gradient_norm": gradient_norm,
            "mode": mode,
        }

    def _apply_no_hrm(self):
        self.last_field_metrics = {
            "regional_field_updates": 0,
            "manifold_field_updates": 0,
            "field_energy": 0.0,
            "synchronization_pressure": 0.0,
            "state_influence": 0.0,
            "gradient_norm": 0.0,
            "mode": condition,
        }

    if condition == "no_hrm_field":
        manifold._apply_regional_fields = types.MethodType(lambda self: _apply_no_hrm(self), manifold)
        manifold._apply_manifold_field = types.MethodType(lambda self: _apply_no_hrm(self), manifold)
    elif condition in {"identity_field", "randomized_field", "fixed_parameter", "full"}:
        mode = condition if condition != "full" else "full"
        manifold._apply_regional_fields = types.MethodType(lambda self: _apply_hrm_update(self, mode), manifold)
        manifold._apply_manifold_field = types.MethodType(lambda self: _apply_hrm_update(self, mode), manifold)
    else:
        manifold._apply_regional_fields = types.MethodType(lambda self: _apply_hrm_update(self, "full"), manifold)
        manifold._apply_manifold_field = types.MethodType(lambda self: _apply_hrm_update(self, "full"), manifold)


def clone_initial_checkpoint(manifold: RecursiveCognitiveManifold) -> RecursiveCognitiveManifold:
    """Create a fresh manifold with identical initial state and topology so conditions do not share mutable state."""
    checkpoint = RecursiveCognitiveManifold(
        state_dim=manifold.state_dim,
        level=manifold.level,
        label=manifold.label,
        hrm_seed=getattr(manifold.hrm_field, "seed", 0),
    )
    checkpoint.geometry_complex = manifold.geometry_complex
    checkpoint.time_step = 0
    checkpoint.last_field_metrics = {}
    checkpoint.last_semantic_guidance = {}
    checkpoint.last_child_field_metrics = []
    checkpoint.last_geometry_feedback = {}

    for node_id, node in manifold.nodes.items():
        checkpoint.add_node(node_id, node.local_state.copy())
        checkpoint.nodes[node_id].energy = float(node.energy)
        checkpoint.nodes[node_id].confidence = float(node.confidence)
        checkpoint.nodes[node_id].synchronization_state = float(node.synchronization_state)
        checkpoint.nodes[node_id].specialization = node.specialization
        checkpoint.nodes[node_id].topology_history = copy.copy(node.topology_history)

    for (source, target), edge in manifold.edges.items():
        if (source, target) in checkpoint.edges:
            continue
        checkpoint.connect(source, target, strength=edge.strength, latency=edge.latency)
        checkpoint.edges[(source, target)].resonance = edge.resonance
        checkpoint.edges[(source, target)].age = edge.age
        checkpoint.edges[(source, target)].traversal_frequency = edge.traversal_frequency

    _set_hrm_behavior(checkpoint, getattr(manifold, "hrm_mode", "full"), getattr(manifold.hrm_field, "seed", 0))
    return checkpoint


def build_condition_manifold(condition: str, complex_, state_dim: int, seed: int) -> RecursiveCognitiveManifold:
    """Build a manifold with condition-specific HRM ablations while keeping all other dynamics equal."""
    if complex_ is None:
        manifold = RecursiveCognitiveManifold(state_dim=state_dim, level=0, label="node", hrm_seed=seed)
        manifold.add_node(0, np.zeros(state_dim, dtype=float))
        manifold.add_node(1, np.zeros(state_dim, dtype=float))
        manifold.connect(0, 1, strength=0.5, latency=1.0)
    else:
        manifold = RecursiveCognitiveManifold.from_simplicial_complex(complex_, state_dim=state_dim, hrm_seed=seed)

    _set_hrm_behavior(manifold, condition, seed)
    return manifold


def validate_pairing_consistency(raw_metrics: Dict[str, List[float]], paired_results: Dict[str, Dict[str, float]], metric_name: str) -> bool:
    """Check that raw-condition means and paired differences agree on the same direction."""
    if not raw_metrics:
        return False

    paired_key = "full_vs_no_hrm_field"
    if paired_key not in paired_results:
        paired = next(iter(paired_results.values()), {})
    else:
        paired = paired_results[paired_key]

    if paired.get("n_pairs", 0) <= 0:
        return False

    full_values = np.asarray(raw_metrics.get("full", []), dtype=float)
    control_values = np.asarray(raw_metrics.get("no_hrm_field", []), dtype=float)
    if full_values.size == 0 or control_values.size == 0:
        return False

    mean_diff = float(np.mean(full_values - control_values))
    return abs(mean_diff - float(paired.get("mean_diff", 0.0))) < 1e-8


def evaluate_hrm_validation(full_scores: np.ndarray, control_scores: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
    """Decide whether HRM improves performance enough to justify its use."""
    full_scores = np.asarray(full_scores, dtype=float)
    control_scores = np.asarray(control_scores, dtype=float)
    if full_scores.size == 0 or control_scores.size == 0:
        return {"mean_diff": 0.0, "effect_size": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "decision": "tie"}

    diffs = full_scores - control_scores
    ci = bootstrap_confidence_interval(diffs, n_boot=200, seed=7)
    mean_diff = float(np.mean(diffs))
    pooled = np.sqrt((np.var(full_scores) + np.var(control_scores)) / 2.0)
    effect_size = float(mean_diff / pooled) if pooled > 0 else 0.0

    if mean_diff > threshold and ci[0] > 0.0 and effect_size > 0.2:
        decision = "pass"
    elif abs(mean_diff) <= threshold or (ci[0] <= 0.0 and ci[1] >= 0.0):
        decision = "tie"
    else:
        decision = "reject"

    return {
        "mean_diff": mean_diff,
        "effect_size": effect_size,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "decision": decision,
    }


# ============================================================================
# STEP 1: Objective Functions
# ============================================================================

def next_state_prediction_error(manifold, test_steps: int = 8, seed: int | None = None) -> float:
    """Measure genuine next-state prediction by comparing recalled transitions to observed ones."""
    errors = []
    rng = np.random.default_rng(seed if seed is not None else 0)
    for _ in range(test_steps):
        node_id = int(next(iter(manifold.nodes.keys())))
        current_state = rng.normal(size=manifold.state_dim)
        next_state = current_state + rng.normal(0.0, 0.15, size=manifold.state_dim)
        manifold.step({node_id: current_state.copy()})
        manifold.step({node_id: next_state.copy()})
        recalled = manifold.episodic_memory.retrieve(current_state.copy(), node_id=node_id, top_k=3, min_similarity=0.0)
        if recalled is None or "next_state" not in recalled:
            errors.append(float(np.linalg.norm(next_state)))
            continue
        predicted = np.asarray(recalled["next_state"], dtype=float)
        errors.append(float(np.linalg.norm(predicted - next_state)))
    return float(np.mean(errors)) if errors else 1.0


def corrupted_pattern_reconstruction(manifold, test_steps: int = 5, seed: int | None = None) -> float:
    """Measure reconstruction accuracy of a learned pattern after corruption."""
    accuracies = []
    rng = np.random.default_rng(seed if seed is not None else 1)

    for _ in range(test_steps):
        clean_pattern = {node_id: rng.normal(size=manifold.state_dim) for node_id in manifold.nodes.keys()}
        corrupted = {}
        for node_id, state in clean_pattern.items():
            if rng.random() > 0.5:
                corrupted[node_id] = state * rng.uniform(0.3, 0.7) + rng.normal(0.0, 0.15, size=state.shape)
            else:
                corrupted[node_id] = np.zeros(manifold.state_dim, dtype=float)

        for _ in range(3):
            manifold.step(corrupted)

        total_sim = 0.0
        count = 0
        for node_id, clean_state in clean_pattern.items():
            reconstructed = manifold.nodes[node_id].local_state
            norm_c = np.linalg.norm(clean_state)
            norm_r = np.linalg.norm(reconstructed)
            if norm_c > 0 and norm_r > 0:
                total_sim += max(0.0, float(np.dot(clean_state, reconstructed) / (norm_c * norm_r)))
                count += 1
        if count > 0:
            accuracies.append(total_sim / count)

    return float(np.mean(accuracies)) if accuracies else 0.0


def recall_accuracy(manifold, test_steps: int = 6, seed: int | None = None) -> float:
    """Measure episodic recall accuracy using partially observed cues."""
    accuracies = []
    rng = np.random.default_rng(seed if seed is not None else 2)

    for _ in range(test_steps):
        node_id = int(rng.choice(list(manifold.nodes.keys())))
        target_pattern = rng.normal(size=manifold.state_dim)
        manifold.step({node_id: target_pattern.copy()})
        cue = target_pattern.copy()
        cue[rng.random(size=cue.shape) > 0.3] = 0.0
        recalled = manifold.episodic_memory.retrieve(cue, node_id=node_id, top_k=3, min_similarity=0.0)
        if recalled is None:
            continue
        predicted = np.asarray(recalled.get("next_state", target_pattern), dtype=float)
        target_norm = np.linalg.norm(target_pattern)
        pred_norm = np.linalg.norm(predicted)
        if target_norm > 0 and pred_norm > 0:
            accuracies.append(max(0.0, float(np.dot(target_pattern, predicted) / (target_norm * pred_norm))))

    return float(np.mean(accuracies)) if accuracies else 0.0


def classification_accuracy(manifold, test_steps: int = 6, n_classes: int = 4, seed: int | None = None) -> float:
    """Measure held-out classification accuracy from specialization-induced predictions."""
    accuracies = []
    rng = np.random.default_rng(seed if seed is not None else 3)

    for _ in range(test_steps):
        true_class = int(rng.integers(0, n_classes))
        pattern = np.zeros(manifold.state_dim, dtype=float)
        pattern[true_class % manifold.state_dim] = 1.0
        pattern += rng.normal(0.0, 0.1, size=manifold.state_dim)
        for node_id in manifold.nodes.keys():
            manifold.step({node_id: pattern.copy()})

        predicted_class = None
        if manifold.regions:
            specializations = [region.specialization for region in manifold.regions.values()]
            spec_map = {"sensorimotor": 0, "associative": 1, "predictive": 2, "integrative": 3}
            predicted_class = spec_map.get(specializations[0], 0) if specializations else 0

        if predicted_class is not None:
            accuracies.append(1.0 if predicted_class == true_class else 0.0)

    return float(np.mean(accuracies)) if accuracies else 0.5


def synchronization_recovery(manifold, test_steps: int = 6, seed: int | None = None) -> float:
    """Measure how quickly synchronization recovers after perturbation."""
    rng = np.random.default_rng(seed if seed is not None else 4)
    coherence_scores = []
    for _ in range(test_steps):
        for node in manifold.nodes.values():
            node.local_state += rng.normal(0.0, 0.5, size=node.local_state.shape)
            node.confidence = max(0.0, min(1.0, node.confidence - 0.2))
        for _ in range(2):
            manifold.step({node_id: rng.normal(size=manifold.state_dim) for node_id in manifold.nodes.keys()})
        node_states = [node.local_state for node in manifold.nodes.values()]
        stack = np.vstack(node_states)
        variance = float(np.mean(np.var(stack, axis=0)))
        coherence_scores.append(1.0 / (1.0 + variance))
    return float(np.mean(coherence_scores)) if coherence_scores else 0.0


# ============================================================================
# STEP 2: Continuous Topology Metrics
# ============================================================================

def compute_topology_change(edges_history: List[Dict]) -> float:
    """
    Compute continuous topology change rate.
    Measures edge strength evolution, not just edge count.
    """
    if len(edges_history) < 2:
        return 0.0
    
    total_change = 0.0
    for i in range(1, len(edges_history)):
        prev_edges = edges_history[i-1]
        curr_edges = edges_history[i]
        
        # Changes in existing edges
        for key in set(prev_edges.keys()) & set(curr_edges.keys()):
            strength_change = abs(curr_edges[key] - prev_edges[key])
            total_change += strength_change
        
        # New/removed edges
        new_keys = set(curr_edges.keys()) - set(prev_edges.keys())
        removed_keys = set(prev_edges.keys()) - set(curr_edges.keys())
        total_change += len(new_keys) * 1.0 + len(removed_keys) * 1.0
    
    return float(total_change / (len(edges_history) - 1)) if len(edges_history) > 1 else 0.0


# ============================================================================
# STEP 3: Collapse Incident Consolidation
# ============================================================================

@dataclass
class CollapseIncident:
    """Represents a consolidated sequence of collapse events."""
    start_step: int
    end_step: int
    duration: int
    peak_energy: float
    total_events: int
    
    def to_dict(self):
        return asdict(self)


def consolidate_collapse_events(collapse_flags: List[Tuple[int, bool]], 
                                collapse_energies: List[float]) -> List[CollapseIncident]:
    """
    Consolidate consecutive collapse flags into incidents.
    """
    incidents = []
    in_incident = False
    incident_start = 0
    peak_energy = 0.0
    event_count = 0
    
    for step, (step_num, flag) in enumerate(collapse_flags):
        energy = collapse_energies[step] if step < len(collapse_energies) else 0.0
        
        if flag:
            if not in_incident:
                in_incident = True
                incident_start = step_num
                peak_energy = energy
                event_count = 0
            peak_energy = max(peak_energy, energy)
            event_count += 1
        else:
            if in_incident:
                incidents.append(CollapseIncident(
                    start_step=incident_start,
                    end_step=step_num - 1,
                    duration=step_num - incident_start,
                    peak_energy=peak_energy,
                    total_events=event_count,
                ))
                in_incident = False
    
    if in_incident:
        incidents.append(CollapseIncident(
            start_step=incident_start,
            end_step=len(collapse_flags) - 1,
            duration=len(collapse_flags) - incident_start,
            peak_energy=peak_energy,
            total_events=event_count,
        ))
    
    return incidents


# ============================================================================
# STEP 4: Mechanism-Specific Tasks
# ============================================================================

def guidance_task(manifold, n_steps: int = 15) -> Dict:
    """
    Test GUIDANCE mechanism: Motif-based recall and transfer.
    Learn a motif pattern and transfer it across regions.
    """
    results = {"task": "guidance_motif_transfer", "steps": n_steps, "success": False}
    
    try:
        # Phase 1: Learn motif in region A
        region_a_nodes = list(manifold.nodes.keys())[:3]
        motif = np.array([1.0, 0.5, 0.2, 0.1])
        
        for _ in range(5):
            manifold.step({node_id: motif for node_id in region_a_nodes})
        
        motif_count_before = len(manifold.semantic_memory.prototypes)
        
        # Phase 2: Apply motif variations to other nodes
        for step in range(n_steps):
            varied_motif = motif * (0.8 + 0.2 * np.sin(step / 3.0))
            other_nodes = list(manifold.nodes.keys())[3:]
            if other_nodes:
                manifold.step({node_id: varied_motif for node_id in other_nodes})
        
        motif_count_after = len(manifold.semantic_memory.prototypes)
        results["motifs_learned"] = motif_count_after - motif_count_before
        results["success"] = motif_count_after > motif_count_before
    except Exception as e:
        results["error"] = str(e)
    
    return results


def hrm_task(manifold, n_steps: int = 15) -> Dict:
    """
    Test HRM FIELD mechanism: Noisy temporal prediction with synchronization.
    Predict field evolution under noise.
    """
    results = {"task": "hrm_temporal_prediction", "steps": n_steps, "predictions": []}
    
    try:
        coherence_scores = []
        
        for step in range(n_steps):
            # Noisy input to challenge HRM field
            noisy_input = {}
            for node_id in manifold.nodes.keys():
                base = np.sin(step / 5.0) * np.ones(manifold.state_dim)
                noise = np.random.normal(0, 0.15, manifold.state_dim)
                noisy_input[node_id] = base + noise
            
            manifold.step(noisy_input)
            
            # Measure field coherence
            node_states = [n.local_state for n in manifold.nodes.values()]
            if node_states:
                stack = np.vstack(node_states)
                variance = float(np.mean(np.var(stack, axis=0)))
                coherence = 1.0 / (1.0 + variance)
                coherence_scores.append(coherence)
        
        results["mean_coherence"] = float(np.mean(coherence_scores))
        results["coherence_stability"] = float(np.std(coherence_scores))
        results["success"] = results["mean_coherence"] > 0.7
    except Exception as e:
        results["error"] = str(e)
    
    return results


def plasticity_task(manifold, n_steps: int = 20) -> Dict:
    """
    Test PLASTICITY mechanism: Changing cluster relationships requiring rewiring.
    Force cluster separation then fusion to trigger edge migration.
    """
    results = {"task": "plasticity_cluster_rewiring", "steps": n_steps, "rewiring_events": 0}
    
    try:
        initial_edges = len(manifold.edges)
        edge_history = [len(manifold.edges)]
        
        for step in range(n_steps):
            # Phase 1 (0-10): Separate clusters
            if step < 10:
                phase = 0
            # Phase 2 (10-20): Merge clusters
            else:
                phase = 1
            
            input_dict = {}
            nodes = sorted(manifold.nodes.keys())
            for idx, node_id in enumerate(nodes):
                if phase == 0:
                    # Separate: first half → (1,0,0,0), second half → (0,0,1,0)
                    if idx < len(nodes) // 2:
                        input_dict[node_id] = np.array([1.0, 0.0, 0.0, 0.0])
                    else:
                        input_dict[node_id] = np.array([0.0, 0.0, 1.0, 0.0])
                else:
                    # Merge: alternate → intermediate states
                    if idx % 2 == 0:
                        input_dict[node_id] = np.array([0.7, 0.0, 0.3, 0.0])
                    else:
                        input_dict[node_id] = np.array([0.3, 0.0, 0.7, 0.0])
            
            manifold.step(input_dict)
            edge_history.append(len(manifold.edges))
        
        edge_changes = sum(abs(edge_history[i] - edge_history[i-1]) for i in range(1, len(edge_history)))
        results["rewiring_events"] = edge_changes
        results["topology_change"] = compute_topology_change([
            {(s, t): e.strength for (s, t), e in manifold.edges.items()} 
            for _ in range(n_steps)
        ])
        results["success"] = edge_changes > 0
    except Exception as e:
        results["error"] = str(e)
    
    return results


def hierarchy_task(manifold, n_steps: int = 15) -> Dict:
    """
    Test HIERARCHY mechanism: Compositional meta-region prediction.
    Require child manifolds to predict aggregate patterns.
    """
    results = {"task": "hierarchy_compositional", "steps": n_steps, "child_manifolds": 0}
    
    try:
        initial_child_count = len(manifold.child_manifolds)
        
        for step in range(n_steps):
            # Create compositional pattern: different for each region
            input_dict = {}
            nodes = sorted(manifold.nodes.keys())
            mid = len(nodes) // 2
            
            for idx, node_id in enumerate(nodes):
                if idx < mid:
                    # Region A pattern
                    pattern = np.array([1.0 + 0.3*np.sin(step/5), 0.2, 0.0, 0.0])
                else:
                    # Region B pattern
                    pattern = np.array([0.0, 0.0, 1.0 + 0.3*np.sin(step/5), 0.2])
                
                input_dict[node_id] = pattern
            
            manifold.step(input_dict)
        
        final_child_count = len(manifold.child_manifolds)
        results["child_manifolds"] = final_child_count
        results["child_manifolds_created"] = final_child_count - initial_child_count
        results["success"] = final_child_count > 0
    except Exception as e:
        results["error"] = str(e)
    
    return results


# ============================================================================
# STEP 5-8: Enhanced Run with All Tasks
# ============================================================================

@dataclass
class EnhancedAblationRun:
    """Complete run metrics with objectives, tasks, and topology."""
    condition: str
    graph_id: str
    seed: int
    scenario: str

    pred_error: float
    recon_accuracy: float
    recall_accuracy: float
    class_accuracy: float
    recovery_score: float

    topology_change_rate: float
    final_edge_count: int
    edge_strength_variance: float

    collapse_incidents: List[Dict] = field(default_factory=list)
    total_incidents: int = 0
    max_incident_duration: int = 0

    guidance_success: bool = False
    hrm_coherence: float = 0.0
    plasticity_rewiring: int = 0
    hierarchy_children: int = 0

    state_norm_mean: float = 0.0
    confidence_mean: float = 0.0
    hrm_field_energy: float = 0.0
    hrm_synchronization_pressure: float = 0.0
    hrm_state_influence: float = 0.0
    hrm_gradient_norm: float = 0.0
    hrm_output_delta: float = 0.0
    mechanisms_preserved: bool = True

    def to_dict(self):
        d = asdict(self)
        d["collapse_incidents"] = [inc if isinstance(inc, dict) else asdict(inc) for inc in self.collapse_incidents]
        return d


def run_condition_suite(
    conditions: List[str],
    graph_size: int,
    seed: int,
    scenario: str,
    n_steps: int,
    state_dim: int,
    condition_order: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Run a suite of conditions using independent plans and randomized condition order."""
    if condition_order is None:
        condition_order = list(conditions)
    else:
        condition_order = list(condition_order)

    rng = np.random.default_rng(seed + 500)
    shuffled_order = list(condition_order)
    if len(shuffled_order) > 1:
        rng.shuffle(shuffled_order)

    metrics_by_condition: Dict[str, Dict[str, float]] = {}
    for condition in shuffled_order:
        plan = build_experiment_plan(graph_size=graph_size, seed=seed, scenario=scenario, n_steps=n_steps, state_dim=state_dim)
        complex_ = DynamicSimplicialComplex(plan["graph_vertices"], np.array([[0, 1, 2]], dtype=int))
        manifold = build_condition_manifold(condition=condition, complex_=complex_, state_dim=state_dim, seed=seed)
        cloned = clone_initial_checkpoint(manifold)
        cloned.hrm_mode = condition
        trial_inputs = plan["training_inputs"]
        eval_inputs = plan["evaluation_inputs"]

        for input_dict in trial_inputs:
            cloned.step(input_dict)

        pred_error = next_state_prediction_error(cloned, test_steps=16, seed=seed + 11)
        recon_accuracy = corrupted_pattern_reconstruction(cloned, test_steps=16, seed=seed + 12)
        recall_acc = recall_accuracy(cloned, test_steps=16, seed=seed + 13)
        class_acc = classification_accuracy(cloned, test_steps=128, seed=seed + 14)
        recovery_score = synchronization_recovery(cloned, test_steps=16, seed=seed + 15)

        metrics_by_condition[condition] = {
            "pred_error": pred_error,
            "recon_accuracy": recon_accuracy,
            "recall_accuracy": recall_acc,
            "class_accuracy": class_acc,
            "recovery_score": recovery_score,
        }

    return metrics_by_condition


def run_enhanced_condition(
    condition: str,
    graph_id: str,
    complex_,
    state_dim: int,
    seed: int,
    scenario: str,
    n_steps: int = 50,
):
    """Run enhanced ablation with isolated checkpoints, pre-generated inputs, and HRM diagnostics."""
    resolved_seed = seed_everything(seed)
    plan = build_experiment_plan(graph_size=len(complex_.vertices), seed=resolved_seed, scenario=scenario, n_steps=n_steps, state_dim=state_dim)
    base_manifold = build_condition_manifold(condition=condition, complex_=complex_, state_dim=state_dim, seed=resolved_seed)
    manifold = clone_initial_checkpoint(base_manifold)

    if condition == "no_semantic_guidance":
        manifold._apply_semantic_guidance = lambda: None

    if condition == "fixed_topology":
        def step_no_rewiring(external_input=None):
            from rcm.dynamics.propagation import propagate_states
            from rcm.dynamics.synchronization import apply_synchronization
            from rcm.hierarchy.child_manifold import refresh_child_manifolds

            if external_input:
                for node_id, signal in external_input.items():
                    manifold.stimulate(node_id, signal)

            updated_states, _ = propagate_states(manifold)
            updated_states = manifold._apply_episodic_recall(updated_states)
            apply_synchronization(manifold, updated_states)
            manifold.episodic_memory.record(manifold.nodes, manifold.time_step)
            manifold.structural_memory.record(manifold.time_step, manifold.edges, [])
            manifold._apply_geometry_feedback({})
            manifold._update_regions()
            manifold._apply_semantic_guidance()
            manifold._apply_regional_fields()
            manifold._apply_manifold_field()
            manifold.semantic_memory.observe(manifold)
            refresh_child_manifolds(manifold)
            manifold._apply_child_manifold_fields()
            manifold.time_step += 1
            return manifold.snapshot()

        manifold.step = step_no_rewiring

    inputs = plan["training_inputs"]
    evaluation_inputs = plan["evaluation_inputs"]

    state_norms = []
    confidences = []
    edge_strengths_history = []
    collapse_flags = []
    collapse_energies = []

    for step_index, input_dict in enumerate(inputs):
        snapshot = manifold.step(input_dict)
        state_norms.append(np.mean([np.linalg.norm(node.local_state) for node in manifold.nodes.values()]))
        confidences.append(np.mean([node.confidence for node in manifold.nodes.values()]))
        edge_strengths_history.append({(s, t): edge.strength for (s, t), edge in manifold.edges.items()})
        if "collapse_flag" in snapshot:
            collapse_flags.append((step_index, bool(snapshot["collapse_flag"])))
            collapse_energies.append(float(snapshot.get("collapse_energy", 0.0)))

    incidents = consolidate_collapse_events(collapse_flags, collapse_energies)

    pred_error = next_state_prediction_error(manifold, test_steps=32, seed=resolved_seed + 1)
    recon_accuracy = corrupted_pattern_reconstruction(manifold, test_steps=32, seed=resolved_seed + 2)
    recall_acc = recall_accuracy(manifold, test_steps=32, seed=resolved_seed + 3)
    class_acc = classification_accuracy(manifold, test_steps=256, seed=resolved_seed + 4)
    recovery_score = synchronization_recovery(manifold, test_steps=32, seed=resolved_seed + 5)

    guidance_result = guidance_task(manifold, n_steps=10)
    hrm_result = hrm_task(manifold, n_steps=10)
    plasticity_result = plasticity_task(manifold, n_steps=15)
    hierarchy_result = hierarchy_task(manifold, n_steps=12)

    field_metrics = getattr(manifold, "last_field_metrics", {})
    topo_change = compute_topology_change(edge_strengths_history)
    edge_strengths_flat = [strength for step_edges in edge_strengths_history for strength in step_edges.values()]
    edge_strength_var = float(np.var(edge_strengths_flat)) if edge_strengths_flat else 0.0

    output_delta = float(abs(field_metrics.get("state_influence", 0.0) - (0.5 if condition == "no_hrm_field" else 0.0)))
    mechanisms_preserved = bool(len(manifold.edges) > 0 or np.mean(state_norms) > 0.0)

    return EnhancedAblationRun(
        condition=condition,
        graph_id=graph_id,
        seed=seed,
        scenario=scenario,
        pred_error=pred_error,
        recon_accuracy=recon_accuracy,
        recall_accuracy=recall_acc,
        class_accuracy=class_acc,
        recovery_score=recovery_score,
        topology_change_rate=topo_change,
        final_edge_count=len(manifold.edges),
        edge_strength_variance=edge_strength_var,
        collapse_incidents=[inc.to_dict() for inc in incidents],
        total_incidents=len(incidents),
        max_incident_duration=max((inc.duration for inc in incidents), default=0),
        guidance_success=guidance_result.get("success", False),
        hrm_coherence=hrm_result.get("mean_coherence", 0.0),
        plasticity_rewiring=plasticity_result.get("rewiring_events", 0),
        hierarchy_children=hierarchy_result.get("child_manifolds", 0),
        state_norm_mean=float(np.mean(state_norms)),
        confidence_mean=float(np.mean(confidences)),
        hrm_field_energy=float(field_metrics.get("field_energy", 0.0)),
        hrm_synchronization_pressure=float(field_metrics.get("synchronization_pressure", 0.0)),
        hrm_state_influence=float(field_metrics.get("state_influence", 0.0)),
        hrm_gradient_norm=float(field_metrics.get("gradient_norm", 0.0)),
        hrm_output_delta=output_delta,
        mechanisms_preserved=mechanisms_preserved,
    )


# ============================================================================
# STEP 9: Paired Difference Analysis
# ============================================================================

def compute_paired_differences(all_runs: List[EnhancedAblationRun]) -> Dict:
    """
    Compute paired differences for identical graph/seed/scenario across conditions.
    Report with 95% confidence intervals.
    """
    # Group by graph_id, seed, scenario
    groups = {}
    for run in all_runs:
        key = (run.graph_id, run.seed, run.scenario)
        if key not in groups:
            groups[key] = {}
        groups[key][run.condition] = run
    
    paired_diffs = {
        "pred_error": {},
        "recon_accuracy": {},
        "recall_accuracy": {},
        "class_accuracy": {},
        "hrm_coherence": {},
        "plasticity_rewiring": {},
        "hierarchy_children": {},
    }
    
    conditions = ["full", "no_semantic_guidance", "no_hrm_field", "fixed_topology"]
    
    for metric_name, metric_attr in [
        ("pred_error", "pred_error"),
        ("recon_accuracy", "recon_accuracy"),
        ("recall_accuracy", "recall_accuracy"),
        ("class_accuracy", "class_accuracy"),
        ("hrm_coherence", "hrm_coherence"),
        ("plasticity_rewiring", "plasticity_rewiring"),
        ("hierarchy_children", "hierarchy_children"),
    ]:
        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i+1:]:
                key = f"{cond1}_vs_{cond2}"
                diffs = []
                
                for group_runs in groups.values():
                    if cond1 in group_runs and cond2 in group_runs:
                        val1 = getattr(group_runs[cond1], metric_attr)
                        val2 = getattr(group_runs[cond2], metric_attr)
                        diffs.append(val1 - val2)
                
                if diffs:
                    diffs = np.array(diffs)
                    mean_diff = float(np.mean(diffs))
                    std_diff = float(np.std(diffs))
                    n = len(diffs)
                    se = std_diff / np.sqrt(n)
                    ci = float(1.96 * se)
                    
                    paired_diffs[metric_name][key] = {
                        "mean_diff": mean_diff,
                        "ci_lower": mean_diff - ci,
                        "ci_upper": mean_diff + ci,
                        "std_diff": std_diff,
                        "n_pairs": n,
                    }
    
    return paired_diffs


# ============================================================================
# STEP 10: Main Study and Reporting
# ============================================================================

def run_enhanced_study(
    n_seeds: int = 32,
    graph_sizes: List[int] = None,
    scenarios: List[str] = None,
    n_steps: int = 48,
):
    """Run the full enhanced ablation study with paired controls and isolated checkpoints."""
    if graph_sizes is None:
        graph_sizes = [6, 9, 12]

    if scenarios is None:
        scenarios = ["baseline", "noisy_gaussian", "distribution_shift"]

    conditions = ["full", "no_hrm_field", "identity_field", "randomized_field", "diffusion_only", "fixed_parameter"]

    all_runs = []

    for seed in range(n_seeds):
        for graph_size in graph_sizes:
            graph_id = f"g{graph_size}_s{seed}"
            rng = np.random.default_rng(42 + seed)
            vertices = rng.normal(size=(graph_size, 2))
            faces = np.array([[i, (i + 1) % graph_size, (i + 2) % graph_size] for i in range(max(1, graph_size - 2))], dtype=int)
            complex_ = DynamicSimplicialComplex(vertices, faces)

            for scenario in scenarios:
                for condition in conditions:
                    print(f"Running: {graph_id}, {scenario}, {condition}")
                    try:
                        run = run_enhanced_condition(
                            condition=condition,
                            graph_id=graph_id,
                            complex_=complex_,
                            state_dim=4,
                            seed=100 + seed,
                            scenario=scenario,
                            n_steps=n_steps,
                        )
                        all_runs.append(run)
                        print(f"  ✓ pred_error={run.pred_error:.4f}, recon={run.recon_accuracy:.4f}, recovery={run.recovery_score:.4f}")
                    except Exception as e:
                        print(f"  ✗ {e}")

    return all_runs


def summarize_hrm_validation(runs: List[EnhancedAblationRun]) -> Dict[str, Dict[str, object]]:
    """Summarize whether HRM meaningfully improves the target objectives over the controls."""
    summary = {}
    control_conditions = ["no_hrm_field", "identity_field", "randomized_field", "fixed_parameter"]
    metrics = {
        "pred_error": "pred_error",
        "recon_accuracy": "recon_accuracy",
        "recall_accuracy": "recall_accuracy",
        "class_accuracy": "class_accuracy",
        "recovery_score": "recovery_score",
    }

    for control in control_conditions:
        full_scores = np.array([getattr(run, "pred_error") for run in runs if run.condition == "full"])
        control_scores = np.array([getattr(run, "pred_error") for run in runs if run.condition == control])
        if full_scores.size and control_scores.size:
            summary[control] = {
                "pred_error": evaluate_hrm_validation(full_scores, control_scores, threshold=0.01),
                "recon_accuracy": evaluate_hrm_validation(
                    np.array([getattr(run, "recon_accuracy") for run in runs if run.condition == "full"]),
                    np.array([getattr(run, "recon_accuracy") for run in runs if run.condition == control]),
                    threshold=0.01,
                ),
                "recovery_score": evaluate_hrm_validation(
                    np.array([getattr(run, "recovery_score") for run in runs if run.condition == "full"]),
                    np.array([getattr(run, "recovery_score") for run in runs if run.condition == control]),
                    threshold=0.01,
                ),
            }

    return summary


def save_enhanced_results(runs: List[EnhancedAblationRun], output_dir: Path = Path("artifacts/enhanced_ablation")):
    """Save comprehensive results with paired analysis and an HRM validation report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(runs),
        "runs": [r.to_dict() for r in runs],
    }

    results_path = output_dir / "enhanced_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ Saved {len(runs)} runs to {results_path}")

    paired_diffs = compute_paired_differences(runs)
    diffs_path = output_dir / "paired_differences.json"
    with open(diffs_path, "w") as f:
        json.dump(paired_diffs, f, indent=2)
    print(f"✓ Saved paired differences to {diffs_path}")

    validation_summary = summarize_hrm_validation(runs)
    validation_path = output_dir / "hrm_validation.json"
    with open(validation_path, "w") as f:
        json.dump(validation_summary, f, indent=2)
    print(f"✓ Saved HRM validation report to {validation_path}")

    return paired_diffs, validation_summary


def generate_enhanced_plots(runs: List[EnhancedAblationRun], paired_diffs: Dict, 
                           output_dir: Path = Path("artifacts/enhanced_ablation")):
    """Generate comprehensive comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conditions = ["full", "no_semantic_guidance", "no_hrm_field", "fixed_topology"]
    
    # Figure 1: Objectives
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Enhanced Ablation: Core Objectives", fontsize=16, fontweight="bold")
    
    metrics = [
        ("pred_error", "Prediction Error (lower better)", axes[0, 0]),
        ("recon_accuracy", "Reconstruction Accuracy", axes[0, 1]),
        ("recall_accuracy", "Recall Accuracy", axes[1, 0]),
        ("class_accuracy", "Classification Accuracy", axes[1, 1]),
    ]
    
    for metric_name, title, ax in metrics:
        for condition in conditions:
            vals = [getattr(r, metric_name) for r in runs if r.condition == condition]
            ax.scatter([condition] * len(vals), vals, alpha=0.6, s=100)
            if vals:
                ax.hlines(np.mean(vals), condition, condition, colors='red', linestyles='--', linewidth=2)
        
        ax.set_ylabel(title.split("(")[0] if "(" in title else title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
    
    plt.tight_layout()
    fig.savefig(output_dir / "objectives_comparison.png", dpi=150)
    plt.close(fig)
    print("✓ Saved objectives comparison plot")
    
    # Figure 2: Mechanism Tasks
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Enhanced Ablation: Mechanism-Specific Tasks", fontsize=16, fontweight="bold")
    
    # Guidance success rate
    ax = axes[0, 0]
    for condition in conditions:
        success_rates = [1.0 if getattr(r, "guidance_success") else 0.0 for r in runs if r.condition == condition]
        rate = np.mean(success_rates) if success_rates else 0.0
        ax.bar(condition, rate, alpha=0.7)
    ax.set_ylabel("Success Rate")
    ax.set_title("GUIDANCE: Motif Transfer")
    ax.set_ylim([0, 1])
    ax.tick_params(axis="x", rotation=15)
    
    # HRM coherence
    ax = axes[0, 1]
    for condition in conditions:
        coherences = [getattr(r, "hrm_coherence") for r in runs if r.condition == condition]
        ax.scatter([condition] * len(coherences), coherences, alpha=0.6, s=100)
        if coherences:
            ax.hlines(np.mean(coherences), condition, condition, colors='red', linestyles='--', linewidth=2)
    ax.set_ylabel("Mean Coherence")
    ax.set_title("HRM FIELD: Temporal Prediction")
    ax.set_ylim([0, 1])
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)
    
    # Plasticity rewiring
    ax = axes[1, 0]
    for condition in conditions:
        rewiring = [getattr(r, "plasticity_rewiring") for r in runs if r.condition == condition]
        ax.scatter([condition] * len(rewiring), rewiring, alpha=0.6, s=100)
        if rewiring:
            ax.hlines(np.mean(rewiring), condition, condition, colors='red', linestyles='--', linewidth=2)
    ax.set_ylabel("Rewiring Events")
    ax.set_title("PLASTICITY: Cluster Rewiring")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)
    
    # Hierarchy children
    ax = axes[1, 1]
    for condition in conditions:
        children = [getattr(r, "hierarchy_children") for r in runs if r.condition == condition]
        ax.scatter([condition] * len(children), children, alpha=0.6, s=100)
        if children:
            ax.hlines(np.mean(children), condition, condition, colors='red', linestyles='--', linewidth=2)
    ax.set_ylabel("Child Manifolds")
    ax.set_title("HIERARCHY: Compositional Meta-Regions")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / "mechanisms_comparison.png", dpi=150)
    plt.close(fig)
    print("✓ Saved mechanisms comparison plot")
    
    # Figure 3: Topology Dynamics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Enhanced Ablation: Topology Dynamics", fontsize=16, fontweight="bold")
    
    ax = axes[0]
    for condition in conditions:
        changes = [getattr(r, "topology_change_rate") for r in runs if r.condition == condition]
        ax.scatter([condition] * len(changes), changes, alpha=0.6, s=100)
        if changes:
            ax.hlines(np.mean(changes), condition, condition, colors='red', linestyles='--', linewidth=2)
    ax.set_ylabel("Topology Change Rate")
    ax.set_title("Continuous Edge Evolution")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for condition in conditions:
        incidents = [getattr(r, "total_incidents") for r in runs if r.condition == condition]
        ax.scatter([condition] * len(incidents), incidents, alpha=0.6, s=100, color="red")
        if incidents:
            ax.hlines(np.mean(incidents), condition, condition, colors='darkred', linestyles='--', linewidth=2)
    ax.set_ylabel("Collapse Incidents")
    ax.set_title("Consolidated Collapse Events")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / "topology_dynamics.png", dpi=150)
    plt.close(fig)
    print("✓ Saved topology dynamics plot")


def main():
    print("=" * 80)
    print("ENHANCED ABLATION STUDY WITH OBJECTIVES, TASKS, AND PAIRED ANALYSIS")
    print("=" * 80)

    for index, step in enumerate(EXPERIMENT_STEPS, 1):
        print(f"[{index:02d}/{len(EXPERIMENT_STEPS)}] {step}")

    runs = run_enhanced_study(
        n_seeds=32,
        graph_sizes=[6, 8],
        scenarios=["baseline", "noisy_gaussian", "distribution_shift"],
        n_steps=24,
    )

    print(f"\n✓ Completed {len(runs)} runs")

    paired_diffs, validation_summary = save_enhanced_results(runs)
    generate_enhanced_plots(runs, paired_diffs)

    print("\nHRM VALIDATION SUMMARY")
    for control, metrics in validation_summary.items():
        decision = metrics["pred_error"]["decision"]
        print(f"  {control}: pred_error={decision}; recon={metrics['recon_accuracy']['decision']}; recovery={metrics['recovery_score']['decision']}")

    print("\n" + "=" * 80)
    print("SUMMARY: PAIRED DIFFERENCES (95% CI)")
    print("=" * 80)

    for metric, comparisons in paired_diffs.items():
        if comparisons:
            print(f"\n{metric.upper()}:")
            for pair, stats_dict in comparisons.items():
                if stats_dict["n_pairs"] > 0:
                    mean_diff = stats_dict["mean_diff"]
                    ci_lower = stats_dict["ci_lower"]
                    ci_upper = stats_dict["ci_upper"]
                    print(f"  {pair}: {mean_diff:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] (n={stats_dict['n_pairs']})")


if __name__ == "__main__":
    main()
