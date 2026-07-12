"""
Comprehensive Ablation Study: Testing RCM robustness across configurations.

Conditions:
1. Full RCM (baseline)
2. No semantic guidance
3. No HRM/regional field
4. Fixed topology (no rewiring)

Test scenarios:
- Randomized initial graphs
- Shuffled node assignments
- Noisy signals
- Partially missing signals
- Multiple seeds for statistical validity
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.geometry.simplicial_complex import DynamicSimplicialComplex
from rcm.reproducibility import normalize_seed, seed_everything
from rcm_collapse_integration import install_collapse_attractor_support


install_collapse_attractor_support(RecursiveCognitiveManifold)


@dataclass
class AblationMetrics:
    """Metrics collected for each condition."""
    condition: str
    seed: int
    scenario: str
    steps: int
    
    # Stability metrics
    mean_state_norm: float
    state_norm_std: float
    energy_mean: float
    energy_std: float
    confidence_mean: float
    confidence_std: float
    
    # Topology metrics
    final_edge_count: int
    edge_rewiring_count: int
    edges_pruned: int
    
    # Regional metrics
    final_region_count: int
    region_stability_mean: float
    region_specialization_entropy: float
    
    # Hierarchy metrics
    child_manifolds_created: int
    
    # Collapse metrics
    collapse_events: int
    peak_collapse_energy: float
    
    # Memory metrics
    episodic_events: int
    semantic_motifs_count: int


def generate_random_graph(n_nodes: int, edge_density: float = 0.3, seed: int = None):
    """Generate random connected graph with specified edge density."""
    if seed is not None:
        np.random.seed(seed)
    
    vertices = np.random.randn(n_nodes, 2).astype(float)
    
    # Create faces to ensure connectivity
    faces = []
    for i in range(n_nodes - 2):
        if np.random.rand() < edge_density:
            faces.append([i, i + 1, (i + 2) % n_nodes])
    
    if not faces:
        faces = [[0, 1, 2]] if n_nodes >= 3 else [[0, 1]]
    
    faces = np.array(faces, dtype=int)
    return DynamicSimplicialComplex(vertices, faces)


def create_ablated_manifold(
    complex_,
    state_dim: int,
    seed: int,
    condition: str,
):
    """Create manifold with specified ablation condition."""
    manifold = RecursiveCognitiveManifold.from_simplicial_complex(
        complex_, state_dim=state_dim, hrm_seed=seed
    )
    
    # Store original methods for ablation
    if condition == "no_semantic_guidance":
        manifold._apply_semantic_guidance = lambda: None
    
    elif condition == "no_hrm_field":
        manifold._apply_regional_fields = lambda: None
        manifold._apply_manifold_field = lambda: None
    
    elif condition == "fixed_topology":
        # Disable edge rewiring
        original_update = manifold.step
        
        def step_no_rewiring(external_input=None):
            # Run normal step but skip edge dynamics updates
            if external_input:
                for node_id, signal in external_input.items():
                    manifold.stimulate(node_id, signal)
            
            from rcm.dynamics.propagation import propagate_states
            from rcm.dynamics.synchronization import apply_synchronization
            from rcm.hierarchy.child_manifold import refresh_child_manifolds
            
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
    
    return manifold


def apply_noise_to_signal(signal: np.ndarray, noise_type: str, noise_level: float = 0.1):
    """Apply various types of noise to input signal."""
    signal = np.asarray(signal, dtype=float).copy()
    
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise
    
    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_level, noise_level, signal.shape)
        return signal + noise
    
    elif noise_type == "missing":
        # Zero out random elements (missing signal)
        mask = np.random.rand(*signal.shape) > noise_level
        return signal * mask
    
    elif noise_type == "dropout":
        # Dropout entire timesteps
        if np.random.rand() < noise_level:
            return np.zeros_like(signal)
        return signal
    
    return signal


def run_condition_scenario(
    condition: str,
    scenario: str,
    complex_,
    state_dim: int,
    seed: int,
    n_steps: int = 40,
    noise_level: float = 0.1,
):
    """Run a single ablation condition under a specific scenario."""
    resolved_seed = seed_everything(seed)
    
    # Create ablated manifold
    manifold = create_ablated_manifold(complex_, state_dim, resolved_seed, condition)
    initial_edge_count = len(manifold.edges)
    
    # Define input pattern based on scenario
    if "shuffled" in scenario:
        node_order = list(manifold.nodes.keys())
        np.random.shuffle(node_order)
    else:
        node_order = sorted(manifold.nodes.keys())
    
    state_norms = []
    energies = []
    confidences = []
    edges_per_step = []
    collapse_events_list = []
    
    for step in range(n_steps):
        # Create input pattern
        phase = (step // 10) % 3
        base_state = np.zeros(state_dim, dtype=float)
        base_state[phase] = 1.0
        
        external_input = {}
        for idx, node_id in enumerate(node_order):
            # Vary signal based on node position
            signal = base_state.copy()
            signal = signal * (0.8 + 0.2 * np.sin(step / 5.0 + idx))
            
            # Apply noise based on scenario
            if "noisy" in scenario:
                signal = apply_noise_to_signal(signal, "gaussian", noise_level)
            elif "missing" in scenario:
                signal = apply_noise_to_signal(signal, "missing", noise_level)
            
            external_input[node_id] = signal
        
        # Step manifold
        snapshot = manifold.step(external_input)
        
        # Collect metrics
        state_norms.append(np.mean([np.linalg.norm(n.local_state) for n in manifold.nodes.values()]))
        energies.append(np.mean([n.energy for n in manifold.nodes.values()]))
        confidences.append(np.mean([n.confidence for n in manifold.nodes.values()]))
        edges_per_step.append(len(manifold.edges))
        
        if "collapse_events" in snapshot:
            collapse_events_list.append(int(snapshot["collapse_events"]))
    
    # Calculate metrics
    edge_rewiring = sum(abs(edges_per_step[i] - edges_per_step[i-1]) for i in range(1, len(edges_per_step)))
    edges_pruned = initial_edge_count - edges_per_step[-1] if edges_per_step else 0
    
    region_stabilities = [r.stability for r in manifold.regions.values()] if manifold.regions else [1.0]
    region_specializations = [r.specialization for r in manifold.regions.values()] if manifold.regions else []
    
    # Calculate entropy of specializations
    if region_specializations:
        unique, counts = np.unique(region_specializations, return_counts=True)
        probs = counts / len(region_specializations)
        specialization_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
    else:
        specialization_entropy = 0.0
    
    metrics = AblationMetrics(
        condition=condition,
        seed=seed,
        scenario=scenario,
        steps=n_steps,
        
        mean_state_norm=float(np.mean(state_norms)),
        state_norm_std=float(np.std(state_norms)),
        energy_mean=float(np.mean(energies)),
        energy_std=float(np.std(energies)),
        confidence_mean=float(np.mean(confidences)),
        confidence_std=float(np.std(confidences)),
        
        final_edge_count=len(manifold.edges),
        edge_rewiring_count=int(edge_rewiring),
        edges_pruned=int(edges_pruned),
        
        final_region_count=len(manifold.regions),
        region_stability_mean=float(np.mean(region_stabilities)),
        region_specialization_entropy=specialization_entropy,
        
        child_manifolds_created=len(manifold.child_manifolds),
        
        collapse_events=int(sum(collapse_events_list)),
        peak_collapse_energy=float(max(collapse_events_list) if collapse_events_list else 0.0),
        
        episodic_events=len(manifold.episodic_memory.transitions),
        semantic_motifs_count=len(manifold.semantic_memory.prototypes),
    )
    
    return metrics


def run_comprehensive_study(
    n_seeds: int = 3,
    graph_sizes: list = None,
    scenarios: list = None,
    n_steps: int = 40,
):
    """Run the full ablation study."""
    if graph_sizes is None:
        graph_sizes = [6, 9]
    
    if scenarios is None:
        scenarios = [
            "baseline",
            "shuffled",
            "noisy_gaussian",
            "missing_signals",
        ]
    
    conditions = [
        "full",
        "no_semantic_guidance",
        "no_hrm_field",
        "fixed_topology",
    ]
    
    all_results = []
    
    for graph_size in graph_sizes:
        for scenario in scenarios:
            for seed in range(n_seeds):
                print(f"Running: graph_size={graph_size}, scenario={scenario}, seed={seed}")
                
                # Generate graph once per seed (deterministic)
                complex_ = generate_random_graph(graph_size, seed=42 + seed)
                
                for condition in conditions:
                    try:
                        metrics = run_condition_scenario(
                            condition=condition,
                            scenario=scenario,
                            complex_=complex_,
                            state_dim=4,
                            seed=100 + seed,
                            n_steps=n_steps,
                            noise_level=0.15,
                        )
                        all_results.append(metrics)
                        print(f"  ✓ {condition}: {metrics.mean_state_norm:.3f} norm, {metrics.final_region_count} regions")
                    except Exception as e:
                        print(f"  ✗ {condition}: {e}")
    
    return all_results


def save_results(results: list, output_dir: Path = Path("artifacts/ablation")):
    """Save results as JSON and generate analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict for JSON
    results_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "study_metadata": {
            "n_conditions": 4,
            "n_results": len(results),
        },
        "results": [asdict(r) for r in results],
    }
    
    # Save JSON
    output_path = output_dir / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"✓ Saved results to {output_path}")
    
    # Generate summary statistics
    summary = {}
    for condition in ["full", "no_semantic_guidance", "no_hrm_field", "fixed_topology"]:
        condition_results = [r for r in results if r.condition == condition]
        if condition_results:
            summary[condition] = {
                "n_runs": len(condition_results),
                "mean_state_norm": float(np.mean([r.mean_state_norm for r in condition_results])),
                "mean_region_count": float(np.mean([r.final_region_count for r in condition_results])),
                "mean_collapse_events": float(np.mean([r.collapse_events for r in condition_results])),
                "mean_edge_rewiring": float(np.mean([r.edge_rewiring_count for r in condition_results])),
            }
    
    summary_path = output_dir / "summary_statistics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")
    
    return summary


def generate_comparison_plots(results: list, output_dir: Path = Path("artifacts/ablation")):
    """Generate comparison plots for ablation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conditions = ["full", "no_semantic_guidance", "no_hrm_field", "fixed_topology"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("RCM Ablation Study: Condition Comparison", fontsize=16, fontweight="bold")
    
    # Plot 1: State norm
    ax = axes[0, 0]
    for condition in conditions:
        vals = [r.mean_state_norm for r in results if r.condition == condition]
        ax.scatter([condition] * len(vals), vals, alpha=0.6, s=80)
    ax.set_ylabel("Mean State Norm")
    ax.set_title("State Stability")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Region count
    ax = axes[0, 1]
    for condition in conditions:
        vals = [r.final_region_count for r in results if r.condition == condition]
        ax.scatter([condition] * len(vals), vals, alpha=0.6, s=80)
    ax.set_ylabel("Region Count")
    ax.set_title("Topology Organization")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Collapse events
    ax = axes[0, 2]
    for condition in conditions:
        vals = [r.collapse_events for r in results if r.condition == condition]
        ax.scatter([condition] * len(vals), vals, alpha=0.6, s=80, color="red")
    ax.set_ylabel("Collapse Events")
    ax.set_title("Collapse Robustness")
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Edge rewiring
    ax = axes[1, 0]
    for condition in conditions:
        vals = [r.edge_rewiring_count for r in results if r.condition == condition]
        ax.scatter([condition] * len(vals), vals, alpha=0.6, s=80)
    ax.set_ylabel("Edge Rewiring Events")
    ax.set_title("Topology Plasticity")
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Confidence
    ax = axes[1, 1]
    for condition in conditions:
        vals = [r.confidence_mean for r in results if r.condition == condition]
        ax.scatter([condition] * len(vals), vals, alpha=0.6, s=80)
    ax.set_ylabel("Mean Confidence")
    ax.set_title("Signal Coherence")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Child manifolds
    ax = axes[1, 2]
    for condition in conditions:
        vals = [r.child_manifolds_created for r in results if r.condition == condition]
        ax.scatter([condition] * len(vals), vals, alpha=0.6, s=80)
    ax.set_ylabel("Child Manifolds Created")
    ax.set_title("Hierarchy Formation")
    ax.grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)
    
    plt.tight_layout()
    plot_path = output_dir / "ablation_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved comparison plot to {plot_path}")


def main():
    print("=" * 70)
    print("COMPREHENSIVE RCM ABLATION STUDY")
    print("=" * 70)
    
    # Run study
    results = run_comprehensive_study(
        n_seeds=3,
        graph_sizes=[6, 9],
        scenarios=["baseline", "shuffled", "noisy_gaussian", "missing_signals"],
        n_steps=40,
    )
    
    print(f"\n✓ Study complete: {len(results)} results collected")
    
    # Save and analyze
    summary = save_results(results)
    generate_comparison_plots(results)
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for condition, stats in summary.items():
        print(f"\n{condition}:")
        for key, value in stats.items():
            if key != "n_runs":
                print(f"  {key}: {value:.4f}")
        print(f"  (n={stats['n_runs']})")


if __name__ == "__main__":
    main()
