import argparse
import copy
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.geometry.simplicial_complex import DynamicSimplicialComplex
from rcm.reproducibility import normalize_seed
from rcm.reproducibility import seed_everything
from rcm_collapse_integration import install_collapse_attractor_support


install_collapse_attractor_support(RecursiveCognitiveManifold)


SPECIALIZATION_COLORS = {
    "sensorimotor": "#d97706",
    "associative": "#2563eb",
    "predictive": "#059669",
    "integrative": "#9333ea",
    "undifferentiated": "#6b7280",
}

TRACKED_EDGES = [
    (0, 1),
    (1, 2),
    (0, 2),
    (3, 4),
    (4, 5),
    (3, 5),
    (2, 3),
    (1, 4),
]


def build_demo_manifold(seed: int = 42):
    seed = normalize_seed(seed)
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.9],
        [2.2, 0.9],
        [3.2, 0.0],
        [4.2, 0.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [3, 4, 5],
    ])
    complex_ = DynamicSimplicialComplex(vertices, faces)
    manifold = RecursiveCognitiveManifold.from_simplicial_complex(complex_, state_dim=4, hrm_seed=seed)
    manifold.connect(2, 3, strength=0.32, latency=2.4)
    manifold.connect(1, 4, strength=0.28, latency=2.8)
    return vertices, manifold


def build_inputs(steps):
    inputs = []
    phase_a = {
        0: np.array([1.0, 0.15, 0.0, 0.0]),
        1: np.array([0.96, 0.18, 0.0, 0.0]),
        2: np.array([0.92, 0.12, 0.0, 0.0]),
        3: np.array([0.0, 0.0, 0.95, 0.1]),
        4: np.array([0.0, 0.0, 0.92, 0.12]),
        5: np.array([0.0, 0.0, 0.88, 0.08]),
    }
    phase_b = {
        0: np.array([0.85, 0.22, 0.0, 0.0]),
        1: np.array([0.82, 0.24, 0.0, 0.0]),
        2: np.array([0.8, 0.2, 0.0, 0.0]),
        3: np.array([0.0, 0.0, 0.75, 0.2]),
        4: np.array([0.0, 0.0, 0.78, 0.24]),
        5: np.array([0.0, 0.0, 0.74, 0.18]),
    }
    phase_c = {
        0: np.array([0.78, 0.24, 0.0, 0.08]),
        1: np.array([0.75, 0.26, 0.0, 0.1]),
        2: np.array([0.72, 0.22, 0.0, 0.12]),
        3: np.array([0.0, 0.04, 0.82, 0.16]),
        4: np.array([0.0, 0.06, 0.8, 0.18]),
        5: np.array([0.0, 0.05, 0.78, 0.2]),
    }
    phases = [phase_a, phase_b, phase_c]
    for step in range(steps):
        inputs.append(phases[min(step // 6, len(phases) - 1)])
    return inputs


def edge_strength(manifold, source, target):
    values = []
    for left, right in ((source, target), (target, source)):
        edge = manifold.edges.get((left, right))
        if edge is not None:
            values.append(edge.strength)
    if not values:
        return 0.0
    return float(np.mean(values))


def canonical_regions(manifold):
    regions = []
    for signature, region in manifold.regions.items():
        normalized = tuple(sorted(int(node_id) for node_id in signature))
        regions.append((normalized, region.specialization, region.stability))
    return sorted(regions)


def collect_history(manifold, history, step_index):
    history["time"].append(step_index)
    history["region_count"].append(len(manifold.regions))
    history["child_count"].append(len(manifold.child_manifolds))
    history["child_nodes"].append(manifold.child_manifolds[0].snapshot()["node_count"] if manifold.child_manifolds else 0)
    history["regions"].append(canonical_regions(manifold))
    history["bridge_strength"].append(edge_strength(manifold, 2, 3))
    history["cross_cluster_strength"].append(edge_strength(manifold, 1, 4))
    history["cluster_a_strength"].append(np.mean([edge_strength(manifold, 0, 1), edge_strength(manifold, 1, 2), edge_strength(manifold, 0, 2)]))
    history["cluster_b_strength"].append(np.mean([edge_strength(manifold, 3, 4), edge_strength(manifold, 4, 5), edge_strength(manifold, 3, 5)]))
    manifold_metrics = manifold.last_field_metrics
    history["manifold_sync_pressure"].append(manifold_metrics.get("synchronization_pressure", 0.0))
    history["manifold_plasticity_pressure"].append(manifold_metrics.get("plasticity_pressure", 0.0))
    region_sync = [region.field_metrics.get("synchronization_pressure", 0.0) for region in manifold.regions.values()]
    region_plasticity = [region.field_metrics.get("plasticity_pressure", 0.0) for region in manifold.regions.values()]
    history["regional_sync_pressure"].append(float(np.mean(region_sync)) if region_sync else 0.0)
    history["regional_plasticity_pressure"].append(float(np.mean(region_plasticity)) if region_plasticity else 0.0)
    child_metrics = [child.last_field_metrics for child in manifold.child_manifolds if getattr(child, "last_field_metrics", None)]
    history["child_sync_pressure"].append(float(np.mean([metrics.get("synchronization_pressure", 0.0) for metrics in child_metrics])) if child_metrics else 0.0)
    history["child_plasticity_pressure"].append(float(np.mean([metrics.get("plasticity_pressure", 0.0) for metrics in child_metrics])) if child_metrics else 0.0)
    feedback_metrics = getattr(manifold, "last_child_field_metrics", [])
    history["feedback_sync_pressure"].append(float(np.mean([metrics.get("sync_pressure", 0.0) for metrics in feedback_metrics])) if feedback_metrics else 0.0)
    history["feedback_plasticity_pressure"].append(float(np.mean([metrics.get("plasticity_pressure", 0.0) for metrics in feedback_metrics])) if feedback_metrics else 0.0)
    collapse_metrics = getattr(manifold, "last_collapse_metrics", {})
    history["collapse_energy"].append(float(collapse_metrics.get("collapse_energy", 0.0)))
    history["collapse_flag"].append(bool(collapse_metrics.get("collapse_flag", False)))
    history["collapse_recovery_ratio"].append(
        float(collapse_metrics.get("distance_to_anchor_before", 0.0) - collapse_metrics.get("distance_to_anchor_after", 0.0))
    )
    history["regional_collapse_count"].append(
        sum(1 for metrics in getattr(manifold, "region_collapse_metrics", {}).values() if metrics.get("collapse_flag"))
    )
    for source, target in TRACKED_EDGES:
        history["edge_series"].setdefault((source, target), []).append(edge_strength(manifold, source, target))


def plot_topology(ax, vertices, manifold, title):
    ax.set_title(title)
    for source, target in TRACKED_EDGES:
        strength = edge_strength(manifold, source, target)
        if strength <= 0.0:
            continue
        line = vertices[[source, target]]
        color = "#111827" if strength >= 0.25 else "#9ca3af"
        alpha = 0.85 if strength >= 0.25 else 0.35
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1.0 + 3.0 * strength, alpha=alpha)

    for node_id, coords in enumerate(vertices):
        node = manifold.nodes[node_id]
        color = SPECIALIZATION_COLORS.get(node.specialization, SPECIALIZATION_COLORS["undifferentiated"])
        ax.scatter(coords[0], coords[1], s=280, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(coords[0], coords[1] + 0.14, f"n{node_id}", ha="center", va="bottom", fontsize=9)

    for signature, region in manifold.regions.items():
        points = vertices[[int(node_id) for node_id in signature]]
        center = np.mean(points, axis=0)
        ax.text(
            center[0],
            center[1] - 0.22,
            f"{region.specialization}\nS={region.stability:.2f}",
            ha="center",
            va="top",
            fontsize=9,
            color=SPECIALIZATION_COLORS.get(region.specialization, SPECIALIZATION_COLORS["undifferentiated"]),
        )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 4.7)
    ax.set_ylim(-0.6, 1.5)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def plot_edge_rewiring(ax, history):
    ax.set_title("Edge Rewiring")
    ax.plot(history["time"], history["cluster_a_strength"], label="cluster A internal", color="#d97706", linewidth=2.5)
    ax.plot(history["time"], history["cluster_b_strength"], label="cluster B internal", color="#059669", linewidth=2.5)
    ax.plot(history["time"], history["bridge_strength"], label="bridge 2-3", color="#111827", linewidth=2.5)
    ax.plot(history["time"], history["cross_cluster_strength"], label="cross edge 1-4", color="#6b7280", linewidth=2.0, linestyle="--")
    ax.axhline(0.25, color="#9ca3af", linestyle=":", linewidth=1.2)
    ax.text(history["time"][0], 0.27, "region split threshold", color="#6b7280", fontsize=8, va="bottom")
    ax.set_xlabel("step")
    ax.set_ylabel("mean strength")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")


def plot_abstraction(ax, history):
    ax.set_title("Meta-Region Emergence")
    ax.step(history["time"], history["region_count"], where="post", label="regions", color="#2563eb", linewidth=2.5)
    ax.step(history["time"], history["child_nodes"], where="post", label="meta nodes", color="#9333ea", linewidth=2.5)
    emergence_step = next((time for time, count in zip(history["time"], history["child_count"]) if count > 0), None)
    if emergence_step is not None:
        ax.axvline(emergence_step, color="#9333ea", linestyle="--", linewidth=1.5)
        ax.text(emergence_step + 0.2, max(history["child_nodes"]) + 0.1, "child manifold appears", color="#9333ea", fontsize=8)
    ax.set_xlabel("step")
    ax.set_ylabel("count")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")


def plot_specialization(ax, history):
    ax.set_title("Region Specialization")
    ax.set_xlim(min(history["time"]) - 0.5, max(history["time"]) + 0.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["region 0", "region 1", "meta"])
    for time_index, regions in zip(history["time"], history["regions"]):
        for region_index, (_, specialization, stability) in enumerate(regions[:2]):
            color = SPECIALIZATION_COLORS.get(specialization, SPECIALIZATION_COLORS["undifferentiated"])
            ax.scatter(time_index, region_index, s=90 + 110 * stability, color=color, alpha=0.85)
        if len(regions) >= 2:
            meta_color = SPECIALIZATION_COLORS.get(regions[0][1], SPECIALIZATION_COLORS["undifferentiated"])
            ax.scatter(time_index, 2, s=60 + 40 * history["child_nodes"][time_index], marker="s", color=meta_color, alpha=0.7)
    ax.set_xlabel("step")
    ax.grid(alpha=0.2)


def plot_field_pressures(ax, history):
    ax.set_title("Field + Collapse Pressures")
    ax.plot(history["time"], history["regional_sync_pressure"], label="regional sync", color="#d97706", linewidth=2.2)
    ax.plot(history["time"], history["regional_plasticity_pressure"], label="regional plasticity", color="#b45309", linewidth=2.0, linestyle="--")
    ax.plot(history["time"], history["manifold_sync_pressure"], label="manifold sync", color="#2563eb", linewidth=2.2)
    ax.plot(history["time"], history["manifold_plasticity_pressure"], label="manifold plasticity", color="#1d4ed8", linewidth=2.0, linestyle="--")
    ax.plot(history["time"], history["feedback_sync_pressure"], label="child feedback sync", color="#9333ea", linewidth=2.0)
    ax.plot(history["time"], history["feedback_plasticity_pressure"], label="child feedback plasticity", color="#7e22ce", linewidth=1.8, linestyle="--")
    ax.plot(history["time"], history["collapse_energy"], label="collapse energy", color="#dc2626", linewidth=2.2)
    flagged_times = [time for time, flag in zip(history["time"], history["collapse_flag"]) if flag]
    flagged_energy = [energy for energy, flag in zip(history["collapse_energy"], history["collapse_flag"]) if flag]
    if flagged_times:
        ax.scatter(flagged_times, flagged_energy, label="collapse flagged", color="#7f1d1d", s=36, zorder=4)
    ax.set_xlabel("step")
    ax.set_ylabel("pressure")
    ax.set_ylim(0.0, max(1.05, max(history["collapse_energy"], default=0.0) + 0.1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")


def collapse_runtime_summary(manifold, history):
    flagged_steps = [time for time, flag in zip(history["time"], history["collapse_flag"]) if flag]
    return {
        "final_collapse_flag": bool(getattr(manifold, "collapse_flag", False)),
        "final_collapse_energy": float(getattr(manifold, "collapse_energy", 0.0)),
        "collapse_events": len(getattr(manifold, "collapse_memory", [])),
        "peak_collapse_energy": max(history["collapse_energy"], default=0.0),
        "peak_regional_collapse_count": max(history["regional_collapse_count"], default=0),
        "flagged_steps": flagged_steps,
        "attractor_initialized": getattr(manifold, "attractor_anchor", None) is not None,
    }


def render_dashboard(vertices, initial_manifold, current_manifold, history, output_path, title_suffix=""):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    plot_topology(axes[0, 0], vertices, initial_manifold, "Initial Topology")
    plot_topology(axes[0, 1], vertices, current_manifold, "Current Topology")
    plot_edge_rewiring(axes[1, 0], history)
    plot_abstraction(axes[1, 1], history)
    plot_specialization(axes[2, 0], history)
    plot_field_pressures(axes[2, 1], history)

    title = "Recursive Cognitive Manifold Demo"
    if title_suffix:
        title = f"{title} | {title_suffix}"
    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_figure(vertices, initial_manifold, final_manifold, history, output_path):
    render_dashboard(vertices, initial_manifold, final_manifold, history, output_path, title_suffix="final state")


def history_prefix(history, frame_index):
    prefix = {}
    for key, value in history.items():
        if key == "edge_series":
            prefix[key] = {edge_key: series[:frame_index + 1] for edge_key, series in value.items()}
        else:
            prefix[key] = value[:frame_index + 1]
    return prefix


def render_frame_sequence(vertices, initial_manifold, frame_manifolds, history, frames_dir):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, manifold in enumerate(frame_manifolds):
        history_slice = history_prefix(history, frame_index)
        output_path = frames_dir / f"frame_{frame_index:03d}.png"
        title_suffix = f"step {frame_index + 1} | regions {history_slice['region_count'][-1]} | meta nodes {history_slice['child_nodes'][-1]}"
        render_dashboard(vertices, initial_manifold, manifold, history_slice, output_path, title_suffix=title_suffix)


def render_mp4_from_frames(frames_dir, output_path, fps):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for mp4 output; use --animation frames or install ffmpeg.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%03d.png"),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def run_demo(steps, capture_frames=False, seed: int = 42):
    seed = normalize_seed(seed)
    vertices, manifold = build_demo_manifold(seed=seed)
    _, initial_manifold = build_demo_manifold(seed=seed)
    history = {
        "time": [],
        "region_count": [],
        "child_count": [],
        "child_nodes": [],
        "regions": [],
        "bridge_strength": [],
        "cross_cluster_strength": [],
        "cluster_a_strength": [],
        "cluster_b_strength": [],
        "manifold_sync_pressure": [],
        "manifold_plasticity_pressure": [],
        "regional_sync_pressure": [],
        "regional_plasticity_pressure": [],
        "child_sync_pressure": [],
        "child_plasticity_pressure": [],
        "feedback_sync_pressure": [],
        "feedback_plasticity_pressure": [],
        "collapse_energy": [],
        "collapse_flag": [],
        "collapse_recovery_ratio": [],
        "regional_collapse_count": [],
        "edge_series": {},
    }
    frame_manifolds = []

    for step_index, external_input in enumerate(build_inputs(steps)):
        manifold.step(external_input)
        collect_history(manifold, history, step_index)
        if capture_frames:
            frame_manifolds.append(copy.deepcopy(manifold))

    return vertices, initial_manifold, manifold, history, frame_manifolds


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize recursive cognitive manifold specialization and rewiring.")
    parser.add_argument("--steps", type=int, default=18, help="Number of deterministic update steps to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic manifold HRM fields.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/recursive_cognitive_manifold_demo.png"), help="Path to the output visualization image.")
    parser.add_argument("--animation", choices=["none", "frames", "mp4"], default="none", help="Emit a frame sequence or mp4 from the same deterministic run.")
    parser.add_argument("--frames-dir", type=Path, default=Path("artifacts/frames"), help="Directory used for animation frame output.")
    parser.add_argument("--mp4-output", type=Path, default=Path("artifacts/recursive_cognitive_manifold_demo.mp4"), help="Path to the mp4 animation output.")
    parser.add_argument("--fps", type=int, default=4, help="Frame rate used for mp4 output.")
    return parser


def main():
    args = build_parser().parse_args()
    resolved_seed = seed_everything(args.seed)
    capture_frames = args.animation in {"frames", "mp4"}
    vertices, initial_manifold, final_manifold, history, frame_manifolds = run_demo(
        args.steps,
        capture_frames=capture_frames,
        seed=resolved_seed,
    )
    render_figure(vertices, initial_manifold, final_manifold, history, args.output)

    if args.animation in {"frames", "mp4"}:
        render_frame_sequence(vertices, initial_manifold, frame_manifolds, history, args.frames_dir)
        print(f"Saved frame sequence to {args.frames_dir}")
    if args.animation == "mp4":
        render_mp4_from_frames(args.frames_dir, args.mp4_output, args.fps)
        print(f"Saved mp4 animation to {args.mp4_output}")

    print(f"Saved visualization to {args.output}")
    print("Final snapshot:")
    print(final_manifold.snapshot())
    collapse_summary = collapse_runtime_summary(final_manifold, history)
    print("Collapse summary:")
    print(
        {
            "final_collapse_flag": collapse_summary["final_collapse_flag"],
            "final_collapse_energy": round(collapse_summary["final_collapse_energy"], 6),
            "collapse_events": collapse_summary["collapse_events"],
            "peak_collapse_energy": round(collapse_summary["peak_collapse_energy"], 6),
            "peak_regional_collapse_count": collapse_summary["peak_regional_collapse_count"],
            "flagged_steps": collapse_summary["flagged_steps"],
            "attractor_initialized": collapse_summary["attractor_initialized"],
        }
    )
    print("Final regions:")
    for signature, specialization, stability in canonical_regions(final_manifold):
        print(f"  {signature}: specialization={specialization}, stability={stability:.3f}")
    if final_manifold.child_manifolds:
        print("Child manifold snapshot:")
        print(final_manifold.child_manifolds[0].snapshot())


if __name__ == "__main__":
    main()