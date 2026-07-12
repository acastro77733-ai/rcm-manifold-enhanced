import argparse

import numpy as np

from rcm.experiments.specialization_demo import build_demo_manifold
from rcm.experiments.specialization_demo import build_inputs
from rcm.experiments.specialization_demo import canonical_regions
from rcm.experiments.specialization_demo import collapse_runtime_summary
from rcm.reproducibility import seed_everything


def lesion_node(manifold, node_id: int):
    manifold.nodes[node_id].local_state = np.zeros(manifold.state_dim, dtype=float)
    manifold.nodes[node_id].energy = 0.2
    manifold.nodes[node_id].confidence = 0.1


def main():
    parser = argparse.ArgumentParser(description="Run lesion recovery on the Recursive Cognitive Manifold demo setup.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic manifold HRM fields.")
    parser.add_argument("--steps", type=int, default=18, help="Number of steps to execute for recovery evaluation.")
    args = parser.parse_args()

    resolved_seed = seed_everything(args.seed)
    _, manifold = build_demo_manifold(seed=resolved_seed)
    history = {
        "time": [],
        "collapse_energy": [],
        "collapse_flag": [],
        "regional_collapse_count": [],
    }
    for step_index, external_input in enumerate(build_inputs(args.steps)):
        if step_index == 8:
            lesion_node(manifold, 2)
        manifold.step(external_input)
        history["time"].append(step_index)
        history["collapse_energy"].append(float(getattr(manifold, "collapse_energy", 0.0)))
        history["collapse_flag"].append(bool(getattr(manifold, "collapse_flag", False)))
        history["regional_collapse_count"].append(
            sum(1 for metrics in getattr(manifold, "region_collapse_metrics", {}).values() if metrics.get("collapse_flag"))
        )

    print("Post-lesion snapshot:")
    print(manifold.snapshot())
    print("Collapse summary:")
    print(collapse_runtime_summary(manifold, history))
    print("Regions after recovery:")
    for signature, specialization, stability in canonical_regions(manifold):
        print(f"  {signature}: specialization={specialization}, stability={stability:.3f}")


if __name__ == "__main__":
    main()