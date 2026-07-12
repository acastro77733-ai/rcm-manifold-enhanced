import argparse

from rcm.experiments.specialization_demo import run_demo
from rcm.memory.recall import recall_node_history
from rcm.memory.recall import recall_recent_episode
from rcm.memory.recall import recall_semantic_motifs
from rcm.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Run memory recall diagnostics on a deterministic RCM demo trace.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic manifold HRM fields.")
    parser.add_argument("--steps", type=int, default=18, help="Number of demo steps to generate before recall queries.")
    args = parser.parse_args()

    resolved_seed = seed_everything(args.seed)
    _, _, manifold, _, _ = run_demo(args.steps, capture_frames=False, seed=resolved_seed)
    print(f"Run configuration: seed={resolved_seed}, steps={args.steps}")
    print("Recent episode:")
    print(recall_recent_episode(manifold))
    print("\nNode 0 history:")
    print(recall_node_history(manifold, 0))
    print("\nSemantic motifs:")
    print(recall_semantic_motifs(manifold, min_count=2))


if __name__ == "__main__":
    main()