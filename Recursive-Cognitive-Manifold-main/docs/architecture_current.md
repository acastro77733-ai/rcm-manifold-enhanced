# Current architecture baseline

This document freezes the repository's current runtime behavior so later architectural work can be compared against a stable reference.

## Runtime layers

- Cognition core: recursive manifold state, nodes, edges, regions, and child manifolds
- Dynamics: propagation, synchronization, plasticity, regional HRM fields, and geometry feedback
- Memory: episodic, semantic, and structural memory
- Experiments: benchmarks, lesion recovery, specialization demo, and perturbation testing
- Persistence: checkpoint serialization and restoration

## Stable public namespaces

The following remain the stable public entry points for external integrations:

- rcm.cognition
- rcm.dynamics
- rcm.geometry
- rcm.hierarchy
- rcm.memory
- rcm.reproducibility

## Current baseline behaviors

- Deterministic seeds are controlled with normalize_seed / seed_everything.
- Unit tests exercise experimental isolation, persistence, geometry feedback, semantic guidance, and active hierarchy behavior.
- Benchmark outputs are serialized as JSON and compared against stored baselines.
- Snapshot output includes node, edge, region, field, semantic guidance, and geometry feedback summaries.

## Baseline release contract

The repository baseline is intended to remain reproducible through:

- fixed random seeds,
- deterministic benchmark metadata,
- serialized reference snapshots, and
- compatibility tests for core APIs.
