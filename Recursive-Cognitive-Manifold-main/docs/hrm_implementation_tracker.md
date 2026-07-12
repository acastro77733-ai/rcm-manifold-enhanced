# HRM Implementation Tracker

## Repository-specific gap analysis

- The runtime control path is centered on `RecursiveCognitiveManifold.step()` and originally executed input, propagation, episodic recall, synchronization, structural updates, geometry feedback, region updates, semantic guidance, field updates, child manifolds, and only then an observational state-engine summary.
- `rcm.dynamics.state_engine.StateEngine` projected node updates but those updates were not written back into the manifold, so unified evolution did not causally govern subsequent node behavior.
- `rcm.state_engine.BoundedStateEngine` duplicated a disconnected summary-only state representation instead of sharing the authoritative runtime state.
- HRM field controls used either weak synthetic application gains in the manifold or synthetic monkeypatched condition behavior in the experiment layer.
- Hierarchy formed meta-regions, but top-down feedback was weak and not explicitly forecast-driven.
- Topology repair and geometry surgery remain heuristic and not yet utility-transactional.
- Representation adaptation, forecasting contracts, immutable data splits, surgical ablations, and multi-seed evaluation still need substantial work.

## Files and APIs changing

- Runtime control: `rcm/cognition/manifold.py`, `rcm/dynamics/state_engine.py`, `rcm/state_engine.py`
- Field dynamics and controls: `rcm/dynamics/fields/base.py`, `rcm/dynamics/fields/nonlinear.py`, `rcm/dynamics/fields/legacy.py`, `rcm/config/field.py`
- Hierarchy: `rcm/hierarchy/abstraction.py`, `rcm/hierarchy/child_manifold.py`
- Experiment control surfaces: `enhanced_ablation_study.py`
- Focused validations: `tests/test_unified_state_engine.py`, `tests/test_state_engine_and_fields.py`, `tests/test_experimental_isolation.py`, `tests/test_active_hierarchy.py`

## Step dependencies

1. Map the execution path before changing control ownership.
2. Establish canonical governed state before changing field, hierarchy, topology, or representation utility.
3. Put coupled evolution into the governed state before trusting ablations or forecast evaluation.
4. Feed governed updates back into nodes before claiming causal activation.
5. Route all field modes through one equation family before evaluating HRM controls.
6. Add influence diagnostics before interpreting field gains.
7. Make hierarchy forecastive and bidirectional before assessing its utility.
8. Make topology transactional before measuring task benefit.
9. Tie representation allocation to forecast/stability utility before adding resource-constrained tasks.
10. Add the common forecast contract before immutable train/validation/test evaluation.
11. Add immutable dataset splits before multi-seed evaluation.
12. Build mechanism-requiring tasks before surgical ablations and large experiments.
13. Build exact causal ablations before claims about mechanism benefit.
14. Run paired multi-seed evaluation only after steps 10-13 are complete.
15. Freeze baseline artifacts after numerical and API regression checks are in place.

## Acceptance criteria

1. Runtime order and disconnected components documented.
2. One governed state owns node, region, geometry, topology, memory, hierarchy, resource, stability, and forecast blocks.
3. Evolution logs separate external, local, diffusion, memory, regional, hierarchy, topology, damping, and residual contributions.
4. Manifold trajectories measurably change when the governed engine is ablated.
5. Full, no-field, identity, diffusion-only, randomized-parameter, linearized, and frozen-parameter controls share one field equation family.
6. Field metrics flag negligible, active, dominant, and clipped influence.
7. Higher-level forecasts and constraints measurably change lower-level dynamics.
8. Topology updates must be accepted or rejected transactionally against utility and cost.
9. Representation width must adapt under finite budgets using forecast/stress/uncertainty utility.
10. All evaluated models must expose `fit`, `observe`, `predict`, `update`, and `reset`.
11. Training, validation, and test streams must be immutable and RNG-isolated.
12. Task suite must include forecasting, reconstruction, topology change, hierarchy, long memory, shift, lesion recovery, and resource-constrained workloads.
13. Ablations must remove only their intended causal pathway.
14. Multi-seed evaluation must report raw runs, paired differences, intervals, effects, failures, and normalized compute.
15. Baseline freeze must include regression, convergence, invariant, scale, precision, manifest, and assessment artifacts.

## Risk register

- Numerical stability: stronger coupled fields and hierarchy can saturate states if gains outpace damping or clipping.
- Evaluation leakage: current benchmark code still mixes runtime mutation and evaluation semantics.
- Computational cost: transactional topology plus forecast-aware evaluation will increase runtime without tighter budgeting.
- Backward compatibility: snapshot and experiment surfaces are being extended; compatibility must be maintained through tests.

## Step status

1. `complete`
Runtime path mapped and recorded here. Disconnected summary-only unified state and synthetic field controls identified.
Validation: targeted code inspection plus runtime-path review.

2. `complete`
Canonical governed state introduced in `rcm/dynamics/state_engine.py` and exposed through the compatibility surface in `rcm/state_engine.py`.
Validation: `PYTHONPATH=. pytest tests/test_unified_state_engine.py tests/test_state_engine_and_fields.py -q`

3. `complete`
Coupled evolution now logs separate external, local reaction, diffusion, memory, regional coupling, hierarchical feedback, topology transport, damping, and residual terms.
Validation: `PYTHONPATH=. pytest tests/test_unified_state_engine.py tests/test_state_engine_and_fields.py -q`

4. `complete`
Governed node updates are written back into the manifold, and ablation of the governed engine changes trajectories.
Validation: `PYTHONPATH=. pytest tests/test_unified_state_engine.py tests/test_state_engine_and_fields.py -q`

5. `complete`
Field controls now run through one equation family with runtime modes instead of synthetic monkeypatched perturbations.
Validation: `PYTHONPATH=. pytest tests/test_state_engine_and_fields.py tests/test_experimental_isolation.py -q`

6. `complete`
Field metrics now report influence magnitude and status (`negligible`, `active`, `dominant`, `clipped`) and use configurable application gains.
Validation: `PYTHONPATH=. pytest tests/test_state_engine_and_fields.py tests/test_experimental_isolation.py -q`

7. `complete`
Hierarchy now uses weighted lower-level summaries and explicit child-level forecasts/constraints that feed back into parent states.
Validation: `PYTHONPATH=. pytest tests/test_active_hierarchy.py tests/test_unified_state_engine.py -q`

8. `complete`
Topology repair and simplicial surgery now pass through utility-gated accept/reject logic with explicit costs, reasons, and rollback on rejected proposals.
Validation: `PYTHONPATH=. pytest tests/test_topology_adaptation.py tests/test_geometry_feedback.py -q`

9. `complete`
Representation allocation now consumes forecast residual, uncertainty, geometric stress, stability, marginal improvement, and budget pressure while preserving the old controller API as a fallback.
Validation: `PYTHONPATH=. pytest tests/test_representation_controller.py -q`

10. `complete`
A shared experimental forecasting contract (`fit`, `observe`, `predict`, `update`, `reset`) now exists for RCM conditions and baseline controls in `rcm/experiments/forecasting.py`.
Validation: `PYTHONPATH=. pytest tests/test_forecasting_pipeline.py -q`

11. `complete`
Deterministic immutable train/validation/test splits with independent RNG streams now exist in the forecasting module.
Validation: `PYTHONPATH=. pytest tests/test_forecasting_pipeline.py -q`

12. `complete`
A mechanism task registry now covers temporal forecasting, reconstruction, topology change adaptation, hierarchy, long-horizon memory, distribution shift, lesion recovery, and resource-constrained computation with explicit rationales.
Validation: `PYTHONPATH=. pytest tests/test_forecasting_pipeline.py -q`

13. `complete`
Experimental ablations now map to explicit runtime pathway toggles or field modes in both the forecasting harness and the legacy benchmark suite, including guidance, HRM, hierarchy, topology, structural plasticity, episodic recall, and fixed-graph controls.
Validation: `PYTHONPATH=. pytest tests/test_forecasting_pipeline.py tests/test_benchmark_forecast_integration.py -q`

14. `complete`
A full paired 30-seed primary evaluation has been executed and persisted at `artifacts/releases/primary_multi_seed_report.json`. The harness reports raw per-run data, means, medians, dispersion, paired differences, bootstrap intervals, effect sizes, failure rates, runtime, peak memory, op counts, success-threshold checks, and Holm-Bonferroni corrected p-values.
Validation: `PYTHONPATH=. pytest tests/test_multi_seed_evaluation.py tests/test_benchmark_forecast_integration.py -q`

15. `complete`
Release artifacts now include the machine-readable experiment manifest, the plain-language baseline assessment, and the persisted primary multi-seed report. The repository has deterministic reference snapshots, serialization round-trip tests, API compatibility checks, numerical convergence checks, invariant tests, scale-sweep coverage, precision comparisons, and CI smoke gates for the new primary evaluation and release artifact validation.
Validation: `PYTHONPATH=. pytest tests/test_release_artifacts.py tests/test_reference_snapshots.py tests/test_numerical_validation.py tests/test_runtime_invariants.py tests/test_scale_sweeps.py -q`
