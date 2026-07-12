import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enhanced_ablation_study import (
    bootstrap_confidence_interval,
    evaluate_hrm_validation,
    generate_trial_inputs,
    build_condition_manifold,
)
from rcm.cognition.manifold import RecursiveCognitiveManifold


def test_generate_trial_inputs_are_reproducible_and_isolated():
    inputs_a = generate_trial_inputs(graph_size=5, seed=11, scenario="baseline", n_steps=4, state_dim=4)
    inputs_b = generate_trial_inputs(graph_size=5, seed=11, scenario="baseline", n_steps=4, state_dim=4)
    assert len(inputs_a) == 4
    assert len(inputs_b) == 4
    assert inputs_a[0].keys() == inputs_b[0].keys()
    for left, right in zip(inputs_a, inputs_b):
        for node_id in left:
            np.testing.assert_allclose(left[node_id], right[node_id])
            assert left[node_id] is not right[node_id]


def test_no_hrm_field_keeps_other_dynamics_active():
    manifold = build_condition_manifold(condition="no_hrm_field", complex_=None, state_dim=4, seed=7)
    assert manifold.hrm_mode == "no_hrm_field"
    manifold.add_node(0, np.array([0.1, 0.2, 0.3, 0.4]))
    manifold.add_node(1, np.array([0.4, 0.3, 0.2, 0.1]))
    manifold.connect(0, 1, strength=0.6, latency=1.0)
    before = manifold.nodes[0].local_state.copy()
    manifold.step({0: np.array([1.0, 0.0, 0.0, 0.0])})
    after = manifold.nodes[0].local_state
    assert not np.allclose(before, after)
    assert manifold.last_field_metrics.get("regional_field_updates", 0) == 0


def test_hrm_validation_summary_is_well_formed():
    full = np.array([0.8, 0.9, 0.95, 0.85])
    control = np.array([0.7, 0.75, 0.78, 0.72])
    ci = bootstrap_confidence_interval(full - control, n_boot=200, seed=1)
    summary = evaluate_hrm_validation(full, control, threshold=0.05)
    assert ci[0] < ci[1]
    assert summary["decision"] in {"pass", "tie", "reject"}


def test_full_and_no_hrm_conditions_diverge_on_cloned_manifolds():
    full = build_condition_manifold(condition="full", complex_=None, state_dim=4, seed=99)
    no_hrm = build_condition_manifold(condition="no_hrm_field", complex_=None, state_dim=4, seed=99)

    full.add_node(2, np.array([0.5, -0.3, 0.4, 0.2]))
    full.add_node(3, np.array([-0.2, 0.7, 0.1, 0.3]))
    full.connect(2, 3, strength=0.7, latency=1.0)

    no_hrm.add_node(2, np.array([0.5, -0.3, 0.4, 0.2]))
    no_hrm.add_node(3, np.array([-0.2, 0.7, 0.1, 0.3]))
    no_hrm.connect(2, 3, strength=0.7, latency=1.0)

    full._apply_regional_fields()
    no_hrm._apply_regional_fields()

    assert full.last_field_metrics["field_energy"] != no_hrm.last_field_metrics["field_energy"]
