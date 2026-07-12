import unittest

import numpy as np

from rcm.cognition.edge import CognitiveEdge
from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.cognition.node import CognitiveNode
from rcm.dynamics.regional_hrm_field import RegionalHRMField
from rcm_collapse_integration import CollapseAttractorMonitor
from rcm_collapse_integration import CollapseConfig
from rcm_collapse_integration import install_collapse_attractor_support


class CollapseIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = CollapseConfig(
            energy_threshold=0.5,
            cooldown_steps=2,
            anchor_update_rate=0.5,
            minimum_rebind_alpha=0.2,
            maximum_rebind_alpha=0.85,
        )
        self.monitor = CollapseAttractorMonitor(self.config)
        self.scope_id = "test-scope"

    def _make_node(self, values):
        return CognitiveNode(local_state=np.asarray(values, dtype=float))

    def _observe(self, nodes, previous_state, field_owner=None, edge_strength=0.5, step=0):
        edge = CognitiveEdge(strength=edge_strength, latency=1.0)
        return self.monitor.observe(
            scope_id=self.scope_id,
            node_items=[(index, node) for index, node in enumerate(nodes)],
            edge_entries=[((0, 1), edge)],
            previous_state=np.asarray(previous_state, dtype=float),
            step=step,
            dt=1.0,
            field_owner=field_owner,
            auto_rebind=True,
        ), edge

    def test_collapse_flagging_records_event_and_sets_cooldown(self):
        nodes = [
            self._make_node([4.0, 4.0, -4.0, -4.0]),
            self._make_node([4.0, 4.0, -4.0, -4.0]),
        ]
        field = RegionalHRMField(seed=0, state_dim=4)
        field.field_state = np.asarray([0.0, 3.0, -3.0, 3.0], dtype=float)

        metrics, _edge = self._observe(nodes, previous_state=[0.0, 0.0, 0.0, 0.0], field_owner=field, step=1)
        state = self.monitor.state_for(self.scope_id)

        self.assertTrue(metrics["collapse_event"])
        self.assertTrue(metrics["collapse_flag"])
        self.assertGreater(metrics["collapse_energy"], self.config.energy_threshold)
        self.assertEqual(metrics["collapse_memory_size"], 1)
        self.assertEqual(len(state.collapse_memory), 1)
        self.assertEqual(metrics["cooldown_remaining"], self.config.cooldown_steps)

    def test_cooldown_keeps_flag_active_before_clearing(self):
        collapse_nodes = [
            self._make_node([4.0, 4.0, -4.0, -4.0]),
            self._make_node([4.0, 4.0, -4.0, -4.0]),
        ]
        field = RegionalHRMField(seed=0, state_dim=4)
        field.field_state = np.asarray([0.0, 3.0, -3.0, 3.0], dtype=float)
        self._observe(collapse_nodes, previous_state=[0.0, 0.0, 0.0, 0.0], field_owner=field, step=1)

        calm_nodes = [
            self._make_node([1.0, 0.0, 1.0, 0.0]),
            self._make_node([1.0, 0.0, 1.0, 0.0]),
        ]
        calm_field = RegionalHRMField(seed=1, state_dim=4)
        calm_field.field_state = np.zeros(4, dtype=float)

        metrics_one, _ = self._observe(calm_nodes, previous_state=[1.0, 0.0, 1.0, 0.0], field_owner=calm_field, step=2)
        metrics_two, _ = self._observe(calm_nodes, previous_state=[1.0, 0.0, 1.0, 0.0], field_owner=calm_field, step=3)
        metrics_three, _ = self._observe(calm_nodes, previous_state=[1.0, 0.0, 1.0, 0.0], field_owner=calm_field, step=4)

        self.assertFalse(metrics_one["collapse_event"])
        self.assertTrue(metrics_one["collapse_flag"])
        self.assertEqual(metrics_one["cooldown_remaining"], 1)

        self.assertFalse(metrics_two["collapse_event"])
        self.assertTrue(metrics_two["collapse_flag"])
        self.assertEqual(metrics_two["cooldown_remaining"], 0)

        self.assertFalse(metrics_three["collapse_event"])
        self.assertFalse(metrics_three["collapse_flag"])
        self.assertEqual(metrics_three["cooldown_remaining"], 0)

    def test_attractor_rebinding_moves_nodes_field_and_edges_toward_anchor(self):
        stable_nodes = [
            self._make_node([1.0, 0.0, 1.0, 0.0]),
            self._make_node([1.0, 0.0, 1.0, 0.0]),
        ]
        stable_field = RegionalHRMField(seed=2, state_dim=4)
        stable_field.field_state = np.zeros(4, dtype=float)
        self._observe(stable_nodes, previous_state=[1.0, 0.0, 1.0, 0.0], field_owner=stable_field, edge_strength=0.8, step=0)

        state = self.monitor.state_for(self.scope_id)
        self.assertIsNotNone(state.attractor_anchor)
        np.testing.assert_allclose(state.attractor_anchor, np.asarray([1.0, 0.0, 1.0, 0.0], dtype=float))
        np.testing.assert_allclose(state.field_anchor, np.zeros(4, dtype=float))
        self.assertAlmostEqual(state.edge_anchor[(0, 1)], 0.8)

        unstable_nodes = [
            self._make_node([5.0, 5.0, -5.0, -5.0]),
            self._make_node([5.0, 5.0, -5.0, -5.0]),
        ]
        unstable_field = RegionalHRMField(seed=3, state_dim=4)
        unstable_field.field_state = np.asarray([2.0, -2.0, 2.0, -2.0], dtype=float)
        aggregate_before = self.monitor.aggregate_nodes(unstable_nodes)
        field_distance_before = float(np.linalg.norm(unstable_field.field_state - state.field_anchor))

        metrics, edge = self._observe(
            unstable_nodes,
            previous_state=state.attractor_anchor,
            field_owner=unstable_field,
            edge_strength=0.1,
            step=1,
        )

        aggregate_after = self.monitor.aggregate_nodes(unstable_nodes)
        anchor = state.attractor_anchor
        distance_before = float(np.linalg.norm(aggregate_before - anchor))
        distance_after = float(np.linalg.norm(aggregate_after - anchor))
        field_distance_after = float(np.linalg.norm(unstable_field.field_state - state.field_anchor))

        self.assertTrue(metrics["collapse_flag"])
        self.assertGreater(metrics["rebind_alpha"], 0.0)
        self.assertLess(distance_after, distance_before)
        self.assertLess(metrics["distance_to_anchor_after"], metrics["distance_to_anchor_before"])
        self.assertLess(field_distance_after, field_distance_before)
        self.assertGreater(edge.strength, 0.1)
        self.assertLess(edge.strength, 0.8)

    def test_installed_wrapper_updates_step_and_snapshot_end_to_end(self):
        install_collapse_attractor_support(RecursiveCognitiveManifold, config=self.config)

        manifold = RecursiveCognitiveManifold(state_dim=4)
        manifold.add_node(0, [1.0, 0.0, 0.0, 0.0])
        manifold.add_node(1, [0.0, 1.0, 0.0, 0.0])
        manifold.connect(0, 1, strength=0.7, latency=1.0)

        result = manifold.step(
            {
                0: np.asarray([1.0, 0.2, 0.0, 0.0], dtype=float),
                1: np.asarray([0.1, 1.0, 0.0, 0.0], dtype=float),
            }
        )
        snapshot = manifold.snapshot()

        self.assertIn("collapse", result)
        self.assertIn("region_collapse", result)
        self.assertIn("child_collapse", result)
        self.assertIsInstance(result["collapse"], dict)
        self.assertIsInstance(result["region_collapse"], dict)
        self.assertIsInstance(result["child_collapse"], list)

        self.assertIn("collapse", snapshot)
        self.assertIn("collapse_flag", snapshot)
        self.assertIn("collapse_energy", snapshot)
        self.assertIn("collapse_events", snapshot)
        self.assertIn("attractor_initialized", snapshot)

        self.assertIsInstance(manifold.last_collapse_metrics, dict)
        self.assertIn("collapse_energy", manifold.last_collapse_metrics)
        self.assertEqual(snapshot["collapse_flag"], manifold.collapse_flag)
        self.assertAlmostEqual(snapshot["collapse_energy"], manifold.collapse_energy)
        self.assertEqual(snapshot["collapse_events"], len(manifold.collapse_memory))
        self.assertEqual(snapshot["attractor_initialized"], manifold.attractor_anchor is not None)

    def test_autonomous_topology_repair_runs_without_manual_callback(self):
        repair_config = CollapseConfig(
            energy_threshold=0.05,
            cooldown_steps=1,
            anchor_update_rate=0.4,
            minimum_rebind_alpha=0.2,
            maximum_rebind_alpha=0.85,
        )
        install_collapse_attractor_support(
            RecursiveCognitiveManifold,
            config=repair_config,
            auto_topology_repair=True,
            max_auto_repairs=2,
        )

        manifold = RecursiveCognitiveManifold(state_dim=4)
        manifold.add_node(0, [1.0, 1.0, 0.0, 0.0])
        manifold.add_node(1, [1.0, 1.0, 0.0, 0.0])
        manifold.connect(0, 1, strength=0.8, latency=1.0)

        manifold.step({0: np.asarray([1.0, 1.0, 0.0, 0.0]), 1: np.asarray([1.0, 1.0, 0.0, 0.0])})
        manifold.edges.pop((0, 1), None)
        manifold.edges.pop((1, 0), None)

        result = manifold.step({0: np.asarray([6.0, 6.0, 0.0, 0.0]), 1: np.asarray([6.0, 6.0, 0.0, 0.0])})

        self.assertIn("autonomous_repair", result)
        self.assertGreaterEqual(len(result["autonomous_repair"]), 1)
        self.assertIn((0, 1), manifold.edges)
        self.assertIn((1, 0), manifold.edges)
        self.assertTrue(manifold.last_autonomous_repairs)

        snapshot = manifold.snapshot()
        self.assertIn("autonomous_repairs", snapshot)
        self.assertGreaterEqual(len(snapshot["autonomous_repairs"]), 1)


if __name__ == "__main__":
    unittest.main()