import copy
import unittest

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.dynamics.state_engine import StateEngine, StateEngineConfig


class NumericalValidationTests(unittest.TestCase):
    def _build_manifold(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="numeric", hrm_seed=17)
        manifold.add_node(0, np.array([0.4, 0.1, -0.2, 0.0], dtype=float))
        manifold.add_node(1, np.array([0.0, 0.3, 0.2, -0.1], dtype=float))
        manifold.connect(0, 1, strength=0.8, latency=1.0)
        return manifold

    def test_governed_state_converges_under_timestep_refinement(self):
        coarse_manifold = self._build_manifold()
        fine_manifold = self._build_manifold()
        signal = {0: np.array([0.6, 0.1, 0.0, 0.0], dtype=float)}

        coarse_engine = StateEngine(config=StateEngineConfig(dt=0.1))
        fine_engine = StateEngine(config=StateEngineConfig(dt=0.05))

        coarse = coarse_engine.step(coarse_manifold, external_input=signal)
        fine_first = fine_engine.step(fine_manifold, external_input=signal)
        for node_id, state in zip(fine_first["node_order"], fine_first["node_states"]):
            fine_manifold.nodes[node_id].local_state = np.asarray(state, dtype=float)
        fine_second = fine_engine.step(fine_manifold, external_input=signal)

        coarse_state = np.concatenate([np.asarray(state, dtype=float) for state in coarse["node_states"]], axis=0)
        fine_state = np.concatenate([np.asarray(state, dtype=float) for state in fine_second["node_states"]], axis=0)
        self.assertLess(np.linalg.norm(coarse_state - fine_state), 0.35)

    def test_float32_and_float64_inputs_remain_close(self):
        manifold64 = self._build_manifold()
        manifold32 = self._build_manifold()

        signal64 = {0: np.array([0.7, -0.1, 0.2, 0.0], dtype=np.float64)}
        signal32 = {0: np.array([0.7, -0.1, 0.2, 0.0], dtype=np.float32)}

        snapshot64 = manifold64.step(signal64)
        snapshot32 = manifold32.step(signal32)

        state64 = np.asarray(snapshot64["unified_state"]["state_vector"], dtype=float)
        state32 = np.asarray(snapshot32["unified_state"]["state_vector"], dtype=float)
        self.assertTrue(np.isfinite(state64).all())
        self.assertTrue(np.isfinite(state32).all())
        self.assertLess(np.linalg.norm(state64 - state32), 0.5)


if __name__ == "__main__":
    unittest.main()
