import tempfile
import unittest
from pathlib import Path

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm_persistence import RCMPersistenceStore


class SerializationRoundtripTests(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_core_state(self):
        manifold = RecursiveCognitiveManifold(state_dim=4, level=0, label="roundtrip")
        manifold.add_node(0, [0.1, 0.2, 0.3, 0.4])
        manifold.add_node(1, [0.2, 0.3, 0.4, 0.5])
        manifold.connect(0, 1, strength=0.6, latency=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RCMPersistenceStore(root_dir=Path(tmpdir))
            checkpoint_dir = store.save(manifold, random_seed=7, configuration={"mode": "roundtrip"})
            restored, metadata = store.load(checkpoint_dir, foundation=__import__("rcm.cognition.manifold", fromlist=["RecursiveCognitiveManifold"]))
            self.assertEqual(restored.state_dim, manifold.state_dim)
            self.assertEqual(restored.label, manifold.label)
            self.assertEqual(len(restored.nodes), len(manifold.nodes))
            self.assertEqual(restored.time_step, manifold.time_step)


if __name__ == "__main__":
    unittest.main()
