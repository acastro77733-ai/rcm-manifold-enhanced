import importlib
import unittest


class APIStabilityTests(unittest.TestCase):
    def test_core_namespaces_import(self):
        for module_name in ("rcm.memory", "rcm.geometry", "rcm.cognition"):
            module = importlib.import_module(module_name)
            self.assertIsNotNone(module)

    def test_memory_exports_stable(self):
        module = importlib.import_module("rcm.memory")
        self.assertTrue(hasattr(module, "EpisodicMemory"))
        self.assertTrue(hasattr(module, "SemanticMemory"))
        self.assertTrue(hasattr(module, "StructuralMemory"))


if __name__ == "__main__":
    unittest.main()
