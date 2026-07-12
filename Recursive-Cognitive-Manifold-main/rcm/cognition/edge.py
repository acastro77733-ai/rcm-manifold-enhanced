from dataclasses import dataclass


@dataclass
class CognitiveEdge:
    strength: float = 0.5
    latency: float = 1.0
    resonance: float = 0.0
    age: int = 0
    traversal_frequency: int = 0