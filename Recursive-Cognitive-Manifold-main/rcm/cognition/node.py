from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CognitiveNode:
    local_state: np.ndarray
    short_term_memory: deque = field(default_factory=lambda: deque(maxlen=8))
    long_term_memory: deque = field(default_factory=lambda: deque(maxlen=32))
    synchronization_state: float = 0.0
    energy: float = 1.0
    confidence: float = 0.5
    topology_history: deque = field(default_factory=lambda: deque(maxlen=16))
    specialization: str = "undifferentiated"