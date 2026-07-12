from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AttractorState:
    state: np.ndarray
    field_state: np.ndarray | None = None
    topology_snapshot: dict[str, Any] = field(default_factory=dict)
    representation_dim: int = 4
    metadata: dict[str, Any] = field(default_factory=dict)
