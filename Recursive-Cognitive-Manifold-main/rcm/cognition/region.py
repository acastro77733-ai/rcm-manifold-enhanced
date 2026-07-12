from collections import deque
from dataclasses import dataclass, field


@dataclass
class CognitiveRegion:
    node_ids: tuple[int, ...]
    specialization: str
    region_id: int = 0
    hrm_field: object | None = None
    activation_trace: deque = field(default_factory=lambda: deque(maxlen=16))
    stability: float = 0.0
    field_metrics: dict = field(default_factory=dict)