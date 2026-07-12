from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RepresentationHistoryEntry:
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    proposal: str = ""
    accepted: bool = False
    previous_dim: int = 0
    new_dim: int = 0
    objective_delta: float = 0.0
    reason: str = ""
