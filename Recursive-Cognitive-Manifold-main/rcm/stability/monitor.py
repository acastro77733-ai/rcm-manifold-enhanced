from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rcm.config.stability import StabilityConfig


@dataclass
class CollapseMonitor:
    config: StabilityConfig = field(default_factory=StabilityConfig)
    history: list[dict[str, Any]] = field(default_factory=list)

    def evaluate(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        state_signature = np.asarray(snapshot.get("state_signature", []), dtype=float).reshape(-1)
        previous_signature = np.asarray(snapshot.get("previous_state_signature", []), dtype=float).reshape(-1)
        width = min(state_signature.size, previous_signature.size)
        if width > 0:
            delta = float(np.linalg.norm(state_signature[:width] - previous_signature[:width]))
        else:
            delta = float(np.linalg.norm(state_signature))

        field_metrics = snapshot.get("field_metrics", {}) or {}
        field_variance = float(field_metrics.get("field_variance", 0.0))
        collapse_energy = float(snapshot.get("collapse_energy", 0.0))
        energy = max(collapse_energy, field_variance, delta * 0.25)
        level = 0
        if energy >= self.config.collapse_threshold:
            level = 4
        elif energy >= self.config.collapse_threshold * 0.75:
            level = 3
        elif energy >= self.config.collapse_threshold * 0.45:
            level = 2
        elif energy >= self.config.collapse_threshold * 0.2:
            level = 1

        metrics = {
            "level": level,
            "collapse_energy": energy,
            "delta": delta,
            "field_variance": field_variance,
            "collapse_event": level >= 3,
        }
        self.history.append(metrics)
        return metrics
