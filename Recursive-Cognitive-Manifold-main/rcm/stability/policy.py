from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rcm.stability.attractor import AttractorState
from rcm.stability.monitor import CollapseMonitor
from rcm.stability.recovery import CollisionRecoveryController


class StabilityLevel(Enum):
    STABLE = 0
    WATCH = 1
    ALERT = 2
    CRITICAL = 3
    RECOVER = 4


@dataclass
class StabilityPolicy:
    monitor: CollapseMonitor = field(default_factory=CollapseMonitor)
    recovery: CollisionRecoveryController = field(default_factory=CollisionRecoveryController)
    attractor: AttractorState | None = None

    def begin_transaction(self, scope: str, payload: dict[str, Any], snapshot: dict[str, Any], *, trial_metrics: dict[str, Any] | None = None, post_change_metrics: dict[str, Any] | None = None, accepted: bool = False):
        return self.recovery.begin_transaction(
            scope,
            payload,
            snapshot,
            trial_metrics=trial_metrics,
            post_change_metrics=post_change_metrics,
            accepted=accepted,
        )

    def complete_transaction(self, transaction_id: str, *, post_change_metrics: dict[str, Any] | None = None, accepted: bool | None = None) -> None:
        self.recovery.complete_transaction(transaction_id, post_change_metrics=post_change_metrics, accepted=accepted)

    def respond_to_instability(self, manifold: Any, level: int = 0) -> dict[str, Any]:
        return self.recovery.respond_to_instability(manifold, level=level)

    def evaluate(self, manifold: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
        monitor_metrics = self.monitor.evaluate(snapshot)
        level = StabilityLevel(monitor_metrics["level"] if monitor_metrics["level"] <= 4 else 4)
        if level is StabilityLevel.CRITICAL or level is StabilityLevel.RECOVER:
            recovery_metrics = self.recovery.respond_to_instability(manifold, level=monitor_metrics["level"])
            return {**monitor_metrics, **recovery_metrics, "stability_level": level.name}
        return {**monitor_metrics, "stability_level": level.name}
