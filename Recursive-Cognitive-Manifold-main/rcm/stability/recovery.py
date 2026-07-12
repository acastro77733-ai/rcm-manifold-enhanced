from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TransactionRecord:
    transaction_id: str
    scope: str
    payload: dict[str, Any]
    snapshot: dict[str, Any]
    trial_metrics: dict[str, Any]
    post_change_metrics: dict[str, Any]
    accepted: bool = False


@dataclass
class CollisionRecoveryController:
    transactions: list[TransactionRecord] = field(default_factory=list)
    rollback_history: list[dict[str, Any]] = field(default_factory=list)

    def begin_transaction(self, scope: str, payload: dict[str, Any], snapshot: dict[str, Any], *, trial_metrics: dict[str, Any] | None = None, post_change_metrics: dict[str, Any] | None = None, accepted: bool = False) -> str:
        tx = TransactionRecord(
            transaction_id=f"tx-{len(self.transactions) + 1}",
            scope=scope,
            payload=payload,
            snapshot=snapshot,
            trial_metrics=trial_metrics or {},
            post_change_metrics=post_change_metrics or {},
            accepted=accepted,
        )
        self.transactions.append(tx)
        return tx.transaction_id

    def complete_transaction(self, transaction_id: str, *, post_change_metrics: dict[str, Any] | None = None, accepted: bool | None = None) -> None:
        for tx in self.transactions:
            if tx.transaction_id != transaction_id:
                continue
            tx.post_change_metrics = post_change_metrics or tx.post_change_metrics
            tx.accepted = accepted if accepted is not None else tx.accepted
            break

    def _rollback(self, manifold: Any, transaction: TransactionRecord) -> dict[str, Any]:
        if transaction.scope == "representation" and hasattr(manifold, "representation_controller"):
            manifold.representation_controller.current_dim = max(
                manifold.representation_controller.min_dim,
                min(manifold.representation_controller.max_dim, int(transaction.snapshot.get("representation_dim", manifold.representation_controller.current_dim))),
            )
            manifold.representation_controller.current_state = np.asarray(transaction.snapshot.get("state_vector", np.zeros(max(2, manifold.representation_controller.current_dim), dtype=float)), dtype=float).reshape(-1)
            manifold.representation_controller.rollback_state = None
        return {"rolled_back": True, "transaction_id": transaction.transaction_id, "scope": transaction.scope}

    def respond_to_instability(self, manifold: Any, level: int = 0) -> dict[str, Any]:
        if not self.transactions:
            return {"rolled_back": False, "level": level, "reason": "no transactions"}
        rollback_results = []
        for tx in reversed(self.transactions):
            if tx.accepted and level >= 3:
                rollback_results.append(self._rollback(manifold, tx))
                tx.accepted = False
                break
        if rollback_results:
            self.rollback_history.append({"level": level, "transactions": rollback_results})
            return {"rolled_back": True, "level": level, "transactions": rollback_results}
        return {"rolled_back": False, "level": level, "reason": "no eligible transaction"}
