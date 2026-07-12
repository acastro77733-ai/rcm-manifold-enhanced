from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .adapters import RepresentationAdapter, IdentityAdapter
from .history import RepresentationHistoryEntry
from .proposal import ProposalType, RepresentationProposal


@dataclass
class RepresentationController:
    adapter: RepresentationAdapter = field(default_factory=IdentityAdapter)
    min_dim: int = 2
    max_dim: int = 8
    trial_duration: int = 2
    minimum_improvement: float = 0.0
    switch_cost: float = 0.03
    cooldown_steps: int = 3
    collapse_threshold: float = 0.35
    current_dim: int = 4
    current_state: np.ndarray | None = None
    last_change_step: int = -1000
    history: list[RepresentationHistoryEntry] = field(default_factory=list)
    rollback_state: np.ndarray | None = None

    def __post_init__(self):
        self.current_state = np.zeros(max(self.min_dim, self.current_dim), dtype=float) if self.current_state is None else np.asarray(self.current_state, dtype=float).ravel()
        self.current_dim = max(self.min_dim, min(self.max_dim, int(self.current_dim)))
        self.current_state = self.adapter.expand(self.current_state, self.current_dim)

    def propose(self, state: np.ndarray, objective: float, collapse_event: bool = False, step: int = 0) -> RepresentationProposal:
        state = np.asarray(state, dtype=float).ravel()
        if collapse_event or (self.history and step - self.last_change_step < self.cooldown_steps):
            return RepresentationProposal(ProposalType.NO_CHANGE, self.current_dim, objective, "cooldown or collapse", {}, None)

        diagnostics = {
            "prediction_residual": float(np.mean(np.abs(state))),
            "temporal_derivative": float(np.std(state)),
            "confidence": float(np.clip(1.0 - np.mean(np.abs(state)), 0.0, 1.0)),
            "field_disagreement": float(np.mean(np.abs(state))),
            "recall_disagreement": float(np.std(state)),
            "instability": float(np.max(np.abs(state))),
        }

        if diagnostics["instability"] > max(self.collapse_threshold, 0.55):
            return RepresentationProposal(ProposalType.NO_CHANGE, self.current_dim, objective, "collapse event", diagnostics, None)

        if objective >= 0.6 and self.current_dim < self.max_dim:
            target_dim = min(self.max_dim, self.current_dim + 2)
            trial_state = self.adapter.expand(state, target_dim)
            return RepresentationProposal(ProposalType.EXPAND, target_dim, objective + 0.05, "objective-driven expansion", diagnostics, trial_state)

        if objective <= 0.5 and self.current_dim > self.min_dim:
            target_dim = max(self.min_dim, self.current_dim - 2)
            trial_state = self.adapter.contract(state, target_dim)
            return RepresentationProposal(ProposalType.CONTRACT, target_dim, objective - 0.02, "objective-driven contraction", diagnostics, trial_state)

        return RepresentationProposal(ProposalType.NO_CHANGE, self.current_dim, objective, "no proposal", diagnostics, None)

    def evaluate(self, proposal: RepresentationProposal, objective: float, current_state: np.ndarray, step: int = 0) -> tuple[bool, float, np.ndarray]:
        if proposal.proposal_type == ProposalType.NO_CHANGE:
            return False, objective, np.asarray(current_state, dtype=float).ravel()

        candidate_state = np.asarray(proposal.trial_state, dtype=float).ravel() if proposal.trial_state is not None else np.asarray(current_state, dtype=float).ravel()
        if proposal.proposal_type == ProposalType.EXPAND:
            candidate_objective = objective + 0.05 - self.switch_cost
        elif proposal.proposal_type == ProposalType.CONTRACT:
            candidate_objective = objective - 0.02 - self.switch_cost
        else:
            candidate_objective = objective
        current_objective = objective
        delta = candidate_objective - current_objective
        accepted = delta >= self.minimum_improvement
        if accepted:
            self.rollback_state = np.asarray(current_state, dtype=float).ravel().copy()
            self.current_state = candidate_state
            self.current_dim = proposal.target_dim
            self.last_change_step = step
            self.history.append(RepresentationHistoryEntry(proposal=str(proposal.proposal_type.value), accepted=True, previous_dim=int(len(np.asarray(current_state, dtype=float).ravel())), new_dim=int(proposal.target_dim), objective_delta=float(delta), reason=proposal.reason))
            return accepted, delta, candidate_state

        self.current_state = np.asarray(self.rollback_state, dtype=float).ravel().copy() if self.rollback_state is not None else np.asarray(current_state, dtype=float).ravel().copy()
        self.history.append(RepresentationHistoryEntry(proposal=str(proposal.proposal_type.value), accepted=False, previous_dim=int(len(np.asarray(current_state, dtype=float).ravel())), new_dim=int(proposal.target_dim), objective_delta=float(delta), reason=proposal.reason))
        return accepted, delta, self.current_state.copy()

    def apply(self, state: np.ndarray, objective: float, collapse_event: bool = False, step: int = 0) -> tuple[bool, RepresentationProposal, np.ndarray]:
        proposal = self.propose(state, objective, collapse_event=collapse_event, step=step)
        accepted, delta, candidate_state = self.evaluate(proposal, objective, self.current_state, step=step)
        if not accepted:
            return False, proposal, self.current_state.copy()
        return True, proposal, candidate_state
