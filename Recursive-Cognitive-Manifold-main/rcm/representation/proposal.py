from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ProposalType(str, Enum):
    EXPAND = "expand"
    CONTRACT = "contract"
    NO_CHANGE = "no_change"


@dataclass
class RepresentationProposal:
    proposal_type: ProposalType
    target_dim: int
    score: float
    reason: str
    diagnostics: dict
    trial_state: np.ndarray | None = None
