from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ManifoldMetrics:
    task_score: float
    recall_score: float
    coherence: float
    collapse_energy: float
    topology_cost: float
    representation_cost: float
    field_energy: float
