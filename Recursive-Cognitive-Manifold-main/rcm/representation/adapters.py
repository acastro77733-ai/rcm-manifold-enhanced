from __future__ import annotations

import numpy as np


class RepresentationAdapter:
    def expand(self, state: np.ndarray, target_dim: int) -> np.ndarray:
        raise NotImplementedError

    def contract(self, state: np.ndarray, target_dim: int) -> np.ndarray:
        raise NotImplementedError


class IdentityAdapter(RepresentationAdapter):
    def expand(self, state: np.ndarray, target_dim: int) -> np.ndarray:
        state = np.asarray(state, dtype=float).ravel()
        if len(state) >= target_dim:
            return state[:target_dim]
        padded = np.zeros(target_dim, dtype=float)
        padded[: len(state)] = state
        return padded

    def contract(self, state: np.ndarray, target_dim: int) -> np.ndarray:
        state = np.asarray(state, dtype=float).ravel()
        if target_dim <= 0:
            return np.zeros(1, dtype=float)
        if len(state) <= target_dim:
            return state.copy()
        return state[:target_dim].copy()
