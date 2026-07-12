from __future__ import annotations

import random

import numpy as np


DEFAULT_SEED = 42


def normalize_seed(seed: int | None, fallback: int = DEFAULT_SEED) -> int:
    if seed is None:
        return int(fallback)
    return int(seed)


def seed_everything(seed: int | None, fallback: int = DEFAULT_SEED) -> int:
    resolved = normalize_seed(seed, fallback=fallback)
    random.seed(resolved)
    np.random.seed(resolved)
    return resolved


def build_rng(seed: int | None, fallback: int = DEFAULT_SEED) -> np.random.Generator:
    resolved = normalize_seed(seed, fallback=fallback)
    return np.random.default_rng(resolved)
