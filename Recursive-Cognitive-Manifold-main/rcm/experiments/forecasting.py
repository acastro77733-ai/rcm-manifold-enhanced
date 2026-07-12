from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from rcm.cognition.manifold import RecursiveCognitiveManifold
from rcm.config.field import FieldConfig
from rcm.reproducibility import normalize_seed


@dataclass(frozen=True)
class MechanismTask:
    name: str
    rationale: str
    scenario: str


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[dict[int, np.ndarray], ...]
    validation: tuple[dict[int, np.ndarray], ...]
    test: tuple[dict[int, np.ndarray], ...]
    seeds: dict[str, int]
    task: MechanismTask


MECHANISM_TASKS = {
    "temporal_forecasting": MechanismTask("temporal_forecasting", "HRM fields and forecasting should help predict smooth but noisy next-step trajectories.", "baseline_noisy"),
    "corrupted_reconstruction": MechanismTask("corrupted_reconstruction", "Hierarchy and memory should help reconstruct partially missing patterns.", "missing"),
    "topology_change_adaptation": MechanismTask("topology_change_adaptation", "Topology adaptation should help when graph structure changes under load.", "distribution_shift"),
    "hierarchical_composition": MechanismTask("hierarchical_composition", "Hierarchical summaries should help compose region-level patterns into meta-patterns.", "baseline"),
    "long_horizon_memory": MechanismTask("long_horizon_memory", "Memory mechanisms should retain informative traces over delayed recall horizons.", "baseline"),
    "distribution_shift": MechanismTask("distribution_shift", "Adaptive fields and topology should help recover after input distribution changes.", "distribution_shift"),
    "structural_lesion_recovery": MechanismTask("structural_lesion_recovery", "Structural recovery should help after targeted edge loss or lesion events.", "noisy"),
    "resource_constrained_computation": MechanismTask("resource_constrained_computation", "Representation allocation should help under finite compute and memory budgets.", "baseline"),
}


class ForecastModel(ABC):
    def __init__(self):
        self.prediction_calls = 0
        self.update_calls = 0
        self.last_observation: dict[int, np.ndarray] | None = None

    @abstractmethod
    def fit(self, sequence: tuple[dict[int, np.ndarray], ...]) -> None:
        raise NotImplementedError

    @abstractmethod
    def observe(self, observation: dict[int, np.ndarray]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self) -> dict[int, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def update(self, target: dict[int, np.ndarray]) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class PersistenceForecastModel(ForecastModel):
    def fit(self, sequence: tuple[dict[int, np.ndarray], ...]) -> None:
        self.reset()
        if sequence:
            self.last_observation = _copy_step(sequence[0])

    def observe(self, observation: dict[int, np.ndarray]) -> None:
        self.last_observation = _copy_step(observation)

    def predict(self) -> dict[int, np.ndarray]:
        self.prediction_calls += 1
        return _copy_step(self.last_observation or {})

    def update(self, target: dict[int, np.ndarray]) -> None:
        self.update_calls += 1
        self.last_observation = _copy_step(target)

    def reset(self) -> None:
        self.prediction_calls = 0
        self.update_calls = 0
        self.last_observation = None


class LinearAutoregressionForecastModel(ForecastModel):
    def __init__(self):
        super().__init__()
        self.coefficients: dict[int, np.ndarray] = {}

    def fit(self, sequence: tuple[dict[int, np.ndarray], ...]) -> None:
        self.reset()
        if len(sequence) < 2:
            return
        node_ids = sorted(sequence[0].keys())
        for node_id in node_ids:
            xs = []
            ys = []
            for left, right in zip(sequence[:-1], sequence[1:]):
                xs.append(np.asarray(left[node_id], dtype=float))
                ys.append(np.asarray(right[node_id], dtype=float))
            x = np.stack(xs, axis=0)
            y = np.stack(ys, axis=0)
            xtx = x.T @ x + 1e-4 * np.eye(x.shape[1])
            self.coefficients[node_id] = np.linalg.solve(xtx, x.T @ y)
        self.last_observation = _copy_step(sequence[-1])

    def observe(self, observation: dict[int, np.ndarray]) -> None:
        self.last_observation = _copy_step(observation)

    def predict(self) -> dict[int, np.ndarray]:
        self.prediction_calls += 1
        if self.last_observation is None:
            return {}
        prediction = {}
        for node_id, observation in self.last_observation.items():
            coeff = self.coefficients.get(node_id)
            if coeff is None:
                prediction[node_id] = np.asarray(observation, dtype=float).copy()
            else:
                prediction[node_id] = np.asarray(observation, dtype=float) @ coeff
        return prediction

    def update(self, target: dict[int, np.ndarray]) -> None:
        self.update_calls += 1
        self.last_observation = _copy_step(target)

    def reset(self) -> None:
        self.prediction_calls = 0
        self.update_calls = 0
        self.last_observation = None
        self.coefficients = {}


class NearestNeighborTransitionForecastModel(ForecastModel):
    def __init__(self):
        super().__init__()
        self.transitions: list[tuple[np.ndarray, dict[int, np.ndarray]]] = []

    def fit(self, sequence: tuple[dict[int, np.ndarray], ...]) -> None:
        self.reset()
        for left, right in zip(sequence[:-1], sequence[1:]):
            self.transitions.append((_flatten_step(left), _copy_step(right)))
        if sequence:
            self.last_observation = _copy_step(sequence[-1])

    def observe(self, observation: dict[int, np.ndarray]) -> None:
        self.last_observation = _copy_step(observation)

    def predict(self) -> dict[int, np.ndarray]:
        self.prediction_calls += 1
        if self.last_observation is None or not self.transitions:
            return _copy_step(self.last_observation or {})
        context = _flatten_step(self.last_observation)
        distances = [float(np.linalg.norm(context - candidate)) for candidate, _target in self.transitions]
        index = int(np.argmin(distances))
        return _copy_step(self.transitions[index][1])

    def update(self, target: dict[int, np.ndarray]) -> None:
        self.update_calls += 1
        self.last_observation = _copy_step(target)

    def reset(self) -> None:
        self.prediction_calls = 0
        self.update_calls = 0
        self.last_observation = None
        self.transitions = []


class MemoryOnlyForecastModel(NearestNeighborTransitionForecastModel):
    pass


class GraphDiffusionForecastModel(ForecastModel):
    def __init__(self, graph_size: int, state_dim: int, diffusion_gain: float = 0.25):
        super().__init__()
        self.graph_size = graph_size
        self.state_dim = state_dim
        self.diffusion_gain = diffusion_gain
        self.adjacency = _chain_adjacency(graph_size)

    def fit(self, sequence: tuple[dict[int, np.ndarray], ...]) -> None:
        self.reset()
        if sequence:
            self.last_observation = _copy_step(sequence[-1])

    def observe(self, observation: dict[int, np.ndarray]) -> None:
        self.last_observation = _copy_step(observation)

    def predict(self) -> dict[int, np.ndarray]:
        self.prediction_calls += 1
        if self.last_observation is None:
            return {}
        stacked = _stack_step(self.last_observation, self.graph_size, self.state_dim)
        transported = self.adjacency @ stacked
        prediction = (1.0 - self.diffusion_gain) * stacked + self.diffusion_gain * transported
        return {node_id: prediction[node_id].copy() for node_id in range(self.graph_size)}

    def update(self, target: dict[int, np.ndarray]) -> None:
        self.update_calls += 1
        self.last_observation = _copy_step(target)

    def reset(self) -> None:
        self.prediction_calls = 0
        self.update_calls = 0
        self.last_observation = None


class FixedGraphControlForecastModel(GraphDiffusionForecastModel):
    pass


class ReservoirForecastModel(ForecastModel):
    def __init__(self, graph_size: int, state_dim: int, seed: int):
        super().__init__()
        self.graph_size = graph_size
        self.state_dim = state_dim
        self.seed = normalize_seed(seed)
        self.rng = np.random.default_rng(self.seed)
        self.hidden_dim = max(8, graph_size * state_dim)
        self.recurrent = self.rng.normal(0.0, 0.2, size=(self.hidden_dim, self.hidden_dim))
        self.input_weights = self.rng.normal(0.0, 0.2, size=(self.hidden_dim, graph_size * state_dim))
        self.readout = np.zeros((self.hidden_dim, graph_size * state_dim), dtype=float)
        self.hidden_state = np.zeros(self.hidden_dim, dtype=float)

    def fit(self, sequence: tuple[dict[int, np.ndarray], ...]) -> None:
        self.reset()
        if len(sequence) < 2:
            return
        hidden_states = []
        targets = []
        for left, right in zip(sequence[:-1], sequence[1:]):
            context = _flatten_step(left)
            self.hidden_state = np.tanh(self.recurrent @ self.hidden_state + self.input_weights @ context)
            hidden_states.append(self.hidden_state.copy())
            targets.append(_flatten_step(right))
        x = np.stack(hidden_states, axis=0)
        y = np.stack(targets, axis=0)
        xtx = x.T @ x + 1e-4 * np.eye(x.shape[1])
        self.readout = np.linalg.solve(xtx, x.T @ y)
        self.last_observation = _copy_step(sequence[-1])

    def observe(self, observation: dict[int, np.ndarray]) -> None:
        self.last_observation = _copy_step(observation)

    def predict(self) -> dict[int, np.ndarray]:
        self.prediction_calls += 1
        if self.last_observation is None:
            return {}
        context = _flatten_step(self.last_observation)
        next_hidden = np.tanh(self.recurrent @ self.hidden_state + self.input_weights @ context)
        prediction = next_hidden @ self.readout
        return _unflatten_step(prediction, self.graph_size, self.state_dim)

    def update(self, target: dict[int, np.ndarray]) -> None:
        self.update_calls += 1
        self.last_observation = _copy_step(target)
        if self.last_observation:
            context = _flatten_step(self.last_observation)
            self.hidden_state = np.tanh(self.recurrent @ self.hidden_state + self.input_weights @ context)

    def reset(self) -> None:
        self.prediction_calls = 0
        self.update_calls = 0
        self.last_observation = None
        self.hidden_state = np.zeros(self.hidden_dim, dtype=float)


class RCMForecastModel(ForecastModel):
    def __init__(self, seed: int, graph_size: int, state_dim: int, condition: str = "full_rcm"):
        super().__init__()
        self.seed = normalize_seed(seed)
        self.graph_size = graph_size
        self.state_dim = state_dim
        self.condition = condition
        self.manifold = self._build_manifold()

    def _build_manifold(self) -> RecursiveCognitiveManifold:
        manifold = RecursiveCognitiveManifold(
            state_dim=self.state_dim,
            level=0,
            label=self.condition,
            hrm_seed=self.seed,
            field_config=FieldConfig(field_model="nonlinear", control_mode="full"),
        )
        for node_id in range(self.graph_size):
            manifold.add_node(node_id, np.zeros(self.state_dim, dtype=float))
        for node_id in range(self.graph_size - 1):
            manifold.connect(node_id, node_id + 1, strength=0.6, latency=1.0)
        manifold.hrm_mode = self.condition
        if self.condition == "current_rcm":
            manifold.field_config.field_model = "legacy"
        if self.condition == "no_guidance":
            manifold.semantic_guidance_enabled = False
        if self.condition == "no_hrm":
            manifold.field_config.control_mode = "no_field"
        if self.condition == "fixed_topology":
            manifold.topology_adaptation_enabled = False
        if self.condition == "no_hierarchy":
            manifold.hierarchy_enabled = False
        manifold.hrm_field.control_mode = manifold.field_config.control_mode
        return manifold

    def fit(self, sequence: tuple[dict[int, np.ndarray], ...]) -> None:
        self.reset()
        for step in sequence:
            self.manifold.step(_copy_step(step))
        if sequence:
            self.last_observation = _copy_step(sequence[-1])

    def observe(self, observation: dict[int, np.ndarray]) -> None:
        self.last_observation = _copy_step(observation)

    def predict(self) -> dict[int, np.ndarray]:
        self.prediction_calls += 1
        if self.last_observation is None:
            return {}
        shadow = copy.deepcopy(self.manifold)
        shadow.step(_copy_step(self.last_observation))
        return {node_id: np.asarray(node.local_state, dtype=float).copy() for node_id, node in shadow.nodes.items()}

    def update(self, target: dict[int, np.ndarray]) -> None:
        self.update_calls += 1
        self.manifold.step(_copy_step(target))
        self.last_observation = _copy_step(target)

    def reset(self) -> None:
        self.prediction_calls = 0
        self.update_calls = 0
        self.last_observation = None
        self.manifold = self._build_manifold()


def build_dataset_split(
    task_name: str,
    graph_size: int,
    seed: int,
    state_dim: int = 4,
    train_steps: int = 8,
    validation_steps: int = 4,
    test_steps: int = 4,
) -> DatasetSplit:
    task = MECHANISM_TASKS[task_name]
    resolved_seed = normalize_seed(seed)
    seeds = {
        "train": resolved_seed + 101,
        "validation": resolved_seed + 202,
        "test": resolved_seed + 303,
    }
    train = tuple(_generate_inputs(graph_size, seeds["train"], task.scenario, train_steps, state_dim))
    validation = tuple(_generate_inputs(graph_size, seeds["validation"], task.scenario, validation_steps, state_dim))
    test = tuple(_generate_inputs(graph_size, seeds["test"], task.scenario, test_steps, state_dim))
    return DatasetSplit(train=train, validation=validation, test=test, seeds=seeds, task=task)


def build_forecast_model(name: str, *, seed: int, graph_size: int, state_dim: int) -> ForecastModel:
    if name in {"full_rcm", "current_rcm", "no_guidance", "no_hrm", "fixed_topology", "no_hierarchy"}:
        return RCMForecastModel(seed=seed, graph_size=graph_size, state_dim=state_dim, condition=name)
    if name == "memory_only":
        return MemoryOnlyForecastModel()
    if name == "persistence":
        return PersistenceForecastModel()
    if name == "linear_autoregression":
        return LinearAutoregressionForecastModel()
    if name == "nearest_neighbor_transition":
        return NearestNeighborTransitionForecastModel()
    if name == "graph_diffusion":
        return GraphDiffusionForecastModel(graph_size=graph_size, state_dim=state_dim)
    if name == "fixed_graph_control":
        return FixedGraphControlForecastModel(graph_size=graph_size, state_dim=state_dim, diffusion_gain=0.18)
    if name == "reservoir_recurrent":
        return ReservoirForecastModel(graph_size=graph_size, state_dim=state_dim, seed=seed)
    raise ValueError(f"Unknown forecast model: {name}")


def describe_model_mechanisms(model: ForecastModel) -> dict[str, Any]:
    if isinstance(model, RCMForecastModel):
        manifold = model.manifold
        return {
            "semantic_guidance": bool(manifold.semantic_guidance_enabled),
            "hrm_mode": str(manifold.field_config.control_mode),
            "hierarchy": bool(manifold.hierarchy_enabled),
            "topology_adaptation": bool(manifold.topology_adaptation_enabled),
        }
    return {"baseline": model.__class__.__name__}


def evaluate_forecast_model(model: ForecastModel, split: DatasetSplit) -> dict[str, Any]:
    model.fit(split.train)
    validation_records = _rollout(model, split.validation)
    test_records = _rollout(model, split.test)
    return {
        "validation": validation_records,
        "test": test_records,
        "validation_mean_error": float(np.mean([record["mse"] for record in validation_records])) if validation_records else 0.0,
        "test_mean_error": float(np.mean([record["mse"] for record in test_records])) if test_records else 0.0,
    }


def _rollout(model: ForecastModel, sequence: tuple[dict[int, np.ndarray], ...]) -> list[dict[str, Any]]:
    if len(sequence) < 2:
        return []
    records = []
    context = _copy_step(sequence[0])
    for target in sequence[1:]:
        model.observe(context)
        prediction = model.predict()
        mse = _mean_squared_error(prediction, target)
        records.append({
            "prediction": prediction,
            "target": _copy_step(target),
            "mse": mse,
        })
        model.update(target)
        context = _copy_step(target)
    return records


def _generate_inputs(graph_size: int, seed: int, scenario: str, n_steps: int, state_dim: int) -> list[dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    steps = []
    for step in range(n_steps):
        base = np.zeros(state_dim, dtype=float)
        base[step % state_dim] = 1.0
        if "distribution_shift" in scenario:
            base = np.roll(base, 1)
        step_dict = {}
        for node_id in range(graph_size):
            signal = base.copy() * (0.85 + 0.15 * np.cos(0.25 * step + node_id))
            if "noisy" in scenario:
                signal += rng.normal(0.0, 0.08, size=state_dim)
            if "missing" in scenario:
                mask = rng.random(state_dim) > 0.2
                signal = signal * mask
            step_dict[node_id] = signal.astype(float)
        steps.append(_copy_step(step_dict))
    return steps


def _copy_step(step: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {int(node_id): np.asarray(value, dtype=float).copy() for node_id, value in (step or {}).items()}


def _flatten_step(step: dict[int, np.ndarray]) -> np.ndarray:
    if not step:
        return np.zeros(0, dtype=float)
    node_ids = sorted(step.keys())
    return np.concatenate([np.asarray(step[node_id], dtype=float).ravel() for node_id in node_ids], axis=0)


def _stack_step(step: dict[int, np.ndarray], graph_size: int, state_dim: int) -> np.ndarray:
    stacked = np.zeros((graph_size, state_dim), dtype=float)
    for node_id, value in step.items():
        stacked[int(node_id)] = np.asarray(value, dtype=float)[:state_dim]
    return stacked


def _unflatten_step(vector: np.ndarray, graph_size: int, state_dim: int) -> dict[int, np.ndarray]:
    vector = np.asarray(vector, dtype=float).ravel()
    output = {}
    for node_id in range(graph_size):
        start = node_id * state_dim
        output[node_id] = vector[start : start + state_dim].copy()
    return output


def _chain_adjacency(graph_size: int) -> np.ndarray:
    adjacency = np.zeros((graph_size, graph_size), dtype=float)
    for node_id in range(graph_size):
        adjacency[node_id, node_id] = 0.5
        if node_id > 0:
            adjacency[node_id, node_id - 1] = 0.25
        if node_id < graph_size - 1:
            adjacency[node_id, node_id + 1] = 0.25
    row_sums = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.maximum(row_sums, 1e-12)


def _mean_squared_error(prediction: dict[int, np.ndarray], target: dict[int, np.ndarray]) -> float:
    if not target:
        return 0.0
    errors = []
    for node_id, target_value in target.items():
        predicted = np.asarray(prediction.get(node_id, np.zeros_like(target_value)), dtype=float)
        errors.append(float(np.mean((predicted - np.asarray(target_value, dtype=float)) ** 2)))
    return float(np.mean(errors)) if errors else 0.0
