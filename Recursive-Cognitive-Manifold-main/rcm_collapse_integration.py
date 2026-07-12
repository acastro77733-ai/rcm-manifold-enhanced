"""Drop-in collapse and attractor diagnostics for Recursive Cognitive Manifold.

Integration (one time, after RecursiveCognitiveManifold is defined):

    from rcm_collapse_integration import install_collapse_attractor_support
    install_collapse_attractor_support(RecursiveCognitiveManifold)

The installer wraps ``__init__``, ``step``, and ``snapshot`` without replacing
the normal RCM update rules. It adds bounded collapse diagnostics and attractor
rebinding for manifolds, regions, and child manifolds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class CollapseConfig:
    energy_threshold: float = 1.60
    anchor_update_rate: float = 0.08
    memory_limit: int = 128
    cooldown_steps: int = 6
    anchor_energy_fraction: float = 0.70
    minimum_anchor_coherence: float = 0.60
    minimum_rebind_alpha: float = 0.20
    maximum_rebind_alpha: float = 0.85
    instability_gain: float = 0.35
    node_rebind_strength: float = 1.00
    field_rebind_strength: float = 0.45
    edge_rebind_strength: float = 0.25
    epsilon: float = 1e-8


@dataclass
class CollapseEvent:
    step: int
    scope_id: str
    collapse_energy: float
    gradient_term: float
    velocity_term: float
    coherence_loss: float
    state_snapshot: Array
    field_snapshot: Optional[Array]
    node_ids: Tuple[Any, ...]
    edge_snapshot: Dict[Tuple[Any, Any], float]


@dataclass
class AttractorState:
    collapse_flag: bool = False
    collapse_energy: float = 0.0
    collapse_memory: List[CollapseEvent] = field(default_factory=list)
    attractor_anchor: Optional[Array] = None
    field_anchor: Optional[Array] = None
    edge_anchor: Dict[Tuple[Any, Any], float] = field(default_factory=dict)
    cooldown_remaining: int = 0
    recovery_steps: int = 0
    last_metrics: Dict[str, float] = field(default_factory=dict)


class CollapseAttractorMonitor:
    """Observe RCM dynamics and guide bounded recovery toward learned attractors."""

    def __init__(self, config: Optional[CollapseConfig] = None):
        self.config = config or CollapseConfig()
        self.scopes: Dict[str, AttractorState] = {}

    def state_for(self, scope_id: str) -> AttractorState:
        return self.scopes.setdefault(scope_id, AttractorState())

    @staticmethod
    def _node_vector(node: Any) -> Array:
        if hasattr(node, "local_state"):
            return np.asarray(node.local_state, dtype=float)
        if hasattr(node, "state_vector"):
            return np.asarray(node.state_vector, dtype=float)
        raise AttributeError("Node must expose local_state or state_vector")

    @staticmethod
    def aggregate_nodes(nodes: Sequence[Any]) -> Array:
        if not nodes:
            return np.zeros(1, dtype=float)
        vectors = [CollapseAttractorMonitor._node_vector(node).reshape(-1) for node in nodes]
        width = min(vector.size for vector in vectors)
        if width == 0:
            return np.zeros(1, dtype=float)
        return np.mean(np.stack([vector[:width] for vector in vectors]), axis=0)

    @staticmethod
    def estimate_coherence(state_vector: Array) -> float:
        vector = np.asarray(state_vector, dtype=float).reshape(-1)
        if vector.size >= 2:
            midpoint = vector.size // 2
            left = vector[:midpoint]
            right = vector[-midpoint:]
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            if denominator > 1e-8:
                cosine = float(np.dot(left, right) / denominator)
                return float(np.clip((cosine + 1.0) * 0.5, 0.0, 1.0))
        variance = float(np.var(vector))
        return float(1.0 / (1.0 + variance))

    @staticmethod
    def gradient_norm(field_value: Optional[Array]) -> float:
        if field_value is None:
            return 0.0
        field_array = np.asarray(field_value, dtype=float)
        if field_array.size <= 1:
            return 0.0
        gradients = np.gradient(field_array)
        if not isinstance(gradients, (list, tuple)):
            gradients = [gradients]
        magnitude_squared = np.zeros_like(field_array, dtype=float)
        for gradient in gradients:
            magnitude_squared += np.asarray(gradient, dtype=float) ** 2
        return float(np.mean(np.sqrt(magnitude_squared)))

    @staticmethod
    def _field_attribute(field_owner: Optional[Any]) -> Optional[str]:
        if field_owner is None:
            return None
        for attribute in ("field", "phi", "field_state"):
            if hasattr(field_owner, attribute):
                return attribute
        return None

    @classmethod
    def _field_value(cls, field_owner: Optional[Any]) -> Optional[Array]:
        attribute = cls._field_attribute(field_owner)
        if attribute is None:
            return None
        return np.asarray(getattr(field_owner, attribute), dtype=float)

    @staticmethod
    def _edge_entries(edge_mapping: Mapping[Tuple[Any, Any], Any]) -> List[Tuple[Tuple[Any, Any], Any]]:
        return list(edge_mapping.items())

    @staticmethod
    def _edge_snapshot(edge_entries: Iterable[Tuple[Tuple[Any, Any], Any]]) -> Dict[Tuple[Any, Any], float]:
        return {
            tuple(key): float(getattr(edge, "strength", 1.0))
            for key, edge in edge_entries
        }

    def calculate_collapse_energy(
        self,
        current_state: Array,
        previous_state: Array,
        dt: float,
        field_value: Optional[Array] = None,
        coherence: Optional[float] = None,
    ) -> Dict[str, float]:
        cfg = self.config
        current = np.asarray(current_state, dtype=float).reshape(-1)
        previous = np.asarray(previous_state, dtype=float).reshape(-1)
        width = min(current.size, previous.size)
        if width == 0:
            current = np.zeros(1, dtype=float)
            previous = np.zeros(1, dtype=float)
        else:
            current = current[:width]
            previous = previous[:width]

        measured_coherence = (
            self.estimate_coherence(current)
            if coherence is None
            else float(np.clip(coherence, 0.0, 1.0))
        )
        coherence_loss = float(np.clip(1.0 - measured_coherence, 0.0, 1.0))
        gradient_term = float(np.tanh(self.gradient_norm(field_value)))
        velocity = (current - previous) / max(float(dt), cfg.epsilon)
        velocity_term = float(np.tanh(np.linalg.norm(velocity)))
        collapse_energy = gradient_term + velocity_term + coherence_loss

        return {
            "collapse_energy": float(collapse_energy),
            "gradient_term": gradient_term,
            "velocity_term": velocity_term,
            "coherence_loss": coherence_loss,
            "coherence": measured_coherence,
        }

    def _update_attractor(
        self,
        state: AttractorState,
        state_vector: Array,
        field_value: Optional[Array],
        edge_entries: Sequence[Tuple[Tuple[Any, Any], Any]],
        collapse_energy: float,
        coherence: float,
    ) -> None:
        cfg = self.config
        if collapse_energy >= cfg.energy_threshold * cfg.anchor_energy_fraction:
            return
        if coherence < cfg.minimum_anchor_coherence:
            return

        vector = np.asarray(state_vector, dtype=float).copy()
        if state.attractor_anchor is None or state.attractor_anchor.shape != vector.shape:
            state.attractor_anchor = vector
        else:
            rate = cfg.anchor_update_rate
            state.attractor_anchor = (1.0 - rate) * state.attractor_anchor + rate * vector

        if field_value is not None:
            field_array = np.asarray(field_value, dtype=float).copy()
            if state.field_anchor is None or state.field_anchor.shape != field_array.shape:
                state.field_anchor = field_array
            else:
                rate = cfg.anchor_update_rate
                state.field_anchor = (1.0 - rate) * state.field_anchor + rate * field_array

        current_edges = self._edge_snapshot(edge_entries)
        if not state.edge_anchor:
            state.edge_anchor = current_edges
        else:
            rate = cfg.anchor_update_rate
            for key, value in current_edges.items():
                prior = state.edge_anchor.get(key, value)
                state.edge_anchor[key] = (1.0 - rate) * prior + rate * value

    def _record_collapse(
        self,
        state: AttractorState,
        scope_id: str,
        step: int,
        terms: Mapping[str, float],
        state_vector: Array,
        field_value: Optional[Array],
        node_ids: Sequence[Any],
        edge_entries: Sequence[Tuple[Tuple[Any, Any], Any]],
    ) -> None:
        state.collapse_memory.append(
            CollapseEvent(
                step=int(step),
                scope_id=scope_id,
                collapse_energy=float(terms["collapse_energy"]),
                gradient_term=float(terms["gradient_term"]),
                velocity_term=float(terms["velocity_term"]),
                coherence_loss=float(terms["coherence_loss"]),
                state_snapshot=np.asarray(state_vector, dtype=float).copy(),
                field_snapshot=(None if field_value is None else np.asarray(field_value, dtype=float).copy()),
                node_ids=tuple(node_ids),
                edge_snapshot=self._edge_snapshot(edge_entries),
            )
        )
        if len(state.collapse_memory) > self.config.memory_limit:
            del state.collapse_memory[:-self.config.memory_limit]

    def _rebind_alpha(self, collapse_energy: float) -> float:
        cfg = self.config
        excess = max(0.0, collapse_energy - cfg.energy_threshold)
        return float(
            np.clip(
                cfg.minimum_rebind_alpha + cfg.instability_gain * excess,
                cfg.minimum_rebind_alpha,
                cfg.maximum_rebind_alpha,
            )
        )

    def _apply_rebinding(
        self,
        state: AttractorState,
        nodes: Sequence[Any],
        field_owner: Optional[Any],
        edge_entries: Sequence[Tuple[Tuple[Any, Any], Any]],
    ) -> Dict[str, float]:
        anchor = state.attractor_anchor
        if anchor is None or not nodes:
            return {
                "rebind_alpha": 0.0,
                "distance_to_anchor_before": 0.0,
                "distance_to_anchor_after": 0.0,
            }

        aggregate_before = self.aggregate_nodes(nodes)
        width = min(aggregate_before.size, anchor.size)
        if width == 0:
            return {
                "rebind_alpha": 0.0,
                "distance_to_anchor_before": 0.0,
                "distance_to_anchor_after": 0.0,
            }
        distance_before = float(np.linalg.norm(aggregate_before[:width] - anchor[:width]))

        alpha = self._rebind_alpha(state.collapse_energy)
        node_alpha = float(np.clip(alpha * self.config.node_rebind_strength, 0.0, 1.0))

        for node in nodes:
            vector = self._node_vector(node).copy()
            flat = vector.reshape(-1)
            node_width = min(flat.size, anchor.size)
            flat[:node_width] += node_alpha * (anchor[:node_width] - flat[:node_width])
            if hasattr(node, "local_state"):
                node.local_state = vector
            else:
                node.state_vector = vector

        field_attribute = self._field_attribute(field_owner)
        if field_attribute is not None and state.field_anchor is not None:
            current_field = np.asarray(getattr(field_owner, field_attribute), dtype=float)
            if current_field.shape == state.field_anchor.shape:
                field_alpha = float(np.clip(alpha * self.config.field_rebind_strength, 0.0, 1.0))
                corrected = current_field + field_alpha * (state.field_anchor - current_field)
                setattr(field_owner, field_attribute, corrected)

        edge_alpha = float(np.clip(alpha * self.config.edge_rebind_strength, 0.0, 1.0))
        for key, edge in edge_entries:
            if key not in state.edge_anchor or not hasattr(edge, "strength"):
                continue
            edge.strength = float(edge.strength + edge_alpha * (state.edge_anchor[key] - edge.strength))

        aggregate_after = self.aggregate_nodes(nodes)
        distance_after = float(np.linalg.norm(aggregate_after[:width] - anchor[:width]))
        state.recovery_steps += 1

        return {
            "rebind_alpha": alpha,
            "distance_to_anchor_before": distance_before,
            "distance_to_anchor_after": distance_after,
        }

    def observe(
        self,
        *,
        scope_id: str,
        node_items: Sequence[Tuple[Any, Any]],
        edge_entries: Sequence[Tuple[Tuple[Any, Any], Any]],
        previous_state: Array,
        step: int,
        dt: float,
        field_owner: Optional[Any] = None,
        coherence: Optional[float] = None,
        auto_rebind: bool = True,
    ) -> Dict[str, float]:
        diagnostic = self.state_for(scope_id)
        node_ids = [node_id for node_id, _ in node_items]
        nodes = [node for _, node in node_items]
        current_state = self.aggregate_nodes(nodes)
        field_value = self._field_value(field_owner)

        terms = self.calculate_collapse_energy(
            current_state=current_state,
            previous_state=previous_state,
            dt=dt,
            field_value=field_value,
            coherence=coherence,
        )
        diagnostic.collapse_energy = float(terms["collapse_energy"])
        triggered = diagnostic.collapse_energy >= self.config.energy_threshold

        if triggered:
            diagnostic.collapse_flag = True
            diagnostic.cooldown_remaining = self.config.cooldown_steps
            self._record_collapse(
                diagnostic,
                scope_id,
                step,
                terms,
                current_state,
                field_value,
                node_ids,
                edge_entries,
            )
        elif diagnostic.cooldown_remaining > 0:
            diagnostic.collapse_flag = True
            diagnostic.cooldown_remaining -= 1
        else:
            diagnostic.collapse_flag = False
            diagnostic.recovery_steps = 0

        rebind_metrics = {
            "rebind_alpha": 0.0,
            "distance_to_anchor_before": 0.0,
            "distance_to_anchor_after": 0.0,
        }
        if diagnostic.collapse_flag and auto_rebind:
            if diagnostic.attractor_anchor is None:
                diagnostic.attractor_anchor = np.asarray(previous_state, dtype=float).copy()
            rebind_metrics = self._apply_rebinding(diagnostic, nodes, field_owner, edge_entries)

        self._update_attractor(
            diagnostic,
            self.aggregate_nodes(nodes),
            self._field_value(field_owner),
            edge_entries,
            diagnostic.collapse_energy,
            float(terms["coherence"]),
        )

        metrics = {
            **{key: float(value) for key, value in terms.items()},
            "collapse_threshold": float(self.config.energy_threshold),
            "collapse_event": bool(triggered),
            "collapse_flag": bool(diagnostic.collapse_flag),
            "collapse_memory_size": len(diagnostic.collapse_memory),
            "cooldown_remaining": diagnostic.cooldown_remaining,
            **rebind_metrics,
        }
        diagnostic.last_metrics = metrics
        return metrics

    def monitor_manifold(
        self,
        manifold: Any,
        previous_state: Array,
        dt: float,
        auto_rebind: bool = True,
    ) -> Dict[str, float]:
        level = int(getattr(manifold, "level", getattr(manifold, "hierarchy_level", 0)))
        label = str(getattr(manifold, "label", "manifold"))
        return self.observe(
            scope_id=f"manifold:{label}:level:{level}",
            node_items=list(manifold.nodes.items()),
            edge_entries=self._edge_entries(manifold.edges),
            previous_state=previous_state,
            step=int(getattr(manifold, "time_step", 0)),
            dt=dt,
            field_owner=getattr(manifold, "hrm_field", None),
            auto_rebind=auto_rebind,
        )

    def monitor_region(
        self,
        manifold: Any,
        region_key: Tuple[Any, ...],
        region: Any,
        previous_state: Array,
        dt: float,
        auto_rebind: bool = True,
    ) -> Dict[str, float]:
        node_ids = tuple(getattr(region, "node_ids", region_key))
        node_items = [(node_id, manifold.nodes[node_id]) for node_id in node_ids if node_id in manifold.nodes]
        edge_entries = [(key, edge) for key, edge in manifold.edges.items() if key[0] in node_ids and key[1] in node_ids]
        return self.observe(
            scope_id=f"region:{tuple(node_ids)}",
            node_items=node_items,
            edge_entries=edge_entries,
            previous_state=previous_state,
            step=int(getattr(manifold, "time_step", 0)),
            dt=dt,
            field_owner=getattr(region, "hrm_field", None),
            coherence=getattr(region, "stability", None),
            auto_rebind=auto_rebind,
        )

    def recovery_ratio(self, scope_id: str) -> float:
        metrics = self.state_for(scope_id).last_metrics
        before = float(metrics.get("distance_to_anchor_before", 0.0))
        after = float(metrics.get("distance_to_anchor_after", before))
        if before <= self.config.epsilon:
            return 1.0
        return float(np.clip((before - after) / before, -1.0, 1.0))


def install_collapse_attractor_support(
    manifold_class: type,
    config: Optional[CollapseConfig] = None,
    *,
    monitor_regions: bool = True,
    monitor_children: bool = True,
    auto_rebind: bool = True,
    auto_topology_repair: bool = True,
    max_auto_repairs: int = 2,
    dt: float = 1.0,
) -> type:
    """Install collapse diagnostics into RecursiveCognitiveManifold once.

    The function is idempotent. Calling it more than once does not stack wrappers.
    """

    if getattr(manifold_class, "_collapse_attractor_installed", False):
        return manifold_class

    original_init = manifold_class.__init__
    original_step = manifold_class.step
    original_snapshot = manifold_class.snapshot

    @functools.wraps(original_init)
    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.collapse_monitor = CollapseAttractorMonitor(config)
        self.collapse_flag = False
        self.collapse_energy = 0.0
        self.collapse_memory: List[CollapseEvent] = []
        self.attractor_anchor: Optional[Array] = None
        self.last_collapse_metrics: Dict[str, Any] = {}
        self.region_collapse_metrics: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.child_collapse_metrics: List[Dict[str, Any]] = []
        self.last_autonomous_repairs: List[Dict[str, Any]] = []

    @functools.wraps(original_step)
    def wrapped_step(
        self: Any,
        external_input: Optional[Dict[Any, Array]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        monitor: CollapseAttractorMonitor = self.collapse_monitor

        previous_global = monitor.aggregate_nodes(list(self.nodes.values()))
        previous_regions: Dict[Tuple[Any, ...], Array] = {}
        if monitor_regions:
            for key, region in self.regions.items():
                nodes = [self.nodes[node_id] for node_id in region.node_ids if node_id in self.nodes]
                previous_regions[tuple(key)] = monitor.aggregate_nodes(nodes)

        previous_children: List[Array] = []
        if monitor_children:
            previous_children = [
                monitor.aggregate_nodes(list(child.nodes.values()))
                for child in getattr(self, "child_manifolds", [])
            ]

        result = original_step(self, external_input, *args, **kwargs)

        self.last_collapse_metrics = monitor.monitor_manifold(
            self,
            previous_state=previous_global,
            dt=dt,
            auto_rebind=auto_rebind,
        )

        level = int(getattr(self, "level", getattr(self, "hierarchy_level", 0)))
        label = str(getattr(self, "label", "manifold"))
        global_scope_id = f"manifold:{label}:level:{level}"
        global_state = monitor.state_for(global_scope_id)
        self.collapse_flag = global_state.collapse_flag
        self.collapse_energy = global_state.collapse_energy
        self.collapse_memory = global_state.collapse_memory
        self.attractor_anchor = global_state.attractor_anchor

        self.region_collapse_metrics = {}
        if monitor_regions:
            for key, region in self.regions.items():
                region_key = tuple(key)
                nodes = [self.nodes[node_id] for node_id in region.node_ids if node_id in self.nodes]
                if not nodes:
                    continue
                previous = previous_regions.get(region_key, monitor.aggregate_nodes(nodes))
                metrics = monitor.monitor_region(
                    self,
                    region_key,
                    region,
                    previous_state=previous,
                    dt=dt,
                    auto_rebind=auto_rebind,
                )
                self.region_collapse_metrics[region_key] = metrics
                try:
                    region.collapse_metrics = metrics
                except Exception:
                    pass

        self.child_collapse_metrics = []
        if monitor_children:
            for child_index, child in enumerate(getattr(self, "child_manifolds", [])):
                if not hasattr(child, "collapse_monitor"):
                    child.collapse_monitor = CollapseAttractorMonitor(config)
                child_previous = (
                    previous_children[child_index]
                    if child_index < len(previous_children)
                    else child.collapse_monitor.aggregate_nodes(list(child.nodes.values()))
                )
                child_metrics = child.collapse_monitor.monitor_manifold(
                    child,
                    previous_state=child_previous,
                    dt=dt,
                    auto_rebind=auto_rebind,
                )
                child.last_collapse_metrics = child_metrics
                self.child_collapse_metrics.append(child_metrics)

        self.last_autonomous_repairs = []
        should_attempt_repair = bool(self.collapse_flag)
        if not should_attempt_repair and self.region_collapse_metrics:
            should_attempt_repair = any(bool(metrics.get("collapse_flag", False)) for metrics in self.region_collapse_metrics.values())
        if not should_attempt_repair and hasattr(self, "_expected_edge_strengths"):
            try:
                expected_edges = self._expected_edge_strengths(repair_decay_horizon=8)
                should_attempt_repair = any(
                    edge_key not in self.edges
                    and edge_key[0] in self.nodes
                    and edge_key[1] in self.nodes
                    for edge_key in expected_edges.keys()
                )
            except Exception:
                should_attempt_repair = False
        if auto_topology_repair and should_attempt_repair and hasattr(self, "attempt_topology_repair"):
            try:
                self.last_autonomous_repairs = list(
                    self.attempt_topology_repair(
                        max_repairs=max_auto_repairs,
                        similarity_threshold=0.0,
                    )
                )
            except Exception:
                self.last_autonomous_repairs = []

        if isinstance(result, dict):
            result = dict(result)
            result["collapse"] = self.last_collapse_metrics
            result["region_collapse"] = self.region_collapse_metrics
            result["child_collapse"] = self.child_collapse_metrics
            result["autonomous_repair"] = list(self.last_autonomous_repairs)
        return result

    @functools.wraps(original_snapshot)
    def wrapped_snapshot(self: Any) -> Dict[str, Any]:
        snapshot = dict(original_snapshot(self))
        snapshot["collapse"] = dict(getattr(self, "last_collapse_metrics", {}))
        snapshot["collapse_flag"] = bool(getattr(self, "collapse_flag", False))
        snapshot["collapse_energy"] = float(getattr(self, "collapse_energy", 0.0))
        snapshot["collapse_events"] = len(getattr(self, "collapse_memory", []))
        snapshot["attractor_initialized"] = getattr(self, "attractor_anchor", None) is not None
        snapshot["autonomous_repairs"] = list(getattr(self, "last_autonomous_repairs", []))
        return snapshot

    manifold_class.__init__ = wrapped_init
    manifold_class.step = wrapped_step
    manifold_class.snapshot = wrapped_snapshot
    manifold_class._collapse_attractor_installed = True
    manifold_class.CollapseConfig = CollapseConfig
    manifold_class.CollapseEvent = CollapseEvent
    manifold_class.CollapseAttractorMonitor = CollapseAttractorMonitor
    return manifold_class


__all__ = [
    "AttractorState",
    "CollapseAttractorMonitor",
    "CollapseConfig",
    "CollapseEvent",
    "install_collapse_attractor_support",
]