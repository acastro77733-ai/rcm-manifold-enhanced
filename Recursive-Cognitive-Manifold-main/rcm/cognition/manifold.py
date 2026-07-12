from collections import defaultdict, deque

import numpy as np

from rcm.cognition.edge import CognitiveEdge
from rcm.cognition.node import CognitiveNode
from rcm.cognition.region import CognitiveRegion
from rcm.dynamics.fields import LegacyFieldModel, NonlinearFieldModel
from rcm.dynamics.plasticity import request_simplicial_surgery
from rcm.dynamics.plasticity import state_similarity
from rcm.dynamics.plasticity import update_edge_dynamics
from rcm.dynamics.propagation import propagate_states
from rcm.dynamics.propagation import stimulate_node
from rcm.dynamics.regional_hrm_field import RegionalHRMField
from rcm.dynamics.regional_hrm_field import summarize_region_nodes
from rcm.dynamics.state_engine import StateEngine, StateEngineConfig
from rcm.dynamics.synchronization import apply_synchronization
from rcm.dynamics.synchronization import coherence
from rcm.hierarchy.abstraction import infer_specialization
from rcm.hierarchy.abstraction import region_activation
from rcm.hierarchy.child_manifold import refresh_child_manifolds
from rcm.config.dynamics import StateDynamicsConfig
from rcm.config.field import FieldConfig
from rcm.config.representation import RepresentationConfig
from rcm.config.stability import StabilityConfig
from rcm.config.topology import TopologyConfig
from rcm.memory.episodic import EpisodicMemory
from rcm.memory.recall import combine_state_with_prediction
from rcm.memory.semantic import SemanticMemory
from rcm.memory.structural import StructuralMemory
from rcm.metrics import ManifoldMetrics
from rcm.representation import RepresentationController
from rcm.stability import StabilityPolicy


class RecursiveCognitiveManifold:
    def __init__(
        self,
        state_dim: int = 4,
        level: int = 0,
        label: str = "node",
        hrm_seed: int | None = None,
        dynamics_config: StateDynamicsConfig | None = None,
        topology_config: TopologyConfig | None = None,
        field_config: FieldConfig | None = None,
        stability_config: StabilityConfig | None = None,
        representation_config: RepresentationConfig | None = None,
    ):
        self.state_dim = state_dim
        self.level = level
        self.label = label
        self.nodes: dict[int, CognitiveNode] = {}
        self.edges: dict[tuple[int, int], CognitiveEdge] = {}
        self.regions: dict[tuple[int, ...], CognitiveRegion] = {}
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.structural_memory = StructuralMemory()
        self.child_manifolds: list[RecursiveCognitiveManifold] = []
        if hrm_seed is None:
            hrm_seed = 1000 if level == 0 else 2000 + level
        self.dynamics_config = dynamics_config or StateDynamicsConfig()
        self.topology_config = topology_config or TopologyConfig()
        self.field_config = field_config or FieldConfig()
        field_model = self.field_config.field_model.lower()
        if field_model == "nonlinear":
            self.hrm_field = NonlinearFieldModel(seed=hrm_seed, state_dim=state_dim)
        else:
            self.hrm_field = LegacyFieldModel(seed=hrm_seed, state_dim=state_dim)
        self.last_child_field_metrics = []
        self.geometry_complex = None
        self.last_geometry_feedback = {}
        self.last_semantic_guidance = {}
        self.stability_config = stability_config or StabilityConfig()
        self.representation_config = representation_config or RepresentationConfig()
        self.metrics = ManifoldMetrics(task_score=0.0, recall_score=0.0, coherence=0.0, collapse_energy=0.0, topology_cost=0.0, representation_cost=0.0, field_energy=0.0)
        self.unified_state_engine = StateEngine(config=StateEngineConfig())
        self.representation_controller = RepresentationController(
            current_dim=state_dim,
            min_dim=self.representation_config.min_dim,
            max_dim=self.representation_config.max_dim,
            trial_duration=self.representation_config.trial_duration,
            minimum_improvement=self.representation_config.minimum_improvement,
            switch_cost=self.representation_config.switch_cost,
            cooldown_steps=self.representation_config.cooldown_steps,
        )
        self.unified_state_history = []
        self.time_step = 0
        self.stability_controller = StabilityPolicy()
        self.last_stability_report = {}
        self._previous_state_signature = np.zeros(state_dim, dtype=float)
        self._current_state_signature = np.zeros(state_dim, dtype=float)
        self.collapse_energy = 0.0

    @classmethod
    def from_simplicial_complex(cls, complex_, state_dim: int = 4, hrm_seed: int | None = None):
        manifold = cls(state_dim=state_dim, level=0, label="node", hrm_seed=hrm_seed)
        manifold.geometry_complex = complex_
        for node_id in range(complex_.n_vertices):
            coords = complex_.vertices[node_id][:state_dim]
            state = np.zeros(state_dim, dtype=float)
            state[: len(coords)] = coords
            manifold.add_node(node_id, state)

        for u, v in {tuple(sorted(edge)) for edge in complex_.half_edges.keys()}:
            distance = np.linalg.norm(complex_.vertices[u] - complex_.vertices[v])
            manifold.connect(u, v, strength=1.0 / (1.0 + distance), latency=1.0 + distance)
        return manifold

    def add_node(self, node_id: int, local_state=None):
        if local_state is None:
            local_state = np.zeros(self.state_dim, dtype=float)
        self.nodes[node_id] = CognitiveNode(local_state=np.asarray(local_state, dtype=float))

    def connect(self, source: int, target: int, strength: float = 0.5, latency: float = 1.0):
        edge = CognitiveEdge(strength=float(strength), latency=float(latency))
        self.edges[(source, target)] = edge
        self.edges[(target, source)] = CognitiveEdge(strength=edge.strength, latency=edge.latency)
        self._record_topology_event(source, f"link:{source}->{target}")
        self._record_topology_event(target, f"link:{target}->{source}")

    def stimulate(self, node_id: int, signal):
        stimulate_node(self, node_id, signal)

    def step(self, external_input=None):
        if external_input:
            for node_id, signal in external_input.items():
                self.stimulate(node_id, signal)

        updated_states, edge_usage = propagate_states(self)
        updated_states = self._apply_episodic_recall(updated_states)
        apply_synchronization(self, updated_states)
        self.episodic_memory.record(self.nodes, self.time_step)
        pruned_edges = update_edge_dynamics(self, edge_usage)
        self.structural_memory.record(self.time_step, self.edges, pruned_edges)
        self._apply_geometry_feedback(edge_usage)
        self._update_regions()
        self._apply_semantic_guidance()
        self._apply_regional_fields()
        self._apply_manifold_field()
        self.semantic_memory.observe(self)
        refresh_child_manifolds(self)
        self._apply_child_manifold_fields()
        unified_result = self.unified_state_engine.step(self, external_input=external_input)
        self.unified_state_history.append(unified_result)
        state_vector = np.asarray([node.local_state[0] if len(node.local_state) else 0.0 for node in self.nodes.values()], dtype=float)
        self._previous_state_signature = self._current_state_signature.copy()
        self._current_state_signature = self._state_signature()
        proposal = self.representation_controller.propose(state_vector, objective=float(self.metrics.task_score), collapse_event=bool(getattr(self, "collapse_flag", False)), step=self.time_step)
        tx_id = self.stability_controller.recovery.begin_transaction(
            "representation",
            {"target_dim": int(proposal.target_dim)},
            self._snapshot_for_recovery(),
            trial_metrics={"objective": float(self.metrics.task_score)},
            post_change_metrics={"objective": float(self.metrics.task_score)},
            accepted=False,
        )
        accepted, _, _ = self.representation_controller.evaluate(
            proposal,
            objective=float(self.metrics.task_score),
            current_state=self.representation_controller.current_state,
            step=self.time_step,
        )
        self.stability_controller.recovery.complete_transaction(
            tx_id,
            post_change_metrics={"accepted": bool(accepted), "objective": float(self.metrics.task_score)},
            accepted=bool(accepted),
        )
        self.time_step += 1
        snapshot = self.snapshot()
        self.last_stability_report = self.stability_controller.evaluate(
            self,
            {
                **self._snapshot_for_recovery(),
                "field_metrics": getattr(self, "last_field_metrics", {}),
                "state_signature": self._current_state_signature,
                "previous_state_signature": self._previous_state_signature,
                "collapse_energy": float(getattr(self, "collapse_energy", 0.0)),
            },
        )
        self.collapse_energy = float(self.last_stability_report.get("collapse_energy", getattr(self, "collapse_energy", 0.0)))
        self.collapse_flag = bool(self.last_stability_report.get("collapse_event", False))
        self.metrics = ManifoldMetrics(
            task_score=float(len(self.regions) + len(self.child_manifolds)),
            recall_score=float(np.mean([node.confidence for node in self.nodes.values()]) if self.nodes else 0.0),
            coherence=float(np.mean([node.synchronization_state for node in self.nodes.values()]) if self.nodes else 0.0),
            collapse_energy=float(getattr(self, "collapse_energy", 0.0)),
            topology_cost=float(len(self.edges)),
            representation_cost=float(len(self.nodes) + len(self.regions)),
            field_energy=float(getattr(self, "last_field_metrics", {}).get("field_energy", 0.0)),
        )
        snapshot["metrics"] = {
            "task_score": self.metrics.task_score,
            "recall_score": self.metrics.recall_score,
            "coherence": self.metrics.coherence,
            "collapse_energy": self.metrics.collapse_energy,
            "topology_cost": self.metrics.topology_cost,
            "representation_cost": self.metrics.representation_cost,
            "field_energy": self.metrics.field_energy,
        }
        snapshot["metrics_object"] = self.metrics.__dict__
        snapshot["unified_state"] = {
            "state_vector": unified_result["state_vector"].tolist(),
            "bounded": unified_result["bounded"],
            "block_names": unified_result["block_names"],
        }
        snapshot["unified_metrics"] = {
            "state_norm": float(np.linalg.norm(unified_result["state_vector"])),
            "contribution_count": len(unified_result["contributions"]),
        }
        snapshot["unified_contributions"] = [
            {"term": entry["term"], "norm": float(np.linalg.norm(np.asarray(entry["value"], dtype=float)))}
            for entry in unified_result["contributions"]
        ]
        return snapshot

    def attempt_topology_repair(
        self,
        max_repairs: int = 2,
        strength_floor: float = 0.26,
        similarity_threshold: float = 0.55,
        repair_decay_horizon: int = 8,
    ):
        if len(self.nodes) < 2:
            return []

        expected_strengths = self._expected_edge_strengths(repair_decay_horizon)
        if not expected_strengths:
            return []

        candidate_repairs = []
        for edge_key, expected_strength in expected_strengths.items():
            source, target = edge_key
            if source not in self.nodes or target not in self.nodes:
                continue
            if edge_key in self.edges:
                edge = self.edges[edge_key]
                edge.strength = max(edge.strength, min(expected_strength, 1.0))
                continue

            similarity = self._state_similarity(self.nodes[source].local_state, self.nodes[target].local_state)
            if similarity < similarity_threshold:
                continue

            source_events = self._topology_event_score(source, target, repair_decay_horizon)
            target_events = self._topology_event_score(target, source, repair_decay_horizon)
            topology_support = max(source_events, target_events)
            if topology_support <= 0.0:
                continue

            candidate_repairs.append(
                {
                    "edge_key": edge_key,
                    "strength": max(strength_floor, min(expected_strength, 1.0)),
                    "latency": self._estimate_latency(source, target),
                    "priority": (0.6 * expected_strength) + (0.25 * similarity) + (0.15 * topology_support),
                }
            )

        candidate_repairs.sort(key=lambda item: item["priority"], reverse=True)
        applied = []
        repaired_pairs = set()
        for candidate in candidate_repairs:
            source, target = candidate["edge_key"]
            undirected = tuple(sorted((source, target)))
            if undirected in repaired_pairs:
                continue
            self.connect(source, target, strength=candidate["strength"], latency=candidate["latency"])
            repaired_pairs.add(undirected)
            applied.append(
                {
                    "source": source,
                    "target": target,
                    "strength": candidate["strength"],
                    "latency": candidate["latency"],
                    "priority": candidate["priority"],
                }
            )
            if len(applied) >= max_repairs:
                break

        if applied:
            self._update_regions()
        return applied

    def _state_signature(self) -> np.ndarray:
        if not self.nodes:
            return np.zeros(self.state_dim, dtype=float)
        signatures = []
        for node in self.nodes.values():
            state = np.asarray(node.local_state, dtype=float).reshape(-1)
            signatures.append(state[: min(self.state_dim, state.size)])
        if not signatures:
            return np.zeros(self.state_dim, dtype=float)
        return np.asarray(np.mean(np.stack(signatures, axis=0), axis=0), dtype=float).reshape(-1)

    def _snapshot_for_recovery(self) -> dict:
        return {
            "level": self.level,
            "label": self.label,
            "time_step": self.time_step,
            "state_signature": self._state_signature(),
            "previous_state_signature": self._previous_state_signature.copy(),
            "representation_dim": int(self.representation_controller.current_dim),
            "state_vector": self._state_signature().tolist(),
            "field_metrics": getattr(self, "last_field_metrics", {}),
            "metrics": getattr(self, "metrics", None),
            "collapse_energy": float(getattr(self, "collapse_energy", 0.0)),
        }

    def snapshot(self):
        return {
            "level": self.level,
            "label": self.label,
            "time_step": self.time_step,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "region_count": len(self.regions),
            "episodic_events": len(self.episodic_memory.events),
            "semantic_motifs": self.semantic_memory.snapshot(),
            "semantic_guidance": getattr(self, "last_semantic_guidance", {}),
            "child_manifolds": len(self.child_manifolds),
            "field_metrics": getattr(self, "last_field_metrics", {}),
            "child_field_metrics": getattr(self, "last_child_field_metrics", []),
            "geometry_feedback": getattr(self, "last_geometry_feedback", {}),
            "metrics": getattr(self, "metrics", None),
        }

    def _record_topology_event(self, node_id: int, event: str):
        if node_id in self.nodes:
            self.nodes[node_id].topology_history.append((self.time_step, event))

    def _expected_edge_strengths(self, repair_decay_horizon: int):
        expected_strengths = {}
        current_time = max(self.time_step, 1)
        for event in self.structural_memory.events:
            edge_strengths = event.get("edge_strengths", {})
            if not edge_strengths:
                continue
            age = max(current_time - event.get("time_step", current_time), 0)
            weight = 1.0 / (1.0 + (age / max(repair_decay_horizon, 1)))
            for edge_key, strength in edge_strengths.items():
                previous = expected_strengths.get(edge_key)
                weighted_strength = weight * float(strength)
                if previous is None or weighted_strength > previous:
                    expected_strengths[edge_key] = weighted_strength

        for source, node in self.nodes.items():
            for _event_time, event in node.topology_history:
                if not event.startswith("link:"):
                    continue
                try:
                    _, edge_text = event.split(":", 1)
                    left_text, right_text = edge_text.split("->", 1)
                    left = int(left_text)
                    right = int(right_text)
                except ValueError:
                    continue
                if left != source or right not in self.nodes:
                    continue
                expected_strengths.setdefault((left, right), self._estimate_repair_strength(left, right))
        return expected_strengths

    def _topology_event_score(self, source: int, target: int, repair_decay_horizon: int) -> float:
        if source not in self.nodes:
            return 0.0
        signal = f"link:{source}->{target}"
        score = 0.0
        current_time = max(self.time_step, 1)
        for event_time, event in self.nodes[source].topology_history:
            if event != signal:
                continue
            age = max(current_time - event_time, 0)
            score = max(score, 1.0 / (1.0 + (age / max(repair_decay_horizon, 1))))
        return score

    def _estimate_latency(self, source: int, target: int) -> float:
        source_state = self.nodes[source].local_state
        target_state = self.nodes[target].local_state
        distance = float(np.linalg.norm(source_state - target_state))
        return 1.0 + distance

    def _estimate_repair_strength(self, source: int, target: int) -> float:
        similarity = max(0.0, self._state_similarity(self.nodes[source].local_state, self.nodes[target].local_state))
        incident_strengths = []
        for edge_key, edge in self.edges.items():
            if source in edge_key or target in edge_key:
                incident_strengths.append(edge.strength)
        if incident_strengths:
            neighborhood_strength = float(np.mean(incident_strengths))
        else:
            neighborhood_strength = 0.35
        return max(0.26, min(1.0, (0.55 * neighborhood_strength) + (0.35 * similarity) + 0.10))

    def _fit_state(self, signal):
        state = np.zeros(self.state_dim, dtype=float)
        signal = signal[: self.state_dim]
        state[: len(signal)] = signal
        return state

    def _coherence(self, memory: deque) -> float:
        return coherence(memory)

    def _apply_episodic_recall(self, updated_states):
        recalled_states = {}
        for node_id, node_state in updated_states.items():
            episode = self.episodic_memory.retrieve(node_state, node_id=node_id)
            if episode is None:
                recalled_states[node_id] = node_state
                continue
            prediction = episode["next_state"]
            confidence = episode["outcome"].get("confidence", 0.5)
            recalled_states[node_id] = combine_state_with_prediction(
                node_state,
                prediction,
                episode["similarity"],
                confidence,
            )
        return recalled_states

    def _update_regions(self):
        adjacency = defaultdict(set)
        for source, target in self.edges:
            if self.edges[(source, target)].strength >= 0.25:
                adjacency[source].add(target)

        visited = set()
        regions = {}
        for region_index, node_id in enumerate(self.nodes):
            if node_id in visited:
                continue
            component = []
            queue = deque([node_id])
            visited.add(node_id)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            signature = tuple(sorted(component))
            specialization = infer_specialization(self, component)
            previous = self.regions.get(signature)
            region_id = previous.region_id if previous is not None else len(regions)
            region = previous if previous is not None else CognitiveRegion(signature, specialization, region_id=region_id)
            region.specialization = specialization
            region.node_ids = signature
            if region.hrm_field is None:
                field_model = self.field_config.field_model.lower()
                if field_model == "nonlinear":
                    region.hrm_field = NonlinearFieldModel(seed=region.region_id, state_dim=self.state_dim)
                else:
                    region.hrm_field = LegacyFieldModel(seed=region.region_id, state_dim=self.state_dim)
            region.activation_trace.append(region_activation(self, component))
            region.stability = coherence(region.activation_trace)
            regions[signature] = region
            for member in component:
                self.nodes[member].specialization = specialization

        self.regions = regions

    def _region_activation(self, component):
        return region_activation(self, component)

    def _apply_semantic_guidance(self):
        region_guidance = []
        for signature, region in self.regions.items():
            suggestion = self.semantic_memory.suggest_region_transition(self, signature, region)
            if suggestion is None:
                continue
            guidance_vector = np.asarray(suggestion["guidance_vector"], dtype=float)
            guidance_gain = float(suggestion["guidance_gain"])
            confidence = float(suggestion["confidence"])

            for node_id in signature:
                if node_id not in self.nodes:
                    continue
                node = self.nodes[node_id]
                width = min(len(node.local_state), len(guidance_vector))
                node.local_state[:width] += guidance_gain * guidance_vector[:width]
                node.confidence = min(1.0, node.confidence + 0.01 * confidence)

            region.field_metrics["semantic_guidance_gain"] = guidance_gain
            region.field_metrics["semantic_guidance_confidence"] = confidence
            region_guidance.append(
                {
                    "region": tuple(int(node_id) for node_id in signature),
                    "motif_signature": tuple(suggestion["motif_signature"]),
                    "guidance_gain": guidance_gain,
                    "confidence": confidence,
                    "topology_alignment": float(suggestion["topology_alignment"]),
                    "usage_count": int(suggestion["usage_count"]),
                }
            )

        self.last_semantic_guidance = {
            "enabled": True,
            "region_guidance": region_guidance,
        }

    def _state_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        return state_similarity(left, right)

    def _apply_regional_fields(self):
        for region in self.regions.values():
            region_nodes = [self.nodes[node_id] for node_id in region.node_ids]
            node_summary, mean_energy, mean_confidence = summarize_region_nodes(region_nodes)
            regional_edges, region_edge_count, region_density = self._region_edge_stats(region.node_ids)
            result = region.hrm_field.step(
                node_summary=node_summary,
                mean_energy=mean_energy,
                mean_confidence=mean_confidence,
                topology_stats={
                    "node_count": len(region.node_ids),
                    "edge_count": region_edge_count,
                    "density": region_density,
                    "stability": region.stability,
                },
            )
            modulation = result["node_modulation"]
            sync_pressure = result["synchronization_pressure"]
            plasticity_pressure = result["plasticity_pressure"]
            for node in region_nodes:
                width = min(len(node.local_state), len(modulation))
                node.local_state[:width] += 0.01 * sync_pressure * modulation[:width]
                node.energy += 0.005 * result["field_variance"]
            for edge in regional_edges:
                edge.strength += 0.002 * plasticity_pressure * edge.resonance
                edge.strength = max(0.0, min(1.0, edge.strength))
            region.field_metrics = result["metrics"]

    def _apply_manifold_field(self):
        region_nodes = list(self.nodes.values())
        node_summary, mean_energy, mean_confidence = summarize_region_nodes(region_nodes)
        _, edge_count, density = self._region_edge_stats(tuple(self.nodes.keys()))
        mean_stability = float(np.mean([region.stability for region in self.regions.values()])) if self.regions else 0.0
        result = self.hrm_field.step(
            node_summary=node_summary,
            mean_energy=mean_energy,
            mean_confidence=mean_confidence,
            topology_stats={
                "node_count": len(self.nodes),
                "edge_count": edge_count,
                "density": density,
                "stability": mean_stability,
            },
        )
        modulation = result["node_modulation"]
        sync_pressure = result["synchronization_pressure"]
        plasticity_pressure = result["plasticity_pressure"]
        for node in self.nodes.values():
            width = min(len(node.local_state), len(modulation))
            node.local_state[:width] += 0.004 * sync_pressure * modulation[:width]
            node.energy += 0.002 * result["field_variance"]
        for edge in self.edges.values():
            edge.strength += 0.001 * plasticity_pressure * edge.resonance
            edge.strength = max(0.0, min(1.0, edge.strength))
        self.last_field_metrics = result["metrics"]

    def _apply_child_manifold_fields(self):
        feedback_metrics = []
        for child_manifold in self.child_manifolds:
            child_nodes = list(child_manifold.nodes.values())
            node_summary, mean_energy, mean_confidence = summarize_region_nodes(child_nodes)
            if node_summary.size == 0:
                continue
            _, edge_count, density = child_manifold._region_edge_stats(tuple(child_manifold.nodes.keys()))
            result = child_manifold.hrm_field.step(
                node_summary=node_summary,
                mean_energy=mean_energy,
                mean_confidence=mean_confidence,
                topology_stats={
                    "node_count": len(child_manifold.nodes),
                    "edge_count": edge_count,
                    "density": density,
                    "stability": 0.0,
                },
            )
            modulation = result["node_modulation"]
            sync_pressure = result["synchronization_pressure"]
            plasticity_pressure = result["plasticity_pressure"]
            for node in child_nodes:
                width = min(len(node.local_state), len(modulation))
                node.local_state[:width] += 0.003 * sync_pressure * modulation[:width]
            child_manifold.last_field_metrics = result["metrics"]
            parent_region_map = getattr(child_manifold, "parent_region_map", {})
            signature_to_meta = {
                tuple(int(node_id) for node_id in signature): int(meta_id)
                for meta_id, signature in parent_region_map.items()
                if int(meta_id) in child_manifold.nodes
            }
            for meta_id, parent_signature in parent_region_map.items():
                if meta_id not in child_manifold.nodes or parent_signature not in self.regions:
                    continue
                parent_region = self.regions[parent_signature]
                parent_nodes = [self.nodes[node_id] for node_id in parent_region.node_ids]
                regional_edges, _, _ = self._region_edge_stats(parent_region.node_ids)
                child_node = child_manifold.nodes[meta_id]
                width = min(len(child_node.local_state), len(modulation))
                feedback_vector = child_node.local_state[:width] + modulation[:width]
                for node in parent_nodes:
                    local_width = min(len(node.local_state), len(feedback_vector))
                    node.local_state[:local_width] += 0.006 * sync_pressure * feedback_vector[:local_width]
                    node.energy += 0.003 * result["field_variance"]
                    node.confidence = min(1.0, node.confidence + 0.002 * sync_pressure)
                for edge in regional_edges:
                    edge.strength += 0.0015 * plasticity_pressure * (1.0 + edge.resonance)
                    edge.strength = max(0.0, min(1.0, edge.strength))
                coupling_pressure = 0.0
                for other_signature, other_meta_id in signature_to_meta.items():
                    if other_signature == tuple(int(node_id) for node_id in parent_signature):
                        continue
                    forward = child_manifold.edges.get((meta_id, other_meta_id))
                    backward = child_manifold.edges.get((other_meta_id, meta_id))
                    if forward is None and backward is None:
                        continue
                    coupling = 0.0
                    if forward is not None:
                        coupling += forward.strength * (1.0 + forward.resonance)
                    if backward is not None:
                        coupling += backward.strength * (1.0 + backward.resonance)
                    if forward is not None and backward is not None:
                        coupling *= 0.5
                    coupling_pressure += coupling
                    for source_id in parent_region.node_ids:
                        for target_id in other_signature:
                            edge = self.edges.get((source_id, target_id))
                            if edge is None:
                                continue
                            edge.strength += 0.0012 * plasticity_pressure * coupling
                            edge.strength = max(0.0, min(1.0, edge.strength))
                parent_region.field_metrics["child_feedback_coupling_pressure"] = coupling_pressure
                parent_region.field_metrics["child_feedback_sync_pressure"] = sync_pressure
                parent_region.field_metrics["child_feedback_plasticity_pressure"] = plasticity_pressure
                feedback_metrics.append(
                    {
                        "parent_region": tuple(int(node_id) for node_id in parent_signature),
                        "sync_pressure": sync_pressure,
                        "plasticity_pressure": plasticity_pressure,
                        "coupling_pressure": coupling_pressure,
                        "field_variance": result["field_variance"],
                    }
                )
        self.last_child_field_metrics = feedback_metrics

    def _apply_geometry_feedback(self, edge_usage):
        complex_ = getattr(self, "geometry_complex", None)
        if complex_ is None:
            self.last_geometry_feedback = {
                "enabled": False,
                "surgery_requests": [],
                "surgery_applied": [],
                "laplacian_energy": 0.0,
            }
            return

        requests = request_simplicial_surgery(self, edge_usage, max_requests=2)
        attempted = []
        applied = []
        for request in requests:
            source, target = request["edge"]
            if source not in self.nodes or target not in self.nodes:
                continue
            attempted.append(
                {
                    "edge": (int(source), int(target)),
                    "priority": float(request["priority"]),
                }
            )
            if not complex_.flip_edge(source, target):
                continue
            applied.append(
                {
                    "edge": (int(source), int(target)),
                    "priority": float(request["priority"]),
                    "similarity": float(request["similarity"]),
                    "usage": float(request["usage"]),
                }
            )
            self._record_topology_event(source, f"surgery:flip:{source}->{target}")
            self._record_topology_event(target, f"surgery:flip:{target}->{source}")

        if applied:
            complex_.compute_cotangent_laplacian()

        laplacian_energy = 0.0
        if hasattr(complex_, "L") and self.nodes:
            ordered_ids = sorted(node_id for node_id in self.nodes if 0 <= int(node_id) < complex_.n_vertices)
            if ordered_ids:
                state_matrix = np.stack([self.nodes[node_id].local_state for node_id in ordered_ids], axis=0)
                laplacian = complex_.L[ordered_ids, :][:, ordered_ids].dot(state_matrix)
                laplacian_energy = float(np.mean(np.linalg.norm(laplacian, axis=1)))
                for row_index, node_id in enumerate(ordered_ids):
                    node = self.nodes[node_id]
                    width = min(len(node.local_state), laplacian.shape[1])
                    node.local_state[:width] -= 0.002 * laplacian[row_index, :width]

        self.last_geometry_feedback = {
            "enabled": True,
            "surgery_requests": [
                {
                    "edge": (int(item["edge"][0]), int(item["edge"][1])),
                    "priority": float(item["priority"]),
                    "similarity": float(item["similarity"]),
                    "usage": float(item["usage"]),
                }
                for item in requests
            ],
            "surgery_attempted": attempted,
            "surgery_applied": applied,
            "laplacian_energy": laplacian_energy,
        }

    def _region_edge_stats(self, node_ids):
        node_set = set(node_ids)
        regional_edges = []
        for (source, target), edge in self.edges.items():
            if source in node_set and target in node_set:
                regional_edges.append(edge)
        node_count = len(node_set)
        max_edges = max(node_count * (node_count - 1), 1)
        edge_count = len(regional_edges)
        density = edge_count / max_edges
        return regional_edges, edge_count, density