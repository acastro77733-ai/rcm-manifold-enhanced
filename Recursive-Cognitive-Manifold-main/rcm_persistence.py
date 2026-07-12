from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

import numpy as np

from rcm.dynamics.regional_hrm_field import RegionalHRMField
from rcm.memory.semantic import MotifPrototype


SCHEMA_NAME = "recursive-cognitive-manifold-checkpoint"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
ARRAYS_NAME = "arrays.npz"
CHECKSUM_NAME = "checksum.sha256"


class PersistenceError(RuntimeError):
    pass


class _ArrayArchiveBuilder:
    def __init__(self):
        self.arrays: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}

    def add(self, prefix: str, array) -> str:
        key = self._unique_key(prefix)
        self.arrays[key] = np.asarray(array)
        return key

    def write(self, path: Path):
        np.savez_compressed(path, **self.arrays)

    def _unique_key(self, prefix: str) -> str:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", prefix).strip("_") or "array"
        count = self._counts.get(base, 0)
        self._counts[base] = count + 1
        if count == 0 and base not in self.arrays:
            return base
        return f"{base}_{count}"


def _json_primitive(value) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _encode_value(value, builder: _ArrayArchiveBuilder, prefix: str):
    if isinstance(value, np.ndarray):
        return {"__array__": builder.add(prefix, value)}
    if isinstance(value, np.generic):
        return value.item()
    if _json_primitive(value):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, deque):
        return {
            "__deque__": {
                "items": [_encode_value(item, builder, f"{prefix}_item") for item in value],
                "maxlen": value.maxlen,
            }
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_value(item, builder, f"{prefix}_item") for item in value]}
    if isinstance(value, list):
        return [_encode_value(item, builder, f"{prefix}_item") for item in value]
    if isinstance(value, set):
        items = sorted(value, key=repr)
        return {"__set__": [_encode_value(item, builder, f"{prefix}_item") for item in items]}
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {
                key: _encode_value(item, builder, f"{prefix}_{key}")
                for key, item in value.items()
            }
        return {
            "__pairs__": [
                {
                    "key": _encode_value(key, builder, f"{prefix}_key"),
                    "value": _encode_value(item, builder, f"{prefix}_value"),
                }
                for key, item in value.items()
            ]
        }
    raise TypeError(f"Unsupported value type: {type(value)!r}")


def _decode_value(value, arrays):
    if isinstance(value, dict):
        if "__array__" in value:
            return arrays[value["__array__"]].copy()
        if "__tuple__" in value:
            return tuple(_decode_value(item, arrays) for item in value["__tuple__"])
        if "__deque__" in value:
            items = [_decode_value(item, arrays) for item in value["__deque__"]["items"]]
            return deque(items, maxlen=value["__deque__"]["maxlen"])
        if "__set__" in value:
            return set(_decode_value(item, arrays) for item in value["__set__"])
        if "__pairs__" in value:
            return {
                _decode_value(item["key"], arrays): _decode_value(item["value"], arrays)
                for item in value["__pairs__"]
            }
        return {key: _decode_value(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item, arrays) for item in value]
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(checkpoint_dir: Path):
    manifest_path = checkpoint_dir / MANIFEST_NAME
    arrays_path = checkpoint_dir / ARRAYS_NAME
    checksum_path = checkpoint_dir / CHECKSUM_NAME
    checksum_path.write_text(
        f"{_hash_file(manifest_path)}  {MANIFEST_NAME}\n{_hash_file(arrays_path)}  {ARRAYS_NAME}\n",
        encoding="ascii",
    )


def _read_checksums(checkpoint_dir: Path) -> dict[str, str]:
    checksum_path = checkpoint_dir / CHECKSUM_NAME
    if not checksum_path.exists():
        raise PersistenceError(f"Missing checksum file: {checksum_path}")
    expected = {}
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            digest, filename = line.split("  ", 1)
        except ValueError as exc:
            raise PersistenceError(f"Malformed checksum entry: {line!r}") from exc
        expected[filename] = digest
    return expected


def _serialize_field(field, builder: _ArrayArchiveBuilder, prefix: str):
    if field is None:
        return None
    attributes = {}
    for name, value in vars(field).items():
        if name == "rng" or callable(value):
            continue
        try:
            attributes[name] = _encode_value(value, builder, f"{prefix}_{name}")
        except TypeError:
            continue
    return {
        "class_name": field.__class__.__name__,
        "attributes": attributes,
    }


def _restore_field(field_state, arrays, field_factory, fallback_factory):
    if field_state is None:
        return None
    field = field_factory() if field_factory is not None else fallback_factory()
    for name, value in field_state.get("attributes", {}).items():
        setattr(field, name, _decode_value(value, arrays))
    seed = getattr(field, "seed", 0)
    try:
        field.rng = np.random.default_rng(seed)
    except Exception:
        pass
    return field


def _serialize_nodes(manifold, builder: _ArrayArchiveBuilder, prefix: str):
    node_ids = sorted(int(node_id) for node_id in manifold.nodes.keys())
    if node_ids:
        node_states = np.vstack([np.asarray(manifold.nodes[node_id].local_state, dtype=float) for node_id in node_ids])
    else:
        node_states = np.zeros((0, manifold.state_dim), dtype=float)

    node_entries = []
    for state_index, node_id in enumerate(node_ids):
        node = manifold.nodes[node_id]
        short_term = [np.asarray(item, dtype=float) for item in node.short_term_memory]
        long_term = [
            {
                "time_step": int(time_step),
                "state": _encode_value(np.asarray(state, dtype=float), builder, f"{prefix}_node_{node_id}_ltm_state"),
                "energy": float(energy),
            }
            for time_step, state, energy in node.long_term_memory
        ]
        node_entries.append(
            {
                "node_id": node_id,
                "state_index": state_index,
                "specialization": node.specialization,
                "synchronization_state": float(node.synchronization_state),
                "energy": float(node.energy),
                "confidence": float(node.confidence),
                "short_term_maxlen": node.short_term_memory.maxlen,
                "short_term_memory": _encode_value(short_term, builder, f"{prefix}_node_{node_id}_stm"),
                "long_term_maxlen": node.long_term_memory.maxlen,
                "long_term_memory": long_term,
                "topology_history_maxlen": node.topology_history.maxlen,
                "topology_history": [[int(time_step), str(event)] for time_step, event in node.topology_history],
            }
        )

    return {
        "state_array": builder.add(f"{prefix}_node_states", node_states),
        "entries": node_entries,
    }


def _serialize_edges(manifold, builder: _ArrayArchiveBuilder, prefix: str):
    edge_keys = sorted((int(source), int(target)) for source, target in manifold.edges.keys())
    if edge_keys:
        edge_pairs = np.asarray(edge_keys, dtype=np.int64)
        strengths = np.asarray([manifold.edges[edge_key].strength for edge_key in edge_keys], dtype=float)
        latencies = np.asarray([manifold.edges[edge_key].latency for edge_key in edge_keys], dtype=float)
        resonances = np.asarray([manifold.edges[edge_key].resonance for edge_key in edge_keys], dtype=float)
        ages = np.asarray([manifold.edges[edge_key].age for edge_key in edge_keys], dtype=np.int64)
        traversals = np.asarray([manifold.edges[edge_key].traversal_frequency for edge_key in edge_keys], dtype=np.int64)
    else:
        edge_pairs = np.zeros((0, 2), dtype=np.int64)
        strengths = np.zeros((0,), dtype=float)
        latencies = np.zeros((0,), dtype=float)
        resonances = np.zeros((0,), dtype=float)
        ages = np.zeros((0,), dtype=np.int64)
        traversals = np.zeros((0,), dtype=np.int64)

    return {
        "pairs_array": builder.add(f"{prefix}_edge_pairs", edge_pairs),
        "strengths_array": builder.add(f"{prefix}_edge_strengths", strengths),
        "latencies_array": builder.add(f"{prefix}_edge_latencies", latencies),
        "resonances_array": builder.add(f"{prefix}_edge_resonances", resonances),
        "ages_array": builder.add(f"{prefix}_edge_ages", ages),
        "traversals_array": builder.add(f"{prefix}_edge_traversals", traversals),
    }


def _serialize_regions(manifold, builder: _ArrayArchiveBuilder, prefix: str):
    region_entries = []
    for signature, region in sorted(manifold.regions.items(), key=lambda item: tuple(int(node_id) for node_id in item[0])):
        trace_items = [np.asarray(item, dtype=float) for item in region.activation_trace]
        if trace_items:
            activation_trace = np.vstack(trace_items)
        else:
            activation_trace = np.zeros((0, manifold.state_dim), dtype=float)
        region_entries.append(
            {
                "signature": [int(node_id) for node_id in signature],
                "region_id": int(region.region_id),
                "specialization": region.specialization,
                "stability": float(region.stability),
                "activation_trace_maxlen": region.activation_trace.maxlen,
                "activation_trace": _encode_value(activation_trace, builder, f"{prefix}_region_{region.region_id}_trace"),
                "field_metrics": _encode_value(region.field_metrics, builder, f"{prefix}_region_{region.region_id}_field_metrics"),
                "hrm_field": _serialize_field(region.hrm_field, builder, f"{prefix}_region_{region.region_id}_field"),
            }
        )
    return region_entries


def _serialize_episodic_memory(manifold, builder: _ArrayArchiveBuilder, prefix: str):
    events = []
    for event_index, event in enumerate(manifold.episodic_memory.events):
        events.append(
            {
                "time_step": int(event["time_step"]),
                "activations": _encode_value(event.get("activations", {}), builder, f"{prefix}_event_{event_index}_activations"),
                "energy": _encode_value(event.get("energy", {}), builder, f"{prefix}_event_{event_index}_energy"),
                "confidence": _encode_value(event.get("confidence", {}), builder, f"{prefix}_event_{event_index}_confidence"),
                "next_activations": _encode_value(event.get("next_activations"), builder, f"{prefix}_event_{event_index}_next_activations"),
                "outcome": _encode_value(event.get("outcome"), builder, f"{prefix}_event_{event_index}_outcome"),
            }
        )

    transitions = []
    for transition_index, transition in enumerate(manifold.episodic_memory.transitions):
        transitions.append(
            {
                "node_id": int(transition["node_id"]),
                "time_step": int(transition["time_step"]),
                "current_state": _encode_value(transition["current_state"], builder, f"{prefix}_transition_{transition_index}_current"),
                "next_state": _encode_value(transition["next_state"], builder, f"{prefix}_transition_{transition_index}_next"),
                "outcome": _encode_value(transition["outcome"], builder, f"{prefix}_transition_{transition_index}_outcome"),
            }
        )

    return {
        "events_maxlen": manifold.episodic_memory.events.maxlen,
        "transitions_maxlen": manifold.episodic_memory.transitions.maxlen,
        "events": events,
        "transitions": transitions,
    }


def _serialize_structural_memory(manifold, builder: _ArrayArchiveBuilder, prefix: str):
    events = []
    for event_index, event in enumerate(manifold.structural_memory.events):
        events.append(
            {
                "time_step": int(event["time_step"]),
                "edge_strengths": _encode_value(event.get("edge_strengths", {}), builder, f"{prefix}_structural_{event_index}_strengths"),
                "pruned_edges": _encode_value(event.get("pruned_edges", []), builder, f"{prefix}_structural_{event_index}_pruned"),
            }
        )
    return {
        "events_maxlen": manifold.structural_memory.events.maxlen,
        "events": events,
    }


def _serialize_semantic_memory(manifold, builder: _ArrayArchiveBuilder, prefix: str):
    prototypes = []
    for prototype_index, (signature, prototype) in enumerate(sorted(manifold.semantic_memory.prototypes.items(), key=lambda item: repr(item[0]))):
        prototypes.append(
            {
                "signature": _encode_value(signature, builder, f"{prefix}_semantic_{prototype_index}_signature"),
                "prototype_activation": _encode_value(prototype.prototype_activation, builder, f"{prefix}_semantic_{prototype_index}_activation"),
                "topological_pattern": _encode_value(prototype.topological_pattern, builder, f"{prefix}_semantic_{prototype_index}_pattern"),
                "expected_transition": _encode_value(prototype.expected_transition, builder, f"{prefix}_semantic_{prototype_index}_transition"),
                "associated_outcome": _encode_value(prototype.associated_outcome, builder, f"{prefix}_semantic_{prototype_index}_outcome"),
                "confidence": float(prototype.confidence),
                "usage_count": int(prototype.usage_count),
            }
        )
    return prototypes


def _serialize_motif_prototypes(manifold, builder: _ArrayArchiveBuilder, prefix: str):
    motif_prototypes = getattr(manifold, "motif_prototypes", None)
    if motif_prototypes is None:
        return None
    return _encode_value(motif_prototypes, builder, f"{prefix}_motif_prototypes")


def _serialize_manifold(manifold, builder: _ArrayArchiveBuilder, prefix: str):
    child_manifolds = [
        _serialize_manifold(child_manifold, builder, f"{prefix}_child_{child_index}")
        for child_index, child_manifold in enumerate(manifold.child_manifolds)
    ]
    return {
        "state_dim": int(manifold.state_dim),
        "level": int(manifold.level),
        "label": str(manifold.label),
        "time_step": int(manifold.time_step),
        "nodes": _serialize_nodes(manifold, builder, f"{prefix}_nodes"),
        "edges": _serialize_edges(manifold, builder, f"{prefix}_edges"),
        "regions": _serialize_regions(manifold, builder, f"{prefix}_regions"),
        "episodic_memory": _serialize_episodic_memory(manifold, builder, f"{prefix}_episodic"),
        "structural_memory": _serialize_structural_memory(manifold, builder, f"{prefix}_structural"),
        "semantic_memory": _serialize_semantic_memory(manifold, builder, f"{prefix}_semantic"),
        "motif_prototypes": _serialize_motif_prototypes(manifold, builder, f"{prefix}_motif"),
        "hrm_field": _serialize_field(manifold.hrm_field, builder, f"{prefix}_hrm_field"),
        "last_field_metrics": _encode_value(getattr(manifold, "last_field_metrics", {}), builder, f"{prefix}_last_field_metrics"),
        "last_child_field_metrics": _encode_value(getattr(manifold, "last_child_field_metrics", []), builder, f"{prefix}_last_child_field_metrics"),
        "parent_region_map": _encode_value(getattr(manifold, "parent_region_map", {}), builder, f"{prefix}_parent_region_map"),
        "child_manifolds": child_manifolds,
    }


def _restore_nodes(manifold, nodes_state, arrays, foundation):
    node_states = arrays[nodes_state["state_array"]]
    for node_entry in nodes_state["entries"]:
        node_id = int(node_entry["node_id"])
        state = np.asarray(node_states[node_entry["state_index"]], dtype=float).copy()
        manifold.add_node(node_id, state)
        node = manifold.nodes[node_id]
        node.specialization = node_entry["specialization"]
        node.synchronization_state = float(node_entry["synchronization_state"])
        node.energy = float(node_entry["energy"])
        node.confidence = float(node_entry["confidence"])
        short_term = _decode_value(node_entry["short_term_memory"], arrays)
        node.short_term_memory = deque([np.asarray(item, dtype=float).copy() for item in short_term], maxlen=node_entry["short_term_maxlen"])
        long_term_items = []
        for memory_entry in node_entry["long_term_memory"]:
            long_term_items.append(
                (
                    int(memory_entry["time_step"]),
                    np.asarray(_decode_value(memory_entry["state"], arrays), dtype=float).copy(),
                    float(memory_entry["energy"]),
                )
            )
        node.long_term_memory = deque(long_term_items, maxlen=node_entry["long_term_maxlen"])
        history = [(int(time_step), str(event)) for time_step, event in node_entry["topology_history"]]
        node.topology_history = deque(history, maxlen=node_entry["topology_history_maxlen"])


def _restore_edges(manifold, edges_state, arrays, foundation):
    edge_pairs = arrays[edges_state["pairs_array"]]
    strengths = arrays[edges_state["strengths_array"]]
    latencies = arrays[edges_state["latencies_array"]]
    resonances = arrays[edges_state["resonances_array"]]
    ages = arrays[edges_state["ages_array"]]
    traversals = arrays[edges_state["traversals_array"]]
    manifold.edges = {}
    for index, pair in enumerate(edge_pairs):
        source, target = int(pair[0]), int(pair[1])
        edge = foundation.CognitiveEdge(
            strength=float(strengths[index]),
            latency=float(latencies[index]),
            resonance=float(resonances[index]),
            age=int(ages[index]),
            traversal_frequency=int(traversals[index]),
        )
        manifold.edges[(source, target)] = edge


def _restore_regions(manifold, regions_state, arrays, field_factory, foundation):
    manifold.regions = {}
    for region_entry in regions_state:
        signature = tuple(int(node_id) for node_id in region_entry["signature"])
        region = foundation.CognitiveRegion(signature, region_entry["specialization"], region_id=int(region_entry["region_id"]))
        region.stability = float(region_entry["stability"])
        activation_trace = _decode_value(region_entry["activation_trace"], arrays)
        region.activation_trace = deque(
            [np.asarray(item, dtype=float).copy() for item in activation_trace],
            maxlen=region_entry["activation_trace_maxlen"],
        )
        region.field_metrics = _decode_value(region_entry["field_metrics"], arrays)
        region.hrm_field = _restore_field(
            region_entry["hrm_field"],
            arrays,
            field_factory,
            lambda: RegionalHRMField(seed=region.region_id, state_dim=manifold.state_dim),
        )
        manifold.regions[signature] = region


def _restore_episodic_memory(manifold, episodic_state, arrays):
    events = []
    for event_entry in episodic_state["events"]:
        events.append(
            {
                "time_step": int(event_entry["time_step"]),
                "activations": _decode_value(event_entry["activations"], arrays),
                "energy": _decode_value(event_entry["energy"], arrays),
                "confidence": _decode_value(event_entry["confidence"], arrays),
                "next_activations": _decode_value(event_entry["next_activations"], arrays),
                "outcome": _decode_value(event_entry["outcome"], arrays),
            }
        )
    transitions = []
    for transition_entry in episodic_state["transitions"]:
        transitions.append(
            {
                "node_id": int(transition_entry["node_id"]),
                "time_step": int(transition_entry["time_step"]),
                "current_state": np.asarray(_decode_value(transition_entry["current_state"], arrays), dtype=float).copy(),
                "next_state": np.asarray(_decode_value(transition_entry["next_state"], arrays), dtype=float).copy(),
                "outcome": _decode_value(transition_entry["outcome"], arrays),
            }
        )
    manifold.episodic_memory.events = deque(events, maxlen=episodic_state["events_maxlen"])
    manifold.episodic_memory.transitions = deque(transitions, maxlen=episodic_state["transitions_maxlen"])


def _restore_structural_memory(manifold, structural_state, arrays):
    events = []
    for event_entry in structural_state["events"]:
        events.append(
            {
                "time_step": int(event_entry["time_step"]),
                "edge_strengths": _decode_value(event_entry["edge_strengths"], arrays),
                "pruned_edges": _decode_value(event_entry["pruned_edges"], arrays),
            }
        )
    manifold.structural_memory.events = deque(events, maxlen=structural_state["events_maxlen"])


def _restore_semantic_memory(manifold, semantic_state, arrays):
    prototypes = {}
    for prototype_entry in semantic_state:
        signature = _decode_value(prototype_entry["signature"], arrays)
        prototypes[signature] = MotifPrototype(
            signature=signature,
            prototype_activation=np.asarray(_decode_value(prototype_entry["prototype_activation"], arrays), dtype=float).copy(),
            topological_pattern=_decode_value(prototype_entry["topological_pattern"], arrays),
            expected_transition=np.asarray(_decode_value(prototype_entry["expected_transition"], arrays), dtype=float).copy(),
            associated_outcome=_decode_value(prototype_entry["associated_outcome"], arrays),
            confidence=float(prototype_entry["confidence"]),
            usage_count=int(prototype_entry["usage_count"]),
        )
    manifold.semantic_memory.prototypes = prototypes


def _restore_manifold(manifold_state, arrays, foundation, field_factory):
    manifold = foundation.RecursiveCognitiveManifold(
        state_dim=int(manifold_state["state_dim"]),
        level=int(manifold_state["level"]),
        label=manifold_state["label"],
    )
    _restore_nodes(manifold, manifold_state["nodes"], arrays, foundation)
    _restore_edges(manifold, manifold_state["edges"], arrays, foundation)
    _restore_regions(manifold, manifold_state["regions"], arrays, field_factory, foundation)
    _restore_episodic_memory(manifold, manifold_state["episodic_memory"], arrays)
    _restore_structural_memory(manifold, manifold_state["structural_memory"], arrays)
    _restore_semantic_memory(manifold, manifold_state["semantic_memory"], arrays)
    manifold.hrm_field = _restore_field(
        manifold_state["hrm_field"],
        arrays,
        field_factory,
        lambda: RegionalHRMField(seed=0, state_dim=manifold.state_dim),
    )
    manifold.last_field_metrics = _decode_value(manifold_state["last_field_metrics"], arrays)
    manifold.last_child_field_metrics = _decode_value(manifold_state["last_child_field_metrics"], arrays)
    manifold.parent_region_map = _decode_value(manifold_state["parent_region_map"], arrays)
    manifold.time_step = int(manifold_state["time_step"])
    motif_prototypes = manifold_state.get("motif_prototypes")
    if motif_prototypes is not None:
        manifold.motif_prototypes = _decode_value(motif_prototypes, arrays)
    manifold.child_manifolds = [
        _restore_manifold(child_state, arrays, foundation, field_factory)
        for child_state in manifold_state["child_manifolds"]
    ]
    return manifold


class RCMPersistenceStore:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        manifold,
        random_seed: int,
        configuration: dict | None = None,
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        notes: str | None = None,
        parent_checkpoint_id: str | None = None,
        checkpoint_id: str | None = None,
    ):
        checkpoint_id = checkpoint_id or self._new_checkpoint_id()
        checkpoint_dir = self.root_dir / checkpoint_id
        if checkpoint_dir.exists():
            raise PersistenceError(f"Checkpoint already exists: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=False)

        created_at = datetime.now(timezone.utc).isoformat()
        builder = _ArrayArchiveBuilder()
        manifest = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "checkpoint_id": checkpoint_id,
            "created_at": created_at,
            "parent_checkpoint_id": parent_checkpoint_id,
            "experiment": {
                "id": experiment_id,
                "name": experiment_name,
                "notes": notes,
            },
            "random_seed": int(random_seed),
            "configuration": configuration or {},
            "manifold": _serialize_manifold(manifold, builder, "root"),
        }

        manifest_path = checkpoint_dir / MANIFEST_NAME
        arrays_path = checkpoint_dir / ARRAYS_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
        builder.write(arrays_path)
        _write_checksums(checkpoint_dir)
        return checkpoint_dir

    def load(self, checkpoint_path, foundation, field_factory=None):
        checkpoint_dir = Path(checkpoint_path)
        self.verify(checkpoint_dir)
        manifest_path = checkpoint_dir / MANIFEST_NAME
        arrays_path = checkpoint_dir / ARRAYS_NAME
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        self._validate_manifest(manifest)
        with np.load(arrays_path, allow_pickle=False) as arrays:
            restored = _restore_manifold(manifest["manifold"], arrays, foundation, field_factory)
        metadata = {
            "schema": manifest["schema"],
            "schema_version": manifest["schema_version"],
            "checkpoint_id": manifest["checkpoint_id"],
            "created_at": manifest.get("created_at"),
            "parent_checkpoint_id": manifest.get("parent_checkpoint_id"),
            "experiment": manifest["experiment"],
            "random_seed": manifest["random_seed"],
            "configuration": manifest["configuration"],
            "path": checkpoint_dir,
        }
        return restored, metadata

    def verify(self, checkpoint_path):
        checkpoint_dir = Path(checkpoint_path)
        manifest_path = checkpoint_dir / MANIFEST_NAME
        arrays_path = checkpoint_dir / ARRAYS_NAME
        if not checkpoint_dir.exists():
            raise PersistenceError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        if not manifest_path.exists():
            raise PersistenceError(f"Missing manifest: {manifest_path}")
        if not arrays_path.exists():
            raise PersistenceError(f"Missing array archive: {arrays_path}")
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        self._validate_manifest(manifest)
        expected = _read_checksums(checkpoint_dir)
        for filename in (MANIFEST_NAME, ARRAYS_NAME):
            if filename not in expected:
                raise PersistenceError(f"Missing checksum entry for {filename}")
            actual = _hash_file(checkpoint_dir / filename)
            if actual != expected[filename]:
                raise PersistenceError(f"Checksum mismatch for {filename}")
        return True

    def list_checkpoints(self):
        checkpoints = []
        for checkpoint_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            manifest_path = checkpoint_dir / MANIFEST_NAME
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="ascii"))
                self._validate_manifest(manifest)
            except Exception:
                continue
            checkpoints.append(
                {
                    "checkpoint_id": manifest["checkpoint_id"],
                    "experiment": manifest["experiment"],
                    "created_at": manifest.get("created_at"),
                    "parent_checkpoint_id": manifest.get("parent_checkpoint_id"),
                    "path": checkpoint_dir,
                }
            )
        checkpoints.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return checkpoints

    def latest(self, experiment_id: str | None = None):
        checkpoints = self.list_checkpoints()
        if experiment_id is not None:
            checkpoints = [
                checkpoint
                for checkpoint in checkpoints
                if checkpoint["experiment"].get("id") == experiment_id
            ]
        if not checkpoints:
            return None
        return checkpoints[0]["path"]

    def _new_checkpoint_id(self):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid.uuid4().hex[:12]}"

    def _validate_manifest(self, manifest):
        if manifest.get("schema") != SCHEMA_NAME:
            raise PersistenceError("Unsupported checkpoint schema")
        if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
            raise PersistenceError("Unsupported checkpoint schema version")
        for key in ("checkpoint_id", "experiment", "random_seed", "configuration", "manifold"):
            if key not in manifest:
                raise PersistenceError(f"Manifest missing required key: {key}")


__all__ = [
    "CHECKSUM_NAME",
    "MANIFEST_NAME",
    "ARRAYS_NAME",
    "PersistenceError",
    "RCMPersistenceStore",
]
