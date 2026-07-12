from collections import deque

import numpy as np


class EpisodicMemory:
    def __init__(self, maxlen: int = 64):
        self.events = deque(maxlen=maxlen)
        self.transitions = deque(maxlen=maxlen * 16)

    def record(self, nodes, time_step: int):
        activations = {node_id: node.local_state.copy() for node_id, node in nodes.items()}
        energy = {node_id: node.energy for node_id, node in nodes.items()}
        confidence = {node_id: node.confidence for node_id, node in nodes.items()}

        if self.events:
            previous_episode = self.events[-1]
            previous_episode["next_activations"] = activations
            previous_episode["outcome"] = {
                "energy": energy,
                "confidence": confidence,
            }
            for node_id, current_state in previous_episode["activations"].items():
                if node_id not in activations:
                    continue
                self.transitions.append(
                    {
                        "node_id": node_id,
                        "time_step": previous_episode["time_step"],
                        "current_state": current_state.copy(),
                        "next_state": activations[node_id].copy(),
                        "outcome": {
                            "energy": energy[node_id],
                            "confidence": confidence[node_id],
                        },
                    }
                )

        episode = {
            "time_step": time_step,
            "activations": activations,
            "energy": energy,
            "confidence": confidence,
            "next_activations": None,
            "outcome": None,
        }
        self.events.append(episode)
        for node in nodes.values():
            node.long_term_memory.append((time_step, node.local_state.copy(), node.energy))
        return episode

    def retrieve(self, current_state, node_id: int | None = None, top_k: int = 5, min_similarity: float = 0.05):
        scored = []
        for transition in self.transitions:
            if node_id is not None and transition["node_id"] != node_id:
                continue
            similarity = self._state_similarity(current_state, transition["current_state"])
            if similarity < min_similarity:
                continue
            scored.append((similarity, transition))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        top_k = max(1, int(top_k))
        selected = scored[:top_k]

        weights = []
        for similarity, transition in selected:
            confidence = max(0.0, float(transition.get("outcome", {}).get("confidence", 0.5)))
            weights.append(max(similarity, 0.0) * confidence)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            weights = np.ones(len(selected), dtype=float)
            weight_sum = float(len(selected))
        else:
            weights = np.asarray(weights, dtype=float)

        predicted_state = np.zeros_like(np.asarray(selected[0][1]["next_state"], dtype=float), dtype=float)
        predicted_energy = 0.0
        predicted_confidence = 0.0
        mean_similarity = 0.0
        for index, (similarity, transition) in enumerate(selected):
            factor = float(weights[index] / weight_sum)
            predicted_state += factor * np.asarray(transition["next_state"], dtype=float)
            outcome = transition.get("outcome", {})
            predicted_energy += factor * float(outcome.get("energy", 0.0))
            predicted_confidence += factor * float(outcome.get("confidence", 0.5))
            mean_similarity += factor * float(similarity)

        best_similarity, best_transition = selected[0]
        recalled = dict(best_transition)
        recalled["next_state"] = predicted_state
        recalled["outcome"] = {
            "energy": predicted_energy,
            "confidence": max(0.0, min(1.0, predicted_confidence)),
        }
        recalled["similarity"] = max(0.0, min(1.0, mean_similarity))
        recalled["best_similarity"] = best_similarity
        recalled["retrieval_count"] = len(selected)
        return recalled

    def _state_similarity(self, left, right) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))