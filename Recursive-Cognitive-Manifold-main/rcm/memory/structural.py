from collections import deque


class StructuralMemory:
    def __init__(self, maxlen: int = 64):
        self.events = deque(maxlen=maxlen)

    def record(self, time_step: int, edges, pruned_edges):
        event = {
            "time_step": time_step,
            "edge_strengths": {edge_key: edge.strength for edge_key, edge in edges.items()},
            "pruned_edges": list(pruned_edges),
        }
        self.events.append(event)
        return event