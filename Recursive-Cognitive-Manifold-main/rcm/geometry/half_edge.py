class HalfEdge:
    __slots__ = ["origin", "twin", "next", "face"]

    def __init__(self, origin_idx: int):
        self.origin = origin_idx
        self.twin = None
        self.next = None
        self.face = None