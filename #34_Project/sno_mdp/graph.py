"""Directed action graph with unavailable boundary actions and safe paths."""

import numpy as np

from .environment import ACTIONS


class TransitionGraph:
    def __init__(self, transitions, available=None):
        t = np.asarray(transitions)
        if t.ndim != 2 or not all(t.shape) or not np.issubdtype(t.dtype, np.integer):
            raise ValueError("Transitions must be a nonempty integer matrix")
        if np.any(t < 0) or np.any(t >= len(t)):
            raise ValueError("Invalid successor index")
        self.transitions = t.copy()
        self.size = len(t)
        self.available = (
            np.ones(t.shape, bool) if available is None else np.array(available, bool)
        )
        if self.available.shape != t.shape:
            raise ValueError("Availability shape differs from transitions")
        self.transitions.setflags(write=False)
        self.available.setflags(write=False)

    @classmethod
    def from_grid(cls, grid):
        return cls(
            [
                [
                    np.ravel_multi_index(grid.next_state(s, a), (grid.rows, grid.cols))
                    for a in ACTIONS
                ]
                for s in grid.states()
            ],
            grid.action_mask(),
        )

    def mask(self, values):
        mask = np.asarray(values, dtype=bool)
        if mask.shape != (self.size,):
            raise ValueError("State mask has wrong shape")
        return mask

    def executable(self, permitted, initial):
        permitted = np.asarray(permitted, bool) & self.available
        if permitted.shape != self.transitions.shape:
            raise ValueError("Action mask has wrong shape")
        initial = self.mask(initial)
        reach = initial.copy()
        frontier = list(np.flatnonzero(initial))
        for s in frontier:
            for target in self.transitions[s, permitted[s]]:
                if not reach[target]:
                    reach[target] = True
                    frontier.append(int(target))
        incoming = [[] for _ in range(self.size)]
        for s, a in zip(*np.where(permitted)):
            incoming[self.transitions[s, a]].append(int(s))
        ret = initial.copy()
        frontier = list(np.flatnonzero(initial))
        for s in frontier:
            for source in incoming[s]:
                if not ret[source]:
                    ret[source] = True
                    frontier.append(source)
        nodes = reach & ret
        actions = permitted & nodes[:, None] & nodes[self.transitions]
        return nodes, actions

    def path(self, start, target, allowed_actions):
        allowed = np.asarray(allowed_actions, bool) & self.available
        if not 0 <= start < self.size or not 0 <= target < self.size:
            raise ValueError("Path endpoint outside graph")
        parents = {start: None}
        frontier = [start]
        for state in frontier:
            if state == target:
                break
            for action in np.flatnonzero(allowed[state]):
                successor = int(self.transitions[state, action])
                if successor not in parents:
                    parents[successor] = state, int(action)
                    frontier.append(successor)
        if target not in parents:
            raise ValueError("No certified path to target")
        result = []
        state = target
        while state != start:
            state, action = parents[state]
            result.append(action)
        return tuple(reversed(result))
