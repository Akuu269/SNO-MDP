"""Source-aligned GP transition certificates, reachability and returnability.

Adapted from SafeMDP GridWorld/SafeMDP_class; see third_party licenses. The
released code recomputes intervals and monotonically retains accepted edges.
This differs from intersecting state-function intervals in the paper's text.
"""

from dataclasses import dataclass

import numpy as np


class ConfidenceConflict(ValueError):
    pass


@dataclass(frozen=True)
class SafeSnapshot:
    lower: np.ndarray
    upper: np.ndarray
    candidates: np.ndarray
    pessimistic: np.ndarray
    optimistic: np.ndarray
    pessimistic_actions: np.ndarray
    optimistic_actions: np.ndarray
    expanders: np.ndarray

    @property
    def widths(self):
        return self.upper - self.lower


class SafetyTracker:
    def __init__(
        self,
        graph,
        initial_actions,
        coordinates,
        threshold,
        multiplier=2.0,
        lipschitz=0.0,
        certificate_mode="retained",
    ):
        self.graph = graph
        if certificate_mode not in ("retained", "current_lcb"):
            raise ValueError("Unknown certificate mode")
        self.certificate_mode = certificate_mode
        self.initial_actions = np.array(initial_actions, bool, copy=True)
        if (
            self.initial_actions.shape != graph.transitions.shape
            or not self.initial_actions[:, 0].any()
        ):
            raise ValueError("Valid initial action certificates required")
        self.initial_nodes = self.initial_actions[:, 0]
        self.threshold, self.multiplier, self.lipschitz = (
            threshold,
            multiplier,
            lipschitz,
        )
        self.distances = np.linalg.norm(
            coordinates[:, None] - coordinates[None, :], axis=2
        )
        self.candidates = self.initial_actions.copy()
        self.candidates[:, 0] = True
        self.snapshot = None

    def update(self, mean, std):
        mean, std = np.asarray(mean, float), np.asarray(std, float)
        if (
            mean.shape != self.graph.transitions.shape
            or std.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std < 0)
        ):
            raise ValueError("Invalid transition posterior")
        lower = mean - self.multiplier * std
        upper = mean + self.multiplier * std
        lower[~self.graph.available] = upper[~self.graph.available] = self.threshold - 1
        lower[:, 0] = upper[:, 0] = (
            self.threshold
        )  # source's always-safe stay convention
        # A certificate supplied before learning cannot be contradicted silently.
        if np.any(self.initial_actions & (upper < self.threshold)):
            raise ConfidenceConflict(
                "Posterior contradicts a supplied initial transition certificate"
            )
        candidates = self.candidates | (
            (lower >= self.threshold) & self.graph.available
        )
        permitted = candidates
        if self.certificate_mode == "current_lcb":
            # Project extension: supplied initial certificates remain known.
            # Historical acceptance is retained only for provenance.
            permitted = self.initial_actions | (
                (lower >= self.threshold) & self.graph.available
            )
        nodes, actions = self.graph.executable(permitted, self.initial_nodes)
        opt_candidates = ((upper >= self.threshold) & self.graph.available) | actions
        opt_nodes, opt_actions = self.graph.executable(
            opt_candidates, self.initial_nodes
        )
        expanders = np.zeros_like(actions)
        for action in range(1, actions.shape[1]):
            outside = ~permitted[:, action]
            if outside.any():
                sources = np.flatnonzero(actions[:, action])
                scores = (
                    upper[sources, action, None]
                    - self.lipschitz * self.distances[np.ix_(sources, outside)]
                )
                expanders[sources, action] = np.any(scores >= self.threshold, axis=1)
        arrays = (
            lower,
            upper,
            candidates,
            nodes,
            opt_nodes,
            actions,
            opt_actions,
            expanders,
        )
        for array in arrays:
            array.setflags(write=False)
        self.candidates = candidates
        self.snapshot = SafeSnapshot(*arrays)
        return self.snapshot
