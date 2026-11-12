"""Belief-only transition-safe controller; no access to privileged maps."""

import numpy as np

from .environment import ACTIONS
from .gp import ExactGP
from .graph import TransitionGraph
from .transition_sets import SafetyTracker


class SafetyExplorer:
    def __init__(self, grid, initial_actions, config):
        self.grid, self.config = grid, config
        self.graph = TransitionGraph.from_grid(grid)
        self.coordinates = np.array(grid.states(), float) * config.grid_spacing
        self.tracker = SafetyTracker(
            self.graph,
            initial_actions,
            self.coordinates,
            config.safety_threshold,
            config.safety_confidence_multiplier,
            config.expander_lipschitz,
            config.certificate_mode,
        )
        self.observations = []
        self.target = None
        self.altitude_prediction = self.transition_prediction = (
            self.reward_prediction
        ) = None

    def update(self, observation):
        if observation.step != len(self.observations):
            raise ValueError("Expected one new observation frame per action")
        self.observations.append(observation)
        samples = [s for o in self.observations for s in o.samples]
        x = np.array([s.state for s in samples], float) * self.config.grid_spacing
        kernel = {
            "length_scale": self.config.gp_length_scale,
            "prior_variance": self.config.gp_prior_variance,
        }
        model = ExactGP(
            self.config.safety_noise_std * self.config.assumed_noise_factor, **kernel
        )
        model.fit(x, [s.altitude for s in samples])
        self.altitude_prediction, self.transition_prediction = (
            model.predict_transitions(self.coordinates, self.graph.transitions)
        )
        self.reward_prediction = (
            ExactGP(self.config.reward_noise_std, **kernel)
            .fit(x, [s.reward for s in samples])
            .predict(self.coordinates)
        )
        return self.tracker.update(
            self.transition_prediction.mean, self.transition_prediction.std
        )

    def safety_action(self, position):
        snapshot = self.tracker.snapshot
        if snapshot is None:
            raise ValueError("Observe before selecting an action")
        current = int(np.ravel_multi_index(position, (self.grid.rows, self.grid.cols)))
        if not snapshot.pessimistic[current]:
            raise ValueError("Current state is not executable")
        eligible = (
            snapshot.expanders
            if snapshot.expanders.any()
            else snapshot.pessimistic_actions
        )
        indices = np.argwhere(eligible)
        if not len(indices):
            raise ValueError("No available certified action")
        scores = snapshot.widths[eligible]
        selected = tuple(int(v) for v in indices[np.argmax(scores)])
        if (
            self.target is None
            or self.target[0] == current
            or not eligible[self.target]
        ):
            self.target = selected
        path = self.graph.path(current, self.target[0], snapshot.pessimistic_actions)
        action_index = path[0] if path else 0
        return ACTIONS[action_index], self.grid.states()[self.target[0]], self.target[1]

    def accuracy_reached(self):
        snapshot = self.tracker.snapshot
        return bool(
            np.max(snapshot.widths[snapshot.pessimistic_actions])
            <= self.config.exploration_accuracy
        )
