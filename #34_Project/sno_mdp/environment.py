"""Discrete benchmark dynamics, authorized observations, and privileged scoring.

Safety is attached to directed actions: altitude(current)-altitude(successor).
The upstream action order uses coordinates (x index, y index), not image rows.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Action(str, Enum):
    STAY = "stay"
    UP = "up"
    RIGHT = "right"
    DOWN = "down"
    LEFT = "left"


ACTIONS = tuple(Action)
DELTAS = ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.rows, self.cols)):
            raise ValueError("Grid dimensions must be positive integers")

    def validate(self, state):
        if (
            len(state) != 2
            or any(type(v) not in (int, np.int64, np.int32) for v in state)
            or not 0 <= state[0] < self.rows
            or not 0 <= state[1] < self.cols
        ):
            raise ValueError("Invalid grid state: " + str(state))
        return tuple(int(v) for v in state)

    def states(self):
        return tuple((x, y) for x in range(self.rows) for y in range(self.cols))

    def next_state(self, state, action):
        state = self.validate(state)
        try:
            index = ACTIONS.index(Action(action))
        except (ValueError, TypeError) as exc:
            raise ValueError("Unknown action") from exc
        dx, dy = DELTAS[index]
        target = state[0] + dx, state[1] + dy
        return (
            target
            if 0 <= target[0] < self.rows and 0 <= target[1] < self.cols
            else state
        )

    def action_mask(self):
        return np.array(
            [
                [i == 0 or self.next_state(s, a) != s for i, a in enumerate(ACTIONS)]
                for s in self.states()
            ],
            dtype=bool,
        )

    def footprint(self, state):
        return tuple(sorted({self.next_state(state, a) for a in ACTIONS}))

    def neighbors(self, state):
        return tuple(s for s in self.footprint(state) if s != tuple(state))


@dataclass(frozen=True)
class WorldMap:
    grid: Grid
    reward: np.ndarray
    altitude: np.ndarray
    start: tuple
    initial_actions: np.ndarray
    threshold: float = -0.25
    map_seed: int = 0
    spacing: float = 0.5
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        shape = (self.grid.rows, self.grid.cols)
        r, z = (
            np.array(self.reward, float, copy=True),
            np.array(self.altitude, float, copy=True),
        )
        initial = np.array(self.initial_actions, bool, copy=True)
        if (
            r.shape != shape
            or z.shape != shape
            or not np.isfinite(r).all()
            or not np.isfinite(z).all()
        ):
            raise ValueError(
                "Reward and altitude must be finite arrays matching the grid"
            )
        if np.any(r < 0):
            raise ValueError(
                "Benchmark rewards must be nonnegative; released zero values are valid"
            )
        if (
            not np.isfinite(self.threshold)
            or self.threshold > 0
            or not np.isfinite(self.spacing)
            or self.spacing <= 0
        ):
            raise ValueError("Invalid safety threshold or grid spacing")
        if initial.shape != (z.size, 5):
            raise ValueError("Initial certificates must have shape (states, 5 actions)")
        start = self.grid.validate(self.start)
        available = self.grid.action_mask()
        successor = np.array(
            [
                [
                    np.ravel_multi_index(self.grid.next_state(s, a), shape)
                    for a in ACTIONS
                ]
                for s in self.grid.states()
            ]
        )
        margins = z.ravel()[:, None] - z.ravel()[successor]
        if np.any(initial & ~available) or np.any(initial & (margins < self.threshold)):
            raise ValueError(
                "Initial action certificates include invalid/unsafe actions"
            )
        if not initial[np.ravel_multi_index(start, shape), 0]:
            raise ValueError("Start must belong to the certified initial region")
        for a in (r, z, initial):
            a.setflags(write=False)
        object.__setattr__(self, "reward", r)
        object.__setattr__(self, "altitude", z)
        object.__setattr__(self, "initial_actions", initial)
        object.__setattr__(self, "start", start)

    @property
    def initial_safe(self):
        return tuple(
            self.grid.states()[i] for i in np.flatnonzero(self.initial_actions[:, 0])
        )

    def true_action_safety(self):
        values = np.array(
            [
                [
                    self.altitude[s] - self.altitude[self.grid.next_state(s, a)]
                    for a in ACTIONS
                ]
                for s in self.grid.states()
            ]
        )
        return (values >= self.threshold) & self.grid.action_mask()


@dataclass(frozen=True)
class SensorSample:
    state: tuple
    reward: float
    altitude: float


@dataclass(frozen=True)
class Observation:
    step: int
    position: tuple
    samples: tuple


class RoverSession:
    """Learner API. Dynamics reject boundaries, never use latent safety to steer."""

    def __init__(
        self,
        world,
        observation_seed=2026,
        reward_noise_std=0.001,
        safety_noise_std=0.001,
    ):
        if type(observation_seed) is not int or observation_seed < 0:
            raise ValueError("Observation seed must be a nonnegative integer")
        if any(
            not np.isfinite(v) or v < 0 for v in (reward_noise_std, safety_noise_std)
        ):
            raise ValueError("Noise standard deviations must be finite and nonnegative")
        self._world = world
        self.grid, self.threshold = world.grid, world.threshold
        self.initial_actions = world.initial_actions.copy()
        self.initial_safe, self.spacing = world.initial_safe, world.spacing
        self.position, self.step_count = world.start, 0
        self._seed = observation_seed
        self._noise = reward_noise_std, safety_noise_std
        self._counts = {}
        self.initial_observation = self._observe()

    def _observe(self):
        samples = []
        for state in self.grid.footprint(self.position):
            occurrence = self._counts.get(state, 0)
            noise = [
                np.random.default_rng(
                    np.random.SeedSequence(
                        [self._seed, self._world.map_seed, *state, occurrence, sensor]
                    )
                ).normal(0, sigma)
                for sensor, sigma in enumerate(self._noise)
            ]
            self._counts[state] = occurrence + 1
            samples.append(
                SensorSample(
                    state,
                    float(self._world.reward[state] + noise[0]),
                    float(self._world.altitude[state] + noise[1]),
                )
            )
        return Observation(self.step_count, self.position, tuple(samples))

    def step(self, action):
        try:
            action = Action(action)
        except ValueError as exc:
            raise ValueError("Unknown action") from exc
        if (
            action != Action.STAY
            and self.grid.next_state(self.position, action) == self.position
        ):
            raise ValueError("Out-of-grid action is unavailable")
        self.position = self.grid.next_state(self.position, action)
        self.step_count += 1
        return self._observe()


class Evaluator:
    def __init__(self, world):
        self.world = world

    def evaluate(self, position, previous=None):
        previous = position if previous is None else previous
        margin = float(self.world.altitude[previous] - self.world.altitude[position])
        return {
            "true_reward": float(self.world.reward[position]),
            "true_altitude": float(self.world.altitude[position]),
            "true_transition_safety": margin,
            "unsafe": margin < self.world.threshold,
            "violation_severity": max(0.0, self.world.threshold - margin),
        }


def world_from_config(config):
    from .gp_safety_gym import load_world

    return load_world(config)
