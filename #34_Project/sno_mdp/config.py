"""Validated settings for the primary GP-Safety-Gym benchmark profile."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    rows: int = 20
    cols: int = 20
    map_source: str = "released"
    map_seed: int = 7
    observation_seed: int = 2026
    grid_spacing: float = 0.5
    safety_threshold: float = -0.25
    reward_noise_std: float = 0.001
    safety_noise_std: float = 0.001
    assumed_noise_factor: float = 1.0
    gp_length_scale: float = 2.0
    gp_prior_variance: float = 1.0
    safety_confidence_multiplier: float = 2.0
    certificate_mode: str = "retained"
    reward_confidence_multiplier: float = 3.0
    expander_lipschitz: float = 0.0
    exploration_accuracy: float = 0.005
    discount: float = 0.99
    horizon: int = 120
    value_tolerance: float = 1e-8
    max_iterations: int = 5000

    def __post_init__(self):
        for name in ("rows", "cols", "horizon", "max_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if min(self.rows, self.cols) < 2:
            raise ValueError("Grid dimensions must be at least two")
        for name in ("map_seed", "observation_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if self.map_source not in ("released", "generated"):
            raise ValueError("map_source must be released or generated")
        if self.certificate_mode not in ("retained", "current_lcb"):
            raise ValueError("certificate_mode must be retained or current_lcb")
        if self.map_source == "released" and (self.rows, self.cols) != (20, 20):
            raise ValueError("The pinned released map is exactly 20 x 20")
        for name in (
            "grid_spacing",
            "assumed_noise_factor",
            "gp_length_scale",
            "gp_prior_variance",
            "safety_confidence_multiplier",
            "reward_confidence_multiplier",
            "exploration_accuracy",
            "value_tolerance",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("reward_noise_std", "safety_noise_std", "expander_lipschitz"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if not np.isfinite(self.safety_threshold) or self.safety_threshold > 0:
            raise ValueError(
                "safety_threshold must be finite and nonpositive (stay is safe)"
            )
        if not np.isfinite(self.discount) or not 0 <= self.discount < 1:
            raise ValueError("discount must be in [0, 1)")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError("Unknown configuration fields: " + str(sorted(unknown)))
        return cls(**data)

    def as_dict(self):
        return asdict(self)
