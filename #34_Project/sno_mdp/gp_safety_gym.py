"""Headless adapter for the pinned authors' synthetic benchmark.

Map generation and seed-certificate rule are adapted from the MIT-licensed
SNO-MDP / SafeMDP sources identified in third_party/upstream.json. Native
Engine_GP rendering (legacy Safety-Gym/MuJoCo) is not imported by this adapter.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .environment import ACTIONS, Grid, WorldMap


ROOT = Path(__file__).resolve().parent.parent


def initial_seed(grid, altitude, center, spacing, threshold):
    """Port of compute_S_hat0: safe star with both coordinate directions present."""
    center = grid.validate(center)
    mask = np.zeros((grid.rows * grid.cols, 5), bool)
    index = np.ravel_multi_index(center, (grid.rows, grid.cols))
    mask[index, 0] = True
    directions = set()
    for action_index, action in enumerate(ACTIONS[1:], 1):
        target = grid.next_state(center, action)
        if (
            target != center
            and -abs(float(altitude[center] - altitude[target])) / spacing >= threshold
        ):
            target_index = np.ravel_multi_index(target, (grid.rows, grid.cols))
            reverse = ((action_index + 1) % 4) + 1
            mask[index, action_index] = True
            mask[target_index, 0] = mask[target_index, reverse] = True
            directions.add(action_index % 2)
    if directions != {0, 1}:
        raise ValueError("No valid two-direction initial safe star at this location")
    return mask


@lru_cache(maxsize=16)
def _sample_fields(rows, cols, spacing, length_scale, variance, seed):
    x = np.array(Grid(rows, cols).states(), float) * spacing
    delta = x[:, None] - x[None, :]
    covariance = variance * np.exp(-np.sum(delta**2, axis=2) / (2 * length_scale**2))
    covariance.flat[:: len(x) + 1] += 1e-10
    chol = np.linalg.cholesky(covariance)
    rng = np.random.default_rng(seed)
    altitude = (chol @ rng.standard_normal(len(x))).reshape(rows, cols)
    reward = (chol @ rng.standard_normal(len(x))).reshape(rows, cols)
    reward -= (
        reward.min()
    )  # matches simple_make_rand_settings.py; do not normalize to [0,1]
    altitude.setflags(write=False)
    reward.setflags(write=False)
    return altitude, reward


def load_world(config):
    grid = Grid(config.rows, config.cols)
    upstream = json.loads((ROOT / "third_party/upstream.json").read_text())
    if config.map_source == "released":
        path = ROOT / "data/gp_safety_gym/released_map.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != upstream["map_sha256"]:
            raise ValueError(
                "Released benchmark map checksum does not match its pinned source"
            )
        with np.load(path, allow_pickle=False) as saved:
            altitude = saved["safety"].reshape(config.rows, config.cols)
            reward = saved["reward"].reshape(config.rows, config.cols)
            start = tuple(
                int(x)
                for x in np.unravel_index(int(saved["start_pos"]), altitude.shape)
            )
        mask = initial_seed(
            grid, altitude, start, config.grid_spacing, config.safety_threshold
        )
        map_seed = 0  # a released fixture has no new map randomness from --map-seed
        details = {
            "map_file_sha256": upstream["map_sha256"],
            "initial_seed": "star centered on released start_pos",
        }
    else:
        altitude, reward = _sample_fields(
            config.rows,
            config.cols,
            config.grid_spacing,
            config.gp_length_scale,
            config.gp_prior_variance,
            config.map_seed,
        )
        # Uniform seeded search among possible centers; reject only invalid seeds,
        # never maps with poor algorithm returns or an inconvenient reward field.
        rng = np.random.default_rng(np.random.SeedSequence([config.map_seed, 471]))
        for index in rng.permutation(config.rows * config.cols):
            start = grid.states()[int(index)]
            try:
                mask = initial_seed(
                    grid, altitude, start, config.grid_spacing, config.safety_threshold
                )
                break
            except ValueError:
                continue
        else:
            raise ValueError(
                "Generated map has no valid initial star; retain this invalid-map outcome"
            )
        map_seed = config.map_seed
        details = {
            "initial_seed": "seeded uniform search for a valid two-direction star"
        }
    provenance = {
        "environment": "GP-Safety-Gym synthetic benchmark, headless port",
        "source_profile": "released transition-altitude-difference implementation",
        "native_engine_executed": False,
        "map_source": config.map_source,
        "upstream_commit": upstream["snomdp"]["commit"],
        "safemdp_commit": upstream["safemdp"]["commit"],
        **details,
    }
    return WorldMap(
        grid,
        reward,
        altitude,
        start,
        mask,
        config.safety_threshold,
        map_seed,
        config.grid_spacing,
        provenance,
    )
