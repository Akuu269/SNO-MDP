"""Matched-map/seed experiments with explicit failures and map-level summaries."""

import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from .artifacts import read_json, save_json
from .config import Config
from .experiment import LEARNING_METHODS, run_experiment
from .experiment import run_reference


def _unique(values, name):
    if not isinstance(values, list) or not values or len(set(values)) != len(values):
        raise ValueError("%s must be a nonempty list without duplicates" % name)


def load_suite(path):
    spec = json.loads(path.read_text())
    required = {
        "base_config",
        "grids",
        "map_seeds",
        "observation_seeds",
        "noise_factors",
        "horizon",
    }
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("Suite must contain exactly: %s" % sorted(required))
    if not isinstance(spec["base_config"], str):
        raise ValueError("base_config must be a JSON configuration path")
    if type(spec["horizon"]) is not int or spec["horizon"] < 1:
        raise ValueError("Suite horizon must be a positive integer")
    if not isinstance(spec["grids"], list) or not spec["grids"]:
        raise ValueError("Suite grids must be nonempty")
    for grid in spec["grids"]:
        if (
            not isinstance(grid, list)
            or len(grid) != 2
            or any(type(x) is not int or x < 2 for x in grid)
        ):
            raise ValueError("Each grid must contain two integers >= 2")
    if len({tuple(g) for g in spec["grids"]}) != len(spec["grids"]):
        raise ValueError("Duplicate grid sizes")
    for name in ("map_seeds", "observation_seeds"):
        _unique(spec[name], name)
        if any(type(v) is not int or v < 0 for v in spec[name]):
            raise ValueError("Seeds must be nonnegative integers")
    _unique(spec["noise_factors"], "noise_factors")
    if any(
        type(v) not in (int, float) or not np.isfinite(v) or v <= 0
        for v in spec["noise_factors"]
    ):
        raise ValueError("Noise factors must be finite positive numbers")
    config = Config.load(path.parent / spec["base_config"])
    return spec, config


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty results table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["grid"], row["method"], row["noise_factor"])].append(row)
    summaries = []
    for (grid, method, factor), group in groups.items():
        maps = defaultdict(list)
        for row in group:
            maps[row["map_seed"]].append(row)

        def map_means(key):
            return [
                float(np.mean([r[key] for r in replicates]))
                for replicates in maps.values()
            ]

        def mean(key):
            return float(np.mean(map_means(key)))

        returns = map_means("discounted_return")
        summaries.append(
            {
                "grid": grid,
                "method": method,
                "noise_factor": factor,
                "runs": len(group),
                "maps": len(maps),
                "discounted_return_mean": float(np.mean(returns)),
                "discounted_return_map_sd": float(np.std(returns, ddof=1))
                if len(returns) > 1
                else None,
                "cumulative_reward_mean": mean("cumulative_reward"),
                "reward_ratio_mean": mean("discounted_return_ratio_to_reference"),
                "failure_rate": mean("failed"),
                "unsafe_run_rate": mean("any_unsafe"),
                "mean_unsafe_actions": mean("unsafe_actions"),
                "false_certificate_run_rate": mean("any_false_certificate"),
                "es2_trigger_rate": mean("es2_triggered"),
                "exploration_budget_exhaustion_rate": mean(
                    "budget_exhausted_during_exploration"
                ),
                "exploration_actions_mean": mean("exploration_actions"),
                "reward_actions_mean": mean("reward_actions"),
                "final_certified_cells_mean": mean("final_certified_cells"),
            }
        )
    return summaries


def paired_differences(rows):
    lookup = {
        (
            r["grid"],
            r["map_seed"],
            r["observation_seed"],
            r["noise_factor"],
            r["method"],
        ): r
        for r in rows
    }
    pairs = []
    for row in rows:
        if row["method"] != "sno_es2":
            continue
        key = (
            row["grid"],
            row["map_seed"],
            row["observation_seed"],
            row["noise_factor"],
            "sno_basic",
        )
        other = lookup[key]
        pairs.append(
            {
                "grid": row["grid"],
                "map_seed": row["map_seed"],
                "observation_seed": row["observation_seed"],
                "noise_factor": row["noise_factor"],
                "es2_minus_basic_return": row["discounted_return"]
                - other["discounted_return"],
                "es2_minus_basic_exploration_actions": row["exploration_actions"]
                - other["exploration_actions"],
                "es2_failed": row["failed"],
                "basic_failed": other["failed"],
                "es2_triggered": row["es2_triggered"],
            }
        )
    return pairs


def noise_slug(factor):
    return str(float(factor)).replace(".", "p")


def run_benchmark(suite_path: Path, output: Path):
    spec, base = load_suite(suite_path)
    if base.map_source == "released" and (
        len(spec["map_seeds"]) != 1 or spec["grids"] != [[20, 20]]
    ):
        raise ValueError(
            "The released fixture is one 20x20 map; multiple seeds do not create independent released maps"
        )
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "suite.json", spec)
    save_json(output / "base_config.json", base.as_dict())
    rows, curves = [], []
    for height, width in spec["grids"]:
        grid = "%dx%d" % (height, width)
        for map_seed in spec["map_seeds"]:
            for observation_seed in spec["observation_seeds"]:
                config = replace(
                    base,
                    rows=height,
                    cols=width,
                    map_seed=map_seed,
                    observation_seed=observation_seed,
                    horizon=spec["horizon"],
                )
                parent = (
                    Path("runs")
                    / grid
                    / ("map_%d" % map_seed)
                    / ("obs_%d" % observation_seed)
                )
                reference_dir = parent / "oracle"
                reference = run_reference(config, output / reference_dir)
                conditions = [("oracle", None, reference, reference_dir)]
                for factor in spec["noise_factors"]:
                    for method in LEARNING_METHODS:
                        relative = parent / method / ("noise_" + noise_slug(factor))
                        summary = run_experiment(
                            replace(config, assumed_noise_factor=factor),
                            output / relative,
                            method,
                        )
                        conditions.append((method, factor, summary, relative))
                for method, factor, summary, relative in conditions:
                    oracle = method == "oracle"
                    failed = summary["stop_reason"] != "horizon_reached"
                    phases = summary.get("phase_actions", {})
                    row = {
                        "grid": grid,
                        "map_seed": map_seed,
                        "observation_seed": observation_seed,
                        "method": method,
                        "noise_factor": factor,
                        "actual_safety_noise_std": config.safety_noise_std,
                        "assumed_safety_noise_std": None
                        if oracle
                        else summary["assumed_safety_noise_std"],
                        "scheduled_horizon": config.horizon,
                        "physical_actions": summary["physical_actions"],
                        "failed": failed,
                        "stop_reason": summary["stop_reason"],
                        "diagnostic": summary.get("diagnostic"),
                        "transition_step": 0 if oracle else summary["transition_step"],
                        "transition_reason": "fully_informed"
                        if oracle
                        else summary["transition_reason"],
                        "budget_exhausted_during_exploration": summary.get(
                            "budget_exhausted_during_exploration", False
                        ),
                        "es2_triggered": summary.get("es2_triggered", False),
                        "exploration_actions": phases.get("safety_exploration", 0),
                        "reward_actions": config.horizon
                        if oracle
                        else phases.get("reward_optimization", 0),
                        "final_certified_cells": summary["final_certified_cells"],
                        "unsafe_actions": summary["unsafe_actions"],
                        "any_unsafe": summary["unsafe_actions"] > 0,
                        "maximum_violation_severity": summary[
                            "maximum_violation_severity"
                        ],
                        "maximum_false_certified_actions": summary[
                            "maximum_false_certified_actions"
                        ],
                        "any_false_certificate": summary[
                            "maximum_false_certified_actions"
                        ]
                        > 0,
                        "cumulative_reward": summary["cumulative_reward"],
                        "discounted_return": summary["discounted_return"],
                        "reference_discounted_return": reference["discounted_return"],
                        "discounted_return_ratio_to_reference": summary[
                            "discounted_return"
                        ]
                        / reference["discounted_return"]
                        if reference["discounted_return"]
                        else 0.0,
                        "replay": (relative / "index.html").as_posix(),
                    }
                    rows.append(row)
                    trajectory = read_json(output / relative / "trajectory.json.gz")
                    # Explicit zero reward credit after a failed prefix: cumulative
                    # reward/return stays flat, without fabricating any executed actions.
                    returns = [f["discounted_return"] for f in trajectory]
                    returns.extend([returns[-1]] * (config.horizon + 1 - len(returns)))
                    curves.append(
                        {
                            "grid": grid,
                            "map_seed": map_seed,
                            "observation_seed": observation_seed,
                            "method": method,
                            "noise_factor": factor,
                            "discounted_return": returns,
                        }
                    )
                print(
                    "Finished %s map=%d observation_seed=%d"
                    % (grid, map_seed, observation_seed),
                    flush=True,
                )
    aggregates = aggregate_rows(rows)
    pairs = paired_differences(rows)
    write_csv(output / "runs.csv", rows)
    write_csv(output / "aggregates.csv", aggregates)
    write_csv(output / "paired_es2_vs_basic.csv", pairs)
    summary = {
        "run_count": len(rows),
        "map_count": len(spec["grids"]) * len(spec["map_seeds"]),
        "failed_runs": sum(r["failed"] for r in rows),
        "runs_with_unsafe_actions": sum(r["any_unsafe"] for r in rows),
        "runs_with_false_certificates": sum(r["any_false_certificate"] for r in rows),
        "es2_triggered_runs": sum(r["es2_triggered"] for r in rows),
        "development_only": True,
        "budget_exhausted_exploration_runs": sum(
            r["budget_exhausted_during_exploration"] for r in rows
        ),
        "methods": ["oracle", *LEARNING_METHODS],
        "suite": spec,
        "aggregates": aggregates,
        "aggregation": "average observation replicates within each map, then average maps; SD across map means, no significance claim",
        "failure_accounting": "all failures retained; executed return prefix, zero reward credit for unexecuted scheduled actions",
        "reference": "infinite-horizon discounted true-map policy rolled out for the same finite horizon; ratios are descriptive, not universal bounds",
        "scope": "GP-Safety-Gym headless source-audited port; current posterior bands and retained accepted transitions; native legacy renderer not executed",
        "baseline": "SafeMDP transition uncertainty sampling throughout the full action horizon; no reward objective",
        "upstream": json.loads(
            (
                Path(__file__).resolve().parent.parent / "third_party/upstream.json"
            ).read_text()
        ),
    }
    save_json(output / "summary.json", summary)
    save_json(output / "runs.json", rows)
    save_json(output / "curves.json", curves)
    from .benchmark_visualization import write_benchmark_visualization

    write_benchmark_visualization(output, summary, rows, curves)
    return summary
