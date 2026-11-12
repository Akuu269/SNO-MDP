"""Controlled calibration, stopping studies, and frozen held-out evaluation.

Cases are physical policy reruns. All planned setup errors and failed prefixes
remain in the inventory. Frozen evaluation refuses changed settings/source.
"""

import csv
import hashlib
import itertools
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from .artifacts import read_json, save_json
from .config import Config
from .experiment import LEARNING_METHODS, run_experiment


def source_hashes():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows):
    if not rows:
        raise ValueError("No planned cases")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_research(path):
    path = Path(path)
    spec = read_json(path)
    required = {
        "stage",
        "base_config",
        "map_seeds",
        "observation_seeds",
        "noise_factors",
        "horizons",
        "profiles",
    }
    if set(spec) != required or spec["stage"] not in (
        "calibration",
        "stopping",
        "final",
    ):
        raise ValueError("Invalid research suite fields or stage")
    base = Config.load(path.parent / spec["base_config"])
    if base.map_source != "generated":
        raise ValueError("Research matrices require independent generated maps")
    for key in ("map_seeds", "observation_seeds", "horizons", "noise_factors"):
        values = spec[key]
        if (
            not isinstance(values, list)
            or not values
            or len(set(values)) != len(values)
        ):
            raise ValueError("Invalid or duplicate " + key)
        if key == "noise_factors":
            valid = all(
                type(v) in (int, float) and np.isfinite(v) and v > 0 for v in values
            )
        else:
            valid = all(
                type(v) is int and v >= (1 if key == "horizons" else 0) for v in values
            )
        if not valid:
            raise ValueError("Invalid " + key)
    names = set()
    for profile in spec["profiles"]:
        if set(profile) != {"name", "certificate_mode", "multiplier", "methods"}:
            raise ValueError("Invalid profile")
        name = profile["name"]
        if (
            not isinstance(name, str)
            or not name.replace("_", "").isalnum()
            or name in names
            or not profile["methods"]
            or len(set(profile["methods"])) != len(profile["methods"])
            or not set(profile["methods"]) <= set(LEARNING_METHODS)
        ):
            raise ValueError("Invalid profile name or methods")
        names.add(name)
        replace(
            base,
            certificate_mode=profile["certificate_mode"],
            safety_confidence_multiplier=profile["multiplier"],
        )
    if not names:
        raise ValueError("No profiles")
    return spec, base


def plan_cases(spec):
    cases = []
    for m, o, h in itertools.product(
        spec["map_seeds"], spec["observation_seeds"], spec["horizons"]
    ):
        cases.append(
            {
                "map_seed": m,
                "observation_seed": o,
                "horizon": h,
                "noise_factor": None,
                "profile": "oracle",
                "method": "oracle",
                "certificate_mode": "retained",
                "multiplier": 2.0,
            }
        )
        for factor, profile in itertools.product(
            spec["noise_factors"], spec["profiles"]
        ):
            for method in profile["methods"]:
                cases.append(
                    {
                        "map_seed": m,
                        "observation_seed": o,
                        "horizon": h,
                        "noise_factor": factor,
                        "profile": profile["name"],
                        "method": method,
                        "certificate_mode": profile["certificate_mode"],
                        "multiplier": profile["multiplier"],
                    }
                )
    return cases


def freeze_research(suite, output):
    spec, base = load_research(suite)
    if spec["stage"] != "final":
        raise ValueError("Only the final suite is frozen")
    output = Path(output)
    if output.exists():
        raise ValueError("Protocol already exists; do not overwrite a frozen protocol")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        output,
        {
            "spec": spec,
            "base_config": base.as_dict(),
            "source_sha256": source_hashes(),
            "planned_cases": plan_cases(spec),
            "rules": {
                "aggregation": "Observation means within each map, then map means; SD across maps.",
                "failures": "All planned failures retained; zero extra reward after a failed prefix.",
                "selection": "Settings selected on calibration only; no final-outcome tuning.",
                "oracle": "Infinite-horizon true-map policy, finite equal rollout; ratio descriptive.",
                "scope": "Headless source-audited port; current_lcb is a project extension.",
            },
        },
    )
    return read_json(output)


def case_path(case):
    factor = (
        "reference"
        if case["noise_factor"] is None
        else "noise_" + str(case["noise_factor"]).replace(".", "p")
    )
    return (
        Path("runs")
        / ("map_%d" % case["map_seed"])
        / ("obs_%d" % case["observation_seed"])
        / ("h_%d" % case["horizon"])
        / case["profile"]
        / case["method"]
        / factor
    )


def aggregate_research(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[
            row["profile"], row["method"], row["noise_factor"], row["horizon"]
        ].append(row)
    aggregates = []
    metrics = (
        "discounted_return",
        "reward_ratio",
        "unsafe_actions",
        "failed",
        "max_false_certificates",
        "exploration_actions",
        "es2_triggered",
        "inference_seconds",
        "decision_seconds",
        "exploration_exhausted",
    )
    for (profile, method, noise, horizon), members in groups.items():
        maps = defaultdict(list)
        for row in members:
            maps[row["map_seed"]].append(row)
        a = {
            "profile": profile,
            "method": method,
            "noise_factor": noise,
            "horizon": horizon,
            "runs": len(members),
            "maps": len(maps),
        }
        for metric in metrics:
            values = [
                np.mean([r[metric] for r in repetitions if r[metric] is not None])
                for repetitions in maps.values()
                if any(r[metric] is not None for r in repetitions)
            ]
            a[metric + "_mean"] = float(np.mean(values)) if values else None
            a[metric + "_map_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        a["unsafe_run_fraction"] = float(
            np.mean([r["unsafe_actions"] > 0 for r in members])
        )
        a["region_lost_runs"] = sum(
            r["stop_reason"] == "certificate_region_lost" for r in members
        )
        a["setup_failure_runs"] = sum(
            r["stop_reason"] == "setup_failure" for r in members
        )
        aggregates.append(a)
    return aggregates


def select_calibration(study):
    if read_json(Path(study) / "summary.json")["stage"] != "calibration":
        raise ValueError("Selection requires development calibration data")
    rows = read_json(Path(study) / "runs.json")
    groups = defaultdict(list)
    for row in rows:
        if row["method"] == "sno_es2" and row["certificate_mode"] == "current_lcb":
            groups[row["multiplier"]].append(row)
    if not groups:
        raise ValueError("No current_lcb calibration candidates")
    scores = []
    for multiplier, members in sorted(groups.items()):
        scores.append(
            {
                "multiplier": multiplier,
                "failure_rate": float(np.mean([r["failed"] for r in members])),
                "unsafe_run_rate": float(
                    np.mean([r["unsafe_actions"] > 0 for r in members])
                ),
                "mean_reward_ratio": float(
                    np.mean([r["reward_ratio"] or 0.0 for r in members])
                ),
            }
        )
    chosen = min(
        scores,
        key=lambda r: (
            r["failure_rate"],
            r["unsafe_run_rate"],
            -r["mean_reward_ratio"],
            r["multiplier"],
        ),
    )
    result = {
        "selected_multiplier": chosen["multiplier"],
        "candidates": scores,
        "rule": "Predeclared lexicographic order: failure rate, unsafe-run rate, negative mean reference ratio, multiplier. Development data only.",
        "calibration_runs_sha256": sha(Path(study) / "runs.json"),
    }
    save_json(Path(study) / "selection.json", result)
    return result


def run_research(suite, output, protocol=None):
    spec, base = load_research(suite)
    output = Path(output)
    cases = plan_cases(spec)
    frozen = None
    if spec["stage"] == "final":
        if protocol is None:
            raise ValueError("Final evaluation requires --protocol from freeze")
        frozen = read_json(protocol)
        if (
            frozen["spec"] != spec
            or frozen["base_config"] != base.as_dict()
            or frozen["source_sha256"] != source_hashes()
            or frozen["planned_cases"] != cases
        ):
            raise ValueError("Frozen protocol/source/config mismatch")
    elif protocol is not None:
        raise ValueError("Protocol is only for final evaluation")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "runs.json").exists():
        raise ValueError("Study already completed; use a new output directory")
    if spec["stage"] == "final":
        save_json(output / "protocol.json", frozen)
    save_json(output / "suite.json", spec)
    save_json(output / "base_config.json", base.as_dict())
    save_json(output / "planned_cases.json", cases)
    save_json(output / "source.json", source_hashes())
    rows, curves, references = [], [], {}
    for index, case in enumerate(cases):
        relative = case_path(case)
        config = replace(
            base,
            map_seed=case["map_seed"],
            observation_seed=case["observation_seed"],
            horizon=case["horizon"],
            assumed_noise_factor=case["noise_factor"] or 1.0,
            certificate_mode=case["certificate_mode"],
            safety_confidence_multiplier=case["multiplier"],
        )
        directory = output / relative
        try:
            summary = run_experiment(
                config, directory, case["method"], write_replay=False
            )
            frames = read_json(directory / "trajectory.json.gz")
            with (directory / "timing.csv").open() as handle:
                times = list(csv.DictReader(handle))
            inference = float(np.mean([float(t["inference_seconds"]) for t in times]))
            decision = float(np.mean([float(t["decision_seconds"]) for t in times]))
            returns = [f["discounted_return"] for f in frames]
            regions = [f["certified_cells"] for f in frames]
            false = [f["false_certified_actions"] for f in frames]
        except (ValueError, RuntimeError) as exc:
            directory.mkdir(parents=True, exist_ok=True)
            summary = {
                "physical_actions": 0,
                "discounted_return": 0.0,
                "unsafe_actions": 0,
                "stop_reason": "setup_failure",
                "diagnostic": str(exc),
                "maximum_false_certified_actions": 0,
                "phase_actions": {},
                "transition_step": None,
                "transition_reason": None,
                "es2_triggered": False,
                "budget_exhausted_during_exploration": False,
            }
            save_json(
                directory / "setup_failure.json",
                {"config": config.as_dict(), "error": str(exc)},
            )
            inference = decision = 0.0
            returns, regions, false = [0.0], [0], [0]
        key = case["map_seed"], case["observation_seed"], case["horizon"]
        if case["method"] == "oracle":
            references[key] = (
                summary["discounted_return"]
                if summary["stop_reason"] != "setup_failure"
                else None
            )
        reference = references[key]
        row = {
            **case,
            "physical_actions": summary["physical_actions"],
            "discounted_return": summary["discounted_return"],
            "reward_ratio": summary["discounted_return"] / reference
            if reference
            else None,
            "unsafe_actions": summary["unsafe_actions"],
            "max_false_certificates": summary["maximum_false_certified_actions"],
            "failed": summary["stop_reason"] != "horizon_reached",
            "stop_reason": summary["stop_reason"],
            "diagnostic": summary["diagnostic"],
            "transition_step": summary["transition_step"],
            "transition_reason": summary["transition_reason"],
            "exploration_actions": summary["phase_actions"].get(
                "safety_exploration", 0
            ),
            "es2_triggered": summary["es2_triggered"],
            "exploration_exhausted": summary["budget_exhausted_during_exploration"],
            "inference_seconds": inference,
            "decision_seconds": decision,
            "directory": relative.as_posix(),
        }
        rows.append(row)
        pad = case["horizon"] + 1 - len(returns)
        curves.append(
            {
                **case,
                "return": returns + [returns[-1]] * pad,
                "cells": regions + [regions[-1]] * pad,
                "false_certificates": false + [false[-1]] * pad,
            }
        )
        # Incremental inventory survives an interrupted batch without claiming completion.
        save_json(
            output / "progress.json",
            {"completed": len(rows), "planned": len(cases), "latest": row},
        )
        print(
            "%s %d/%d | %s %s c=%g noise=%s map=%d obs=%d h=%d | %s return=%.3f unsafe=%d"
            % (
                spec["stage"],
                index + 1,
                len(cases),
                case["profile"],
                case["method"],
                case["multiplier"],
                case["noise_factor"],
                case["map_seed"],
                case["observation_seed"],
                case["horizon"],
                row["stop_reason"],
                row["discounted_return"],
                row["unsafe_actions"],
            ),
            flush=True,
        )
    if source_hashes() != read_json(output / "source.json"):
        raise ValueError(
            "Source changed during the study; do not report mixed revisions"
        )
    aggregates = aggregate_research(rows)
    save_json(output / "runs.json", rows)
    csv_write(output / "runs.csv", rows)
    csv_write(output / "aggregates.csv", aggregates)
    save_json(output / "curves.json.gz", curves, compact=True)
    summary = {
        "stage": spec["stage"],
        "planned_runs": len(cases),
        "completed_runs": len(rows),
        "maps": len(spec["map_seeds"]),
        "failed_runs": sum(r["failed"] for r in rows),
        "unsafe_actions": sum(r["unsafe_actions"] for r in rows),
        "aggregates": aggregates,
    }
    save_json(output / "summary.json", summary)
    from .research_report import export_research

    export_research(output)
    return summary


def verify_research(study):
    study = Path(study)
    spec, base = (
        read_json(study / "suite.json"),
        Config(**read_json(study / "base_config.json")),
    )
    rows = read_json(study / "runs.json")
    planned = plan_cases(spec)
    if len(rows) != len(planned) or read_json(study / "planned_cases.json") != planned:
        raise ValueError("Incomplete inventory")
    from .environment import ACTIONS, Grid
    from .graph import TransitionGraph
    import numpy.testing as npt

    count = unsafe = 0
    map_hashes = {}
    hashes = read_json(study / "source.json")
    for case, row in zip(planned, rows):
        if any(case[k] != row[k] for k in case):
            raise ValueError("Inventory ordering/conditions changed")
        run = study / row["directory"]
        if row["stop_reason"] == "setup_failure":
            if not (run / "setup_failure.json").exists():
                raise ValueError("Missing setup failure record")
            continue
        manifest, maps = read_json(run / "manifest.json"), read_json(run / "map.json")
        config = read_json(run / "config.json")
        expected = replace(
            base,
            map_seed=case["map_seed"],
            observation_seed=case["observation_seed"],
            horizon=case["horizon"],
            assumed_noise_factor=case["noise_factor"] or 1.0,
            certificate_mode=case["certificate_mode"],
            safety_confidence_multiplier=case["multiplier"],
        )
        if config != expected.as_dict() or manifest["config"] != config:
            raise ValueError("Case settings differ from planned configuration")
        if manifest["source_sha256"] != hashes:
            raise ValueError("Mixed source revisions")
        if (
            manifest["map_sha256"]
            != hashlib.sha256(
                json.dumps(maps, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
        ):
            raise ValueError("Map checksum mismatch")
        previous_hash = map_hashes.setdefault(case["map_seed"], manifest["map_sha256"])
        if previous_hash != manifest["map_sha256"]:
            raise ValueError("Matched cases use different maps")
        frames = read_json(run / "trajectory.json.gz")
        observations = read_json(run / "observations.json.gz")
        if len(frames) != row["physical_actions"] + 1 or len(observations) != len(
            frames
        ):
            raise ValueError("Physical action accounting mismatch")
        grid = Grid(base.rows, base.cols)
        graph = TransitionGraph.from_grid(grid)
        z, reward = np.array(maps["altitude"]), np.array(maps["reward"])
        value = 0.0
        for a, b in zip(frames, frames[1:]):
            s = int(np.ravel_multi_index(a["position"], z.shape))
            action = [v.value for v in ACTIONS].index(b["action"])
            t = int(np.ravel_multi_index(b["position"], z.shape))
            if (
                not graph.available[s, action]
                or graph.transitions[s, action] != t
                or not a["certified_actions"][s][action]
            ):
                raise ValueError("Uncertified or unavailable executed action")
            if (
                row["certificate_mode"] == "current_lcb"
                and not a["pessimistic"][a["position"][0]][a["position"][1]]
            ):
                raise ValueError("Revalidated policy moved outside executable region")
            if (
                row["certificate_mode"] == "current_lcb"
                and action != 0
                and not a["next_action_initial_certificate"]
                and a["next_action_safety_lower"] < base.safety_threshold
            ):
                raise ValueError("Revalidated policy used a failing current LCB")
            truth = float(z[tuple(a["position"])] - z[tuple(b["position"])])
            if b["unsafe"] != (truth < base.safety_threshold):
                raise ValueError("Incorrect evaluator safety")
            value += base.discount ** a["step"] * reward[tuple(b["position"])]
            count += 1
            unsafe += int(b["unsafe"])
        npt.assert_allclose(value, row["discounted_return"], rtol=1e-10, atol=1e-10)
    if spec["stage"] == "final":
        protocol = read_json(study / "protocol.json")
        if protocol["spec"] != spec or protocol["source_sha256"] != hashes:
            raise ValueError("Final protocol mismatch")
    result = {
        "verified_runs": len(rows),
        "verified_physical_actions": count,
        "unsafe_actions_retained": unsafe,
        "source_sha256": hashes,
    }
    save_json(study / "verification.json", result)
    return result
