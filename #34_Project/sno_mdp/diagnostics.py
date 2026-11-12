"""Evaluator-only audits of saved observations, certificates, and actions.

Reconstructs joint transition posteriors, never approximating them from marginal
standard deviations. No diagnostic quantity is returned to a running policy.
Original run artifacts are read-only; analysis outputs live under safety/.
"""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from .agent import SafetyExplorer
from .artifacts import build_manifest, read_json, save_json
from .config import Config
from .environment import ACTIONS, Observation, SensorSample, Grid, WorldMap
from .graph import TransitionGraph


MODEL_FILES = (
    "agent.py",
    "config.py",
    "environment.py",
    "gp.py",
    "graph.py",
    "transition_sets.py",
)
DEFAULT_MULTIPLIERS = (1.0, 2.0, 3.0, 4.0, 6.0)
SUBSETS = ("available_nonstay", "executable_nonstay")
EXECUTION_COLUMNS = (
    "decision_step",
    "execution_step",
    "phase",
    "source",
    "action",
    "successor",
    "mean",
    "std",
    "lower",
    "upper",
    "true_g",
    "threshold",
    "unsafe",
    "violation_severity",
    "support",
    "first_accepted_step",
    "first_executable_step",
    "certificate_age",
    "acceptance_mean",
    "acceptance_std",
    "acceptance_lower",
    "acceptance_upper",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, columns=None):
    columns = columns or tuple(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def interval_statistics(mean, std, truth, multiplier):
    """Pointwise counts, without independence or uniform-coverage claims."""
    mean, std, truth = (np.asarray(a, float) for a in (mean, std, truth))
    if (
        mean.shape != std.shape
        or mean.shape != truth.shape
        or not all(np.isfinite(a).all() for a in (mean, std, truth))
        or np.any(std < 0)
        or not np.isfinite(multiplier)
        or multiplier <= 0
    ):
        raise ValueError("Invalid confidence diagnostic inputs")
    lower, upper = mean - multiplier * std, mean + multiplier * std
    below, above = truth < lower, truth > upper
    positive = std > 0
    residual = np.abs(truth - mean)
    return {
        "count": mean.size,
        "covered": int((~below & ~above).sum()),
        "below_lower": int(below.sum()),
        "above_upper": int(above.sum()),
        "coverage": float((~below & ~above).mean()) if mean.size else None,
        "rmse": float(np.sqrt(np.mean(residual**2))) if mean.size else None,
        "mean_std": float(std.mean()) if mean.size else None,
        "max_abs_standardized_error": float(np.max(residual[positive] / std[positive]))
        if positive.any()
        else None,
        "zero_std_count": int((~positive).sum()),
        "zero_std_misses": int(((~positive) & (below | above)).sum()),
    }


class SafetyAudit:
    """Truth-aware recorder owned by the evaluator, not the learner."""

    def __init__(self, world, config, multipliers=DEFAULT_MULTIPLIERS):
        self.world, self.config = world, config
        self.graph = TransitionGraph.from_grid(world.grid)
        self.valid = self.graph.available.copy()
        self.valid[:, 0] = False  # stay is structurally safe, not GP evidence
        z = world.altitude.ravel()
        self.truth = z[:, None] - z[self.graph.transitions]
        self.initial = world.initial_actions
        self.accepted = world.initial_actions.copy()
        values = tuple(multipliers) + (config.safety_confidence_multiplier,)
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Diagnostic multipliers must be finite and positive")
        self.multipliers = sorted(set(float(v) for v in values))
        self.history = {}
        for s, a in np.argwhere(self.initial & self.valid):
            self.history[int(s), int(a)] = self._certificate(s, a, None, None, None)
        self.coverage, self.executions = [], []
        self.totals = defaultdict(lambda: defaultdict(int))
        for subset in SUBSETS:
            for multiplier in self.multipliers:
                for key in (
                    "count",
                    "covered",
                    "below_lower",
                    "above_upper",
                    "zero_std_count",
                    "zero_std_misses",
                    "current_lcb_passes",
                    "false_current_lcb_passes",
                ):
                    self.totals[subset, multiplier][key] = 0

    def _certificate(self, s, a, step, mean, std):
        c = self.config.safety_confidence_multiplier
        initial = step is None
        return {
            "source": self.world.grid.states()[s],
            "action": ACTIONS[a].value,
            "successor": self.world.grid.states()[self.graph.transitions[s, a]],
            "origin": "initial_certificate" if initial else "posterior_lcb",
            "first_accepted_step": step,
            "first_executable_step": None,
            "first_executed_step": None,
            "mean": None if initial else float(mean),
            "std": None if initial else float(std),
            "lower": None if initial else float(mean - c * std),
            "upper": None if initial else float(mean + c * std),
            "true_g": float(self.truth[s, a]),
            "false_certificate": bool(self.truth[s, a] < self.world.threshold),
        }

    def record(self, step, phase, prediction, snapshot, position, next_action):
        mean, std = prediction.mean, prediction.std
        c, h = self.config.safety_confidence_multiplier, self.world.threshold
        lower = mean - c * std
        certified = snapshot.pessimistic_actions
        if np.any(self.accepted & ~snapshot.candidates):
            raise ValueError("Accepted certificate history is not monotone")
        for s, a in np.argwhere(snapshot.candidates & ~self.accepted & self.valid):
            self.history[int(s), int(a)] = self._certificate(
                s, a, step, mean[s, a], std[s, a]
            )
        self.accepted = snapshot.candidates.copy()
        for s, a in np.argwhere(certified & self.valid):
            item = self.history[int(s), int(a)]
            if item["first_executable_step"] is None:
                item["first_executable_step"] = step
        false = certified & self.valid & (self.truth < h)
        retained = certified & self.valid & ~self.initial & (lower < h)
        counts = {
            "false_executable": int(false.sum()),
            "false_current_lcb_supported": int((false & (lower >= h)).sum()),
            "false_retained": int((false & retained).sum()),
            "retained_without_current_lcb": int(retained.sum()),
            "newly_accepted": int(
                sum(v["first_accepted_step"] == step for v in self.history.values())
            ),
        }
        for subset, mask in zip(SUBSETS, (self.valid, certified & self.valid)):
            stats = interval_statistics(mean[mask], std[mask], self.truth[mask], c)
            self.coverage.append(
                {"step": step, "phase": phase, "subset": subset, **stats, **counts}
            )
            for multiplier in self.multipliers:
                values = interval_statistics(
                    mean[mask], std[mask], self.truth[mask], multiplier
                )
                total = self.totals[subset, multiplier]
                for key in (
                    "count",
                    "covered",
                    "below_lower",
                    "above_upper",
                    "zero_std_count",
                    "zero_std_misses",
                ):
                    total[key] += values[key]
                passes = mean[mask] - multiplier * std[mask] >= h
                total["current_lcb_passes"] += int(passes.sum())
                total["false_current_lcb_passes"] += int(
                    (passes & (self.truth[mask] < h)).sum()
                )
        if next_action is not None:
            s = int(
                np.ravel_multi_index(
                    position, (self.world.grid.rows, self.world.grid.cols)
                )
            )
            a = [action.value for action in ACTIONS].index(next_action)
            if not certified[s, a] or not self.graph.available[s, a]:
                raise ValueError("Saved execution is not a certified available action")
            certificate = self.history.get((s, a))
            if a == 0:
                support = "stay"
            elif self.initial[s, a]:
                support = "initial_certificate"
            elif lower[s, a] >= h:
                support = "current_lcb"
            else:
                support = "retained_certificate"
            if certificate and certificate["first_executed_step"] is None:
                certificate["first_executed_step"] = step + 1
            accepted_step = certificate["first_accepted_step"] if certificate else None
            self.executions.append(
                {
                    "decision_step": step,
                    "execution_step": step + 1,
                    "phase": phase,
                    "source": self.world.grid.states()[s],
                    "action": next_action,
                    "successor": self.world.grid.states()[self.graph.transitions[s, a]],
                    "mean": float(mean[s, a]),
                    "std": float(std[s, a]),
                    "lower": float(lower[s, a]),
                    "upper": float(mean[s, a] + c * std[s, a]),
                    "true_g": float(self.truth[s, a]),
                    "threshold": h,
                    "unsafe": bool(self.truth[s, a] < h),
                    "violation_severity": max(0.0, h - float(self.truth[s, a])),
                    "support": support,
                    "first_accepted_step": accepted_step,
                    "first_executable_step": certificate["first_executable_step"]
                    if certificate
                    else None,
                    "certificate_age": step - accepted_step
                    if accepted_step is not None
                    else None,
                    **{
                        "acceptance_" + key: certificate[key] if certificate else None
                        for key in ("mean", "std", "lower", "upper")
                    },
                }
            )

    def calibration_rows(self):
        return [
            {
                "subset": subset,
                "multiplier": c,
                **dict(counts),
                "coverage": counts["covered"] / counts["count"]
                if counts["count"]
                else None,
            }
            for (subset, c), counts in sorted(self.totals.items())
        ]

    def summary(self):
        actual = [
            row
            for row in self.calibration_rows()
            if row["multiplier"] == self.config.safety_confidence_multiplier
        ]
        unique_steps = [r for r in self.coverage if r["subset"] == SUBSETS[0]]
        unsafe = [row for row in self.executions if row["unsafe"]]
        return {
            "schema_version": 1,
            "analyzed_frames": len(unique_steps),
            "executed_actions": len(self.executions),
            "unsafe_actions": len(unsafe),
            "unsafe_current_lcb_supported": sum(
                r["support"] == "current_lcb" for r in unsafe
            ),
            "unsafe_retained_certificate": sum(
                r["support"] == "retained_certificate" for r in unsafe
            ),
            "accepted_nonstay_certificates": len(self.history),
            "false_accepted_certificates": sum(
                v["false_certificate"] for v in self.history.values()
            ),
            "false_ever_executable_certificates": sum(
                v["false_certificate"] and v["first_executable_step"] is not None
                for v in self.history.values()
            ),
            "maximum_false_executable": max(
                (r["false_executable"] for r in unique_steps), default=0
            ),
            "coverage": {r["subset"]: r for r in actual},
            "interpretation": "Pointwise edge-frame coverage; correlated edges and repeated frames are not independent trials. Alternative multipliers hold the trajectory and posterior fixed; they are not policy reruns or safety guarantees.",
        }


def diagnose_run(run, multipliers=DEFAULT_MULTIPLIERS, plots=True):
    """Audit saved inputs without rewriting the experiment or changing settings."""
    run = Path(run)
    names = (
        "config.json",
        "map.json",
        "summary.json",
        "manifest.json",
        "observations.json.gz",
        "trajectory.json.gz",
    )
    hashes = {name: _sha(run / name) for name in names}
    config = Config.load(run / "config.json")
    maps, result, manifest = (read_json(run / name) for name in names[1:4])
    if result["method"] == "oracle":
        raise ValueError(
            "Oracle has no learned confidence bands; diagnose a learning method"
        )
    analysis_manifest = build_manifest(config, maps)
    if (
        manifest["map_sha256"] != analysis_manifest["map_sha256"]
        or manifest["config"] != config.as_dict()
    ):
        raise ValueError("Saved map/config does not match its run manifest")
    for name in MODEL_FILES:
        if (
            manifest["source_sha256"].get(name)
            != analysis_manifest["source_sha256"][name]
        ):
            raise ValueError(
                "Replay model source differs from the recorded run: " + name
            )
    world = WorldMap(
        Grid(maps["rows"], maps["cols"]),
        maps["reward"],
        maps["altitude"],
        tuple(maps["start"]),
        maps["initial_actions"],
        maps["threshold"],
        config.map_seed,
        maps["spacing"],
        maps["provenance"],
    )
    if (
        (world.grid.rows, world.grid.cols) != (config.rows, config.cols)
        or world.threshold != config.safety_threshold
        or world.spacing != config.grid_spacing
    ):
        raise ValueError("Saved world and configuration disagree")
    frames = read_json(run / "trajectory.json.gz")
    observations = read_json(run / "observations.json.gz")
    if (
        len(frames) != len(observations)
        or len(frames) != result["physical_actions"] + 1
    ):
        raise ValueError("Saved frame/observation counts disagree")
    learner = SafetyExplorer(world.grid, world.initial_actions, config)
    audit = SafetyAudit(world, config, multipliers)
    skipped = []
    for step, (frame, raw) in enumerate(zip(frames, observations)):
        if (
            frame["step"] != step
            or raw["step"] != step
            or frame["position"] != raw["position"]
            or not raw["used_for_gp"]
        ):
            raise ValueError("Saved observation frame is inconsistent")
        observation = Observation(
            step,
            tuple(raw["position"]),
            tuple(
                SensorSample(tuple(s["state"]), s["reward"], s["altitude"])
                for s in raw["samples"]
            ),
        )
        try:
            snapshot = learner.update(observation)
        except (ValueError, RuntimeError) as exc:
            if (
                step == len(frames) - 1
                and frame["stop_reason"] in ("confidence_conflict", "model_failure")
                and frame["diagnostic"] == str(exc)
            ):
                skipped.append(step)
                break  # no fresh posterior is reported for a failed update
            raise
        for key, reconstructed in (
            ("certified_actions", snapshot.pessimistic_actions),
            ("optimistic_actions", snapshot.optimistic_actions),
            ("expander_actions", snapshot.expanders),
        ):
            if not np.array_equal(frame[key], reconstructed):
                raise ValueError(
                    "Reconstructed " + key + " differs at step " + str(step)
                )
        for key, reconstructed in (
            ("altitude_mean", learner.altitude_prediction.mean),
            ("altitude_std", learner.altitude_prediction.std),
        ):
            if not np.allclose(
                np.asarray(frame[key]).ravel(), reconstructed, atol=1e-10, rtol=1e-7
            ):
                raise ValueError("Reconstructed GP differs at step " + str(step))
        audit.record(
            step,
            frame["phase"],
            learner.transition_prediction,
            snapshot,
            tuple(frame["position"]),
            frame["next_action"],
        )
        if step + 1 < len(frames):
            if frame["next_action"] is None:
                raise ValueError("Missing next action before a saved execution")
            action = audit.executions[-1]
            arrival = frames[step + 1]
            if (
                action["action"] != arrival["action"]
                or tuple(action["successor"]) != tuple(arrival["position"])
                or action["unsafe"] != arrival["unsafe"]
                or not np.isclose(action["true_g"], arrival["true_transition_safety"])
            ):
                raise ValueError(
                    "Saved action outcome disagrees with reconstructed transition"
                )
        elif frame["next_action"] is not None:
            raise ValueError("Last frame must not contain an unexecuted next action")
    summary = {
        **audit.summary(),
        "method": result["method"],
        "safety_multiplier": config.safety_confidence_multiplier,
        "noise_factor": config.assumed_noise_factor,
        "skipped_failed_update_steps": skipped,
        "status": "partial_failed_update" if skipped else "complete",
    }
    if (
        summary["unsafe_actions"] != result["unsafe_actions"]
        or summary["executed_actions"] != result["physical_actions"]
    ):
        raise ValueError("Diagnostic action totals disagree with the run")
    # A concurrent overwrite must never produce a mixed-version audit.
    if hashes != {name: _sha(run / name) for name in names}:
        raise ValueError("Run inputs changed during analysis; rerun on a stable run")
    output = run / "safety"
    output.mkdir(exist_ok=True)
    save_json(output / "summary.json", summary)
    save_json(
        output / "manifest.json",
        {
            "input_sha256": hashes,
            "model_sources_verified": list(MODEL_FILES),
            "analysis": analysis_manifest,
            "multipliers": audit.multipliers,
        },
    )
    coverage_columns = (
        "step",
        "phase",
        "subset",
        *interval_statistics([], [], [], config.safety_confidence_multiplier),
        "false_executable",
        "false_current_lcb_supported",
        "false_retained",
        "retained_without_current_lcb",
        "newly_accepted",
    )
    write_csv(output / "coverage_by_step.csv", audit.coverage, coverage_columns)
    write_csv(output / "coverage_by_multiplier.csv", audit.calibration_rows())
    write_csv(output / "executed_actions.csv", audit.executions, EXECUTION_COLUMNS)
    unsafe = [r for r in audit.executions if r["unsafe"]]
    write_csv(output / "unsafe_actions.csv", unsafe, EXECUTION_COLUMNS)
    save_json(
        output / "certificate_history.json.gz",
        list(audit.history.values()),
        compact=True,
    )
    from .diagnostic_report import write_run_report

    write_run_report(
        output, summary, audit.coverage, audit.calibration_rows(), unsafe, plots
    )
    return summary


def diagnose_study(study, multipliers=DEFAULT_MULTIPLIERS, plots=True):
    """Analyze all learner runs, keeping matched maps and existing failures."""
    study = Path(study)
    rows = []
    for run in read_json(study / "runs.json"):
        if run["method"] == "oracle":
            continue
        path = (study / run["replay"]).parent
        summary = diagnose_run(path, multipliers, plots=False)
        if (
            summary["method"] != run["method"]
            or summary["noise_factor"] != run["noise_factor"]
        ):
            raise ValueError("Study inventory differs from its run")
        coverage = summary["coverage"]
        rows.append(
            {
                "grid": run["grid"],
                "map_seed": run["map_seed"],
                "observation_seed": run["observation_seed"],
                "method": run["method"],
                "noise_factor": run["noise_factor"],
                "available_coverage": coverage[SUBSETS[0]]["coverage"],
                "executable_coverage": coverage[SUBSETS[1]]["coverage"],
                "unsafe_actions": summary["unsafe_actions"],
                "unsafe_current_lcb_supported": summary["unsafe_current_lcb_supported"],
                "unsafe_retained_certificate": summary["unsafe_retained_certificate"],
                "false_ever_executable_certificates": summary[
                    "false_ever_executable_certificates"
                ],
                "failed_run": run["failed"],
                "diagnostic_status": summary["status"],
                "report": (path / "safety/index.html").relative_to(study).as_posix(),
            }
        )
        print("Diagnosed", path.relative_to(study), flush=True)
    if not rows:
        raise ValueError("Study contains no learning runs")
    groups = defaultdict(list)
    for row in rows:
        groups[row["grid"], row["method"], row["noise_factor"]].append(row)
    aggregates = []
    for (grid, method, factor), group in sorted(groups.items()):
        by_map = defaultdict(list)
        for row in group:
            by_map[row["map_seed"]].append(row)
        aggregate = {
            "grid": grid,
            "method": method,
            "noise_factor": factor,
            "runs": len(group),
            "maps": len(by_map),
        }
        for key in ("available_coverage", "executable_coverage"):
            means = [
                np.mean([r[key] for r in members if r[key] is not None])
                for members in by_map.values()
                if any(r[key] is not None for r in members)
            ]
            aggregate[key + "_mean"] = float(np.mean(means)) if means else None
            aggregate[key + "_map_sd"] = (
                float(np.std(means, ddof=1)) if len(means) > 1 else None
            )
            aggregate[key + "_measured_runs"] = sum(r[key] is not None for r in group)
            aggregate[key + "_measured_maps"] = len(means)
        for key in (
            "unsafe_actions",
            "unsafe_current_lcb_supported",
            "unsafe_retained_certificate",
        ):
            aggregate[key] = sum(row[key] for row in group)
        aggregates.append(aggregate)
    output = study / "safety"
    output.mkdir(exist_ok=True)
    summary = {
        "learner_runs": len(rows),
        "unsafe_actions": sum(r["unsafe_actions"] for r in rows),
        "aggregates": aggregates,
        "aggregation": "Each run pools its correlated edge-frame counts. Average measured observation replicates within each map, then average measured maps; SD across map means is not a confidence interval. Missing coverage is not imputed: measured-run/map denominators are saved separately; all runs and failure statuses remain in runs.csv.",
        "scope": "Posthoc diagnostics only; no policy settings changed and no multiplier selected.",
    }
    save_json(output / "summary.json", summary)
    save_json(
        output / "manifest.json",
        {
            "runs_json_sha256": _sha(study / "runs.json"),
            "run_analysis_manifests": {
                row["report"].replace("index.html", "manifest.json"): _sha(
                    study / row["report"].replace("index.html", "manifest.json")
                )
                for row in rows
            },
        },
    )
    write_csv(output / "runs.csv", rows)
    write_csv(output / "aggregates.csv", aggregates)
    from .diagnostic_report import write_study_report

    write_study_report(output, summary, rows, plots)
    return summary
