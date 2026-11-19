"""Run discovery, bounded experiment jobs, exact inspection, and comparisons."""

import hashlib
import fcntl
import itertools
import json
from collections import OrderedDict, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import queue
import secrets
import threading
import time

import numpy as np

from sno_mdp.artifacts import read_json, save_json
from sno_mdp.config import Config
from sno_mdp.environment import ACTIONS, Grid
from sno_mdp.experiment import run_experiment
from sno_mdp.gp import ExactGP
from sno_mdp.graph import TransitionGraph


ROOT = Path(__file__).resolve().parent.parent
METHODS = ("safemdp", "sno_basic", "sno_es2", "oracle")
PRESETS = {
    "source": ("Retained · c=2", "sno_es2", "retained", 2),
    "confidence": ("Retained · c=4", "sno_es2", "retained", 4),
    "revalidated": ("Revalidated · c=4", "sno_es2", "current_lcb", 4),
    "basic": ("Basic SNO-MDP · c=2", "sno_basic", "retained", 2),
    "safemdp": ("SafeMDP · c=2", "safemdp", "retained", 2),
    "oracle": ("True-field oracle", "oracle", "retained", 2),
}
MAX_CASES = 48
ALLOWED_ARTIFACTS = {
    "config.json",
    "map.json",
    "summary.json",
    "manifest.json",
    "progress.csv",
    "timing.csv",
    "trajectory.json.gz",
    "observations.json.gz",
    "es2_checks.json.gz",
    "index.html",
    "overview.png",
    "overview.svg",
    "reproduction.json",
    "safety/index.html",
    "safety/summary.json",
    "safety/unsafe_actions.csv",
    "safety/executed_actions.csv",
    "safety/coverage_by_step.csv",
    "safety/coverage_by_multiplier.csv",
    "safety/certificate_history.json.gz",
    "safety/diagnostics.png",
    "safety/diagnostics.svg",
    "safety/manifest.json",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field_limits(config):
    """Keep an interactive request within a finite local CPU/memory budget."""
    if config.rows > 25 or config.cols > 25:
        raise ValueError("Interactive grids are limited to 25 × 25.")
    if config.horizon > 400:
        raise ValueError("Interactive horizon must be at most 400 actions.")
    if config.gp_length_scale > 50 or config.gp_prior_variance > 100:
        raise ValueError("Kernel parameters exceed interactive limits.")
    if (
        max(config.safety_confidence_multiplier, config.reward_confidence_multiplier)
        > 12
    ):
        raise ValueError("Confidence multipliers must be at most 12.")
    if config.assumed_noise_factor > 20:
        raise ValueError("Assumed-noise factor must be at most 20.")
    if max(config.safety_noise_std, config.reward_noise_std) > 10:
        raise ValueError("Sensor noise exceeds the interactive limit.")
    if max(config.map_seed, config.observation_seed) > 2**32 - 1:
        raise ValueError("Seeds must fit in an unsigned 32-bit integer.")
    if config.max_iterations != 5000:
        raise ValueError("Use the fixed 5000-iteration planning limit in this UI.")
    return config


def values(request, key, default, integer=False):
    result = request.get(key, default)
    if not isinstance(result, list) or not result or len(result) > MAX_CASES:
        raise ValueError(key + " must be a nonempty bounded list.")
    for item in result:
        if integer:
            if type(item) is not int or item < 0:
                raise ValueError(key + " requires nonnegative integers.")
        elif type(item) not in (int, float) or not np.isfinite(item) or item <= 0:
            raise ValueError(key + " requires positive finite numbers.")
    if len(set(result)) != len(result):
        raise ValueError("Remove duplicate " + key + ".")
    return result


def plan_request(request):
    if not isinstance(request, dict) or set(request) - {
        "kind",
        "config",
        "method",
        "presets",
        "map_seeds",
        "observation_seeds",
        "multipliers",
        "noise_factors",
        "certificate_modes",
        "methods",
    }:
        raise ValueError("Unknown experiment request fields.")
    kind = request.get("kind", "single")
    data = request.get("config", {})
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object.")
    unknown = set(data) - set(Config().as_dict())
    if unknown:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
    base = field_limits(Config(**data))
    for name, value in data.items():
        if name not in ("map_source", "certificate_mode") and (
            type(value) not in (int, float) or not np.isfinite(value)
        ):
            raise ValueError(
                name + " must be a finite number, not a boolean or string."
            )
    cases = []
    if kind == "single":
        method = request.get("method", "sno_es2")
        cases.append(
            {
                "label": method + " · " + base.certificate_mode,
                "method": method,
                "config": base.as_dict(),
            }
        )
    elif kind == "matched":
        names = request.get(
            "presets", ["source", "confidence", "revalidated", "oracle"]
        )
        if not isinstance(names, list) or not names or len(set(names)) != len(names):
            raise ValueError("Select distinct comparison presets.")
        for name in names:
            if name not in PRESETS:
                raise ValueError("Unknown comparison preset.")
            label, method, mode, c = PRESETS[name]
            config = replace(
                base, certificate_mode=mode, safety_confidence_multiplier=c
            )
            cases.append({"label": label, "method": method, "config": config.as_dict()})
    elif kind == "sweep":
        maps = values(request, "map_seeds", [base.map_seed], integer=True)
        observations = values(
            request, "observation_seeds", [base.observation_seed], integer=True
        )
        multipliers = values(request, "multipliers", [2, 4])
        factors = values(request, "noise_factors", [0.5, 1, 2])
        modes = request.get("certificate_modes", ["retained", "current_lcb"])
        methods = request.get("methods", ["sno_es2"])
        if (
            not isinstance(modes, list)
            or not modes
            or len(set(modes)) != len(modes)
            or not set(modes) <= {"retained", "current_lcb"}
            or not isinstance(methods, list)
            or not methods
            or len(set(methods)) != len(methods)
            or not set(methods) <= set(METHODS)
        ):
            raise ValueError(
                "Choose valid, distinct sweep methods and certificate modes."
            )
        if base.map_source == "released" and maps != [base.map_seed]:
            raise ValueError(
                "The released fixture is one map; seeds do not create independent maps."
            )
        count = (
            len(maps)
            * len(observations)
            * (
                len(multipliers)
                * len(factors)
                * len(modes)
                * sum(m != "oracle" for m in methods)
                + int("oracle" in methods)
            )
        )
        if count > MAX_CASES:
            raise ValueError(
                f"Sweep has {count} cases; interactive maximum is {MAX_CASES}."
            )
        for m, o in itertools.product(maps, observations):
            if "oracle" in methods:
                config = replace(base, map_seed=m, observation_seed=o)
                cases.append(
                    {
                        "label": "True-field oracle",
                        "method": "oracle",
                        "config": config.as_dict(),
                    }
                )
            for c, factor, mode, method in itertools.product(
                multipliers, factors, modes, [m for m in methods if m != "oracle"]
            ):
                config = field_limits(
                    replace(
                        base,
                        map_seed=m,
                        observation_seed=o,
                        safety_confidence_multiplier=c,
                        assumed_noise_factor=factor,
                        certificate_mode=mode,
                    )
                )
                cases.append(
                    {
                        "label": f"{method} · {mode} · c={c:g} · noise={factor:g}",
                        "method": method,
                        "config": config.as_dict(),
                    }
                )
    else:
        raise ValueError("Unknown experiment kind.")
    if not cases or len(cases) > MAX_CASES:
        raise ValueError("Invalid interactive case count.")
    for case in cases:
        if case["method"] not in METHODS:
            raise ValueError("Unknown method.")
        field_limits(Config(**case["config"]))
    return cases


class Lab:
    def __init__(self, root=ROOT, start_worker=True, recover_interrupted=None):
        self.root = Path(root).resolve()
        self.engine_sources = {
            p.name: digest(p) for p in (ROOT / "sno_mdp").glob("*.py")
        }
        self.output = self.root / "results/interactive"
        self.output.mkdir(parents=True, exist_ok=True)
        self.process_lock = None
        if start_worker:
            self.process_lock = (self.output / ".server.lock").open("a+")
            try:
                fcntl.flock(self.process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.process_lock.close()
                raise ValueError(
                    "A dashboard worker already owns this results directory. Open the existing dashboard or stop it before restarting."
                ) from exc
        if recover_interrupted is None:
            recover_interrupted = start_worker
        self.lock = threading.RLock()
        self.catalog_lock = threading.RLock()
        self.inspect_lock = threading.Lock()
        self.records = {}
        self.jobs = {}
        self.payloads = OrderedDict()
        self.posteriors = OrderedDict()
        self.queue = queue.Queue()
        self.refresh()
        for path in self.output.glob("*/job.json"):
            try:
                job = read_json(path)
                if recover_interrupted and job["status"] in (
                    "queued",
                    "running",
                    "cancel_requested",
                ):
                    job["status"] = "interrupted"
                    job["message"] = (
                        "Server stopped before this batch finished. Completed cases remain available."
                    )
                    save_json(path, job)
                self.jobs[job["id"]] = job
            except (ValueError, KeyError, OSError):
                continue
        if start_worker:
            self.worker = threading.Thread(
                target=self._work, name="snomdp-lab-worker", daemon=True
            )
            self.worker.start()

    def frozen_status(self):
        path = self.root / "report/frozen_protocol.json"
        if not path.exists():
            return {"available": False, "matches": False}
        protocol = read_json(path)
        hashes = {p.name: digest(p) for p in (self.root / "sno_mdp").glob("*.py")}
        return {"available": True, "matches": hashes == protocol["source_sha256"]}

    def _inside_results(self, path):
        path = Path(path).resolve()
        try:
            path.relative_to((self.root / "results").resolve())
        except ValueError as exc:
            raise ValueError("Run is outside the project results directory.") from exc
        return path

    def _record(self, directory, group, label=None, metadata=None):
        directory = self._inside_results(directory)
        try:
            c = read_json(directory / "config.json")
            s = read_json(directory / "summary.json")
            if not (directory / "trajectory.json.gz").exists():
                return
            Config(**c)
            relative = directory.relative_to(self.root).as_posix()
            identifier = hashlib.sha256(relative.encode()).hexdigest()[:20]
            oracle = s["method"] == "oracle"
            manifest = read_json(directory / "manifest.json")
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "config": c,
                        "method": s["method"],
                        "map": manifest["map_sha256"],
                        "model": manifest["source_sha256"],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            self.records[identifier] = {
                "id": identifier,
                "path": relative,
                "group": group,
                "label": label or (s["method"] + " · " + c["certificate_mode"]),
                "method": s["method"],
                "mode": c["certificate_mode"],
                "multiplier": c["safety_confidence_multiplier"],
                "noise_factor": None if oracle else c["assumed_noise_factor"],
                "map_seed": c["map_seed"],
                "observation_seed": c["observation_seed"],
                "rows": c["rows"],
                "cols": c["cols"],
                "map_source": c["map_source"],
                "horizon": c["horizon"],
                "actual_actions": s["physical_actions"],
                "return": s["discounted_return"],
                "cumulative_reward": s["cumulative_reward"],
                "unsafe_actions": s["unsafe_actions"],
                "false_certificates": s["maximum_false_certified_actions"],
                "failed": s["stop_reason"] != "horizon_reached",
                "stop_reason": s["stop_reason"],
                "transition_step": s["transition_step"],
                "transition_reason": s["transition_reason"],
                "es2_triggered": s["es2_triggered"],
                "exploration_actions": s["phase_actions"].get("safety_exploration", 0),
                "reward_actions": s["phase_actions"].get("reward_optimization", 0),
                "fingerprint": fingerprint,
                "config": c,
                "batch": directory.relative_to(self.output).parts[0]
                if group == "interactive"
                else None,
                "map_hash": manifest["map_sha256"],
                "model_hash": hashlib.sha256(
                    json.dumps(manifest["source_sha256"], sort_keys=True).encode()
                ).hexdigest(),
                "inference_seconds": (metadata or {}).get("inference_seconds"),
                "decision_seconds": (metadata or {}).get("decision_seconds"),
            }
        except (OSError, ValueError, KeyError):
            return

    def refresh(self):
        with self.catalog_lock:
            self.records = {}
            inventory = self.root / "results/final/runs.json"
            if inventory.exists():
                for row in read_json(inventory):
                    self._record(
                        self.root / "results/final" / row["directory"],
                        "frozen",
                        row["profile"] + " · " + row["method"],
                        row,
                    )
            for directory in sorted((self.root / "results").glob("*")):
                if (
                    directory.is_dir()
                    and directory.name != "interactive"
                    and (directory / "config.json").exists()
                ):
                    self._record(directory, "saved", directory.name)
            for directory in sorted(self.output.glob("*/cases/*")):
                if directory.is_dir():
                    label_path = directory / "ui_case.json"
                    label = (
                        read_json(label_path)["label"]
                        if label_path.exists()
                        else directory.name
                    )
                    self._record(directory, "interactive", label)
            with self.lock:
                self.payloads.clear()
            with self.inspect_lock:
                self.posteriors.clear()
            return list(self.records.values())

    def catalog(self):
        with self.catalog_lock:
            return sorted(
                self.records.values(),
                key=lambda r: (
                    {"saved": 0, "interactive": 1, "frozen": 2}[r["group"]],
                    r["path"],
                ),
            )

    def resolve_run(self, identifier):
        with self.catalog_lock:
            record = self.records.get(identifier)
        if record is None:
            raise ValueError("Unknown run identifier. Refresh the library.")
        return self._inside_results(self.root / record["path"]), dict(record)

    def payload(self, identifier):
        run, record = self.resolve_run(identifier)
        with self.lock:
            if identifier in self.payloads:
                self.payloads.move_to_end(identifier)
                return self.payloads[identifier]
        result = {
            "record": record,
            "config": read_json(run / "config.json"),
            "map": read_json(run / "map.json"),
            "summary": read_json(run / "summary.json"),
            "frames": read_json(run / "trajectory.json.gz"),
        }
        with self.lock:
            self.payloads[identifier] = result
            while len(self.payloads) > 3:
                self.payloads.popitem(last=False)
        return result

    def series(self, identifier):
        p = self.payload(identifier)
        keys = (
            "step",
            "discounted_return",
            "cumulative_reward",
            "certified_cells",
            "false_certified_actions",
            "unsafe_actions",
            "phase",
        )
        return {
            "record": p["record"],
            "frames": [{k: f[k] for k in keys} for f in p["frames"]],
        }

    def inspect(self, identifier, step, x, y):
        p = self.payload(identifier)
        c = Config(**p["config"])
        if not 0 <= step < len(p["frames"]):
            raise ValueError("Step outside the saved replay.")
        grid = Grid(c.rows, c.cols)
        position = grid.validate((x, y))
        run, _ = self.resolve_run(identifier)
        manifest = read_json(run / "manifest.json")
        if p["record"]["method"] != "oracle" and any(
            manifest["source_sha256"].get(name) != self.engine_sources[name]
            for name in ("gp.py", "agent.py")
        ):
            raise ValueError(
                "Recorded GP source differs. Saved fields remain viewable, but exact bound inspection is unavailable."
            )
        key = identifier, step
        with self.inspect_lock:
            if key not in self.posteriors and p["record"]["method"] != "oracle":
                observations = read_json(run / "observations.json.gz")
                samples = [
                    s
                    for frame in observations[: step + 1]
                    if frame.get("used_for_gp", True)
                    for s in frame["samples"]
                ]
                coordinates = (
                    np.array([s["state"] for s in samples], float) * c.grid_spacing
                )
                graph = TransitionGraph.from_grid(grid)
                model = ExactGP(
                    c.safety_noise_std * c.assumed_noise_factor,
                    length_scale=c.gp_length_scale,
                    prior_variance=c.gp_prior_variance,
                )
                model.fit(coordinates, [s["altitude"] for s in samples])
                _, prediction = model.predict_transitions(
                    np.array(grid.states(), float) * c.grid_spacing, graph.transitions
                )
                self.posteriors[key] = prediction
                while len(self.posteriors) > 8:
                    self.posteriors.popitem(last=False)
            prediction = self.posteriors.get(key)
        f = p["frames"][step]
        source = int(np.ravel_multi_index(position, (c.rows, c.cols)))
        z = np.array(p["map"]["altitude"])
        initial = np.array(p["map"]["initial_actions"])
        rows = []
        for a, action in enumerate(ACTIONS):
            target = grid.next_state(position, action)
            available = bool(grid.action_mask()[source, a])
            g = float(z[position] - z[target])
            certified = bool(f["certified_actions"][source][a])
            mean = float(prediction.mean[source, a]) if prediction else None
            std = float(prediction.std[source, a]) if prediction else None
            lower = mean - c.safety_confidence_multiplier * std if prediction else None
            upper = mean + c.safety_confidence_multiplier * std if prediction else None
            support = "unavailable" if not available else "not certified"
            if certified:
                support = (
                    "oracle truth"
                    if not prediction
                    else "structural stay"
                    if a == 0
                    else "initial certificate"
                    if initial[source, a]
                    else "current lower bound"
                    if lower >= c.safety_threshold
                    else "retained certificate"
                )
            rows.append(
                {
                    "action": action.value,
                    "target": target,
                    "available": available,
                    "certified": certified,
                    "initial": bool(initial[source, a]),
                    "mean": mean,
                    "std": std,
                    "lower": lower,
                    "upper": upper,
                    "true_g": g if available else None,
                    "true_safe": bool(g >= c.safety_threshold) if available else None,
                    "support": support,
                }
            )
        return {
            "step": step,
            "position": position,
            "threshold": c.safety_threshold,
            "actions": rows,
            "scope": "Evaluator-only inspection; never sent to a policy.",
        }

    def compare(self, identifiers):
        if not isinstance(identifiers, list) or not 1 <= len(identifiers) <= 1000:
            raise ValueError("Select between 1 and 1000 runs.")
        records, duplicates, seen = [], 0, set()
        for identifier in identifiers:
            _, r = self.resolve_run(identifier)
            if r["fingerprint"] in seen:
                duplicates += 1
                continue
            seen.add(r["fingerprint"])
            records.append(r)
        groups = defaultdict(list)
        for r in records:
            c = r["config"]
            path, _ = self.resolve_run(r["id"])
            key = (
                r["method"],
                json.dumps(
                    {
                        k: v
                        for k, v in c.items()
                        if k not in ("map_seed", "observation_seed")
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    read_json(path / "manifest.json")["source_sha256"], sort_keys=True
                ),
            )
            groups[key].append(r)
        result = []
        for members in groups.values():
            by_map = defaultdict(list)
            for r in members:
                path, _ = self.resolve_run(r["id"])
                by_map[read_json(path / "manifest.json")["map_sha256"]].append(r)
            item = {
                "condition": members[0]["label"],
                "method": members[0]["method"],
                "mode": members[0]["mode"],
                "multiplier": members[0]["multiplier"],
                "noise_factor": members[0]["noise_factor"],
                "horizon": members[0]["horizon"],
                "runs": len(members),
                "maps": len(by_map),
                "condition_id": "C" + str(len(result) + 1),
                "model_hash": members[0]["model_hash"],
                "cohort": sorted(
                    {
                        (r["map_hash"], r["map_seed"], r["observation_seed"])
                        for r in members
                    }
                ),
                "configuration": json.dumps(members[0]["config"], sort_keys=True),
                "scientific_source": json.dumps(
                    read_json(self.resolve_run(members[0]["id"])[0] / "manifest.json")[
                        "source_sha256"
                    ],
                    sort_keys=True,
                ),
            }
            for metric in (
                "return",
                "cumulative_reward",
                "unsafe_actions",
                "failed",
                "false_certificates",
                "es2_triggered",
                "exploration_actions",
                "inference_seconds",
                "decision_seconds",
            ):
                means = [
                    np.mean([r[metric] for r in values if r[metric] is not None])
                    for values in by_map.values()
                    if any(r[metric] is not None for r in values)
                ]
                item[metric + "_mean"] = float(np.mean(means)) if means else None
                item[metric + "_sd"] = (
                    float(np.std(means, ddof=1)) if len(means) > 1 else None
                )
            item["unsafe_fraction"] = float(
                np.mean(
                    [
                        np.mean([r["unsafe_actions"] > 0 for r in values])
                        for values in by_map.values()
                    ]
                )
            )
            result.append(item)
        paired = None
        changes = []
        if len(records) == 2:
            a, b = records
            changes = [
                {"field": k, "left": a["config"].get(k), "right": b["config"].get(k)}
                for k in a["config"]
                if a["config"].get(k) != b["config"].get(k)
            ]
            if a["method"] != b["method"]:
                changes.insert(
                    0, {"field": "method", "left": a["method"], "right": b["method"]}
                )
            ra, _ = self.resolve_run(a["id"])
            rb, _ = self.resolve_run(b["id"])
            compatible = (
                all(
                    a["config"][k] == b["config"][k]
                    for k in (
                        "observation_seed",
                        "map_seed",
                        "horizon",
                        "discount",
                        "safety_noise_std",
                        "reward_noise_std",
                        "safety_threshold",
                        "grid_spacing",
                    )
                )
                and read_json(ra / "manifest.json")["map_sha256"]
                == read_json(rb / "manifest.json")["map_sha256"]
            )
            compatible = (
                compatible
                and read_json(ra / "manifest.json")["source_sha256"]
                == read_json(rb / "manifest.json")["source_sha256"]
            )
            if compatible:
                paired = {
                    "left": {k: a[k] for k in ("id", "label", "path")},
                    "right": {k: b[k] for k in ("id", "label", "path")},
                    "right_minus_left_return": b["return"] - a["return"],
                    "right_minus_left_unsafe_actions": b["unsafe_actions"]
                    - a["unsafe_actions"],
                    "scope": "One matched pair; not a significance estimate.",
                }
        varying = [
            k
            for k in records[0]["config"]
            if k not in ("map_seed", "observation_seed")
            and len({json.dumps(r["config"][k]) for r in records}) > 1
        ]
        background = [
            k
            for k in varying
            if k
            not in (
                "certificate_mode",
                "safety_confidence_multiplier",
                "assumed_noise_factor",
            )
        ]
        warnings = []
        if background:
            warnings.append(
                "Other settings differ: "
                + ", ".join(background)
                + ". Raw return differences do not isolate the method or certificate rule."
            )
        if len({r["model_hash"] for r in records}) > 1:
            warnings.append(
                "Scientific source revisions differ; these groups do not share one implementation."
            )
        if len({json.dumps(item["cohort"]) for item in result}) > 1:
            warnings.append(
                "Conditions use different map/observation cohorts. Group means are descriptive; use a matched cohort for a controlled comparison."
            )
        if len(records) == 2 and len(changes) > 1:
            warnings.append(
                "Multiple controls differ in this pair: "
                + ", ".join(change["field"] for change in changes)
                + ". The difference cannot be attributed to one change alone."
            )
        return {
            "aggregates": result,
            "duplicates_excluded": duplicates,
            "unique_runs": len(records),
            "changes": changes,
            "paired": paired,
            "varying_fields": varying,
            "warnings": warnings,
            "rule": "Observation replicates averaged within each true map, then maps equally; SD across map means, not a confidence interval.",
        }

    def jobs_list(self):
        with self.lock:
            return sorted(
                json.loads(json.dumps(list(self.jobs.values()))),
                key=lambda v: v["created"],
                reverse=True,
            )

    def submit(self, request):
        cases = plan_request(request)
        with self.lock:
            if (
                sum(
                    j["status"] in ("queued", "running", "cancel_requested")
                    for j in self.jobs.values()
                )
                >= 5
            ):
                raise ValueError(
                    "Five batches are already active/queued. Finish or cancel one first."
                )
            identifier = datetime.now(timezone.utc).strftime(
                "%Y%m%d_%H%M%S_"
            ) + secrets.token_hex(4)
            directory = self.output / identifier
            directory.mkdir()
            save_json(directory / "request.json", request)
            save_json(directory / "plan.json", cases)
            save_json(
                directory / "lab_manifest.json",
                {
                    "created": now(),
                    "frozen_source": self.frozen_status(),
                    "analysis_ui_sha256": {
                        p.relative_to(ROOT / "analysis_ui").as_posix(): digest(p)
                        for p in (ROOT / "analysis_ui").rglob("*")
                        if p.is_file() and p.suffix in (".py", ".js", ".html", ".css")
                    },
                },
            )
            job = {
                "id": identifier,
                "status": "queued",
                "created": now(),
                "total": len(cases),
                "completed": 0,
                "cases": [],
                "message": "Queued; cases execute sequentially on the local engine.",
                "cancel": False,
                "kind": request.get("kind", "single"),
            }
            self.jobs[identifier] = job
            self._save_job(job)
            self.queue.put(identifier)
            return dict(job)

    def cancel(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            if not job:
                raise ValueError("Unknown batch.")
            if job["status"] in ("queued", "running", "cancel_requested"):
                job["cancel"] = True
                job["status"] = "cancel_requested"
                job["message"] = (
                    "Cancellation requested. The current case finishes; remaining cases will not start."
                )
                self._save_job(job)
            return dict(job)

    def _save_job(self, job):
        path = self.output / job["id"] / "job.json"
        temporary = path.with_suffix(".tmp")
        save_json(temporary, job)
        temporary.replace(path)

    def _work(self):
        while True:
            identifier = self.queue.get()
            try:
                self.execute(identifier)
            except Exception as exc:
                with self.lock:
                    job = self.jobs[identifier]
                    job.update(status="error", message=str(exc))
                    self._save_job(job)
            finally:
                self.queue.task_done()

    def execute(self, identifier):
        job_dir = self.output / identifier
        cases = read_json(job_dir / "plan.json")
        for index, case in enumerate(cases):
            with self.lock:
                job = self.jobs[identifier]
                if job["cancel"]:
                    break
                job.update(
                    status="running",
                    message=f"Running case {index + 1}/{len(cases)}: {case['label']}",
                )
                self._save_job(job)
            config = Config(**case["config"])
            output = job_dir / "cases" / ("%03d" % (index + 1))
            began = time.perf_counter()
            try:
                if {
                    p.name: digest(p) for p in (ROOT / "sno_mdp").glob("*.py")
                } != self.engine_sources:
                    raise ValueError(
                        "Scientific source changed while the server was running. Restart the dashboard before running another case."
                    )
                summary = run_experiment(
                    config, output, case["method"], write_replay=True
                )
                save_json(
                    output / "ui_case.json",
                    {
                        "label": case["label"],
                        "exploratory": True,
                        "batch_id": identifier,
                        "case_number": index + 1,
                    },
                )
                item = {
                    "label": case["label"],
                    "path": output.relative_to(self.root).as_posix(),
                    "stop_reason": summary["stop_reason"],
                    "return": summary["discounted_return"],
                    "unsafe_actions": summary["unsafe_actions"],
                    "error": None,
                }
            except (ValueError, RuntimeError, OSError) as exc:
                output.mkdir(parents=True, exist_ok=True)
                save_json(
                    output / "setup_failure.json",
                    {"config": config.as_dict(), "error": str(exc)},
                )
                item = {
                    "label": case["label"],
                    "path": output.relative_to(self.root).as_posix(),
                    "stop_reason": "setup_failure",
                    "return": 0.0,
                    "unsafe_actions": 0,
                    "error": str(exc),
                }
            item["elapsed_seconds"] = time.perf_counter() - began
            with self.lock:
                job = self.jobs[identifier]
                job["cases"].append(item)
                job["completed"] += 1
                self._save_job(job)
            self.refresh()
        with self.lock:
            job = self.jobs[identifier]
            status = (
                "cancelled"
                if job["cancel"] and job["completed"] < job["total"]
                else "completed"
            )
            job.update(
                status=status,
                finished=now(),
                message=f"{job['completed']}/{job['total']} cases finished. Exploratory results saved separately from the frozen study.",
            )
            self._save_job(job)
