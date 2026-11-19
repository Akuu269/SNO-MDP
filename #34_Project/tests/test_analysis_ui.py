"""UI jobs must preserve evidence, match conditions, and expose honest comparisons."""

import json
import fcntl
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from analysis_ui.lab import Lab, ROOT, digest, plan_request
from analysis_ui.server import make_server
from sno_mdp.artifacts import read_json, save_json
from sno_mdp.config import Config
from sno_mdp.experiment import run_experiment


SMALL = Config(
    rows=6,
    cols=6,
    map_source="generated",
    map_seed=3,
    observation_seed=2026,
    horizon=4,
    discount=0.5,
)


class PlanningTests(unittest.TestCase):
    def test_matched_cases_change_only_declared_conditions(self):
        cases = plan_request({"kind": "matched", "config": SMALL.as_dict()})
        self.assertEqual(len(cases), 4)
        for case in cases:
            actual = dict(case["config"])
            for key in ("certificate_mode", "safety_confidence_multiplier"):
                actual[key] = SMALL.as_dict()[key]
            self.assertEqual(actual, SMALL.as_dict())
        self.assertEqual(cases[2]["config"]["certificate_mode"], "current_lcb")

    def test_sweep_counts_oracle_once_per_map_observation(self):
        request = {
            "kind": "sweep",
            "config": SMALL.as_dict(),
            "map_seeds": [3, 4],
            "observation_seeds": [10, 11],
            "multipliers": [2, 4],
            "noise_factors": [1],
            "methods": ["sno_es2", "oracle"],
        }
        cases = plan_request(request)
        self.assertEqual(len(cases), 20)
        self.assertEqual(sum(c["method"] == "oracle" for c in cases), 4)

    def test_unbounded_or_invalid_requests_are_rejected(self):
        requests = [
            {"config": {**SMALL.as_dict(), "safety_noise_std": True}},
            {"config": {**SMALL.as_dict(), "horizon": 401}},
            {"config": {**SMALL.as_dict(), "rows": 26}},
            {"output": "../report", "config": SMALL.as_dict()},
            {"method": "custom_python", "config": SMALL.as_dict()},
            {"kind": "sweep", "config": SMALL.as_dict(), "map_seeds": [True]},
            {"kind": "sweep", "config": SMALL.as_dict(), "noise_factors": [[1]]},
            {"kind": "sweep", "config": SMALL.as_dict(), "multipliers": [2, 2]},
            {"kind": "sweep", "config": SMALL.as_dict(), "map_seeds": list(range(10))},
            {"kind": "sweep", "config": Config().as_dict(), "map_seeds": [7, 8]},
        ]
        for request in requests:
            with (
                self.subTest(request=request),
                self.assertRaises((ValueError, TypeError)),
            ):
                plan_request(request)


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.fixture_run = Path(cls.fixture.name) / "fixture"
        run_experiment(SMALL, cls.fixture_run, "sno_es2", write_replay=False)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "results/demo"
        shutil.copytree(self.fixture_run, self.directory)
        self.lab = Lab(self.root, start_worker=False)
        self.identifier = self.lab.catalog()[0]["id"]

    def test_inspection_matches_recorded_next_action_bound_and_cannot_write_evidence(
        self,
    ):
        before = {p.name: digest(p) for p in self.directory.iterdir() if p.is_file()}
        frame = read_json(self.directory / "trajectory.json.gz")[0]
        data = self.lab.inspect(self.identifier, 0, *frame["position"])
        chosen = next(a for a in data["actions"] if a["action"] == frame["next_action"])
        self.assertAlmostEqual(
            chosen["lower"], frame["next_action_safety_lower"], places=8
        )
        self.assertEqual(data["actions"][0]["std"], 0)
        self.assertEqual(data["actions"][0]["true_g"], 0)
        after = {p.name: digest(p) for p in self.directory.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_historical_model_is_not_silently_reinterpreted(self):
        manifest = read_json(self.directory / "manifest.json")
        manifest["source_sha256"]["gp.py"] = "earlier-version"
        save_json(self.directory / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "differs"):
            self.lab.inspect(self.identifier, 0, 0, 0)

    def test_aliases_are_deduplicated(self):
        shutil.copytree(self.directory, self.root / "results/copied_demo")
        self.lab.refresh()
        result = self.lab.compare([r["id"] for r in self.lab.catalog()])
        self.assertEqual(result["unique_runs"], 1)
        self.assertEqual(result["duplicates_excluded"], 1)
        self.assertEqual(result["aggregates"][0]["maps"], 1)

    def test_map_level_aggregation_weights_maps_equally(self):
        paths = []
        for i, (map_hash, observation, value) in enumerate(
            [("A", 1, 0), ("A", 2, 2), ("B", 1, 9)]
        ):
            path = self.root / "results" / ("case_" + str(i))
            shutil.copytree(self.fixture_run, path)
            config = read_json(path / "config.json")
            config.update(
                observation_seed=observation, map_seed=1 if map_hash == "A" else 2
            )
            save_json(path / "config.json", config)
            summary = read_json(path / "summary.json")
            summary["discounted_return"] = value
            save_json(path / "summary.json", summary)
            manifest = read_json(path / "manifest.json")
            manifest["map_sha256"] = map_hash
            save_json(path / "manifest.json", manifest)
            paths.append(path.relative_to(self.root).as_posix())
        self.lab.refresh()
        rows = self.lab.compare(
            [r["id"] for r in self.lab.catalog() if r["path"] in paths]
        )
        a = rows["aggregates"][0]
        self.assertEqual(a["runs"], 3)
        self.assertEqual(a["maps"], 2)
        self.assertEqual(a["return_mean"], 5)
        self.assertAlmostEqual(a["return_sd"], np.std([1, 9], ddof=1))
        self.assertEqual(a["condition_id"], "C1")
        self.assertEqual(len(a["cohort"]), 3)

    def test_different_model_settings_and_revisions_are_separate_groups(self):
        other = self.root / "results/other"
        shutil.copytree(self.fixture_run, other)
        config = read_json(other / "config.json")
        config["exploration_accuracy"] *= 2
        save_json(other / "config.json", config)
        self.lab.refresh()
        result = self.lab.compare([r["id"] for r in self.lab.catalog()])
        self.assertEqual(len(result["aggregates"]), 2)
        self.assertTrue(any("Other settings differ" in w for w in result["warnings"]))
        self.assertIn("exploration_accuracy", result["varying_fields"])
        manifest = read_json(other / "manifest.json")
        manifest["source_sha256"]["gp.py"] = "historical"
        save_json(other / "manifest.json", manifest)
        save_json(other / "config.json", SMALL.as_dict())
        self.lab.refresh()
        result = self.lab.compare([r["id"] for r in self.lab.catalog()])
        self.assertEqual(len(result["aggregates"]), 2)
        self.assertIsNone(result["paired"])
        self.assertTrue(any("source revisions" in w for w in result["warnings"]))

    def test_pair_checks_map_seed_because_it_changes_the_sensor_stream(self):
        other = self.root / "results/same_map_new_seed"
        shutil.copytree(self.fixture_run, other)
        config = read_json(other / "config.json")
        config["map_seed"] += 1
        save_json(other / "config.json", config)
        self.lab.refresh()
        result = self.lab.compare([r["id"] for r in self.lab.catalog()])
        self.assertIsNone(result["paired"])

    def test_partial_cohorts_are_disclosed(self):
        other = self.root / "results/other_cohort"
        shutil.copytree(self.fixture_run, other)
        config = read_json(other / "config.json")
        config.update(observation_seed=99, safety_confidence_multiplier=4)
        save_json(other / "config.json", config)
        self.lab.refresh()
        result = self.lab.compare([r["id"] for r in self.lab.catalog()])
        self.assertTrue(
            any("different map/observation cohorts" in w for w in result["warnings"])
        )

    def test_two_control_changes_are_not_presented_as_one_effect(self):
        other = self.root / "results/two_controls"
        shutil.copytree(self.fixture_run, other)
        config = read_json(other / "config.json")
        config.update(safety_confidence_multiplier=4, certificate_mode="current_lcb")
        save_json(other / "config.json", config)
        self.lab.refresh()
        result = self.lab.compare([r["id"] for r in self.lab.catalog()])
        self.assertIsNotNone(result["paired"])
        self.assertTrue(
            any("Multiple controls differ" in w for w in result["warnings"])
        )

    def test_oracle_truth_inspection_does_not_require_a_gp_revision(self):
        summary = read_json(self.directory / "summary.json")
        summary["method"] = "oracle"
        save_json(self.directory / "summary.json", summary)
        manifest = read_json(self.directory / "manifest.json")
        manifest["source_sha256"]["gp.py"] = "unused by oracle"
        save_json(self.directory / "manifest.json", manifest)
        self.lab.refresh()
        result = self.lab.inspect(self.identifier, 0, 0, 0)
        self.assertIsNone(result["actions"][0]["mean"])
        self.assertEqual(result["actions"][0]["true_g"], 0)

    def test_loaded_source_change_blocks_an_unreproducible_new_case(self):
        job = self.lab.submit({"config": SMALL.as_dict()})
        self.lab.engine_sources["gp.py"] = "changed after startup"
        self.lab.execute(job["id"])
        case = self.lab.jobs[job["id"]]["cases"][0]
        self.assertEqual(case["stop_reason"], "setup_failure")
        self.assertIn("Restart the dashboard", case["error"])

    def test_recorded_failure_keeps_return_prefix_in_aggregate(self):
        summary = read_json(self.directory / "summary.json")
        summary.update(
            stop_reason="model_failure", physical_actions=2, discounted_return=1.25
        )
        save_json(self.directory / "summary.json", summary)
        self.lab.refresh()
        row = self.lab.compare([self.identifier])["aggregates"][0]
        self.assertEqual(row["return_mean"], 1.25)
        self.assertEqual(row["failed_mean"], 1)

    def test_refresh_invalidates_replay_cache(self):
        self.lab.payload(self.identifier)
        summary = read_json(self.directory / "summary.json")
        summary["discounted_return"] = 123
        save_json(self.directory / "summary.json", summary)
        self.lab.refresh()
        self.assertEqual(
            self.lab.payload(self.identifier)["summary"]["discounted_return"], 123
        )

    def test_second_worker_cannot_take_over_an_active_queue(self):
        with (self.lab.output / ".server.lock").open("a+") as owner:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "already owns"):
                Lab(self.root, start_worker=True)

    def test_worker_saves_separate_reproducible_matched_cases(self):
        job = self.lab.submit(
            {
                "kind": "matched",
                "presets": ["source", "revalidated"],
                "config": SMALL.as_dict(),
            }
        )
        self.lab.execute(job["id"])
        completed = self.lab.jobs_list()[0]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["completed"], 2)
        hashes = [
            read_json(self.root / c["path"] / "manifest.json")["map_sha256"]
            for c in completed["cases"]
        ]
        self.assertEqual(*hashes)
        self.assertTrue(
            all(
                c["path"].startswith("results/interactive/") for c in completed["cases"]
            )
        )
        self.assertTrue((self.lab.output / job["id"] / "lab_manifest.json").exists())

    def test_cancellation_setup_failure_and_restart_are_explicit(self):
        request = {"config": SMALL.as_dict()}
        job = self.lab.submit(request)
        self.lab.cancel(job["id"])
        self.lab.execute(job["id"])
        self.assertEqual(self.lab.jobs[job["id"]]["status"], "cancelled")
        self.assertEqual(self.lab.jobs[job["id"]]["completed"], 0)
        job = self.lab.submit(request)
        with patch(
            "analysis_ui.lab.run_experiment",
            side_effect=ValueError("fixture setup failed"),
        ):
            self.lab.execute(job["id"])
        self.assertEqual(
            self.lab.jobs[job["id"]]["cases"][0]["stop_reason"], "setup_failure"
        )
        self.assertEqual(
            self.lab.jobs[job["id"]]["cases"][0]["error"], "fixture setup failed"
        )
        queued = self.lab.submit(request)
        observer = Lab(self.root, start_worker=False)
        self.assertEqual(observer.jobs[queued["id"]]["status"], "queued")
        restarted = Lab(self.root, start_worker=False, recover_interrupted=True)
        self.assertEqual(restarted.jobs[queued["id"]]["status"], "interrupted")


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        run_experiment(
            SMALL, Path(cls.temp.name) / "results/http_fixture", write_replay=False
        )
        cls.server = make_server(Path(cls.temp.name), port=0, start_worker=False)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:" + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.temp.cleanup()

    def request(self, path, data=None, headers=None):
        req = Request(
            self.base + path,
            data=None if data is None else json.dumps(data).encode(),
            headers=headers or {},
        )
        with urlopen(req, timeout=10) as response:
            return response.read()

    def test_static_and_status_work(self):
        self.assertIn(b"Experiment Lab", self.request("/"))
        self.assertIn(b"drawMap", self.request("/app.js"))
        self.assertEqual(json.loads(self.request("/api/status"))["max_cases"], 48)

    def test_occupied_port_does_not_start_an_unused_queue_owner(self):
        with self.assertRaises(OSError):
            make_server(
                Path(self.temp.name), port=self.server.server_port, start_worker=True
            )
        with (self.server.lab.output / ".server.lock").open("a+") as owner:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_artifact_json_and_compressed_trace_are_downloaded_unchanged(self):
        record = self.server.lab.catalog()[0]
        prefix = "/artifact/" + record["id"] + "/"
        self.assertEqual(
            json.loads(self.request(prefix + "config.json")), SMALL.as_dict()
        )
        self.assertEqual(self.request(prefix + "trajectory.json.gz")[:2], b"\x1f\x8b")
        self.assertEqual(
            len(json.loads(self.request("/api/series/" + record["id"]))["frames"]), 5
        )

    def test_job_endpoint_persists_config_and_cancellation(self):
        headers = {"X-Lab-Token": self.server.token}
        job = json.loads(
            self.request("/api/jobs", {"config": SMALL.as_dict()}, headers)
        )
        self.assertEqual(job["status"], "queued")
        cancelled = json.loads(
            self.request("/api/jobs/" + job["id"] + "/cancel", {}, headers)
        )
        self.assertEqual(cancelled["status"], "cancel_requested")

    def test_mutations_require_token_and_local_origin(self):
        for headers in (
            {},
            {"X-Lab-Token": self.server.token, "Origin": "https://another.example"},
        ):
            with self.assertRaises(HTTPError) as cm:
                self.request("/api/jobs", {"config": SMALL.as_dict()}, headers)
            self.assertEqual(cm.exception.code, 403)
        data = json.loads(
            self.request(
                "/api/plan",
                {"config": SMALL.as_dict()},
                {"X-Lab-Token": self.server.token},
            )
        )
        self.assertEqual(data["count"], 1)

    def test_repository_files_and_unknown_artifacts_are_not_served(self):
        for path in (
            "/../sno_mdp/gp.py",
            "/sno_mdp/gp.py",
            "/artifact/unknown/../../report/report.md",
        ):
            with self.assertRaises(HTTPError) as cm:
                self.request(path)
            self.assertIn(cm.exception.code, (400, 404))

    def test_scientific_source_still_matches_frozen_protocol(self):
        self.assertTrue(Lab(ROOT, start_worker=False).frozen_status()["matches"])


if __name__ == "__main__":
    unittest.main()
