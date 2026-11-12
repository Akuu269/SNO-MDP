"""Independent coverage arithmetic, certificate provenance, and replay integrity."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sno_mdp.artifacts import read_json, save_json
from sno_mdp.benchmark import run_benchmark
from sno_mdp.config import Config
from sno_mdp.diagnostics import (
    SafetyAudit,
    diagnose_run,
    diagnose_study,
    interval_statistics,
)
from sno_mdp.environment import Grid, WorldMap
from sno_mdp.experiment import run_experiment
from sno_mdp.gp import Prediction
from sno_mdp.transition_sets import SafetyTracker


def config():
    return replace(
        Config(),
        rows=6,
        cols=6,
        map_source="generated",
        map_seed=3,
        horizon=5,
        discount=0.5,
    )


def inputs(directory):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.iterdir()
        if p.is_file()
    }


class CoverageTests(unittest.TestCase):
    def test_hand_calculated_intervals_and_zero_variance(self):
        stats = interval_statistics([0, 1, 1], [1, 0, 0.1], [-3, 1, 1.3], 2)
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["covered"], 1)
        self.assertEqual(stats["below_lower"], 1)
        self.assertEqual(stats["above_upper"], 1)
        self.assertEqual(stats["zero_std_count"], 1)
        self.assertEqual(stats["zero_std_misses"], 0)
        self.assertAlmostEqual(stats["max_abs_standardized_error"], 3)
        stats = interval_statistics([1], [0], [2], 2)
        self.assertEqual(stats["zero_std_misses"], 1)
        self.assertIsNone(stats["max_abs_standardized_error"])
        self.assertEqual(stats["coverage"], 0)

    def test_empty_subset_is_not_perfect_coverage(self):
        stats = interval_statistics([], [], [], 2)
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["coverage"])

    def test_invalid_inputs_are_rejected(self):
        for args in (
            ([0], [-1], [0], 2),
            ([0], [1], [np.nan], 2),
            ([0], [1, 2], [0], 2),
            ([0], [1], [0], 0),
        ):
            with self.assertRaises(ValueError):
                interval_statistics(*args)


class CertificateTraceTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(config(), rows=2, cols=2)
        initial = np.zeros((4, 5), bool)
        initial[0, 0] = initial[1, 0] = initial[0, 1] = initial[1, 3] = True
        self.world = WorldMap(
            Grid(2, 2), np.ones((2, 2)), np.array([[0, 0.1], [1, 1.1]]), (0, 0), initial
        )
        self.audit = SafetyAudit(self.world, self.config)
        self.tracker = SafetyTracker(
            self.audit.graph, initial, np.array(self.world.grid.states(), float), -0.25
        )
        self.mean, self.std = np.zeros((4, 5)), np.full((4, 5), 10.0)
        self.std[:, 0] = 0
        # Incorrectly accepts 0 -> 2, while 2 -> 0 permits returnability.
        self.mean[0, 2], self.std[0, 2] = 0.1, 0.1
        self.mean[2, 4], self.std[2, 4] = 1, 0.1

    def record(self, step, action=None):
        snapshot = self.tracker.update(self.mean, self.std)
        self.audit.record(
            step,
            "safety_exploration",
            Prediction(self.mean, self.std),
            snapshot,
            (0, 0),
            action,
        )

    def test_false_current_bound_and_later_retained_certificate(self):
        self.record(0)
        self.mean[0, 2] = -1  # posterior corrects, but the certificate remains
        self.record(1, "right")
        action = self.audit.executions[0]
        self.assertTrue(action["unsafe"])
        self.assertEqual(action["support"], "retained_certificate")
        self.assertEqual(action["first_accepted_step"], 0)
        self.assertEqual(action["first_executable_step"], 0)
        self.assertEqual(action["certificate_age"], 1)
        self.assertEqual(action["execution_step"], 2)
        self.assertAlmostEqual(action["acceptance_lower"], -0.1)
        self.assertAlmostEqual(action["lower"], -1.2)
        self.assertAlmostEqual(action["true_g"], -1)
        rows = [r for r in self.audit.coverage if r["subset"] == "available_nonstay"]
        self.assertEqual([r["count"] for r in rows], [8, 8])  # no stays or boundaries
        self.assertEqual(rows[0]["false_current_lcb_supported"], 1)
        self.assertEqual(rows[1]["false_retained"], 1)
        self.assertEqual(self.audit.summary()["unsafe_retained_certificate"], 1)
        self.assertEqual(self.audit.history[0, 2]["first_executed_step"], 2)

    def test_current_lcb_support_and_counterfactual_coverage(self):
        original_mean, original_std = self.mean.copy(), self.std.copy()
        self.record(0, "right")
        self.assertEqual(self.audit.executions[0]["support"], "current_lcb")
        self.assertEqual(self.audit.summary()["unsafe_current_lcb_supported"], 1)
        rows = [
            r
            for r in self.audit.calibration_rows()
            if r["subset"] == "available_nonstay"
        ]
        self.assertEqual(
            [r["covered"] for r in rows], sorted(r["covered"] for r in rows)
        )
        self.assertEqual(
            [r["current_lcb_passes"] for r in rows],
            sorted((r["current_lcb_passes"] for r in rows), reverse=True),
        )
        np.testing.assert_array_equal(self.mean, original_mean)
        np.testing.assert_array_equal(self.std, original_std)

    def test_initial_certificate_is_not_a_posterior_acceptance(self):
        self.record(0, "up")
        row = self.audit.executions[0]
        self.assertEqual(row["support"], "initial_certificate")
        self.assertIsNone(row["first_accepted_step"])
        self.assertIsNone(row["acceptance_lower"])
        self.assertIsNone(row["certificate_age"])


class ReplayAuditTests(unittest.TestCase):
    def test_original_artifacts_unchanged_and_analysis_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            original = run_experiment(config(), run, "sno_es2")
            before = inputs(run)
            first = diagnose_run(run, plots=False)
            output_hashes = inputs(run / "safety")
            second = diagnose_run(run, plots=False)
            self.assertEqual(inputs(run), before)
            self.assertEqual(first, second)
            self.assertEqual(inputs(run / "safety"), output_hashes)
            self.assertEqual(first["unsafe_actions"], original["unsafe_actions"])
            self.assertEqual(first["analyzed_frames"], config().horizon + 1)
            self.assertEqual(first["executed_actions"], config().horizon)
            self.assertTrue((run / "safety/index.html").exists())

    def test_changed_model_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            run_experiment(config(), run)
            manifest = read_json(run / "manifest.json")
            manifest["source_sha256"]["gp.py"] = "different"
            save_json(run / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "model source differs"):
                diagnose_run(run, plots=False)

    def test_saved_certificate_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            run_experiment(config(), run)
            frames = read_json(run / "trajectory.json.gz")
            frames[0]["certified_actions"][0][0] = not frames[0]["certified_actions"][
                0
            ][0]
            save_json(run / "trajectory.json.gz", frames, compact=True)
            with self.assertRaisesRegex(ValueError, "certified_actions differs"):
                diagnose_run(run, plots=False)

    def test_first_update_failure_is_retained_without_fabricated_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with patch(
                "sno_mdp.agent.SafetyExplorer.update",
                side_effect=ValueError("fixture failure"),
            ):
                result = run_experiment(config(), run)
                summary = diagnose_run(run, plots=False)
            self.assertEqual(result["stop_reason"], "model_failure")
            self.assertEqual(summary["status"], "partial_failed_update")
            self.assertEqual(summary["skipped_failed_update_steps"], [0])
            self.assertEqual(summary["executed_actions"], 0)
            self.assertIsNone(summary["coverage"]["available_nonstay"]["coverage"])

    def test_oracle_has_no_confidence_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            run_experiment(config(), run, "oracle")
            with self.assertRaisesRegex(ValueError, "Oracle"):
                diagnose_run(run, plots=False)

    def test_study_includes_every_learner_and_keeps_map_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / "base.json", config().as_dict())
            save_json(
                root / "suite.json",
                {
                    "base_config": "base.json",
                    "grids": [[6, 6]],
                    "map_seeds": [3, 4],
                    "observation_seeds": [2026],
                    "noise_factors": [1],
                    "horizon": 3,
                },
            )
            with patch("sno_mdp.benchmark_visualization.write_benchmark_visualization"):
                run_benchmark(root / "suite.json", root / "study")
            summary = diagnose_study(root / "study", plots=False)
            self.assertEqual(summary["learner_runs"], 6)
            self.assertEqual(len(summary["aggregates"]), 3)
            self.assertTrue(
                all(a["maps"] == 2 and a["runs"] == 2 for a in summary["aggregates"])
            )
            rows = json.loads((root / "study/safety/summary.json").read_text())
            self.assertEqual(rows, summary)

    def test_study_reports_no_measurements_when_all_updates_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / "base.json", config().as_dict())
            save_json(
                root / "suite.json",
                {
                    "base_config": "base.json",
                    "grids": [[6, 6]],
                    "map_seeds": [3],
                    "observation_seeds": [2026],
                    "noise_factors": [1],
                    "horizon": 3,
                },
            )
            with patch(
                "sno_mdp.agent.SafetyExplorer.update",
                side_effect=ValueError("fixture failure"),
            ):
                with patch(
                    "sno_mdp.benchmark_visualization.write_benchmark_visualization"
                ):
                    run_benchmark(root / "suite.json", root / "study")
                summary = diagnose_study(root / "study", plots=False)
            self.assertEqual(summary["learner_runs"], 3)
            for row in summary["aggregates"]:
                self.assertEqual(row["runs"], 1)
                self.assertEqual(row["available_coverage_measured_maps"], 0)
                self.assertEqual(row["available_coverage_measured_runs"], 0)
                self.assertIsNone(row["available_coverage_mean"])
