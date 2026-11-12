"""Revalidation, exact repeated-reading conditioning, and frozen evaluation."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sno_mdp.agent import SafetyExplorer
from sno_mdp.artifacts import read_json, save_json
from sno_mdp.config import Config
from sno_mdp.environment import Grid
from sno_mdp.experiment import run_experiment
from sno_mdp.gp import ExactGP
from sno_mdp.graph import TransitionGraph
from sno_mdp.research import (
    freeze_research,
    run_research,
    verify_research,
    select_calibration,
)
from sno_mdp.submission import reproduce_run
from sno_mdp.transition_sets import SafetyTracker


def small_config():
    return replace(
        Config(),
        rows=6,
        cols=6,
        map_source="generated",
        map_seed=3,
        horizon=4,
        discount=0.5,
    )


class RevalidationTests(unittest.TestCase):
    def test_withdrawn_edge_removed_and_returnability_recomputed(self):
        grid = Grid(1, 3)
        graph = TransitionGraph.from_grid(grid)
        initial = np.zeros((3, 5), bool)
        initial[0, 0] = True
        trackers = [
            SafetyTracker(
                graph, initial, np.arange(3)[:, None], -0.25, certificate_mode=mode
            )
            for mode in ("retained", "current_lcb")
        ]
        mean, std = np.zeros((3, 5)), np.full((3, 5), 10.0)
        std[:, 0] = 0
        # All physical directions initially pass.
        std[0, 1] = std[1, 1] = std[1, 3] = std[2, 3] = 0.01
        for tracker in trackers:
            self.assertTrue(tracker.update(mean, std).pessimistic.all())
        mean[2, 3] = -1  # withdraw return edge, leaving state 2 a one-way trap
        retained, current = [t.update(mean, std) for t in trackers]
        self.assertTrue(retained.pessimistic[2])
        self.assertFalse(current.pessimistic[2])
        self.assertTrue(current.candidates[2, 3])  # historical provenance persists
        self.assertFalse(current.pessimistic_actions[1, 1])
        mean[2, 3] = 0  # current evidence can restore the edge
        self.assertTrue(trackers[1].update(mean, std).pessimistic[2])

    def test_region_loss_halts_motion_and_retains_prefix(self):
        original = SafetyExplorer.update

        def lose_region(learner, observation):
            snapshot = original(learner, observation)
            if observation.step == 1:
                snapshot = replace(
                    snapshot,
                    pessimistic=np.zeros_like(snapshot.pessimistic),
                    pessimistic_actions=np.zeros_like(snapshot.pessimistic_actions),
                )
                learner.tracker.snapshot = snapshot
            return snapshot

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(SafetyExplorer, "update", lose_region):
                result = run_experiment(
                    replace(small_config(), certificate_mode="current_lcb"), Path(tmp)
                )
            frames = read_json(Path(tmp) / "trajectory.json.gz")
            self.assertEqual(result["stop_reason"], "certificate_region_lost")
            self.assertEqual(result["physical_actions"], 1)
            self.assertIsNone(frames[-1]["next_action"])
            self.assertAlmostEqual(
                result["discounted_return"], frames[1]["true_reward"]
            )

    def test_sufficient_statistics_equal_independent_full_gaussian_solution(self):
        x = np.array([[0.0], [0.0], [0.5], [1.0], [1.0], [1.0]])
        y = np.array([1.0, 1.2, 0.5, -0.1, 0.2, 0.0])
        query = np.array([[0.2], [0.7]])
        noise, jitter = 0.03, 1e-10

        def kernel(a, b):
            return np.exp(-np.sum((a[:, None] - b[None, :]) ** 2, axis=2) / 8)

        covariance = kernel(x, x) + (noise**2 + jitter) * np.eye(len(x))
        cross = kernel(x, query)
        mean = cross.T @ np.linalg.solve(covariance, y)
        variance = 1 - np.sum(cross * np.linalg.solve(covariance, cross), axis=0)
        prediction = ExactGP(noise).fit(x, y).predict(query)
        np.testing.assert_allclose(prediction.mean, mean, atol=1e-10)
        np.testing.assert_allclose(prediction.std**2, variance, atol=1e-10)


class ProtocolTests(unittest.TestCase):
    def suite(self, root, stage="final"):
        save_json(root / "base.json", small_config().as_dict())
        path = root / "suite.json"
        save_json(
            path,
            {
                "stage": stage,
                "base_config": "base.json",
                "map_seeds": [3],
                "observation_seeds": [2026],
                "noise_factors": [1.0],
                "horizons": [4],
                "profiles": [
                    {
                        "name": "source",
                        "certificate_mode": "retained",
                        "multiplier": 2,
                        "methods": ["sno_es2"],
                    },
                    {
                        "name": "revalidated",
                        "certificate_mode": "current_lcb",
                        "multiplier": 3,
                        "methods": ["sno_es2"],
                    },
                ],
            },
        )
        return path

    def test_frozen_study_verifies_and_reproduces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = self.suite(root)
            protocol = root / "protocol.json"
            freeze_research(suite, protocol)
            with patch("sno_mdp.research_report.export_research"):
                result = run_research(suite, root / "study", protocol)
            self.assertEqual(result["completed_runs"], 3)
            self.assertEqual(verify_research(root / "study")["verified_runs"], 3)
            row = read_json(root / "study/runs.json")[2]
            self.assertEqual(
                reproduce_run(root / "study" / row["directory"])["result"],
                "Byte-identical algorithm artifacts.",
            )
            with self.assertRaises(ValueError):
                freeze_research(suite, protocol)

    def test_changed_settings_and_unfrozen_final_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = self.suite(root)
            protocol = root / "protocol.json"
            freeze_research(suite, protocol)
            with self.assertRaises(ValueError):
                run_research(suite, root / "study")
            spec = read_json(suite)
            spec["horizons"] = [5]
            save_json(suite, spec)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                run_research(suite, root / "study", protocol)

    def test_setup_failures_remain_in_all_planned_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = self.suite(root, "calibration")
            with patch(
                "sno_mdp.research.run_experiment",
                side_effect=ValueError("invalid initial region"),
            ):
                with patch("sno_mdp.research_report.export_research"):
                    result = run_research(suite, root / "study")
            self.assertEqual(result["completed_runs"], 3)
            self.assertEqual(result["failed_runs"], 3)
            self.assertEqual(
                verify_research(root / "study")["verified_physical_actions"], 0
            )
            selected = select_calibration(root / "study")
            self.assertEqual(selected["selected_multiplier"], 3)
