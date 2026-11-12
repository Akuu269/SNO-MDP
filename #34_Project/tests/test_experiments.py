"""Planning, ES2, online phases, and reproducible matched evaluation."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sno_mdp.artifacts import read_json
from sno_mdp.benchmark import aggregate_rows, run_benchmark
from sno_mdp.config import Config
from sno_mdp.environment import ACTIONS, Grid, WorldMap
from sno_mdp.experiment import run_experiment
from sno_mdp.graph import TransitionGraph
from sno_mdp.planning import plan_reference
from sno_mdp.reward_planning import check_es2, solve_discounted


def small_config(**changes):
    return replace(
        Config(),
        map_source="generated",
        rows=6,
        cols=6,
        map_seed=3,
        horizon=8,
        discount=0.5,
        **changes,
    )


class PlanningTests(unittest.TestCase):
    def test_hand_solved_three_state_values_and_reference(self):
        graph = TransitionGraph.from_grid(Grid(1, 3))
        plan = solve_discounted(graph, [0.1, 0.2, 1], [1, 1, 1], discount=0.5)
        np.testing.assert_allclose(plan.values, [1.2, 2, 2], atol=1e-9)
        initial = np.zeros((3, 5), bool)
        initial[0, 0] = True
        world = WorldMap(
            Grid(1, 3), np.array([[0.1, 0.2, 1]]), np.zeros((1, 3)), (0, 0), initial
        )
        reference = plan_reference(world, discount=0.5)
        np.testing.assert_allclose(reference.values, [[1.2, 2, 2]], atol=1e-9)

    def test_forbidden_high_reward_and_negative_rewards(self):
        graph = TransitionGraph.from_grid(Grid(1, 3))
        allowed = graph.available.copy()
        allowed[1, 1] = False
        plan = solve_discounted(
            graph, [0.1, 0.2, 100], [1, 1, 0], discount=0.5, action_mask=allowed
        )
        self.assertAlmostEqual(plan.values[0], 0.4, places=8)
        plan = solve_discounted(graph, [-2, -1, -3], [1, 1, 1], discount=0.5)
        np.testing.assert_allclose(plan.values, [-2, -2, -2], atol=1e-9)

    def test_invalid_domain_and_nonconvergence_fail_explicitly(self):
        graph = TransitionGraph.from_grid(Grid(1, 2))
        with self.assertRaises(ValueError):
            solve_discounted(graph, [1, 1], [0, 0])
        with self.assertRaises(ValueError):
            solve_discounted(graph, [1, 1], [1, 1], discount=1)
        with self.assertRaises(RuntimeError):
            solve_discounted(graph, [1, 1], [1, 1], max_iterations=1)

    def test_es2_checks_all_certified_sources(self):
        graph = TransitionGraph.from_grid(Grid(1, 3))
        optimistic = graph.available.copy()
        pessimistic = optimistic.copy()
        pessimistic[2, :] = False
        pessimistic[1, 1] = False
        result = check_es2(
            graph, pessimistic, optimistic, [2, 0, 3], [0, 0, 0], discount=0.5
        )
        self.assertEqual(ACTIONS[result.plan.policy[0]].value, "stay")
        self.assertFalse(result.stop)
        self.assertTrue(result.successors[2])

    def test_es2_accepts_internal_policy_and_has_correct_action_rewards(self):
        graph = TransitionGraph.from_grid(Grid(1, 3))
        optimistic = graph.available.copy()
        pessimistic = optimistic.copy()
        pessimistic[2, :] = False
        pessimistic[1, 1] = False
        mean = np.array([1, 1, 0.1])
        std = np.full(3, 0.01)
        result = check_es2(
            graph, pessimistic, optimistic, mean, std, multiplier=2, discount=0.5
        )
        self.assertTrue(result.stop)
        expected = (
            mean[graph.transitions]
            + np.where(pessimistic, -2, 2) * std[graph.transitions]
        )
        np.testing.assert_allclose(result.auxiliary_reward, expected)


class ExperimentTests(unittest.TestCase):
    def test_online_reward_phase_and_certification_before_every_action(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            config = small_config(exploration_accuracy=10)
            summary = run_experiment(config, out, "sno_basic")
            frames = read_json(out / "trajectory.json.gz")
            raw = read_json(out / "observations.json.gz")
            self.assertEqual(summary["transition_step"], 0)
            self.assertEqual(summary["transition_reason"], "certified_action_accuracy")
            self.assertEqual(
                summary["phase_actions"]["reward_optimization"], config.horizon
            )
            self.assertNotEqual(frames[0]["reward_std"], frames[-1]["reward_std"])
            self.assertTrue(all(o["used_for_gp"] for o in raw))
            for a, b in zip(frames, frames[1:]):
                source = np.ravel_multi_index(a["position"], (config.rows, config.cols))
                action = [x.value for x in ACTIONS].index(b["action"])
                self.assertTrue(a["certified_actions"][source][action])
            self.assertAlmostEqual(
                summary["cumulative_reward"], sum(f["true_reward"] for f in frames[1:])
            )
            self.assertAlmostEqual(
                summary["discounted_return"],
                sum(
                    config.discount**i * f["true_reward"]
                    for i, f in enumerate(frames[1:])
                ),
            )
            self.assertTrue(
                all(
                    set(s) == {"state", "reward", "altitude"}
                    for o in raw
                    for s in o["samples"]
                )
            )

    def test_safemdp_keeps_sampling_without_a_reward_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(
                small_config(exploration_accuracy=10), Path(directory), "safemdp"
            )
            self.assertEqual(result["phase_actions"]["safety_exploration"], 8)
            self.assertEqual(result["phase_actions"]["reward_optimization"], 0)
            self.assertIsNone(result["transition_step"])
            self.assertTrue(result["budget_exhausted_during_exploration"])

    def test_saved_es2_successors_equal_stored_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            config = small_config()
            summary = run_experiment(config, out, "sno_es2")
            checks = read_json(out / "es2_checks.json.gz")
            frames = read_json(out / "trajectory.json.gz")
            graph = TransitionGraph.from_grid(Grid(6, 6))
            for c in checks:
                nodes = np.array(frames[c["step"]]["pessimistic"]).ravel()
                sources = np.flatnonzero(nodes)
                targets = graph.transitions[sources, np.array(c["policy"])[sources]]
                self.assertEqual(c["successor_indices"], sorted(set(targets.tolist())))
                self.assertEqual(
                    c["outside_source_indices"], sources[~nodes[targets]].tolist()
                )
                self.assertEqual(c["stop"], bool(nodes[targets].all()))
            self.assertEqual(summary["stop_reason"], "horizon_reached")

    def test_deterministic_artifacts_except_separate_runtime_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            first = run_experiment(small_config(), a)
            second = run_experiment(small_config(), b)
            self.assertEqual(first, second)
            for p in a.iterdir():
                if p.name != "timing.csv":
                    self.assertEqual(p.read_bytes(), (b / p.name).read_bytes(), p.name)

    def test_oracle_return_is_independent_of_sensor_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            first = run_experiment(small_config(), Path(directory) / "a", "oracle")
            second = run_experiment(
                replace(small_config(), safety_noise_std=1),
                Path(directory) / "b",
                "oracle",
            )
            self.assertEqual(first["discounted_return"], second["discounted_return"])
            self.assertEqual(first["unsafe_actions"], 0)


class BenchmarkTests(unittest.TestCase):
    def test_matched_suite_keeps_maps_references_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.json").write_text(json.dumps(small_config().as_dict()))
            spec = {
                "base_config": "base.json",
                "grids": [[6, 6]],
                "map_seeds": [3, 4],
                "observation_seeds": [2026],
                "noise_factors": [1],
                "horizon": 4,
            }
            (root / "suite.json").write_text(json.dumps(spec))
            with patch("sno_mdp.benchmark_visualization.write_benchmark_visualization"):
                result = run_benchmark(root / "suite.json", root / "out")
            self.assertEqual(result["run_count"], 8)
            rows = json.loads((root / "out/runs.json").read_text())
            for seed in spec["map_seeds"]:
                group = [r for r in rows if r["map_seed"] == seed]
                self.assertEqual(
                    len({r["reference_discounted_return"] for r in group}), 1
                )
                maps = [
                    json.loads(
                        ((root / "out" / r["replay"]).parent / "map.json").read_text()
                    )
                    for r in group
                ]
                self.assertTrue(
                    all(
                        m["altitude"] == maps[0]["altitude"]
                        and m["reward"] == maps[0]["reward"]
                        for m in maps
                    )
                )
            changed = [dict(r) for r in rows]
            bad = next(r for r in changed if r["method"] == "sno_es2")
            bad["failed"] = True
            bad["discounted_return"] = 0
            stats = next(g for g in aggregate_rows(changed) if g["method"] == "sno_es2")
            self.assertEqual(stats["failure_rate"], 0.5)

    def test_released_map_seed_changes_cannot_fake_independent_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.json").write_text(json.dumps(Config().as_dict()))
            spec = {
                "base_config": "base.json",
                "grids": [[20, 20]],
                "map_seeds": [1, 2],
                "observation_seeds": [2026],
                "noise_factors": [1],
                "horizon": 1,
            }
            (root / "suite.json").write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "independent released maps"):
                run_benchmark(root / "suite.json", root / "out")

    def test_configuration_rejects_legacy_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"exploration_actions":32}')
            with self.assertRaises(ValueError):
                Config.load(path)


if __name__ == "__main__":
    unittest.main()
