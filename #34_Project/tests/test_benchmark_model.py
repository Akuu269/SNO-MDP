"""Benchmark provenance and independent numerical/source equivalence checks."""

import ast
import hashlib
import json
import unittest
import zipfile
from dataclasses import replace

import numpy as np

from sno_mdp.config import Config
from sno_mdp.environment import ACTIONS, Action, Evaluator, Grid, RoverSession, WorldMap
from sno_mdp.gp import ExactGP
from sno_mdp.gp_safety_gym import ROOT, load_world
from sno_mdp.graph import TransitionGraph
from sno_mdp.planning import plan_reference
from sno_mdp.transition_sets import ConfidenceConflict, SafetyTracker


def reviewed_source_functions(names):
    """Compile ONLY the explicitly reviewed pure helpers, not upstream imports/drivers."""
    with zipfile.ZipFile(ROOT / "third_party/reference_sources.zip") as archive:
        source = archive.read("safemdp/safemdp/grid_world.py").decode()
    tree = ast.parse(source)
    subset = ast.Module(
        body=[
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        ],
        type_ignores=[],
    )
    namespace = {"np": np}
    exec(compile(subset, "reviewed-upstream-helpers", "exec"), namespace)
    return namespace


class BenchmarkSourceTests(unittest.TestCase):
    def test_pinned_release_arrays_are_used_without_normalization(self):
        config = Config()
        world = load_world(config)
        upstream = json.loads((ROOT / "third_party/upstream.json").read_text())
        path = ROOT / "data/gp_safety_gym/released_map.npz"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), upstream["map_sha256"]
        )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / "third_party/reference_sources.zip").read_bytes()
            ).hexdigest(),
            upstream["reference_archive_sha256"],
        )
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(world.altitude.ravel(), data["safety"])
            np.testing.assert_array_equal(world.reward.ravel(), data["reward"])
            self.assertEqual(
                np.ravel_multi_index(world.start, (20, 20)), int(data["start_pos"])
            )
        self.assertEqual(world.reward.min(), 0)
        self.assertGreater(world.reward.max(), 1)
        self.assertFalse(world.provenance["native_engine_executed"])

    def test_initial_star_equals_reviewed_upstream_helper(self):
        world = load_world(Config())
        namespace = reviewed_source_functions(
            {"compute_S_hat0", "dynamics_vec_ind", "reverse_action"}
        )
        original = namespace["compute_S_hat0"](
            148, (20, 20), 4, world.altitude.ravel(), (0.5, 0.5), -0.25
        )
        np.testing.assert_array_equal(original, world.initial_actions)

    def test_true_transition_masks_equal_reviewed_upstream_helper(self):
        world = load_world(Config())
        original = reviewed_source_functions({"compute_true_safe_set"})[
            "compute_true_safe_set"
        ]
        np.testing.assert_array_equal(
            original((20, 20), world.altitude.ravel(), -0.25),
            world.true_action_safety(),
        )

    def test_generator_is_repeatable_and_preserves_valid_seed(self):
        c = replace(Config(), rows=6, cols=6, map_source="generated", map_seed=3)
        first, second = load_world(c), load_world(c)
        np.testing.assert_array_equal(first.altitude, second.altitude)
        np.testing.assert_array_equal(first.reward, second.reward)
        np.testing.assert_array_equal(first.initial_actions, second.initial_actions)
        other = load_world(replace(c, map_seed=4))
        self.assertFalse(np.array_equal(first.altitude, other.altitude))
        self.assertTrue(np.all(~first.initial_actions | first.true_action_safety()))

    def test_invalid_release_shape_and_obsolete_parameters_fail(self):
        for args in (
            {"rows": 6},
            {"map_source": "unknown"},
            {"discount": 1},
            {"safety_threshold": 1},
        ):
            with self.assertRaises(ValueError):
                Config(**args)


class DynamicsAndSensorTests(unittest.TestCase):
    def setUp(self):
        self.world = load_world(
            replace(Config(), map_source="generated", rows=6, cols=6, map_seed=3)
        )

    def test_coordinate_action_convention_and_blocked_boundaries(self):
        g = Grid(3, 4)
        self.assertEqual(g.next_state((1, 1), Action.UP), (1, 2))
        self.assertEqual(g.next_state((1, 1), Action.RIGHT), (2, 1))
        self.assertEqual(g.next_state((1, 1), Action.DOWN), (1, 0))
        self.assertEqual(g.next_state((1, 1), Action.LEFT), (0, 1))
        self.assertFalse(g.action_mask()[0, 4])
        self.assertEqual(len(g.footprint((0, 0))), 3)

    def test_noisy_data_is_paired_by_location_and_occurrence(self):
        a = RoverSession(self.world, 42, 0.1, 0.2)
        b = RoverSession(self.world, 42, 0.1, 0.2)
        self.assertEqual(a.initial_observation, b.initial_observation)
        self.assertEqual(a.step(Action.STAY), b.step(Action.STAY))
        self.assertEqual(
            set(a.initial_observation.samples[0].__dict__),
            {"state", "reward", "altitude"},
        )
        self.assertFalse(hasattr(a, "altitude"))

    def test_zero_noise_and_no_clipping(self):
        a = RoverSession(self.world, 42, 0, 0)
        for s in a.initial_observation.samples:
            self.assertEqual(s.altitude, self.world.altitude[s.state])
        noisy = RoverSession(self.world, 42, 100, 100)
        self.assertTrue(
            any(abs(s.altitude) > 10 for s in noisy.initial_observation.samples)
        )

    def test_evaluator_scores_directed_transition_not_cell_height(self):
        e = Evaluator(self.world)
        start = self.world.start
        target = self.world.grid.neighbors(start)[0]
        a = e.evaluate(target, start)
        b = e.evaluate(start, target)
        self.assertAlmostEqual(
            a["true_transition_safety"], -b["true_transition_safety"]
        )
        self.assertEqual(e.evaluate(start, start)["true_transition_safety"], 0)

    def test_unsafe_action_is_recorded_without_an_oracle_block(self):
        initial = np.zeros((2, 5), bool)
        initial[0, 0] = True
        world = WorldMap(
            Grid(1, 2), np.array([[0.1, 0.2]]), np.array([[0.0, 1.0]]), (0, 0), initial
        )
        session = RoverSession(world, 0, 0, 0)
        previous = session.position
        observation = session.step(Action.UP)
        self.assertEqual(observation.position, (0, 1))
        result = Evaluator(world).evaluate(observation.position, previous)
        self.assertTrue(result["unsafe"])
        self.assertAlmostEqual(result["violation_severity"], 0.75)


class PosteriorDifferenceTests(unittest.TestCase):
    def test_difference_variance_matches_independent_full_covariance(self):
        query = np.array([[0.0], [0.5], [1.0]])
        training = np.array([[0.0], [1.0]])
        targets = np.array([0.3, -0.2])
        noise = 0.1
        model = ExactGP(noise, length_scale=1).fit(training, targets)
        transitions = np.array([[0, 1], [1, 2], [2, 1]])
        node, edge = model.predict_transitions(query, transitions)

        def kernel(a, b):
            return np.exp(-((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2) / 2)

        train = kernel(training, training) + np.eye(2) * (noise**2 + 1e-10)
        cross = kernel(training, query)
        covariance = kernel(query, query) - cross.T @ np.linalg.inv(train) @ cross
        mean = cross.T @ np.linalg.inv(train) @ targets
        expected = np.array(
            [
                [
                    covariance[i, i] + covariance[j, j] - 2 * covariance[i, j]
                    for j in row
                ]
                for i, row in enumerate(transitions)
            ]
        )
        np.testing.assert_allclose(node.mean, mean, atol=1e-10)
        np.testing.assert_allclose(edge.std**2, expected, atol=1e-10)
        np.testing.assert_allclose(
            edge.mean, mean[:, None] - mean[transitions], atol=1e-10
        )

    def test_stay_difference_has_zero_uncertainty(self):
        query = np.array([[0.0], [1.0]])
        _, edge = ExactGP(0.1).predict_transitions(query, np.array([[0], [1]]))
        np.testing.assert_array_equal(edge.mean, np.zeros((2, 1)))
        np.testing.assert_array_equal(edge.std, np.zeros((2, 1)))


class SafeGraphTests(unittest.TestCase):
    def test_one_way_trap_and_disconnected_island_are_excluded(self):
        graph = TransitionGraph([[0, 1], [1, 2], [2, 2]])
        nodes, actions = graph.executable(np.ones((3, 2), bool), [1, 0, 0])
        np.testing.assert_array_equal(nodes, [1, 0, 0])
        self.assertFalse(actions[0, 1])
        graph = TransitionGraph([[0, 1], [1, 0], [2, 2]])
        nodes, _ = graph.executable(np.ones((3, 2), bool), [1, 0, 0])
        np.testing.assert_array_equal(nodes, [1, 1, 0])

    def test_safe_path_cannot_use_an_excluded_direction(self):
        graph = TransitionGraph.from_grid(Grid(2, 2))
        allowed = graph.available.copy()
        allowed[0, 1] = False
        actions = graph.path(0, 1, allowed)
        self.assertEqual(len(actions), 3)
        with self.assertRaises(ValueError):
            graph.path(0, 1, np.zeros((4, 5), bool))

    def test_current_intervals_and_monotone_accepted_actions(self):
        graph = TransitionGraph.from_grid(Grid(1, 3))
        initial = np.zeros((3, 5), bool)
        initial[0, 0] = True
        tracker = SafetyTracker(graph, initial, np.array([[0.0], [1.0], [2.0]]), -0.25)
        first = tracker.update(np.zeros((3, 5)), np.full((3, 5), 0.01))
        second_mean = np.zeros((3, 5))
        second_mean[0, 1] = -1
        second = tracker.update(second_mean, np.full((3, 5), 0.01))
        self.assertTrue(first.pessimistic_actions[0, 1])
        self.assertTrue(second.pessimistic_actions[0, 1])
        self.assertLess(second.upper[0, 1], -0.25)

    def test_initial_certificate_contradiction_is_not_silently_reset(self):
        graph = TransitionGraph.from_grid(Grid(1, 2))
        initial = graph.available.copy()
        tracker = SafetyTracker(graph, initial, np.array([[0.0], [1.0]]), -0.25)
        with self.assertRaises(ConfidenceConflict):
            tracker.update(-np.ones((2, 5)), np.zeros((2, 5)))
        self.assertIsNone(tracker.snapshot)

    def test_reference_uses_only_true_executable_actions(self):
        world = load_world(Config())
        plan = plan_reference(world, discount=0.5)
        true = world.true_action_safety()
        for s, a in plan.policy.items():
            self.assertTrue(true[np.ravel_multi_index(s, (20, 20)), ACTIONS.index(a)])


if __name__ == "__main__":
    unittest.main()
