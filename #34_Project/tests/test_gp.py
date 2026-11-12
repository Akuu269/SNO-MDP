"""Closed-form regression checks and fixed-data noise-comparison invariants."""

import math
import unittest

import numpy as np

from sno_mdp.gp import ExactGP


class GaussianProcessTests(unittest.TestCase):
    def test_prior_without_data(self):
        p = ExactGP(0.1, prior_mean=0.3, prior_variance=4).predict([[0], [9]])
        np.testing.assert_array_equal(p.mean, [0.3, 0.3])
        np.testing.assert_array_equal(p.std, [2, 2])

    def test_hand_calculated_single_observation(self):
        model = ExactGP(noise_std=0.5, length_scale=1)
        model.fit([[0]], [1])
        p = model.predict([[0], [math.sqrt(2 * math.log(2))]])
        np.testing.assert_allclose(p.mean, [0.8, 0.4], atol=1e-9)
        np.testing.assert_allclose(p.std**2, [0.2, 0.8], atol=1e-9)

    def test_standard_deviation_is_latent_not_measurement_uncertainty(self):
        p = ExactGP(1).fit([[0]], [1]).predict([[0]])
        self.assertAlmostEqual(p.std[0] ** 2, 0.5, places=8)
        self.assertNotAlmostEqual(p.std[0] ** 2, 1.5)

    def test_repeated_observations_match_precision_averaging_identity(self):
        raw = ExactGP(1).fit([[0], [0]], [1, 3]).predict([[0], [1]])
        averaged = ExactGP(1 / math.sqrt(2)).fit([[0]], [2]).predict([[0], [1]])
        np.testing.assert_allclose(raw.mean, averaged.mean, atol=1e-9)
        np.testing.assert_allclose(raw.std, averaged.std, atol=1e-9)

    def test_additional_data_reduces_latent_uncertainty(self):
        model = ExactGP(0.1)
        before = model.predict([[0], [1]])
        after = model.fit([[0]], [0.8]).predict([[0], [1]])
        self.assertTrue(np.all(after.std < before.std))

    def test_larger_assumed_noise_increases_posterior_uncertainty(self):
        small = ExactGP(0.05).fit([[0], [1]], [0.3, 0.7]).predict([[0], [0.5], [3]])
        large = ExactGP(0.2).fit([[0], [1]], [0.3, 0.7]).predict([[0], [0.5], [3]])
        self.assertTrue(np.all(large.std > small.std))

    def test_nonzero_prior_mean_and_zero_noise(self):
        p = ExactGP(0, prior_mean=2).fit([[0]], [2]).predict([[0], [1]])
        np.testing.assert_allclose(p.mean, [2, 2], atol=1e-9)
        self.assertLess(p.std[0], 1e-4)

    def test_invalid_inputs_fail_explicitly(self):
        for args in (
            {"noise_std": -1},
            {"noise_std": 0.1, "length_scale": 0},
            {"noise_std": float("nan")},
            {"noise_std": 0.1, "jitter": -1},
        ):
            with self.assertRaises(ValueError):
                ExactGP(**args)
        model = ExactGP(0.1)
        for x, y in (([], []), ([[0]], [1, 2]), ([[0]], [float("inf")])):
            with self.assertRaises(ValueError):
                model.fit(x, y)
        model.fit([[0]], [1])
        with self.assertRaises(ValueError):
            model.predict([[0, 1]])

    def test_fit_does_not_depend_on_later_mutation_of_training_arrays(self):
        x, y = np.array([[0.0]]), np.array([1.0])
        model = ExactGP(0.5).fit(x, y)
        x[0, 0], y[0] = 20, 100
        self.assertAlmostEqual(model.predict([[0]]).mean[0], 0.8, places=8)


if __name__ == "__main__":
    unittest.main()
