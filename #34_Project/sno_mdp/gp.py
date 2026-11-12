"""Exact Gaussian-process regression with a fixed RBF kernel and Gaussian noise.

The prediction standard deviation concerns the latent function, not a future
noisy sensor reading. Models accept coordinates and observations, never maps.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prediction:
    mean: np.ndarray
    std: np.ndarray


class ExactGP:
    def __init__(
        self,
        noise_std: float,
        length_scale: float = 2.0,
        prior_variance: float = 1.0,
        prior_mean: float = 0.0,
        jitter: float = 1e-10,
    ):
        for name, value in (("noise_std", noise_std), ("jitter", jitter)):
            if not np.isfinite(value) or value < 0:
                raise ValueError("%s must be finite and nonnegative" % name)
        for name, value in (
            ("length_scale", length_scale),
            ("prior_variance", prior_variance),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        if not np.isfinite(prior_mean):
            raise ValueError("prior_mean must be finite")
        self.noise_std = noise_std
        self.length_scale = length_scale
        self.prior_variance = prior_variance
        self.prior_mean = prior_mean
        self.jitter = jitter
        self._x = None
        self._chol = None
        self._weights = None

    @staticmethod
    def _inputs(x):
        x = np.array(x, dtype=float, copy=True)
        if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError(
                "Inputs must be a nonempty finite matrix (samples, features)"
            )
        return x

    def _kernel(self, a, b):
        distance_sq = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return self.prior_variance * np.exp(-distance_sq / (2 * self.length_scale**2))

    def fit(self, x, y):
        x = self._inputs(x)
        y = np.array(y, dtype=float, copy=True)
        if y.shape != (x.shape[0],) or not np.isfinite(y).all():
            raise ValueError("Targets must be a finite vector with one value per input")
        # Independent repeated readings at a coordinate have sufficient
        # statistics (count, mean). This is exact Gaussian conditioning,
        # including the original per-reading jitter, not sparse GP inference.
        x, inverse, counts = np.unique(
            x, axis=0, return_inverse=True, return_counts=True
        )
        y = np.bincount(inverse, weights=y) / counts
        covariance = self._kernel(x, x)
        covariance.flat[:: len(x) + 1] += (self.noise_std**2 + self.jitter) / counts
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "GP covariance is not positive definite; check noise and jitter"
            ) from exc
        weights = np.linalg.solve(chol.T, np.linalg.solve(chol, y - self.prior_mean))
        x.setflags(write=False)
        self._x, self._chol, self._weights = x, chol, weights
        return self

    def predict(self, x) -> Prediction:
        x = self._inputs(x)
        if self._x is None:
            mean = np.full(len(x), self.prior_mean, dtype=float)
            variance = np.full(len(x), self.prior_variance, dtype=float)
        else:
            if x.shape[1] != self._x.shape[1]:
                raise ValueError("Prediction inputs have the wrong feature dimension")
            cross_covariance = self._kernel(self._x, x)
            mean = self.prior_mean + cross_covariance.T @ self._weights
            projected = np.linalg.solve(self._chol, cross_covariance)
            variance = self.prior_variance - np.sum(projected**2, axis=0)
            if np.min(variance) < -1e-8 * self.prior_variance:
                raise ValueError(
                    "GP prediction produced a materially negative variance"
                )
            variance = np.maximum(variance, 0.0)  # floating-point roundoff only
        std = np.sqrt(variance)
        mean.setflags(write=False)
        std.setflags(write=False)
        return Prediction(mean, std)

    def predict_transitions(self, x, transitions):
        """Joint altitude and directed-difference posterior, using covariance.

        Var[z(s)-z(t)] = Var[z(s)]+Var[z(t)]-2 Cov[z(s),z(t)].
        Independent marginal standard deviations must not be added instead.
        """
        x = self._inputs(x)
        transitions = np.asarray(transitions)
        if (
            transitions.ndim != 2
            or len(transitions) != len(x)
            or not np.issubdtype(transitions.dtype, np.integer)
            or np.any(transitions < 0)
            or np.any(transitions >= len(x))
        ):
            raise ValueError("Invalid transition indices")
        if self._x is None:
            mean = np.full(len(x), self.prior_mean)
            variance = np.full(len(x), self.prior_variance)
            projected = np.zeros((0, len(x)))
        else:
            cross = self._kernel(self._x, x)
            projected = np.linalg.solve(self._chol, cross)
            mean = self.prior_mean + cross.T @ self._weights
            variance = self.prior_variance - np.sum(projected**2, axis=0)
        delta = x[:, None, :] - x[transitions]
        prior_diff = (
            2
            * self.prior_variance
            * (1 - np.exp(-np.sum(delta**2, axis=2) / (2 * self.length_scale**2)))
        )
        reduction = np.sum(
            (projected[:, :, None] - projected[:, transitions]) ** 2, axis=0
        )
        diff_variance = prior_diff - reduction
        if (
            min(float(variance.min()), float(diff_variance.min()))
            < -1e-8 * self.prior_variance
        ):
            raise ValueError("Materially negative posterior variance")
        node = Prediction(np.array(mean), np.sqrt(np.maximum(variance, 0)))
        edge = Prediction(
            mean[:, None] - mean[transitions], np.sqrt(np.maximum(diff_variance, 0))
        )
        for array in (node.mean, node.std, edge.mean, edge.std):
            array.setflags(write=False)
        return node, edge
