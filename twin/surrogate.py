# Copyright (c) 2026 Bertwin Kurisinkal Shine. All rights reserved. See LICENSE.
"""POD + Gaussian-process surrogate: operating parameters -> full melt fields."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .data import FIELDS, Dataset


class FieldModel:
    """One field: POD basis from snapshots, one GP per retained mode."""

    def __init__(self, energy: float = 0.9999, max_modes: int = 60):
        self.energy = energy
        self.max_modes = max_modes

    def fit(self, Xn: np.ndarray, Y: np.ndarray) -> FieldModel:
        self.mean = Y.mean(axis=0)
        U, s, Vt = np.linalg.svd(Y - self.mean, full_matrices=False)
        cum = np.cumsum(s**2) / np.sum(s**2)
        k = min(int(np.searchsorted(cum, self.energy)) + 1, self.max_modes, len(s))
        self.basis = Vt[:k]
        coeffs = U[:, :k] * s[:k]
        self.scale = coeffs.std(axis=0)
        self.gps = []
        kernel = (ConstantKernel(1.0) * Matern([0.5] * Xn.shape[1], nu=2.5)
                  + WhiteKernel(1e-6, (1e-10, 1e-1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for j in range(k):
                gp = GaussianProcessRegressor(kernel, normalize_y=False, n_restarts_optimizer=3,
                                              random_state=0)
                self.gps.append(gp.fit(Xn, coeffs[:, j] / self.scale[j]))
        return self

    def predict(self, Xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.empty((len(Xn), len(self.gps)))
        sd = np.empty_like(mu)
        for j, gp in enumerate(self.gps):
            with warnings.catch_warnings():
                # Round-off can give tiny negative variances at training points.
                warnings.filterwarnings("ignore", "Predicted variances smaller than 0")
                m, s = gp.predict(Xn, return_std=True)
            mu[:, j], sd[:, j] = m * self.scale[j], s * self.scale[j]
        field = self.mean + mu @ self.basis
        # Modes are orthonormal and treated as independent.
        std = np.sqrt((sd**2) @ (self.basis**2))
        return field, std


class Surrogate:
    def fit(self, ds: Dataset) -> Surrogate:
        X = ds.X
        self.lo, self.hi = X.min(axis=0), X.max(axis=0)
        Xn = self.normalise(X)
        self.models = {f: FieldModel().fit(Xn, ds.fields[f]) for f in FIELDS}
        return self

    def normalise(self, X: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(X) - self.lo) / (self.hi - self.lo)

    def predict(self, X: np.ndarray, field: str) -> tuple[np.ndarray, np.ndarray]:
        return self.models[field].predict(self.normalise(X))

    def n_modes(self) -> dict[str, int]:
        return {f: len(m.gps) for f, m in self.models.items()}
