# Copyright (c) 2026 Bertwin Kurisinkal Shine. All rights reserved. See LICENSE.
"""Virtual sensors and state estimation: infer operating state from sensor readings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .data import PARAMS
from .surrogate import Surrogate


@dataclass
class Sensors:
    names: list[str]
    idx: np.ndarray

    @classmethod
    def at(cls, r: np.ndarray, z: np.ndarray, points: dict[str, tuple[float, float]]) -> Sensors:
        """Snap each named (r, z) location to the nearest mesh node."""
        idx = [int(np.argmin((r - pr) ** 2 + (z - pz) ** 2)) for pr, pz in points.values()]
        return cls(list(points), np.array(idx))

    def read(self, T_field: np.ndarray, noise_k: float, rng: np.random.Generator) -> np.ndarray:
        return T_field[self.idx] + rng.normal(0.0, noise_k, len(self.idx))


@dataclass
class Estimate:
    params: dict[str, float]
    std: dict[str, float]
    rms_residual_k: float
    at_bound: bool


def estimate(sur: Surrogate, sensors: Sensors, readings: np.ndarray, noise_k: float,
             known: dict[str, float]) -> Estimate:
    """Least-squares fit of the unknown parameters to the sensor temperatures.

    Rotation rates are normally known from the drive setpoints, so a typical call
    passes them in `known` and only the thermal state is estimated.
    """
    free = [p for p in PARAMS if p not in known]
    if not free:
        raise ValueError("nothing to estimate: all parameters are known")
    fi = [PARAMS.index(p) for p in free]
    base = np.array([known.get(p, 0.0) for p in PARAMS], float)
    lo, hi = sur.lo[fi], sur.hi[fi]

    def to_x(u: np.ndarray) -> np.ndarray:
        x = base.copy()
        x[fi] = lo + u * (hi - lo)
        return x

    def resid(u: np.ndarray) -> np.ndarray:
        T, _ = sur.predict(to_x(u), "T")
        return (T[0, sensors.idx] - readings) / noise_k

    best = None
    for start in np.linspace(0.15, 0.85, 4):
        sol = least_squares(resid, np.full(len(fi), start), bounds=(0.0, 1.0), diff_step=1e-3)
        if best is None or sol.cost < best.cost:
            best = sol

    J = best.jac * (1.0 / (hi - lo))  # back to physical units
    cov = np.linalg.pinv(J.T @ J)
    x = to_x(best.x)
    return Estimate(
        params={p: float(x[PARAMS.index(p)]) for p in free},
        std={p: float(np.sqrt(cov[i, i])) for i, p in enumerate(free)},
        rms_residual_k=float(np.sqrt(np.mean(best.fun**2)) * noise_k),
        at_bound=bool(np.any((best.x < 1e-3) | (best.x > 1 - 1e-3))),
    )
