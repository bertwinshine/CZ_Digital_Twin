# Copyright (c) 2026 Bertwin Kurisinkal Shine. All rights reserved. See LICENSE.
"""Is the surrogate allowed to answer at this operating point?"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .data import PARAMS, Dataset
from .surrogate import Surrogate


@dataclass
class Verdict:
    level: str  # "ok", "caution" or "out"
    reasons: list[str]


@dataclass(frozen=True)
class _Reference:
    spacing: float     # typical distance between neighbouring training points (normalised)
    t_std_k: float     # typical max temperature uncertainty across the training set


_CACHE: dict[tuple[int, int], _Reference] = {}


def _reference(sur: Surrogate, ds: Dataset) -> _Reference:
    key = (id(sur), id(ds))
    if key not in _CACHE:
        _CACHE[key] = _build_reference(sur, ds)
    return _CACHE[key]


def _build_reference(sur: Surrogate, ds: Dataset) -> _Reference:
    train = sur.normalise(ds.X[(ds.cases.split == "train").to_numpy()])
    d = np.linalg.norm(train[:, None] - train[None], axis=2)
    np.fill_diagonal(d, np.inf)
    # Uncertainty is judged against the model's typical level between training points
    # (it collapses at the points themselves), so the limit follows the data.
    probe = np.random.default_rng(0).random((40, train.shape[1]))
    t_std = np.median(sur.models["T"].predict(probe)[1].max(axis=1))
    return _Reference(float(np.median(d.min(axis=1))), float(t_std))


def check(x: np.ndarray, sur: Surrogate, ds: Dataset, std_factor: float = 1.5) -> Verdict:
    x = np.asarray(x, float)
    outside = [p for p, v, lo, hi in zip(PARAMS, x, sur.lo, sur.hi) if not lo <= v <= hi]
    if outside:
        return Verdict("out", [f"outside the trained range ({', '.join(outside)})"])

    ref = _reference(sur, ds)
    reasons = []
    if len(ds.excluded):
        bad = sur.normalise(ds.excluded[PARAMS].to_numpy(float))
        if np.min(np.linalg.norm(bad - sur.normalise(x)[0], axis=1)) < 1.5 * ref.spacing:
            reasons.append("close to settings where Fluent found no steady solution")

    _, sd = sur.predict(x, "T")
    if sd.max() > std_factor * ref.t_std_k:
        reasons.append(f"model uncertainty high (T std up to {sd.max():.1f} K)")
    return Verdict("caution" if reasons else "ok", reasons)
