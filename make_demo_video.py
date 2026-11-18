# Copyright (c) 2026 Bertwin Kurisinkal Shine. All rights reserved. See LICENSE.
"""Render the demo video straight from a trained twin (no screen recording).

Usage: python make_demo_video.py twin.pkl [demo/cz_twin_demo.mp4] [--drift "TC3 under crystal"]

Sections: intro, what-if sweep, live replay of unseen Fluent cases (with an
optional drifting sensor), trust map, result card.
"""

from __future__ import annotations

import argparse
import pickle
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from twin.data import PARAMS, Dataset  # noqa: E402
from twin.estimator import Sensors, estimate  # noqa: E402
from twin.sensors import SENSOR_NOISE_K, SENSORS  # noqa: E402
from twin.surrogate import Surrogate  # noqa: E402
from twin.validity import check  # noqa: E402

FPS = 4
FOOTER = ("Prototype: steady, 2D axisymmetric, laminar Fluent reference cases; simulated sensors; "
          "not an industrial prediction.   © 2026 Bertwin Kurisinkal Shine. All rights reserved.")
BADGE = {"ok": ("Inside validated range", "#2e7d32"),
         "caution": ("Use with caution", "#f9a825"),
         "out": ("Outside model range: do not trust", "#c62828")}


class Video:
    def __init__(self, path: str):
        self.fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        self.writer = FFMpegWriter(fps=FPS, metadata={"title": "CZ melt digital twin demonstrator"})
        self.writer.setup(self.fig, path, dpi=100)

    def frame(self, seconds: float = 1 / FPS) -> None:
        self.fig.text(0.5, 0.008, FOOTER, ha="center", fontsize=8, color="0.35")
        for _ in range(max(1, round(seconds * FPS))):
            self.writer.grab_frame()
        self.fig.clear()

    def card(self, title: str, body: str, seconds: float) -> None:
        self.fig.text(0.5, 0.64, title, ha="center", fontsize=22, weight="bold")
        self.fig.text(0.5, 0.42, body, ha="center", va="center", fontsize=13.5, linespacing=1.7)
        self.frame(seconds)

    def finish(self) -> None:
        self.writer.finish()


def melt(ax, ds: Dataset, values, title, cmap, vlim=None, sensors=None):
    kw = {} if vlim is None else {"vmin": vlim[0], "vmax": vlim[1]}
    tc = ax.tricontourf(ds.r, ds.z, values, levels=30, cmap=cmap, **kw)
    if sensors is not None:
        ax.scatter(ds.r[sensors.idx], ds.z[sensors.idx], c="cyan", s=22, marker="s", edgecolors="k")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return tc


def what_if(v: Video, sur: Surrogate, ds: Dataset) -> None:
    lo, hi = sur.lo, sur.hi
    t_mid = 0.5 * (lo[2] + hi[2])
    path = [np.array([8.0, c, t_mid]) for c in np.linspace(hi[1], lo[1], 24)]
    if len(ds.excluded):
        bad = ds.excluded[PARAMS].to_numpy(float)
        target = bad[np.argmin(np.linalg.norm(sur.normalise(bad) - sur.normalise(path[-1]), axis=1))]
        path += [path[-1] + (target - path[-1]) * s for s in np.linspace(0, 1, 12)[1:]]
    end = path[-1].copy()
    path += [np.array([c, end[1], end[2]]) for c in np.linspace(end[0], hi[0] + 0.5 * (hi[0] - lo[0]), 16)[1:]]

    T_all = ds.fields["T"]
    vlim = (T_all.min(), T_all.max())
    for x in path:
        t0 = time.perf_counter()
        T, _ = sur.predict(x, "T")
        ur, _ = sur.predict(x, "u_r")
        uz, _ = sur.predict(x, "u_z")
        ms = 1e3 * (time.perf_counter() - t0)
        verdict = check(x, sur, ds)

        fig = v.fig
        fig.suptitle("1 · What-if: change the furnace settings, get the whole melt instantly", fontsize=14)
        ax1, ax2 = fig.add_axes([0.03, 0.3, 0.4, 0.5]), fig.add_axes([0.53, 0.3, 0.4, 0.5])
        tc = melt(ax1, ds, T[0], "Melt temperature (K)", "inferno", vlim)
        fig.colorbar(tc, ax=ax1, fraction=0.03)
        tc = melt(ax2, ds, 1e3 * np.hypot(ur[0], uz[0]), "Melt flow speed (mm/s)", "viridis")
        fig.colorbar(tc, ax=ax2, fraction=0.03)
        fig.text(0.5, 0.86, f"crystal {x[0]:5.1f} rpm    crucible {x[1]:+5.1f} rpm    "
                 f"hot wall {x[2]:6.1f} K        answer in {ms:.0f} ms (a Fluent run takes minutes)",
                 ha="center", fontsize=11, family="monospace")
        label, colour = BADGE[verdict.level]
        fig.text(0.5, 0.2, label + ("" if not verdict.reasons else ": " + "; ".join(verdict.reasons)),
                 ha="center", fontsize=12, color="w", weight="bold",
                 bbox={"fc": colour, "ec": "none", "boxstyle": "round,pad=0.5"})
        fig.text(0.5, 0.11, "The twin also says when not to trust it: near settings where Fluent found no "
                 "steady solution, and outside the trained range.", ha="center", fontsize=10, color="0.25")
        v.frame(0.25 if verdict.level == "ok" else 0.5)


def replay(v: Video, sur: Surrogate, ds: Dataset, drift: str | None, max_steps: int) -> dict:
    test = ds.subset((ds.cases.split == "test").to_numpy())
    test = test.subset(((test.X >= sur.lo) & (test.X <= sur.hi)).all(axis=1))
    # Real wall-temperature changes first, so a later sensor drift is not confused with them.
    t_varies = (test.cases.t_hot != test.cases.t_hot.median()).to_numpy()
    test = test.subset(np.concatenate([np.flatnonzero(t_varies), np.flatnonzero(~t_varies)]))
    n = min(len(test.cases), max_steps)
    if n == 0:
        raise SystemExit("no in-range hold-out cases to replay")

    sensors = Sensors.at(ds.r, ds.z, SENSORS)
    T_all = test.fields["T"]
    # The twin must expect its own error at the sensors, or clean steps get flagged.
    model_err = np.concatenate([sur.predict(test.X[i], "T")[0][0, sensors.idx] - T_all[i, sensors.idx]
                                for i in range(n)])
    sigma = float(np.hypot(np.sqrt(np.mean(model_err**2)), SENSOR_NOISE_K))
    k_drift = sensors.names.index(drift) if drift else None
    drift_start = 2 * n // 3
    rng = np.random.default_rng(0)
    vlim = (T_all.min(), T_all.max())
    h = {"true": [], "est": [], "sd": [], "flag": []}

    for step in range(n):
        case = test.cases.iloc[step]
        readings = sensors.read(T_all[step], SENSOR_NOISE_K, rng)
        drifting = k_drift is not None and step >= drift_start
        if drifting:
            readings[k_drift] += 2.5 * (step - drift_start + 1)
        est = estimate(sur, sensors, readings, sigma,
                       known={"crystal_rpm": case.crystal_rpm, "crucible_rpm": case.crucible_rpm})
        T_est = sur.predict(np.array([case.crystal_rpm, case.crucible_rpm, est.params["t_hot"]]), "T")[0][0]
        h["true"].append(case.t_hot)
        h["est"].append(est.params["t_hot"])
        h["sd"].append(2 * est.std["t_hot"])
        h["flag"].append(est.rms_residual_k > 3 * sigma)

        fig = v.fig
        fig.suptitle(f"2 · Live twin: unseen Fluent case {step + 1}/{n}  "
                     f"(crystal {case.crystal_rpm:g} rpm, crucible {case.crucible_rpm:g} rpm, wall T hidden)",
                     fontsize=13)
        axes = [fig.add_axes([0.03 + 0.325 * i, 0.55, 0.3, 0.33]) for i in range(3)]
        melt(axes[0], ds, T_all[step], "Fluent result (hidden from the twin)", "inferno", vlim, sensors)
        melt(axes[1], ds, T_est, "Twin's rebuild from the 4 cyan sensors", "inferno", vlim, sensors)
        melt(axes[2], ds, T_est - T_all[step], "Difference (K)", "coolwarm", None, sensors)
        axes[2].text(0.02, 0.04, f"max |diff| {np.abs(T_est - T_all[step]).max():.1f} K",
                     transform=axes[2].transAxes, fontsize=9, bbox={"fc": "w", "alpha": 0.85})
        fig.text(0.5, 0.5, "Sensor TC3 is now drifting (simulated fault)" if drifting else
                 "Each second: a new furnace state. The twin only sees 4 noisy sensor readings.",
                 ha="center", fontsize=11, color="#c62828" if drifting else "0.2",
                 weight="bold" if drifting else "normal")

        ax = fig.add_axes([0.07, 0.1, 0.9, 0.34])
        s = np.arange(step + 1)
        ax.plot(s, h["true"], "k-o", ms=4, label="true wall temperature (hidden)")
        ax.errorbar(s, h["est"], yerr=h["sd"], fmt="s", ms=4, color="tab:orange", capsize=3,
                    label="twin's estimate ±2σ")
        f = np.array(h["flag"])
        ax.scatter(s[f], np.array(h["est"])[f], s=170, facecolors="none", edgecolors="red", lw=2,
                   label="flagged: sensors disagree with the model")
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xlabel("furnace state")
        ax.set_ylabel("wall T (K)")
        ax.legend(fontsize=8, loc="upper right")
        v.frame(1.0)

    return {"n": n, "sigma": sigma, "drift_start": drift_start if k_drift is not None else None, **h,
            "case_ids": set(test.cases.case_id[:n])}


def trust_map(v: Video, ds: Dataset, holdout) -> None:
    fig = v.fig
    fig.suptitle("3 · Where can you trust it?", fontsize=14)
    pairs = [(0, 1), (0, 2), (1, 2)]
    names = {"crystal_rpm": "crystal (rpm)", "crucible_rpm": "crucible (rpm)", "t_hot": "hot wall (K)"}
    m = ds.cases
    for k, (a, b) in enumerate(pairs):
        ax = fig.add_axes([0.06 + 0.31 * k, 0.55, 0.26, 0.33])
        for lab, mask, c, mk in (("training (Fluent)", m.split == "train", "tab:blue", "o"),
                                 ("hold-out (unseen)", m.split == "test", "tab:orange", "s")):
            ax.scatter(ds.X[mask, a], ds.X[mask, b], s=10, c=c, marker=mk, label=lab)
        if len(ds.excluded):
            ex = ds.excluded[PARAMS].to_numpy(float)
            ax.scatter(ex[:, a], ex[:, b], s=45, c="red", marker="x", label="no steady solution")
        ax.set_xlabel(names[PARAMS[a]], fontsize=9)
        ax.set_ylabel(names[PARAMS[b]], fontsize=9)
        if k == 0:
            ax.legend(fontsize=7, loc="lower right")

    if holdout is not None:
        r = holdout.sort_values("T_maxabs_K").reset_index(drop=True)
        ood = r.case_id.str.startswith("ood").to_numpy()
        ax = fig.add_axes([0.08, 0.1, 0.88, 0.3])
        ax.bar(np.arange(len(r)), r.T_maxabs_K, color=np.where(ood, "#c62828", "#2e7d32"))
        ax.set_ylabel("worst-point T error (K)")
        ax.set_xticks([])
        ax.set_xlabel("each bar = one unseen Fluent case   ·   green: inside the trained range   "
                      "·   red: outside it", fontsize=9)
    v.frame(8.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("twin")
    ap.add_argument("out", nargs="?", default="demo/cz_twin_demo.mp4")
    ap.add_argument("--drift", help="sensor that drifts in the last third of the replay")
    ap.add_argument("--max-steps", type=int, default=30)
    a = ap.parse_args()

    with open(a.twin, "rb") as fh:
        b = pickle.load(fh)
    sur, ds, holdout = b["surrogate"], b["dataset"], b["holdout"]
    n_train = int((ds.cases.split == "train").sum())

    v = Video(a.out)
    v.card("Digital twin of a silicon crystal-growth melt",
           f"Czochralski process · trained on {n_train} ANSYS Fluent simulations\n\n"
           "1  What-if: new furnace settings → full melt field in milliseconds\n"
           "2  Live twin: from 4 temperature sensors, recover the hidden wall temperature\n"
           "    and catch a faulty sensor\n"
           "3  Trust map: where the twin is valid, and where it is not", 7)
    what_if(v, sur, ds)
    rep = replay(v, sur, ds, a.drift, a.max_steps)
    trust_map(v, ds, holdout)

    err = np.abs(np.array(rep["est"]) - np.array(rep["true"]))
    lines = [f"Hidden wall temperature recovered from 4 sensors: median error {np.median(err):.1f} K "
             f"over {rep['n']} unseen Fluent cases."]
    if holdout is not None:
        inside = holdout[holdout.case_id.isin(rep["case_ids"])]
        outside = holdout[holdout.case_id.str.startswith("ood")]
        lines.append(f"Whole melt field, worst point per case: median {inside.T_maxabs_K.median():.1f} K "
                     f"inside the trained range (worst {inside.T_maxabs_K.max():.1f} K),")
        lines.append(f"rising to {outside.T_maxabs_K.max():.0f} K outside it, which the twin flags.")
    if rep["drift_start"] is not None:
        ds_ = rep["drift_start"]
        lines.append(f"Drifting sensor flagged in {sum(rep['flag'][ds_:])} of {rep['n'] - ds_} steps; "
                     f"false alarms on clean steps: {sum(rep['flag'][:ds_])}.")
    lines.append("\nScope: steady, 2D, laminar reference simulations and simulated sensors.\n"
                 "A method demonstrator, not a validated furnace model.")
    v.card("Result", "\n".join(lines), 9)
    v.finish()
    print(f"wrote {a.out}  (replay: {rep['n']} cases, expected sensor error {rep['sigma']:.2f} K)")


if __name__ == "__main__":
    main()
