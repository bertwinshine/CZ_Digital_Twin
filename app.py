# Copyright (c) 2026 Bertwin Kurisinkal Shine. All rights reserved. See LICENSE.
"""Interactive dashboard for the CZ melt twin. Run: streamlit run app.py -- twin.pkl"""

from __future__ import annotations

import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from twin.data import PARAMS
from twin.estimator import Sensors, estimate
from twin.sensors import SENSORS
from twin.validity import check

LABELS = {"crystal_rpm": "Crystal rotation (rpm)", "crucible_rpm": "Crucible rotation (rpm)",
          "t_hot": "Hot-wall temperature (K)"}
BADGE = {"ok": "🟢 Inside validated range", "caution": "🟡 Use with caution", "out": "🔴 Outside model range"}


@st.cache_resource
def load_twin(path: str):
    with open(path, "rb") as fh:
        b = pickle.load(fh)
    ds = b["dataset"]
    return b["surrogate"], ds, b["holdout"], Sensors.at(ds.r, ds.z, SENSORS)


def field_plot(ds, values, title, cmap, sensors=None, unit=""):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    tc = ax.tricontourf(ds.r, ds.z, values, levels=30, cmap=cmap)
    fig.colorbar(tc, ax=ax, label=unit)
    if sensors is not None:
        ax.scatter(ds.r[sensors.idx], ds.z[sensors.idx], c="k", s=25, marker="s")
        for name, i in zip(sensors.names, sensors.idx):
            ax.annotate(name.split()[0], (ds.r[i], ds.z[i]), xytext=(3, 3),
                        textcoords="offset points", fontsize=7)
    ax.set_xlabel("r (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def kpis(sur, x):
    T, T_sd = sur.predict(x, "T")
    ur, _ = sur.predict(x, "u_r")
    uz, _ = sur.predict(x, "u_z")
    return T[0], T_sd[0], np.hypot(ur[0], uz[0])


def explore_tab(sur, ds, sensors):
    st.subheader("What-if: set the operating point, get the full melt field instantly")
    cols = st.columns(3)
    x = []
    for c, p, lo, hi in zip(cols, PARAMS, sur.lo, sur.hi):
        pad = 0.1 * (hi - lo)
        x.append(c.slider(LABELS[p], float(lo - pad), float(hi + pad), float((lo + hi) / 2), 0.1))
    x = np.array(x)

    v = check(x, sur, ds)
    st.markdown(f"**{BADGE[v.level]}**" + ("" if not v.reasons else " — " + "; ".join(v.reasons)))

    t0 = time.perf_counter()
    T, T_sd, umag = kpis(sur, x)
    dt = 1e3 * (time.perf_counter() - t0)

    m = st.columns(4)
    m[0].metric("Max melt T", f"{T.max():.1f} K")
    m[1].metric("Mean melt T", f"{T.mean():.1f} K")
    m[2].metric("Max meridional |u|", f"{1e3 * umag.max():.2f} mm/s")
    m[3].metric("Twin answer time", f"{dt:.0f} ms")

    a, b, c = st.columns(3)
    a.pyplot(field_plot(ds, T, "Temperature", "inferno", sensors, "K"))
    b.pyplot(field_plot(ds, 1e3 * umag, "Meridional speed", "viridis", unit="mm/s"))
    c.pyplot(field_plot(ds, T_sd, "Model uncertainty (T std)", "Greys", unit="K"))


def replay_tab(sur, ds, sensors):
    st.subheader("Live twin: the furnace sends sensor data, the twin works out the hidden state")
    st.caption("The 'furnace' replays hold-out CFD cases the twin never saw, one operating point "
               "per step (a sequence of steady states). Rotation setpoints are known, as they are "
               "from the drives; the hot-wall temperature is hidden and estimated from four "
               "thermocouples. An injected sensor drift starts in the last third.")
    test = ds.subset((ds.cases.split == "test").to_numpy())
    test = test.subset(((test.X >= sur.lo) & (test.X <= sur.hi)).all(axis=1))
    # Real wall-temperature changes first, so a later sensor drift is not confused with them.
    t_varies = (test.cases.t_hot != test.cases.t_hot.median()).to_numpy()
    test = test.subset(np.concatenate([np.flatnonzero(t_varies), np.flatnonzero(~t_varies)]))
    if not len(test.cases):
        st.warning("No hold-out cases in this dataset (the case list needs split = test).")
        return

    c = st.columns(4)
    noise = c[0].slider("Sensor noise σ (K)", 0.1, 3.0, 0.5, 0.1)
    fault_on = c[1].checkbox("Inject sensor drift")
    fault_sensor = c[2].selectbox("Drifting sensor", sensors.names, disabled=not fault_on)
    drift_rate = c[3].slider("Drift per step (K)", 0.5, 5.0, 2.5, 0.5, disabled=not fault_on)

    # Expect model error at the sensors as well as sensor noise, or clean steps get flagged.
    model_err = np.concatenate([
        sur.predict(test.X[i], "T")[0][0, sensors.idx] - test.fields["T"][i, sensors.idx]
        for i in range(len(test.cases))])
    sigma = float(np.hypot(np.sqrt(np.mean(model_err**2)), noise))
    st.caption(f"Twin expects ±{sigma:.2f} K at each sensor (model error + sensor noise).")

    if st.button("▶ Run replay"):
        rng = np.random.default_rng(0)
        k_fault = sensors.names.index(fault_sensor)
        drift_start = 2 * len(test.cases) // 3
        rows = []
        chart, table, figs = st.empty(), st.empty(), st.empty()
        for step, (i, case) in enumerate(test.cases.iterrows()):
            readings = sensors.read(test.fields["T"][i], noise, rng)
            if fault_on:
                if step >= drift_start:
                    readings[k_fault] += drift_rate * (step - drift_start + 1)
            known = {"crystal_rpm": case.crystal_rpm, "crucible_rpm": case.crucible_rpm}
            est = estimate(sur, sensors, readings, sigma, known)
            x_est = np.array([case.crystal_rpm, case.crucible_rpm, est.params["t_hot"]])
            T_est, _ = sur.predict(x_est, "T")
            field_err = np.abs(T_est[0] - test.fields["T"][i]).max()
            flag = "⚠ sensor/model mismatch" if est.rms_residual_k > 3 * sigma else (
                "ok (at range limit)" if est.at_bound else "ok")
            rows.append({"step": step, "true T_hot": case.t_hot, "twin T_hot": est.params["t_hot"],
                         "±2σ": 2 * est.std["t_hot"], "sensor residual (K)": est.rms_residual_k,
                         "max field error (K)": field_err, "status": flag})
            df = pd.DataFrame(rows)

            fig, ax = plt.subplots(figsize=(7, 2.6))
            ax.plot(df.step, df["true T_hot"], "k-o", ms=4, label="true (hidden CFD case)")
            ax.errorbar(df.step, df["twin T_hot"], yerr=df["±2σ"], fmt="s", ms=4, capsize=3,
                        color="tab:orange", label="twin estimate ±2σ")
            bad = df.status.str.startswith("⚠")
            ax.scatter(df.step[bad], df["twin T_hot"][bad], s=120, facecolors="none",
                       edgecolors="red", label="flagged")
            ax.set_xlabel("step")
            ax.set_ylabel("T_hot (K)")
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            chart.pyplot(fig)
            plt.close(fig)
            table.dataframe(df.style.format(precision=2), hide_index=True)

            with figs.container():
                a, b = st.columns(2)
                a.pyplot(field_plot(ds, T_est[0], f"Twin reconstruction, step {step}", "inferno",
                                    sensors, "K"))
                b.pyplot(field_plot(ds, T_est[0] - test.fields["T"][i], "Twin − CFD truth",
                                    "coolwarm", unit="K"))
            plt.close("all")
            time.sleep(0.6)


def validation_tab(sur, ds, holdout):
    st.subheader("How far can you trust it?")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    pairs = [(0, 1), (0, 2), (1, 2)]
    groups = [("train", ds.cases.split == "train", "tab:blue", "o"),
              ("hold-out", ds.cases.split == "test", "tab:orange", "s")]
    for ax, (a, b) in zip(axs, pairs):
        for name, mask, col, mk in groups:
            ax.scatter(ds.X[mask, a], ds.X[mask, b], s=12, c=col, marker=mk, label=name)
        if len(ds.excluded):
            ex = ds.excluded[PARAMS].to_numpy(float)
            ax.scatter(ex[:, a], ex[:, b], s=40, c="red", marker="x", label="no steady solution")
        ax.set_xlabel(LABELS[PARAMS[a]], fontsize=8)
        ax.set_ylabel(LABELS[PARAMS[b]], fontsize=8)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(f"POD modes kept per field: {sur.n_modes()}")
    if holdout is not None:
        st.markdown("**Hold-out cases (never used in training)**")
        st.dataframe(holdout.style.format(precision=4), hide_index=True)


def main(path: str) -> None:
    st.set_page_config("CZ melt digital twin — demonstrator", layout="wide")
    st.title("Czochralski silicon melt — digital twin demonstrator")
    st.info("Prototype. Built on steady, 2D axisymmetric, laminar ANSYS Fluent reference cases "
            "(CZ v2 DOE dataset). Sensors are simulated. Not a validated industrial prediction: "
            "no turbulence, no interface shape, no oxygen transport, no time dynamics.")
    sur, ds, holdout, sensors = load_twin(path)
    t1, t2, t3 = st.tabs(["What-if explorer", "Live twin replay", "Validation"])
    with t1:
        explore_tab(sur, ds, sensors)
    with t2:
        replay_tab(sur, ds, sensors)
    with t3:
        validation_tab(sur, ds, holdout)
    st.caption("© 2026 Bertwin Kurisinkal Shine. All rights reserved. Viewing and evaluation only; "
               "see LICENSE.")


main(sys.argv[1] if len(sys.argv) > 1 else "twin.pkl")
