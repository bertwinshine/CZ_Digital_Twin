# Copyright (c) 2026 Bertwin Kurisinkal Shine. All rights reserved. See LICENSE.
"""TEST ONLY: write a fake dataset with a known answer on the real 8181-node mesh.

The fields are an analytic function of the parameters, not CFD. They exist only
to prove the pipeline can recover something it should recover.

Usage: python tests/make_synthetic.py samples/sweep_anchor.csv <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

R_CRU, Z_TOP = 0.30, 0.15


def fields(r, z, cry, cru, th):
    a, b = r / R_CRU, z / Z_TOP
    T = 1685 + (th - 1685) * (0.6 * a + 0.4 * (1 - b)) + 0.05 * cry * a * b * (1 - a)
    psi = np.sin(np.pi * a) * np.sin(np.pi * b)
    u_r = 1e-3 * (th - 1700) / 50 * psi * np.cos(np.pi * b) - 2e-4 * cry / 10 * psi
    u_z = -1e-3 * (th - 1700) / 50 * psi * np.cos(np.pi * a)
    u_sw = (cru * 2 * np.pi / 60) * r * (1 - b) + (cry * 2 * np.pi / 60) * r * b * (a < 0.6)
    return T, u_r, u_z, u_sw


def main(real_case: str, out: str) -> None:
    mesh = pd.read_csv(real_case)[["r", "z"]]
    out_dir = Path(out)
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    lo, hi = [4, -10, 1730], [20, -1, 1785]
    pts = qmc.scale(qmc.Sobol(3, seed=1).random(64), lo, hi)
    test = qmc.scale(qmc.Sobol(3, seed=7).random(8), lo, hi)
    rows = []
    for split, P in (("train", pts), ("test", test)):
        for cry, cru, th in P:
            cid = f"{split}_{len(rows):03d}"
            T, ur, uz, usw = fields(mesh.r, mesh.z, cry, cru, th)
            d = mesh.assign(u_r=ur, u_z=uz, u_swirl=usw, p=0.0, T=T)
            d.sample(frac=1, random_state=len(rows)).to_csv(out_dir / "cases" / f"{cid}.csv", index=False)
            rows.append(dict(case_id=cid, filename=f"cases/{cid}.csv", crystal_rpm=cry, crucible_rpm=cru,
                             t_hot=th, split=split, status="ok"))
    rows.append(dict(case_id="tr_000", filename="", crystal_rpm=18, crucible_rpm=-9, t_hot=1780,
                     split="train", status="excluded"))
    pd.DataFrame(rows).to_csv(out_dir / "all_cases.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
