# Copyright (c) 2026 Bertwin Kurisinkal Shine. All rights reserved. See LICENSE.
"""Load the CZ dataset: a case list (all_cases.csv) plus one node-field CSV per case."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = ["crystal_rpm", "crucible_rpm", "t_hot"]
FIELDS = ["T", "u_r", "u_z", "u_swirl"]
_RENAME = {"pressure": "p"}


@dataclass
class Dataset:
    cases: pd.DataFrame
    r: np.ndarray
    z: np.ndarray
    fields: dict[str, np.ndarray]
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def X(self) -> np.ndarray:
        return self.cases[PARAMS].to_numpy(float)

    def subset(self, mask: np.ndarray) -> Dataset:
        return replace(
            self,
            cases=self.cases.iloc[mask].reset_index(drop=True),
            fields={k: v[mask] for k, v in self.fields.items()},
        )


def load(cases_path: str | Path, case_dir: str | Path) -> Dataset:
    """Read every usable case listed in the case list.

    Required columns: case_id, filename, crystal_rpm, crucible_rpm, t_hot.
    Optional: split ("train" / "test"), status ("ok" / "excluded").
    Excluded cases carry no field data; they are kept so the validity map can
    mark where no steady solution exists.
    """
    man = pd.read_csv(cases_path)
    if "split" not in man:
        man["split"] = "train"
    if "status" not in man:
        man["status"] = "ok"

    ok = man[man.status == "ok"].reset_index(drop=True)
    case_dir = Path(case_dir)
    r = z = None
    stacks: dict[str, list[np.ndarray]] = {f: [] for f in FIELDS}
    for fname in ok.filename:
        d = pd.read_csv(case_dir / fname).rename(columns=_RENAME)
        # Fluent export order can change between sessions, so fix it by coordinates.
        d = d.sort_values(["z", "r"], kind="stable").reset_index(drop=True)
        if r is None:
            r, z = d.r.to_numpy(), d.z.to_numpy()
        elif not (np.allclose(d.r, r) and np.allclose(d.z, z)):
            raise ValueError(f"{fname}: mesh differs from the first case")
        for f in FIELDS:
            stacks[f].append(d[f].to_numpy(float))

    return Dataset(
        cases=ok,
        r=r,
        z=z,
        fields={f: np.vstack(v) for f, v in stacks.items()},
        excluded=man[man.status != "ok"].reset_index(drop=True),
    )
