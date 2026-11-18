# Copyright (c) 2026 Bertwin Kurisinkal Shine. All rights reserved. See LICENSE.
"""Merge the case lists into all_cases.csv and check every case file before training.

Usage:
  python check_cases.py --root <dataset_folder> --train <training_cases.csv>
                        [--test <test_cases.csv>] [--excluded <excluded_cases.txt>]
                        [--out <all_cases.csv>]

Each case list needs: case_id, filename, crystal_rpm, crucible_rpm, t_hot.
`filename` is relative to --root. The excluded file lists one case_id per line
(runs with no steady solution). Writes --out (default <root>/all_cases.csv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from twin.data import PARAMS

NEEDED = ["case_id", "filename", *PARAMS]
FIELD_COLS = {"r", "z", "u_r", "u_z", "u_swirl", "T"}


def read_case_list(path: Path, split: str) -> pd.DataFrame:
    m = pd.read_csv(path)
    missing = [c for c in NEEDED if c not in m]
    if missing:
        sys.exit(f"{path}: missing columns {missing}")
    return m[NEEDED].assign(split=split, status="ok")


def check_case(path: Path) -> str | None:
    if not path.exists():
        return "file not found"
    d = pd.read_csv(path)
    cols = set(d.columns) | ({"p"} if "pressure" in d else set())
    if not FIELD_COLS <= cols:
        return f"missing columns {sorted(FIELD_COLS - cols)}"
    if not np.isfinite(d[sorted(FIELD_COLS)].to_numpy()).all():
        return "NaN or inf in fields"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test", type=Path)
    ap.add_argument("--excluded", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    parts = [read_case_list(a.train, "train")]
    if a.test:
        parts.append(read_case_list(a.test, "test"))
    man = pd.concat(parts, ignore_index=True)

    if man.case_id.duplicated().any():
        sys.exit(f"duplicate case_id: {man.case_id[man.case_id.duplicated()].tolist()}")

    if a.excluded:
        ids = {s.strip() for s in a.excluded.read_text().splitlines() if s.strip()}
        unknown = ids - set(man.case_id)
        if unknown:
            sys.exit(f"excluded ids not in any case list: {sorted(unknown)}")
        man.loc[man.case_id.isin(ids), "status"] = "excluded"

    problems, n_nodes = [], set()
    for _, row in man[man.status == "ok"].iterrows():
        err = check_case(a.root / row.filename)
        if err:
            problems.append(f"  {row.case_id}: {err}")
        else:
            n_nodes.add(sum(1 for _ in open(a.root / row.filename)) - 1)
    if len(n_nodes) > 1:
        problems.append(f"  node counts differ between cases: {sorted(n_nodes)}")
    if problems:
        sys.exit("cases with problems (fix them or list them as excluded):\n" + "\n".join(problems))

    lo = man[man.split == "train"][PARAMS].min()
    hi = man[man.split == "train"][PARAMS].max()
    test = man[(man.split == "test") & (man.status == "ok")]
    inside = ((test[PARAMS] >= lo) & (test[PARAMS] <= hi)).all(axis=1)

    out = a.out or a.root / "all_cases.csv"
    man.to_csv(out, index=False)
    counts = man.groupby(["split", "status"]).size().to_dict()
    print(f"wrote {out}")
    print(f"  {counts}")
    print(f"  nodes per case: {n_nodes.pop() if n_nodes else '-'}")
    print(f"  usable hold-out cases inside training range: {int(inside.sum())}, outside (OOD): {int((~inside).sum())}")


if __name__ == "__main__":
    main()
