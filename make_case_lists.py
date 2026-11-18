# Copyright (c) 2026 Bertwin Kurisinkal Shine. All rights reserved. See LICENSE.
"""Write the twin's case lists from the dataset's metadata files.

Usage: python make_case_lists.py <dataset_folder> [out_folder]
Writes training_cases.csv, test_cases.csv and excluded_cases.txt to out_folder
(default: case_lists). Nothing is written into the dataset folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

COLS = ["case_id", "crystal_rpm", "crucible_rpm", "t_hot"]


def main(root: str, out: str = "case_lists") -> None:
    root, out = Path(root), Path(out)
    out.mkdir(exist_ok=True)

    # case_matrix / holdout_cases carry full-precision parameters; the run-status
    # files say which runs converged.
    doe = pd.read_csv(root / "case_matrix.csv")[COLS]
    doe_status = pd.read_csv(root / "doe_run_status.csv")[["case_id", "status"]]
    ho = pd.read_csv(root / "holdout_cases.csv")[COLS]
    ho_status = pd.read_csv(root / "holdout_run_status.csv")[["case_id", "status"]]

    doe = doe.merge(doe_status, on="case_id", validate="1:1")
    ho = ho.merge(ho_status, on="case_id", validate="1:1")
    doe["filename"] = "cz_v2_doe/doe/" + doe.case_id + ".csv"
    ho["filename"] = "cz_v2_holdout/holdout/" + ho.case_id + ".csv"

    doe.drop(columns="status").to_csv(out / "training_cases.csv", index=False)
    ho.drop(columns="status").to_csv(out / "test_cases.csv", index=False)
    excluded = pd.concat([doe, ho]).query("status != 'kept'").case_id
    (out / "excluded_cases.txt").write_text("\n".join(excluded) + "\n")

    print(f"DOE: {len(doe)} cases, {int((doe.status == 'kept').sum())} kept")
    print(f"hold-out: {len(ho)} cases, {int((ho.status == 'kept').sum())} kept")
    print(f"excluded (no steady solution): {len(excluded)}")
    print(f"wrote {out}/training_cases.csv, test_cases.csv, excluded_cases.txt")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:])
