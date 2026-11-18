# Copyright (c) 2026 Bertwin Kurisinkal Shine. All rights reserved. See LICENSE.
"""Fit the twin on the train split, score it on the test split, save it to disk.

Usage: python build_twin.py <all_cases.csv> <dataset_folder> [out.pkl]
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

from twin.data import FIELDS, PARAMS, load
from twin.surrogate import Surrogate


def score(sur: Surrogate, test) -> pd.DataFrame:
    rows = []
    for i, case in test.cases.iterrows():
        row = {"case_id": case.case_id, **{p: case[p] for p in PARAMS}}
        for f in FIELDS:
            pred, _ = sur.predict(test.X[i], f)
            true = test.fields[f][i]
            row[f"{f}_relL2"] = np.linalg.norm(pred[0] - true) / np.linalg.norm(true)
        pred_T, _ = sur.predict(test.X[i], "T")
        row["T_maxabs_K"] = np.abs(pred_T[0] - test.fields["T"][i]).max()
        rows.append(row)
    return pd.DataFrame(rows)


def main(cases_csv: str, case_dir: str, out: str = "twin.pkl") -> None:
    ds = load(cases_csv, case_dir)
    is_train = (ds.cases.split == "train").to_numpy()
    train, test = ds.subset(is_train), ds.subset(~is_train)
    print(f"cases: {len(train.cases)} train, {len(test.cases)} test, "
          f"{len(ds.excluded)} excluded (no steady solution)")

    sur = Surrogate().fit(train)
    print(f"POD modes kept: {sur.n_modes()}")

    report = None
    if len(test.cases):
        report = score(sur, test)
        report.to_csv("holdout_report.csv", index=False)
        cols = [c for c in report if c.endswith("relL2")] + ["T_maxabs_K"]
        print("\nhold-out error (median / worst):")
        for c in cols:
            print(f"  {c:14s} {report[c].median():.3e} / {report[c].max():.3e}")

    with open(out, "wb") as fh:
        pickle.dump({"surrogate": sur, "dataset": ds, "holdout": report}, fh)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
