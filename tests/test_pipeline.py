# Copyright (c) 2026 Bertwin Kurisinkal Shine. All rights reserved. See LICENSE.
"""End-to-end check on synthetic fields. Run from the repo root: python tests/test_pipeline.py synthetic/twin.pkl"""

import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from twin.estimator import Sensors, estimate  # noqa: E402
from twin.sensors import SENSORS  # noqa: E402
from twin.validity import check  # noqa: E402


def main(pkl: str) -> None:
    b = pickle.load(open(pkl, "rb"))
    sur, ds = b["surrogate"], b["dataset"]
    test = ds.subset((ds.cases.split == "test").to_numpy())
    sensors = Sensors.at(ds.r, ds.z, SENSORS)
    rng = np.random.default_rng(0)

    for i, case in test.cases.iterrows():
        readings = sensors.read(test.fields["T"][i], noise_k=0.5, rng=rng)
        est = estimate(sur, sensors, readings, 0.5,
                       known={"crystal_rpm": case.crystal_rpm, "crucible_rpm": case.crucible_rpm})
        err = est.params["t_hot"] - case.t_hot
        print(f"{case.case_id}: true {case.t_hot:7.2f} K  est {est.params['t_hot']:7.2f} "
              f"± {est.std['t_hot']:.2f} K  (err {err:+.2f})")
        assert abs(err) < 4 * est.std["t_hot"] + 0.5, "estimate outside its own uncertainty"

    assert check([12, -5, 1760], sur, ds).level in ("ok", "caution")
    assert check([25, -5, 1760], sur, ds).level == "out"
    v = check([18, -9, 1780], sur, ds)
    assert any("steady" in r for r in v.reasons), v
    print("all checks passed")


if __name__ == "__main__":
    main(sys.argv[1])
