# CZ melt digital twin — demonstrator

A small digital twin of the silicon melt in a Czochralski (CZ) crystal puller, built on
254 steady ANSYS Fluent simulations. It shows the full twin workflow on a real CFD dataset:
a fast surrogate model, virtual temperature sensors, estimation of a hidden furnace state,
sensor-fault detection, and an honest map of where the model can be trusted.

![Demo preview](demo/preview.gif)

**Full demo video (67 s):** [`demo/cz_twin_demo.mp4`](demo/cz_twin_demo.mp4)

> **Prototype.** Steady, 2D axisymmetric, laminar reference simulations and simulated sensors.
> This is a method demonstrator, not a validated model of a real furnace. See [Scope and limitations](#scope-and-limitations)
> and [Towards industrial use](#towards-industrial-use).
>
> © 2026 Bertwin Kurisinkal Shine. All rights reserved. Viewing only — see [Licence](#licence).

## What it does

1. **What-if.** Set crystal rotation, crucible rotation and hot-wall temperature. The twin
   returns the full temperature and velocity field of the melt in about 10–20 ms, where a
   Fluent run takes minutes.
2. **Live twin.** Hold-out Fluent cases the twin has never seen play the role of the furnace.
   Four simulated thermocouples send noisy readings. Rotation rates are known (as they are
   from the drives on a real puller); the hot-wall temperature is hidden. The twin estimates
   it, rebuilds the whole melt field and compares it with the Fluent result. A drifting
   sensor can be injected, and the twin flags it when the readings stop agreeing with the model.
3. **Trust map.** Every answer is labelled: inside the validated range, *use with caution*
   (close to settings where Fluent found no steady solution, or high model uncertainty), or
   *outside the trained range*.

## Results

Trained on 254 DOE cases (Sobol design plus corners and centre). Tested on 38 hold-out cases
that were never used in training: 26 inside the trained parameter range and 12 outside it.

| Check | Result |
|---|---|
| Hidden wall temperature recovered from 4 sensors (26 in-range cases) | median error **0.5 K** |
| Whole temperature field, worst node per case, inside the range | median **3.4 K**, worst 9.9 K |
| Same, outside the range | median 10.8 K, worst 33 K (flagged as *outside*) |
| Velocity fields inside the range (relative L2 error, median) | u_r 3.8 %, u_z 4.8 %, swirl 0.4 % |
| Drifting sensor (2.5 K per step) | flagged in 8 of 9 steps, **0 false alarms** on clean steps |
| Answer time for one full field | about 10–20 ms on a laptop |

For context: the melt temperature spans about 1685–1785 K, so a 3.4 K worst-node error is
about 3–4 % of that range. The errors are largest in the circulation cell under the crystal.
Raising the number of POD modes did not reduce them; the limit is the number of training cases.

Per-case numbers: [`results/holdout_report.csv`](results/holdout_report.csv).

## How it works

```
Fluent cases ──► POD (one basis per field) ──► one Gaussian process per mode
                                                     │
operating point (crystal, crucible, T_wall) ────────►├──► full T, u_r, u_z, u_swirl fields + uncertainty
                                                     │
4 noisy sensor readings + known rotations ──► least-squares fit ──► estimated T_wall ± σ
                                                     │                 └─► residual check → sensor fault flag
                                                     └──► validity check → ok / caution / out
```

- **Surrogate** (`twin/surrogate.py`): proper orthogonal decomposition (POD) of each field,
  keeping 99.99 % of the energy (22 modes for temperature), with one Gaussian process
  (Matérn 5/2 kernel) per mode.
- **State estimation** (`twin/estimator.py`): finds the hot-wall temperature that best
  explains the four sensor readings. The expected sensor error combines sensor noise
  (0.5 K) with the model's own error at the sensor positions (0.86 K in total), so normal
  model error is not mistaken for a fault.
- **Validity check** (`twin/validity.py`): outside the training box, close to a DOE point
  that had no steady solution, or model uncertainty well above its typical level.

## Scope and limitations

**In scope:** demonstrating the digital-twin workflow — surrogate, virtual sensors, state
estimation, fault flagging and validity checking — on a controlled, well-documented CFD
dataset, with honest hold-out testing.

**Not in scope, and not claimed:**

- **Physics of the reference cases.** The CFD cases are steady, 2D axisymmetric and laminar.
  The rotational Reynolds numbers in this range are at or above the laminar–turbulent
  transition, so a real industrial melt is turbulent, 3D and time-dependent. The cases are
  controlled reference solutions for method development, not predictions of a real puller.
- **Missing physics.** No crystal–melt interface shape, no oxygen transport, no Marangoni
  flow, no radiation model, no magnetic field.
- **No steady solution in part of the range.** At high hot-wall temperature and low rotation,
  Fluent found no steady solution (16 cases excluded). The twin cannot answer there and
  marks it as *use with caution*.
- **Simulated sensors.** Sensor readings are Fluent values plus Gaussian noise. Nothing has
  been compared with measurements from a real furnace.
- **No time dynamics.** The live replay is a sequence of steady states. The twin cannot
  forecast how fast the melt responds to a change.
- **One hidden variable.** Only the hot-wall temperature is estimated; rotation rates are
  assumed known.
- **Accuracy limited by data.** With 254 training cases the typical worst-node temperature
  error is a few kelvin. Outside the trained range the error grows quickly (up to 33 K).

## Towards industrial use

The demonstrator separates two things: the **twin workflow** (fast surrogate, virtual sensors,
state estimation, fault detection, validity checking) and the **reference physics** it is
trained on. The workflow is general. The reference physics here is deliberately simple.

### Would it work with real sensor data?

The estimation and fault-detection steps accept real thermocouple readings without change.
What does not transfer directly is the reference model: it was trained on laminar, 2D, steady
melt simulations, and a production puller is turbulent, 3D, time-dependent and coupled to the
whole hot zone. Fed with real readings today, the twin would report a sensor–model mismatch
rather than a reliable estimate. Making it work on a real furnace means rebuilding the
reference model for that furnace and calibrating it against that furnace's own data, which
is what the steps below set out.

### Where a furnace twin can add value

- **Virtual sensing:** estimating melt and hot-zone temperatures that cannot be measured
  directly, from the thermocouples and heater data already logged.
- **Sensor health:** detecting a drifting or failed thermocouple from its disagreement with
  the model and the other sensors, before it affects the process.
- **Recipe what-if:** screening changes to rotation rates and heater settings in seconds
  before committing furnace time to a trial run.
- **Operating window:** mapping where the process is stable and where it is not, as the
  demonstrator does for the no-steady-solution region.
- **Training and documentation:** a fast, visual model of the melt for process engineers
  and operators.

### Path from demonstrator to a furnace-specific twin

| Stage | Work | Needed from an industrial partner |
|---|---|---|
| 1. Scoping | Choose one puller type and the questions the twin should answer; select the sensors and process data to use | Process engineers' time; list of logged signals |
| 2. Reference model | Global heat-transfer model of the furnace (radiation, heaters, hot zone) coupled to a turbulent melt model; mesh and model studies against published benchmarks | Furnace geometry, material data, typical recipes (under NDA) |
| 3. Simulation campaign | Design-of-experiments CFD runs over the real operating range, with convergence and mesh checks, as done for this dataset | Agreement on parameter ranges |
| 4. Calibration | Tune uncertain model inputs (heater efficiency, contact resistances, emissivities) so the model matches logged thermocouple data | Historical furnace logs (temperatures, heater power, rotations) |
| 5. Dynamics | Add a reduced-order transient model so the twin can follow and forecast changes, not only steady states | Logs from setpoint changes |
| 6. Shadow-mode pilot | Run the twin alongside one furnace, read-only, and compare its estimates and alarms with what operators see | Data connection to one furnace; operator feedback |
| 7. Integration | Live data link, dashboard and alerting within the plant's IT and security rules | IT and automation contacts |

Each stage has a clear check before the next one starts: hold-out testing for the surrogate
(as in this repository), agreement with logged data for the calibrated model, and comparison
with real events for the pilot.

### What carries over from this demonstrator

- A tested pipeline from CFD cases to a surrogate model with hold-out validation.
- State estimation that accounts for the model's own error, so normal model error is not
  reported as a fault.
- A validity check that refuses to answer outside the trained range or near unstable settings.
- Experience building a certified, convergence-checked CFD dataset and its automation.

## Dataset

The twin was trained and tested on a CFD dataset of the CZ silicon melt (version 2):
292 steady ANSYS Fluent cases on a fixed 8,181-node mesh — 254 DOE training cases and
38 hold-out cases (1D parameter sweeps and out-of-range cases), plus 16 excluded cases
with no steady solution. Parameter ranges: crystal rotation 4–20 rpm, crucible rotation
−10 to −1 rpm, hot-wall temperature 1730–1785 K.

**The full dataset is not included in this repository because of its size. It is available
from the author on request** (see [Contact](#contact)).

Four sample cases are included in [`samples/`](samples/) to show the format
(two training cases, one in-range hold-out case, one out-of-range hold-out case), listed in
[`samples/samples.csv`](samples/samples.csv).

### Expected data layout

```
<dataset_folder>/
├── case_matrix.csv            case_id, group, crystal_rpm, crucible_rpm, t_hot  (DOE cases)
├── doe_run_status.csv         case_id, ..., status ("kept" or "excluded")
├── holdout_cases.csv          case_id, group, crystal_rpm, crucible_rpm, t_hot
├── holdout_run_status.csv     case_id, ..., status
├── cz_v2_doe/doe/<case_id>.csv
└── cz_v2_holdout/holdout/<case_id>.csv
```

Each case file has one row per mesh node with columns `r, z, u_r, u_z, u_swirl, p, T`
(SI units: m, m/s, Pa, K). Columns are read by name and nodes are sorted by coordinates,
so the export order does not matter.

## How to run

Python 3.10 or newer.

```
pip install -r requirements.txt
```

**Without the dataset** (pipeline test on synthetic fields, using the sample mesh):

```
python tests/make_synthetic.py samples/sweep_anchor.csv synthetic
python build_twin.py synthetic/all_cases.csv synthetic synthetic/twin.pkl
python tests/test_pipeline.py synthetic/twin.pkl
```

The synthetic fields are an analytic function, not CFD. They only check that the code
recovers a known answer.

**With the dataset:**

```
python make_case_lists.py <dataset_folder> case_lists
python check_cases.py --root <dataset_folder> --train case_lists/training_cases.csv --test case_lists/test_cases.csv --excluded case_lists/excluded_cases.txt --out case_lists/all_cases.csv
python build_twin.py case_lists/all_cases.csv <dataset_folder> twin.pkl
streamlit run app.py -- twin.pkl
python make_demo_video.py twin.pkl demo/cz_twin_demo.mp4 --drift "TC3 under crystal"
```

`check_cases.py` stops with the case name if a file is missing, has the wrong columns,
contains NaN values or uses a different mesh. The video needs `ffmpeg` on the PATH.

## Repository layout

```
twin/                 data loading, surrogate, state estimator, validity check, sensor layout
app.py                interactive Streamlit dashboard
make_case_lists.py    builds the case lists from the dataset metadata
check_cases.py        merges the lists and checks every case file
build_twin.py         trains the twin and writes the hold-out report
make_demo_video.py    renders the demo video from a trained twin
tests/                pipeline test on synthetic fields
samples/              four example cases from the dataset
results/              hold-out errors per case
demo/                 demo video and preview
```

## Notes

The code in this repository was written with AI assistance and checked against the
hold-out cases described above.

## Contact

For collaboration, data access or any permission under the licence, contact the author through
the GitHub profile that owns this repository.

## Licence

**© 2026 Bertwin Kurisinkal Shine. All rights reserved.**

This repository is published for viewing only. It is not open source. Without written
permission you may not use, copy, modify, distribute or present any part of it — code,
video, figures, results, sample data or text — including as part of your own demo, pitch,
proposal or publication. Prospective collaborators may run it privately to evaluate a
collaboration with the author. Full terms: [`LICENSE`](LICENSE).
