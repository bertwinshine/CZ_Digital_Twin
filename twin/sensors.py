# Copyright (c) 2026 Bertwin Kurisinkal Shine. All rights reserved. See LICENSE.
"""Virtual thermocouple layout shared by the dashboard, the demo video and the tests."""

# name -> (r, z) in metres; melt domain is r 0..0.30, z 0..0.15 (crystal face at the top).
SENSORS = {
    "TC1 crucible wall": (0.29, 0.075),
    "TC2 crucible bottom": (0.15, 0.005),
    "TC3 under crystal": (0.05, 0.12),
    "TC4 near free surface": (0.24, 0.145),
}
SENSOR_NOISE_K = 0.5
