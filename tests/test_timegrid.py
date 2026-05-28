from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.grids import TimeGrid


def test_timegrid_from_dt_covers_requested_interval() -> None:
    grid = TimeGrid.from_dt(t_min=-1.0, t_max=1.0, dt=0.3)

    assert np.isclose(grid.t_min, -1.0)
    assert np.isclose(grid.dt, 0.3)
    assert grid.t_max >= 1.0
    assert np.isclose(grid.generate()[1] - grid.generate()[0], 0.3)


def test_timegrid_frequency_axis_is_angular_frequency_in_atomic_units() -> None:
    grid = TimeGrid(0.0, 3.0, 4, zero_padding=False)
    omega = grid.frequency_axis()

    expected = 2.0 * np.pi * np.fft.fftfreq(4, d=1.0)
    np.testing.assert_allclose(omega, expected)
