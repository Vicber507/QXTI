from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.physics import Laser

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
else:
    plt = None


# Edit these values when you want to preview a different pulse by running
# `python tests/test_laser.py`.
def cycles_for_fwhm(omega: float, fwhm: float) -> float:
    return fwhm * omega / (2.0 * np.pi)


PREVIEW_LASER_PARAMS = {
    "omega": 0.03,
    "E0": 0.004,
    "cep": 0.0,
    "ellip": 1.0,
    "ncycles": 3,
    "envname": "gauss",
    "t0": 0.0,
    "thetaz": 0.0,
    "phiz": 0.0,
    "phix": 0.5 * np.pi,
}

PREVIEW_TIME_PARAMS = {
    "cycles": 10,
    "points": 800,
}


def build_reference_laser() -> Laser:
    return Laser(
        omega=0.85,
        E0=0.06,
        cep=0.2,
        ellip=1.0,
        ncycles=cycles_for_fwhm(0.85, 24.0),
        envname="gauss",
        t0=0.0,
        theta=0.0,
        phi=0.75,
    )


def build_preview_laser() -> Laser:
    return Laser(**PREVIEW_LASER_PARAMS)


def build_visualization_time(laser: Laser, cycles: float = 1.5, points: int = 800) -> np.ndarray:
    period = 2.0 * np.pi / laser.omega
    half_span = 0.5 * cycles * period
    return np.linspace(laser.t0 - half_span, laser.t0 + half_span, points)


def set_equal_2d_limits(axis, x: np.ndarray, y: np.ndarray, pad: float = 0.15) -> None:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    radius = 0.5 * max(x_max - x_min, y_max - y_min)
    radius = max(radius * (1.0 + pad), 1e-8)
    axis.set_xlim(x_center - radius, x_center + radius)
    axis.set_ylim(y_center - radius, y_center + radius)
    axis.set_aspect("equal", adjustable="box")


def plot_laser_preview(laser: Laser, t: np.ndarray, output_path: Path) -> Path:
    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib is not installed in this environment. "
            "Install it to generate the laser preview."
        )

    electric_field = laser.electric_field(t)
    envelope = laser.envelope_function(t)
    ex = electric_field[:, 0]
    ey = electric_field[:, 1]
    ez = electric_field[:, 2]
    field_norm = np.linalg.norm(electric_field, axis=1)
    time_colors = np.linspace(0.0, 1.0, len(t))
    centered_time = t - laser.t0
    time_scale = max(np.max(np.abs(centered_time)), 1e-12)
    normalized_time = centered_time / time_scale
    amplitude_bound = laser.E0 * np.asarray(envelope, dtype=float)

    figure = plt.figure(figsize=(13, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15))

    axis_xy = figure.add_subplot(grid[0, 0])
    axis_xy.plot(ex, ey, color="tab:blue", linewidth=1.6)
    axis_xy.scatter(ex, ey, c=time_colors, cmap="viridis", s=10)
    axis_xy.set_title("Projection: Ex vs Ey")
    axis_xy.set_xlabel("Ex")
    axis_xy.set_ylabel("Ey")
    axis_xy.grid(alpha=0.25)
    set_equal_2d_limits(axis_xy, ex, ey)

    axis_xz = figure.add_subplot(grid[0, 1])
    axis_xz.plot(ex, ez, color="tab:orange", linewidth=1.6)
    axis_xz.scatter(ex, ez, c=time_colors, cmap="viridis", s=10)
    axis_xz.set_title("Projection: Ex vs Ez")
    axis_xz.set_xlabel("Ex")
    axis_xz.set_ylabel("Ez")
    axis_xz.grid(alpha=0.25)
    set_equal_2d_limits(axis_xz, ex, ez)

    axis_time = figure.add_subplot(grid[1, :])
    axis_time.plot(normalized_time, ex, color="tab:blue", linewidth=1.8, label="Ex(t)")
    axis_time.plot(normalized_time, ey, color="tab:green", linewidth=1.8, label="Ey(t)")
    axis_time.plot(normalized_time, ez, color="tab:orange", linewidth=1.8, label="Ez(t)")
    axis_time.plot(
        normalized_time,
        field_norm,
        color="tab:red",
        linewidth=1.8,
        label="|E(t)|",
    )
    axis_time.plot(
        normalized_time,
        amplitude_bound,
        color="0.10",
        linewidth=1.9,
        linestyle="--",
        label="+ amplitude bound",
    )
    axis_time.plot(
        normalized_time,
        -amplitude_bound,
        color="0.10",
        linewidth=1.4,
        linestyle="--",
        label="- amplitude bound",
    )
    axis_time.set_title("Field components in time")
    axis_time.set_xlabel("normalized time")
    axis_time.set_ylabel("field amplitude")
    axis_time.set_xlim(-1.0, 1.0)
    axis_time.grid(alpha=0.25)
    axis_time.legend(loc="upper right")

    figure.suptitle(
        (
            "Laser preview\n"
            f"ellipticity={laser.ellipticity:.2f}, "
            f"theta={laser.theta:.2f}, phi={laser.phi:.2f}, "
            f"omega={laser.omega:.2f}, E0={laser.E0:.2f}"
        ),
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def test_laser_shapes_and_summary() -> None:
    laser = build_reference_laser()
    t = np.linspace(-40.0, 40.0, 600)

    electric_field = laser.electric_field(t)
    vector_potential = laser.vector_potential(t)
    summary = laser.summary()

    assert electric_field.shape == (600, 3)
    assert vector_potential.shape == (600, 3)
    assert np.isclose(np.linalg.norm(laser.polarization_vector()), 1.0)
    assert summary["envelope"] == "gauss"
    assert np.isclose(summary["theta"], 0.0)
    assert np.isclose(summary["phi"], 0.75)
    assert np.isclose(summary["fwhm"], laser.fwhm)
    assert np.isclose(summary["period"], 2.0 * np.pi / laser.omega)
    assert summary["intensity"] > 0.0


def test_laser_temporal_bounds_are_inferred_in_atomic_units() -> None:
    laser = build_reference_laser()

    t_min, t_max = laser.temporal_bounds()

    assert np.isclose(t_min, laser.t0 - 2.0 * laser.fwhm)
    assert np.isclose(t_max, laser.t0 + 2.0 * laser.fwhm)


def test_linear_polarization_orientation_from_angles() -> None:
    linear_x = Laser(
        omega=1.0,
        E0=1.0,
        ellip=0.0,
        ncycles=cycles_for_fwhm(1.0, 10.0),
        envname="constant",
        thetaz=0.0,
        phiz=0.0,
        phix=0.0,
    )
    linear_y = Laser(
        omega=1.0,
        E0=1.0,
        ellip=0.0,
        ncycles=cycles_for_fwhm(1.0, 10.0),
        envname="constant",
        thetaz=0.0,
        phiz=0.0,
        phix=0.5 * np.pi,
    )
    linear_z = Laser(
        omega=1.0,
        E0=1.0,
        ellip=0.0,
        ncycles=cycles_for_fwhm(1.0, 10.0),
        envname="constant",
        theta=0.0,
        phi=0.0,
    )

    assert np.allclose(linear_x.electric_field(0.0), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(linear_y.electric_field(0.0), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(linear_z.electric_field(0.0), np.array([0.0, 0.0, 1.0]))


def test_ellipticity_sign_sets_rotation_sense() -> None:
    right_circular = Laser(
        omega=1.0,
        E0=1.0,
        ellip=1.0,
        ncycles=cycles_for_fwhm(1.0, 10.0),
        envname="constant",
        thetaz=0.0,
        phiz=0.0,
        phix=0.0,
    )
    left_circular = Laser(
        omega=1.0,
        E0=1.0,
        ellip=-1.0,
        ncycles=cycles_for_fwhm(1.0, 10.0),
        envname="constant",
        thetaz=0.0,
        phiz=0.0,
        phix=0.0,
    )

    quarter_cycle = 0.5 * np.pi
    right_field = right_circular.electric_field(quarter_cycle)
    left_field = left_circular.electric_field(quarter_cycle)

    assert np.allclose(right_field, np.array([0.0, right_circular.E0y, 0.0]), atol=1e-12)
    assert np.allclose(left_field, np.array([0.0, left_circular.E0y, 0.0]), atol=1e-12)


def test_rotation_matrix_is_orthonormal() -> None:
    laser = build_reference_laser()
    rotation = laser.rotation_matrix()

    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.allclose(rotation[:, 0], laser.polarization_vector(), atol=1e-12)


def test_field_and_vector_potential_have_expected_center_values() -> None:
    laser = build_preview_laser()
    field_center = laser.electric_field(laser.t0)
    potential_center = laser.vector_potential(laser.t0)

    np.testing.assert_allclose(field_center, laser.E0x * laser.xdir, atol=1.0e-12)
    np.testing.assert_allclose(potential_center, laser.A0y * laser.ydir, atol=1.0e-12)


def test_laser_preview_generation(tmp_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return

    laser = build_preview_laser()
    t = build_visualization_time(laser, **PREVIEW_TIME_PARAMS)
    output_path = tmp_path / "laser_preview.png"

    plot_laser_preview(laser, t, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


if __name__ == "__main__":
    test_laser_shapes_and_summary()
    test_linear_polarization_orientation_from_angles()
    test_ellipticity_sign_sets_rotation_sense()
    test_rotation_matrix_is_orthonormal()
    test_field_and_vector_potential_have_expected_center_values()
    print("Core Laser checks passed.")

    if HAS_MATPLOTLIB:
        preview_path = Path(__file__).with_name("laser_preview.png")
        preview_laser = build_preview_laser()
        preview_time = build_visualization_time(preview_laser, **PREVIEW_TIME_PARAMS)
        plot_laser_preview(preview_laser, preview_time, preview_path)
        print(f"Saved laser preview to {preview_path}")
    else:
        print("matplotlib not found: skipped preview generation.")

    test_laser_preview_generation(Path(tempfile.mkdtemp()))
