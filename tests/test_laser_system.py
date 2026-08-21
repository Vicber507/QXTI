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

from qxti.physics import Laser, LaserSystem

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
else:
    plt = None


# Edit these values when you want to preview a composed laser system by running
# `python tests/test_laser_system.py`.
def cycles_for_fwhm(omega: float, fwhm: float) -> float:
    return fwhm * omega / (2.0 * np.pi)


PREVIEW_SYSTEM_LASERS = [
    {
        "omega": 1.0,
        "E0": 1.0,
        "cep": 0.0,
        "ellip": 1.0,
        "ncycles": cycles_for_fwhm(1.0, 18.0),
        "envname": "gauss",
        "t0": 0.0,
        "thetaz": 0.0,
        "phiz": 0.0,
        "phix": 0.0,
    },
    {
        "omega": 2.0,
        "E0": 1.0,
        "cep": 0.0,
        "ellip": -1.0,
        "ncycles": cycles_for_fwhm(2.0, 18.0),
        "envname": "gauss",
        "t0": 0.0,
        "thetaz": 0.0,
        "phiz": 0.0,
        "phix": 0.0,
    },
]

PREVIEW_TIME_PARAMS = {
    "cycles": 10.0,
    "points": 1200,
}

PREVIEW_LASER_COLORS = [
    "tab:purple",
    "tab:red",
    "tab:olive",
    "tab:brown",
    "tab:pink",
]


def build_laser_x() -> Laser:
    return Laser(
        omega=1.0,
        E0=1.0,
        ellip=0.0,
        ncycles=cycles_for_fwhm(1.0, 12.0),
        envname="constant",
        thetaz=0.0,
        phiz=0.0,
        phix=0.0,
    )


def build_laser_y() -> Laser:
    return Laser(
        omega=1.0,
        E0=0.5,
        cep=0.25 * np.pi,
        ellip=0.0,
        ncycles=cycles_for_fwhm(1.0, 12.0),
        envname="constant",
        thetaz=0.0,
        phiz=0.0,
        phix=0.5 * np.pi,
    )


def build_preview_system() -> LaserSystem:
    return LaserSystem([Laser(**params) for params in PREVIEW_SYSTEM_LASERS])


def build_visualization_time(
    system: LaserSystem,
    cycles: float = 10.0,
    points: int = 1200,
) -> np.ndarray:
    if system.number_of_lasers() == 0:
        return np.linspace(-1.0, 1.0, points)

    omegas = [laser.omega for laser in system.lasers]
    t0_values = [laser.t0 for laser in system.lasers]
    reference_omega = min(omegas)
    period = 2.0 * np.pi / reference_omega
    center = 0.5 * (min(t0_values) + max(t0_values))
    half_span = 0.5 * cycles * period + 0.5 * (max(t0_values) - min(t0_values))
    return np.linspace(center - half_span, center + half_span, points)


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


def plot_laser_system_preview(system: LaserSystem, t: np.ndarray, output_path: Path) -> Path:
    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib is not installed in this environment. "
            "Install it to generate the laser-system preview."
        )
    if system.number_of_lasers() == 0:
        raise ValueError("LaserSystem preview requires at least one Laser.")

    total_field = system.electric_field(t)
    ex_total = total_field[:, 0]
    ey_total = total_field[:, 1]
    ez_total = total_field[:, 2]
    time_colors = np.linspace(0.0, 1.0, len(t))

    system_center = 0.5 * (
        min(laser.t0 for laser in system.lasers) + max(laser.t0 for laser in system.lasers)
    )
    centered_time = t - system_center
    time_scale = max(np.max(np.abs(centered_time)), 1e-12)
    normalized_time = centered_time / time_scale

    figure = plt.figure(figsize=(13, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15))

    axis_xy = figure.add_subplot(grid[0, 0])
    for index, laser in enumerate(system.lasers):
        field = laser.electric_field(t)
        axis_xy.plot(
            field[:, 0],
            field[:, 1],
            color=PREVIEW_LASER_COLORS[index % len(PREVIEW_LASER_COLORS)],
            linewidth=1.0,
            alpha=0.35,
        )
    axis_xy.plot(ex_total, ey_total, color="tab:blue", linewidth=1.8, alpha=0.95)
    axis_xy.scatter(ex_total, ey_total, c=time_colors, cmap="viridis", s=10)
    axis_xy.set_title("Total projection: Ex vs Ey")
    axis_xy.set_xlabel("Ex")
    axis_xy.set_ylabel("Ey")
    axis_xy.grid(alpha=0.25)
    set_equal_2d_limits(axis_xy, ex_total, ey_total)

    axis_xz = figure.add_subplot(grid[0, 1])
    for index, laser in enumerate(system.lasers):
        field = laser.electric_field(t)
        axis_xz.plot(
            field[:, 0],
            field[:, 2],
            color=PREVIEW_LASER_COLORS[index % len(PREVIEW_LASER_COLORS)],
            linewidth=1.0,
            alpha=0.35,
        )
    axis_xz.plot(ex_total, ez_total, color="tab:orange", linewidth=1.8, alpha=0.95)
    axis_xz.scatter(ex_total, ez_total, c=time_colors, cmap="viridis", s=10)
    axis_xz.set_title("Total projection: Ex vs Ez")
    axis_xz.set_xlabel("Ex")
    axis_xz.set_ylabel("Ez")
    axis_xz.grid(alpha=0.25)
    set_equal_2d_limits(axis_xz, ex_total, ez_total)

    axis_time = figure.add_subplot(grid[1, :])
    axis_time.plot(normalized_time, ex_total, color="tab:blue", linewidth=1.9, label="Ex total")
    axis_time.plot(normalized_time, ey_total, color="tab:green", linewidth=1.9, label="Ey total")
    axis_time.plot(normalized_time, ez_total, color="tab:orange", linewidth=1.9, label="Ez total")

    total_amplitude_bound = np.zeros_like(t, dtype=float)
    for index, laser in enumerate(system.lasers):
        amplitude_bound = laser.E0 * np.asarray(laser.envelope_function(t), dtype=float)
        total_amplitude_bound += amplitude_bound
        color = PREVIEW_LASER_COLORS[index % len(PREVIEW_LASER_COLORS)]
        axis_time.plot(
            normalized_time,
            amplitude_bound,
            color=color,
            linewidth=1.3,
            linestyle="--",
            alpha=0.85,
            label=f"L{index + 1} bound",
        )
        axis_time.plot(
            normalized_time,
            -amplitude_bound,
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.4,
        )
    axis_time.plot(
        normalized_time,
        total_amplitude_bound,
        color="0.10",
        linewidth=2.0,
        linestyle=":",
        label="+ total bound",
    )
    axis_time.plot(
        normalized_time,
        -total_amplitude_bound,
        color="0.10",
        linewidth=1.5,
        linestyle=":",
        label="- total bound",
    )

    axis_time.set_title("Total field components in time")
    axis_time.set_xlabel("normalized time")
    axis_time.set_ylabel("field amplitude")
    axis_time.set_xlim(-1.0, 1.0)
    axis_time.grid(alpha=0.25)
    axis_time.legend(loc="upper right", ncol=2)

    figure.suptitle(
        (
            "LaserSystem preview\n"
            f"number of lasers = {system.number_of_lasers()}, "
            f"total intensity = {system.total_intensity():.3f}"
        ),
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def test_laser_system_starts_empty() -> None:
    system = LaserSystem()

    assert system.number_of_lasers() == 0
    assert system.total_intensity() == 0.0
    assert np.allclose(system.electric_field(0.0), np.zeros(3))
    assert np.allclose(system.vector_potential(0.0), np.zeros(3))


def test_laser_system_add_remove_and_count() -> None:
    system = LaserSystem()
    laser_x = build_laser_x()
    laser_y = build_laser_y()

    system.add_laser(laser_x)
    system.add_laser(laser_y)
    assert system.number_of_lasers() == 2

    system.remove_laser(0)
    assert system.number_of_lasers() == 1
    assert system.lasers[0] is laser_y


def test_laser_system_rejects_non_laser_objects() -> None:
    system = LaserSystem()

    try:
        system.add_laser("not a laser")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("LaserSystem accepted a non-Laser object.")


def test_total_electric_field_matches_sum_of_lasers() -> None:
    laser_x = build_laser_x()
    laser_y = build_laser_y()
    system = LaserSystem([laser_x, laser_y])
    t = np.linspace(-2.0, 2.0, 200)

    expected = laser_x.electric_field(t) + laser_y.electric_field(t)
    total = system.electric_field(t)

    assert total.shape == (200, 3)
    assert np.allclose(total, expected)


def test_total_vector_potential_matches_sum_of_lasers() -> None:
    laser_x = build_laser_x()
    laser_y = build_laser_y()
    system = LaserSystem([laser_x, laser_y])
    t = np.linspace(-2.0, 2.0, 200)

    expected = laser_x.vector_potential(t) + laser_y.vector_potential(t)
    total = system.vector_potential(t)

    assert total.shape == (200, 3)
    assert np.allclose(total, expected)


def test_total_intensity_is_sum_of_individual_intensities() -> None:
    laser_x = build_laser_x()
    laser_y = build_laser_y()
    system = LaserSystem([laser_x, laser_y])

    expected = laser_x.intensity() + laser_y.intensity()
    assert np.isclose(system.total_intensity(), expected)


def test_laser_system_temporal_bounds_cover_all_pulses() -> None:
    system = LaserSystem(
        [
            Laser(
                omega=1.0,
                E0=1.0,
                ellip=0.0,
                ncycles=cycles_for_fwhm(1.0, 2.0),
                envname="gauss",
                t0=-1.0,
            ),
            Laser(
                omega=1.0,
                E0=1.0,
                ellip=0.0,
                ncycles=cycles_for_fwhm(1.0, 3.0),
                envname="gauss",
                t0=4.0,
            ),
        ]
    )

    t_min, t_max = system.temporal_bounds()

    assert np.isclose(t_min, -5.0)
    assert np.isclose(t_max, 10.0)


def test_system_does_not_inject_offset() -> None:
    """Regression: LaserSystem returns the PURE analytic sum of the pulses — it no longer
    subtracts E(atmin)/A(atmin).  The old subtraction forced the field to exactly zero at
    the window edge but INJECTED a spurious DC (~1e-5 of the peak) → spurious even
    harmonics (pfddm) and a DC ramp when integrated (tddm)."""
    lx, ly = build_laser_x(), build_laser_y()
    system = LaserSystem([lx, ly], blaser=1.5, alaser=0.5)
    t = np.linspace(system.atmin, system.atmax, 1000)

    # no constant subtracted anywhere: system field/A == raw per-pulse sum, exactly
    assert np.allclose(system.electric_field(t), lx.electric_field(t) + ly.electric_field(t))
    assert np.allclose(system.vector_potential(t), lx.vector_potential(t) + ly.vector_potential(t))

    # the window edge is NO LONGER forced to exactly zero: it keeps the natural raw value
    edge_sys = system.electric_field(system.atmin)
    edge_raw = lx.electric_field(system.atmin) + ly.electric_field(system.atmin)
    assert np.allclose(edge_sys, edge_raw)


def test_realistic_gaussian_pulse_has_no_injected_dc() -> None:
    """A well-resolved Gaussian pulse has ~zero DC; the LaserSystem must not add one back
    (the removed E(atmin)-subtraction used to inject ~1.5e-5 of the peak)."""
    laser = Laser(omega=1.0, E0=1.0, ellip=0.0, ncycles=cycles_for_fwhm(1.0, 8.0), envname="gauss")
    system = LaserSystem([laser])
    t = np.linspace(*system.temporal_bounds(), 40001)
    E = system.electric_field(t)
    peak = np.abs(E).max()
    assert np.abs(E.mean(axis=0)).max() < 1e-6 * peak, "electric field carries a spurious DC"


def test_scalar_time_is_supported() -> None:
    laser_x = build_laser_x()
    laser_y = build_laser_y()
    system = LaserSystem([laser_x, laser_y])

    electric_field = system.electric_field(0.0)
    vector_potential = system.vector_potential(0.0)

    assert electric_field.shape == (3,)
    assert vector_potential.shape == (3,)


def test_total_field_norm_is_bounded_by_sum_of_amplitude_envelopes() -> None:
    system = build_preview_system()
    t = build_visualization_time(system, **PREVIEW_TIME_PARAMS)

    total_field = system.electric_field(t)
    total_field_norm = np.linalg.norm(total_field, axis=1)
    total_bound = sum(
        laser.E0 * np.asarray(laser.envelope_function(t), dtype=float)
        for laser in system.lasers
    )

    assert np.all(total_field_norm <= total_bound + 1e-12)


def test_laser_system_preview_generation(tmp_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return

    system = build_preview_system()
    t = build_visualization_time(system, **PREVIEW_TIME_PARAMS)
    output_path = tmp_path / "laser_system_preview.png"

    plot_laser_system_preview(system, t, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


if __name__ == "__main__":
    test_laser_system_starts_empty()
    test_laser_system_add_remove_and_count()
    test_laser_system_rejects_non_laser_objects()
    test_total_electric_field_matches_sum_of_lasers()
    test_total_vector_potential_matches_sum_of_lasers()
    test_total_intensity_is_sum_of_individual_intensities()
    test_scalar_time_is_supported()
    test_total_field_norm_is_bounded_by_sum_of_amplitude_envelopes()
    print("LaserSystem checks passed.")

    if HAS_MATPLOTLIB:
        preview_path = Path(__file__).with_name("laser_system_preview.png")
        preview_system = build_preview_system()
        preview_time = build_visualization_time(preview_system, **PREVIEW_TIME_PARAMS)
        plot_laser_system_preview(preview_system, preview_time, preview_path)
        print(f"Saved laser-system preview to {preview_path}")
    else:
        print("matplotlib not found: skipped preview generation.")

    test_laser_system_preview_generation(Path(tempfile.mkdtemp()))
