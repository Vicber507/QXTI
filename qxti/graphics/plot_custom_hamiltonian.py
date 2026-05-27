from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qxti.physics import Hamiltonian


FloatArray = NDArray[np.float64]

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
else:
    plt = None


def sample_band_energies(
    hamiltonian: Hamiltonian,
    kx_values: ArrayLike,
    ky: float = 0.0,
    kz: float = 0.0,
) -> FloatArray:
    """Return the band energies along one 1D cut in k-space."""

    kx_line = np.asarray(kx_values, dtype=float)
    energies = np.empty((len(kx_line), hamiltonian.basis_size), dtype=float)
    for index, kx in enumerate(kx_line):
        energies[index] = hamiltonian.eigenvalues(float(kx), float(ky), float(kz))
    return energies


def sample_offdiagonal_magnitude(
    hamiltonian: Hamiltonian,
    kx_values: ArrayLike,
    ky: float = 0.0,
    kz: float = 0.0,
) -> FloatArray:
    """Return the largest off-diagonal matrix magnitude along one cut."""

    kx_line = np.asarray(kx_values, dtype=float)
    magnitude = np.empty(len(kx_line), dtype=float)
    for index, kx in enumerate(kx_line):
        matrix = np.asarray(hamiltonian.H(float(kx), float(ky), float(kz)), dtype=complex)
        off_diagonal = matrix - np.diag(np.diag(matrix))
        magnitude[index] = float(np.max(np.abs(off_diagonal))) if off_diagonal.size else 0.0
    return magnitude


def sample_band_maps(
    hamiltonian: Hamiltonian,
    kx_values: ArrayLike,
    ky_values: ArrayLike,
    kz: float = 0.0,
) -> tuple[NDArray[np.float64], FloatArray]:
    """Return eigenvalue maps and one hermiticity-error map on a 2D grid."""

    kx_grid = np.asarray(kx_values, dtype=float)
    ky_grid = np.asarray(ky_values, dtype=float)
    band_maps = np.empty((len(ky_grid), len(kx_grid), hamiltonian.basis_size), dtype=float)
    hermiticity_error = np.empty((len(ky_grid), len(kx_grid)), dtype=float)

    for iy, ky in enumerate(ky_grid):
        for ix, kx in enumerate(kx_grid):
            matrix = np.asarray(hamiltonian.H(float(kx), float(ky), float(kz)), dtype=complex)
            band_maps[iy, ix] = np.linalg.eigvalsh(matrix)
            hermiticity_error[iy, ix] = float(np.linalg.norm(matrix - matrix.conj().T))

    return band_maps, hermiticity_error


def direct_gap_map(band_maps: NDArray[np.float64]) -> FloatArray:
    """Return the minimum adjacent-band gap at each k-point."""

    if band_maps.shape[-1] <= 1:
        return np.zeros(band_maps.shape[:2], dtype=float)
    return np.min(np.diff(band_maps, axis=2), axis=2)


def plot_hamiltonian_diagnostics(
    equilibrium_hamiltonian: Hamiltonian,
    output_path: Path,
    *,
    comparison_hamiltonian: Hamiltonian | None = None,
    comparison_label: str = "comparison",
    k_extent: float = 0.18,
    line_points: int = 401,
    map_points: int = 121,
    ky_cut: float = 0.0,
    kz_cut: float = 0.0,
    time_values: ArrayLike | None = None,
    time_energies: NDArray[np.float64] | None = None,
    time_label: str = "time",
) -> Path:
    """Create a multi-panel verification figure for one Hamiltonian."""

    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib is not installed in this environment. "
            "Install it to generate Hamiltonian diagnostics."
        )

    k_line = np.linspace(-k_extent, k_extent, line_points)
    k_grid = np.linspace(-k_extent, k_extent, map_points)

    equilibrium_line = sample_band_energies(equilibrium_hamiltonian, k_line, ky=ky_cut, kz=kz_cut)
    offdiagonal_line = sample_offdiagonal_magnitude(
        equilibrium_hamiltonian,
        k_line,
        ky=ky_cut,
        kz=kz_cut,
    )
    equilibrium_maps, hermiticity_map = sample_band_maps(
        equilibrium_hamiltonian,
        k_grid,
        k_grid,
        kz=kz_cut,
    )
    gap_map = direct_gap_map(equilibrium_maps)

    comparison_line = None
    if comparison_hamiltonian is not None:
        comparison_line = sample_band_energies(
            comparison_hamiltonian,
            k_line,
            ky=ky_cut,
            kz=kz_cut,
        )

    figure = plt.figure(figsize=(15, 9))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 1.15))

    axis_line = figure.add_subplot(grid[0, 0])
    for band in range(equilibrium_line.shape[1]):
        axis_line.plot(
            k_line,
            equilibrium_line[:, band],
            linewidth=2.0,
            label=f"eq band {band + 1}",
        )
        if comparison_line is not None:
            axis_line.plot(
                k_line,
                comparison_line[:, band],
                linewidth=1.6,
                linestyle="--",
                label=f"{comparison_label} band {band + 1}",
            )
    axis_line.set_title("Band cut along kx")
    axis_line.set_xlabel("kx")
    axis_line.set_ylabel("energy")
    axis_line.grid(alpha=0.25)
    axis_line.legend(loc="best", fontsize=8)

    axis_offdiag = figure.add_subplot(grid[0, 1])
    axis_offdiag.plot(k_line, offdiagonal_line, color="tab:red", linewidth=2.0)
    axis_offdiag.set_title("Largest off-diagonal |H_ij|")
    axis_offdiag.set_xlabel("kx")
    axis_offdiag.set_ylabel("magnitude")
    axis_offdiag.grid(alpha=0.25)

    axis_time = figure.add_subplot(grid[0, 2])
    if time_values is not None and time_energies is not None:
        time_axis = np.asarray(time_values, dtype=float)
        for band in range(time_energies.shape[1]):
            axis_time.plot(time_axis, time_energies[:, band], linewidth=1.9, label=f"band {band + 1}")
        axis_time.set_title("Band energies at fixed k vs time")
        axis_time.set_xlabel(time_label)
        axis_time.set_ylabel("energy")
        axis_time.legend(loc="best", fontsize=8)
    else:
        hermiticity_cut = hermiticity_map[hermiticity_map.shape[0] // 2]
        axis_time.plot(k_grid, hermiticity_cut, color="tab:green", linewidth=2.0)
        axis_time.set_title("Hermiticity error cut")
        axis_time.set_xlabel("kx")
        axis_time.set_ylabel(r"$||H - H^\dagger||$")
    axis_time.grid(alpha=0.25)

    axis_lower = figure.add_subplot(grid[1, 0])
    lower_image = axis_lower.imshow(
        equilibrium_maps[:, :, 0],
        origin="lower",
        extent=[k_grid[0], k_grid[-1], k_grid[0], k_grid[-1]],
        aspect="auto",
        cmap="coolwarm",
    )
    axis_lower.set_title("Lowest band map")
    axis_lower.set_xlabel("kx")
    axis_lower.set_ylabel("ky")
    figure.colorbar(lower_image, ax=axis_lower, shrink=0.9)

    axis_upper = figure.add_subplot(grid[1, 1])
    upper_image = axis_upper.imshow(
        equilibrium_maps[:, :, -1],
        origin="lower",
        extent=[k_grid[0], k_grid[-1], k_grid[0], k_grid[-1]],
        aspect="auto",
        cmap="coolwarm",
    )
    axis_upper.set_title("Highest band map")
    axis_upper.set_xlabel("kx")
    axis_upper.set_ylabel("ky")
    figure.colorbar(upper_image, ax=axis_upper, shrink=0.9)

    axis_gap = figure.add_subplot(grid[1, 2])
    gap_image = axis_gap.imshow(
        gap_map,
        origin="lower",
        extent=[k_grid[0], k_grid[-1], k_grid[0], k_grid[-1]],
        aspect="auto",
        cmap="magma",
    )
    axis_gap.set_title(
        "Direct gap map\n"
        f"max hermiticity error = {np.max(hermiticity_map):.2e}"
    )
    axis_gap.set_xlabel("kx")
    axis_gap.set_ylabel("ky")
    figure.colorbar(gap_image, ax=axis_gap, shrink=0.9)

    figure.suptitle(
        (
            f"Hamiltonian diagnostics: {equilibrium_hamiltonian.model_name}\n"
            f"basis={equilibrium_hamiltonian.basis_size}, dimension={equilibrium_hamiltonian.dimension}, "
            f"ky cut={ky_cut:.3f}, kz cut={kz_cut:.3f}"
        ),
        fontsize=14,
    )
    figure.tight_layout()
    output_path = Path(output_path)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
