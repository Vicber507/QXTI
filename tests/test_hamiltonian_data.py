from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.data import HamiltonianData
from qxti.grids import KGrid
from qxti.physics import Hamiltonian


class TwoBandHamiltonian(Hamiltonian):
    def default_params(self) -> dict[str, float]:
        return {"mass": 1.0, "coupling": 2.0}

    def H(self, kx: float, ky: float, kz: float) -> np.ndarray:
        del kz
        mass = float(self.params["mass"])
        coupling = float(self.params["coupling"])
        return np.array(
            [
                [mass + kx**2, coupling * (kx - 1j * ky)],
                [coupling * (kx + 1j * ky), -mass + ky**2],
            ],
            dtype=complex,
        )


def build_two_band_hamiltonian() -> TwoBandHamiltonian:
    return TwoBandHamiltonian(
        model_name="two-band",
        basis_size=2,
        dimension=2,
        dk_derivative=1.0e-5,
    )


def test_kgrid_uniform_points_and_summary() -> None:
    grid = KGrid.uniform(dimension=2, k_min=-0.2, k_max=0.2, num_points=5)
    points = grid.points()
    summary = grid.summary()

    assert grid.shape == (5, 5, 1)
    assert points.shape == (25, 3)
    assert summary["dimension"] == 2
    assert summary["total_points"] == 25


def test_band_structure_2d_supports_direct_and_diagonal_paths() -> None:
    data = HamiltonianData(build_two_band_hamiltonian())

    kx_cut = data.band_structure_2d_data(path_type="kx", k_min=-0.3, k_max=0.3, num_points=7)
    diagonal_cut = data.band_structure_2d_data(
        path_type="diagonal_kx_ky",
        k_min=-0.2,
        k_max=0.2,
        num_points=5,
    )

    assert kx_cut["bands"].shape == (7, 2)
    assert diagonal_cut["bands"].shape == (5, 2)
    np.testing.assert_allclose(diagonal_cut["k_points"][:, 0], diagonal_cut["k_points"][:, 1])
    np.testing.assert_allclose(diagonal_cut["k_points"][:, 2], np.zeros(5))


def test_band_surface_velocity_and_velocity_magnitude_shapes() -> None:
    builder = HamiltonianData(build_two_band_hamiltonian())
    band_surface = builder.band_surface_3d_data(
        plane="kx_ky",
        num_points_1=9,
        num_points_2=7,
    )
    velocity_field = builder.velocity_field_3d_data(
        plane="kx_ky",
        num_points_1=8,
        num_points_2=6,
    )
    magnitude = builder.velocity_magnitude_data(
        plane="kx_ky",
        num_points_1=8,
        num_points_2=6,
    )

    assert band_surface["energy_surfaces"].shape == (2, 7, 9)
    assert band_surface["band_indices"] == (0, 1)
    assert velocity_field["vx"].shape == (2, 6, 8)
    assert velocity_field["plane_component_1"].shape == (2, 6, 8)
    assert magnitude["magnitude"].shape == (2, 6, 8)
    assert np.all(magnitude["magnitude"] >= 0.0)


def test_velocity_2d_matches_band_derivative_along_kx_cut() -> None:
    builder = HamiltonianData(build_two_band_hamiltonian())
    data = builder.velocity_2d_data(
        path_type="kx",
        k_min=-0.15,
        k_max=0.15,
        num_points=121,
        fixed_ky=0.08,
    )

    path_coordinate = np.asarray(data["path_coordinate"], dtype=float)
    bands = np.asarray(data["bands"], dtype=float)
    vx = np.asarray(data["vx"], dtype=float)

    finite_difference = np.gradient(bands, path_coordinate, axis=0)
    np.testing.assert_allclose(vx[5:-5], finite_difference[5:-5], atol=7.0e-3)


def test_hamiltonian_data_rejects_invalid_requests() -> None:
    builder = HamiltonianData(build_two_band_hamiltonian())

    with pytest.raises(ValueError, match="path_type"):
        builder.band_structure_2d_data(path_type="bad-path")
    with pytest.raises(ValueError, match="band_index"):
        builder.band_surface_3d_data(band_index=4)


def test_hamiltonian_graphics_can_write_basic_outputs(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available in this environment.")

    from qxti.graphics import HamiltonianGraphics

    builder = HamiltonianData(build_two_band_hamiltonian())
    band_path = HamiltonianGraphics.plot_band_structure_2d(
        builder.band_structure_2d_data(num_points=31),
        tmp_path / "bands.png",
    )

    assert band_path.exists()
    assert band_path.stat().st_size > 0
