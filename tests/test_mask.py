from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.graphics import HarmonicGraphics
from qxti.grids import FrequencyGrid, KGrid, TimeGrid
from qxti.physics import Hamiltonian, OperatorFactory
from qxti.response import XTP


DEFAULT_TEST_MASK_CONFIG = {
    "grid": {
        "points_per_axis": 201,
        "k_extent": 0.5 * np.pi,
    },
    "mask": {
        "enabled": True,
        "radius_percent": 70.0,
        "sigma": 0.8,
    },
    "plot": {
        "output_file": "test_mask.png",
        "cmap": "inferno",
    },
}

# Edit this dictionary directly if you want to change the test image.
TEST_MASK_CONFIG = copy.deepcopy(DEFAULT_TEST_MASK_CONFIG)


class ToyTwoBandHamiltonian(Hamiltonian):
    def default_params(self) -> dict[str, float]:
        return {"mass": 0.4, "velocity": 1.2}

    def default_lattice(self) -> dict[str, object]:
        return {
            "lattice_constants": {
                "a0": 2.0,
                "a": 2.0,
                "b": 2.0,
            },
            "real_space_vectors": {
                "a1": [2.0, 0.0],
                "a2": [0.0, 2.0],
            },
        }

    def H(self, kx: float, ky: float, kz: float) -> np.ndarray:
        del kz
        mass = float(self.params["mass"])
        velocity = float(self.params["velocity"])
        return np.array(
            [
                [mass, velocity * (kx - 1j * ky)],
                [velocity * (kx + 1j * ky), -mass],
            ],
            dtype=complex,
        )


def resolve_test_mask_config(overrides: dict[str, object] | None = None) -> dict[str, object]:
    config = copy.deepcopy(TEST_MASK_CONFIG)
    if overrides:
        _deep_update(config, overrides)
    return config


def _build_mask_xtp(config: dict[str, object]) -> XTP:
    grid_cfg = dict(config["grid"])
    mask_cfg = dict(config["mask"])
    hamiltonian = ToyTwoBandHamiltonian(
        model_name="toy-mask",
        basis_size=2,
        dimension=2,
        dk_derivative=1.0e-5,
    )
    operator_factory = OperatorFactory(hamiltonian=hamiltonian, basis="band")
    points_per_axis = int(grid_cfg["points_per_axis"])
    k_extent = float(grid_cfg["k_extent"])
    kgrid = KGrid(
        kx_values=np.linspace(-k_extent, k_extent, points_per_axis),
        ky_values=np.linspace(-k_extent, k_extent, points_per_axis),
        kz_values=np.array([0.0]),
        dimension=2,
    )
    timegrid = TimeGrid(0.0, 1.0, 2, zero_padding=False, padding_factor=2)
    frequencygrid = FrequencyGrid(0.0, 5.0, 8)
    rho_orders = {
        0: np.zeros(
            (kgrid.total_points, len(timegrid), hamiltonian.basis_size, hamiltonian.basis_size),
            dtype=np.complex128,
        )
    }
    return XTP(
        hamiltonian=hamiltonian,
        rho_orders=rho_orders,
        kgrid=kgrid,
        timegrid=timegrid,
        frequencygrid=frequencygrid,
        operator_factory=operator_factory,
        directions=["x"],
        orders=[0],
        bz_mask_enabled=bool(mask_cfg["enabled"]),
        bz_mask_radius_percent=float(mask_cfg["radius_percent"]),
        bz_mask_sigma=float(mask_cfg["sigma"]),
    )


def test_mask_generates_image() -> None:
    if "matplotlib" not in sys.modules and importlib.util.find_spec("matplotlib") is None:
        return

    config = resolve_test_mask_config()
    plot_cfg = dict(config["plot"])
    xtp = _build_mask_xtp(config)
    plot_data = xtp.bz_mask_plot_data()

    output_path = Path(__file__).with_name(str(plot_cfg["output_file"]))
    saved_path = HarmonicGraphics.plot_bz_mask(
        np.asarray(plot_data["kx_grid"], dtype=float),
        np.asarray(plot_data["ky_grid"], dtype=float),
        np.asarray(plot_data["integration_region"], dtype=float),
        np.asarray(plot_data["mask_weights"], dtype=float),
        output_path,
        cmap=str(plot_cfg["cmap"]),
        metadata=plot_data["mask_metadata"],
    )

    assert saved_path.exists()
    assert saved_path.name == str(plot_cfg["output_file"])
    assert saved_path.parent == Path(__file__).resolve().parent
    assert saved_path.stat().st_size > 0


def _deep_update(target: dict[str, object], updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index]
        else:
            target[key] = value
