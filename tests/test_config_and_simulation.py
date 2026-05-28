from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from textwrap import dedent

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.core import QXTIConfig, QXTISimulation


def write_model_file(tmp_path: Path) -> Path:
    model_path = tmp_path / "toy_surface_model.py"
    model_path.write_text(
        dedent(
            """
            from __future__ import annotations

            import numpy as np

            MODEL_NAME = "toy-surface"
            BASIS_SIZE = 2
            DIMENSION = 2
            BASIS_TYPE = "spin"
            IS_PERIODIC = True

            DEFAULT_PARAMS = {"mass": 0.8, "coupling": 1.5}
            DEFAULT_LATTICE = {
                "lattice_constants": {"a": 1.0, "b": 1.0},
                "real_space_vectors": {"a1": [1.0, 0.0], "a2": [0.0, 1.0]},
            }

            def H(kx, ky, kz, params):
                del kz
                mass = params["mass"]
                coupling = params["coupling"]
                return np.array(
                    [
                        [mass + 0.5 * (kx**2 + ky**2), coupling * (kx - 1j * ky)],
                        [coupling * (kx + 1j * ky), -mass - 0.25 * (kx**2 + ky**2)],
                    ],
                    dtype=complex,
                )

            def H00(kx, ky, kz, params):
                del kz
                onsite = params["mass"] + 0.2 * np.cos(kx) + 0.2 * np.cos(ky)
                return np.array([[onsite, 0.0], [0.0, -onsite]], dtype=complex)

            def H01(kx, ky, kz, params):
                del kx, ky, kz
                return np.array([[0.2, 0.0], [0.0, 0.2]], dtype=complex)
            """
        )
    )
    return model_path


def write_config_file(tmp_path: Path, model_path: Path) -> Path:
    config_path = tmp_path / "inputParams.cfg"
    config_path.write_text(
        dedent(
            f"""
            [hamiltonian]
            source_file = {model_path}
            function_name = H
            basis_size = 2
            dimension = 2
            basis_type = spin
            mass = 1.1
            param.coupling = 1.8
            lattice = {{"notes": "test lattice"}}

            [hamiltonian_plots]
            enabled = true
            plots = band_structure_2d, band_surface_3d, velocity_2d, velocity_magnitude
            output_dir = {tmp_path / "plots"}
            path_type = diagonal_kx_ky
            plane = kx_ky
            k_min = -0.15
            k_max = 0.15
            nk_path = 41
            mesh_points = 15
            band_indices = all
            surface_style = colormap
            """
        )
    )
    return config_path


def test_qxti_config_parses_hamiltonian_and_plot_sections(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(tmp_path, model_path)

    config = QXTIConfig.from_file(config_path)

    assert config.hamiltonian.source_file == str(model_path)
    assert np.isclose(config.hamiltonian.params["mass"], 1.1)
    assert np.isclose(config.hamiltonian.params["coupling"], 1.8)
    assert config.hamiltonian.lattice["notes"] == "test lattice"
    assert config.hamiltonian_plots.plots == (
        "band_structure_2d",
        "band_surface_3d",
        "velocity_2d",
        "velocity_magnitude",
    )
    assert config.hamiltonian_plots.band_indices is None


def test_simulation_builds_custom_hamiltonian_from_config(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(tmp_path, model_path)

    simulation = QXTISimulation.from_file(config_path)
    hamiltonian = simulation.build_hamiltonian()

    assert hamiltonian.model_name == "toy-surface"
    assert hamiltonian.basis_size == 2
    assert hamiltonian.dimension == 2
    assert np.isclose(hamiltonian.params["mass"], 1.1)
    assert np.isclose(hamiltonian.params["coupling"], 1.8)
    assert hamiltonian.lattice["notes"] == "test lattice"


def test_simulation_generates_requested_outputs(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available in this environment.")

    model_path = write_model_file(tmp_path)
    config_path = write_config_file(tmp_path, model_path)
    simulation = QXTISimulation.from_file(config_path)

    outputs = simulation.run()

    assert set(outputs) == {
        "band_structure_2d",
        "band_surface_3d",
        "velocity_2d",
        "velocity_magnitude",
    }
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0
