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
from qxti.data import load_dataset_npz


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
    num_cycles = 2.0 * 0.8 / (2.0 * np.pi)
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

            [kgrid]
            dimension = 2
            k_points = [3, 3]

            [timegrid]
            dt = 0.2
            zero_padding = true
            padding_factor = 2

            [laser]
            omega = 0.8
            E0 = 0.05
            ellip = 0.0
            ncycles = {num_cycles}
            envname = gauss
            cep = 0.0
            t0 = 2.0
            phix = 0.0
            thetaz = 0.0
            phiz = 0.0
            blaser = 0.5
            alaser = 0.5

            [cmd]
            enabled = true
            output_dir = {tmp_path / "cmd"}
            max_order = 1
            rho_storage_dtype = complex64
            population_time = 50.0
            coherence_time = 20.0
            temperature = 0.02
            fermi_level = 0.0
            distribution = fermi_dirac
            basis = band
            gauge = length
            include_intraband = true
            include_interband = true
            include_dephasing = true
            solver = rkf45
            solver_tolerance = 1.0e-4
            solver_max_iterations = 100000

            [cmd_plots]
            enabled = true
            output_path = {tmp_path / "cmd" / "population_heatmap.png"}
            orders = [0, 1]
            k_aggregation = mean
            save_animation = false

            [xtp]
            bz_mask_enabled = true
            bz_mask_radius_percent = 80.0
            bz_mask_sigma = 0.75
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
    assert config.kgrid.dimension == 2
    assert config.kgrid.k_points == (3, 3)
    assert config.kgrid.points_per_axis is None
    assert config.kgrid.kx_values == []
    assert np.isclose(config.timegrid.dt, 0.2)
    assert config.timegrid.Nt is None
    assert np.isclose(config.laser.omega, 0.8)
    assert np.isclose(config.laser.ncycles, 2.0 * 0.8 / (2.0 * np.pi))
    assert np.isclose(config.laser.phix, 0.0)
    assert np.isclose(config.laser.thetaz, 0.0)
    assert np.isclose(config.laser.phiz, 0.0)
    assert np.isclose(config.laser.blaser, 0.5)
    assert np.isclose(config.laser.alaser, 0.5)
    assert config.cmd.enabled is True
    assert config.cmd.max_order == 1
    assert config.cmd.rho_storage_dtype == "complex64"
    assert config.cmd.solver == "rkf45"
    assert config.cmd.distribution == "fermi_dirac"
    assert np.isclose(config.cmd.population_time, 50.0)
    assert np.isclose(config.cmd.coherence_time, 20.0)
    assert config.cmd_plots.enabled is True
    assert config.cmd_plots.orders == (0, 1)
    assert config.cmd_plots.k_aggregation == "mean"
    assert config.xtp.bz_mask_enabled is True
    assert np.isclose(config.xtp.bz_mask_radius_percent, 80.0)
    assert np.isclose(config.xtp.bz_mask_sigma, 0.75)


def test_simulation_builds_custom_hamiltonian_from_config(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(tmp_path, model_path)

    simulation = QXTISimulation.from_file(config_path)
    hamiltonian = simulation.build_hamiltonian()
    kgrid = simulation.build_kgrid(hamiltonian)

    assert hamiltonian.model_name == "toy-surface"
    assert hamiltonian.basis_size == 2
    assert hamiltonian.dimension == 2
    assert np.isclose(hamiltonian.params["mass"], 1.1)
    assert np.isclose(hamiltonian.params["coupling"], 1.8)
    assert hamiltonian.lattice["notes"] == "test lattice"
    reciprocal_bounds = np.asarray(hamiltonian.reciprocal_box_bounds(), dtype=float)
    np.testing.assert_allclose(
        reciprocal_bounds,
        np.array([[-np.pi, np.pi], [-np.pi, np.pi]], dtype=float),
    )
    assert kgrid.shape == (3, 3, 1)


def test_simulation_builds_rkf45_with_full_window_default_hmax(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(tmp_path, model_path)

    simulation = QXTISimulation.from_file(config_path)
    hamiltonian = simulation.build_hamiltonian()
    cmd = simulation.build_cmd(hamiltonian)

    assert np.isclose(
        cmd.solver.h_max,  # type: ignore[attr-defined]
        cmd.timegrid.t_max - cmd.timegrid.t_min,
    )


def test_config_parses_unlimited_rkf45_iterations_and_adaptation_controls(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = tmp_path / "inputParams_unlimited.cfg"
    config_path.write_text(
        dedent(
            f"""
            [hamiltonian]
            source_file = {model_path}

            [kgrid]
            dimension = 2
            k_points = [3, 3]

            [timegrid]
            dt = 0.2

            [laser]
            omega = 0.8
            E0 = 0.05
            ellip = 0.0
            ncycles = 1.0
            envname = gauss

            [cmd]
            enabled = true
            solver = rkf45
            solver_tolerance = 5.0e-4
            solver_max_iterations = 0
            solver_max_rejections = 25000
            solver_safety_factor = 0.95
            solver_min_factor = 0.25
            solver_max_factor = 6.0
            """
        )
    )

    config = QXTIConfig.from_file(config_path)
    assert config.cmd.solver_max_iterations is None
    assert config.cmd.solver_max_rejections == 25000
    assert np.isclose(config.cmd.solver_safety_factor, 0.95)
    assert np.isclose(config.cmd.solver_min_factor, 0.25)
    assert np.isclose(config.cmd.solver_max_factor, 6.0)

    simulation = QXTISimulation.from_file(config_path)
    hamiltonian = simulation.build_hamiltonian()
    cmd = simulation.build_cmd(hamiltonian)
    assert cmd.solver.max_iterations is None
    assert np.isclose(cmd.solver.safety_factor, 0.95)  # type: ignore[attr-defined]
    assert np.isclose(cmd.solver.min_factor, 0.25)  # type: ignore[attr-defined]
    assert np.isclose(cmd.solver.max_factor, 6.0)  # type: ignore[attr-defined]


def test_simulation_generates_requested_outputs(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(tmp_path, model_path)
    simulation = QXTISimulation.from_file(config_path)

    outputs = simulation.run()

    assert set(outputs) == {
        "band_structure_2d_data",
        "band_surface_3d_data",
        "velocity_2d_data",
        "velocity_magnitude_data",
        "rho_order_0",
        "rho_order_1",
        "rho_population_kxky_data",
        "rho_coherence_kxky_data",
        "xtp_current_spectrum_data",
    }
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0
    harmonic_dataset = load_dataset_npz(outputs["xtp_current_spectrum_data"])
    assert np.asarray(harmonic_dataset["electric_field_time"], dtype=float).shape[1] == 3
    assert harmonic_dataset["bz_mask"]["enabled"] is True
    assert np.isclose(float(harmonic_dataset["bz_mask"]["radius_percent"]), 80.0)
    assert np.isclose(float(harmonic_dataset["bz_mask"]["sigma"]), 0.75)
    assert bool(harmonic_dataset["current_decomposition_available"]) is True
    assert np.asarray(harmonic_dataset["current_total_magnitude"], dtype=float).ndim == 1
    assert np.asarray(harmonic_dataset["current_total_magnitude_intraband"], dtype=float).ndim == 1
    assert np.asarray(harmonic_dataset["current_total_magnitude_interband"], dtype=float).ndim == 1
def test_graphics_runners_generate_outputs_from_config(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available in this environment.")

    from qxti.graphics.graphics import (
        plot_harmonic_graphics_from_saved_data,
        plot_hamiltonian_graphics_from_saved_data,
        plot_response_graphics_from_saved_data,
    )

    model_path = write_model_file(tmp_path)
    config_path = write_config_file(tmp_path, model_path)
    simulation = QXTISimulation.from_file(config_path)
    simulation.run()

    hamiltonian_outputs = plot_hamiltonian_graphics_from_saved_data(config_path)
    response_outputs = plot_response_graphics_from_saved_data(config_path)
    harmonic_outputs = plot_harmonic_graphics_from_saved_data(config_path)

    assert "band_structure_2d" in hamiltonian_outputs
    assert "velocity_magnitude" in hamiltonian_outputs
    assert "rho_population_snapshots" in response_outputs
    assert "rho_coherence_snapshots" in response_outputs
    assert "current_total_spectrum" in harmonic_outputs
    assert "current_components_spectrum" in harmonic_outputs
    assert "current_inter_intra_spectrum" in harmonic_outputs
    assert "current_circular_spectrum" in harmonic_outputs
    assert "current_overview_spectrum" in harmonic_outputs
    for path in (*hamiltonian_outputs.values(), *response_outputs.values(), *harmonic_outputs.values()):
        assert path.exists()
        assert path.stat().st_size > 0
