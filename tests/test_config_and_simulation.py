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

from qxti.core import QXTIConfig, QXTISimulation, SusceptibilityScanRunner
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


def write_config_file(
    tmp_path: Path,
    model_path: Path,
    *,
    keep_rho_orders: bool = True,
    scratch_rho_storage_dtype: str = "auto",
    dataset_time_stride: int = 1,
    save_population_dataset: bool = True,
    save_coherence_dataset: bool = True,
    save_xtp_dataset: bool = True,
    xtp_susceptibility_enabled: bool = False,
    xtp_susceptibility_orders: str = "[1]",
    xtp_susceptibility_num_frequencies: int = 2,
    xtp_susceptibility_plot_enabled: bool = False,
) -> Path:
    num_cycles = 2.0 * 0.8 / (2.0 * np.pi)
    config_path = tmp_path / "inputParams.cfg"
    susceptibility_sweep_block = ""
    if xtp_susceptibility_enabled:
        susceptibility_sweep_block = f"""
            susceptibility_output_dir = {tmp_path / "susceptibility"}
            susceptibility_orders = {xtp_susceptibility_orders}
            susceptibility_omega_min = 0.7
            susceptibility_omega_max = 0.9
            susceptibility_num_frequencies = {xtp_susceptibility_num_frequencies}
            susceptibility_eps = 1.0e-14
"""
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
            scratch_rho_storage_dtype = {scratch_rho_storage_dtype}
            keep_rho_orders = {"true" if keep_rho_orders else "false"}
            dataset_time_stride = {dataset_time_stride}
            save_population_dataset = {"true" if save_population_dataset else "false"}
            save_coherence_dataset = {"true" if save_coherence_dataset else "false"}
            save_xtp_dataset = {"true" if save_xtp_dataset else "false"}
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
            susceptibility_enabled = {"true" if xtp_susceptibility_enabled else "false"}
            {susceptibility_sweep_block.rstrip()}
            susceptibility_plot_enabled = {"true" if xtp_susceptibility_plot_enabled else "false"}
            susceptibility_plot_overview_enabled = true
            susceptibility_plot_grid_enabled = true
            susceptibility_plot_components_enabled = true
            susceptibility_plot_conductivity_enabled = true
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
    assert config.cmd.scratch_rho_storage_dtype == "auto"
    assert config.cmd.dataset_time_stride == 1
    assert config.cmd.save_population_dataset is True
    assert config.cmd.save_coherence_dataset is True
    assert config.cmd.save_xtp_dataset is True
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
    assert config.xtp.susceptibility_enabled is False
    assert config.xtp.susceptibility_output_dir == "outputs/susceptibility"
    assert config.xtp.susceptibility_orders == (1,)
    assert np.isclose(config.xtp.susceptibility_omega_min, 0.05)
    assert np.isclose(config.xtp.susceptibility_omega_max, 0.15)
    assert config.xtp.susceptibility_num_frequencies == 11
    assert np.isclose(config.xtp.susceptibility_eps, 1.0e-14)
    assert config.xtp.susceptibility_plot_enabled is False
    assert config.xtp.susceptibility_plot_conductivity_enabled is True


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


def test_simulation_can_drop_rho_orders_after_generating_compact_datasets(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(tmp_path, model_path, keep_rho_orders=False)
    simulation = QXTISimulation.from_file(config_path)

    outputs = simulation.run()

    assert "rho_order_0" not in outputs
    assert "rho_order_1" not in outputs
    assert "rho_population_kxky_data" in outputs
    assert "rho_coherence_kxky_data" in outputs
    assert "xtp_current_spectrum_data" in outputs
    assert not any((tmp_path / "cmd").glob("rho_order_*.npy"))
    assert not (tmp_path / "cmd" / ".scratch_rho").exists()

    population_dataset = load_dataset_npz(outputs["rho_population_kxky_data"])
    coherence_dataset = load_dataset_npz(outputs["rho_coherence_kxky_data"])
    assert "equilibrium_population_frame" in population_dataset
    assert "coherence_frames_complex" in coherence_dataset
    assert np.iscomplexobj(np.asarray(coherence_dataset["coherence_frames_complex"]))


def test_simulation_auto_uses_float16_scratch_for_temporary_rho_storage(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(
        tmp_path,
        model_path,
        keep_rho_orders=False,
        save_population_dataset=False,
        save_coherence_dataset=False,
        save_xtp_dataset=False,
    )
    simulation = QXTISimulation.from_file(config_path)

    runtime_cmd_cfg = simulation._cmd_runtime_config()

    assert runtime_cmd_cfg.rho_storage_dtype == "float16_complex"


def test_simulation_cmd_datasets_can_subsample_time_axis(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(tmp_path, model_path, keep_rho_orders=False, dataset_time_stride=3)
    simulation = QXTISimulation.from_file(config_path)

    outputs = simulation.run()

    population_dataset = load_dataset_npz(outputs["rho_population_kxky_data"])
    coherence_dataset = load_dataset_npz(outputs["rho_coherence_kxky_data"])
    population_frames = np.asarray(population_dataset["population_frames"])
    coherence_frames = np.asarray(coherence_dataset["coherence_frames_complex"])
    time_axis = np.asarray(population_dataset["time_axis"], dtype=float)

    assert len(time_axis) < simulation.build_cmd(simulation.build_hamiltonian()).timegrid.Nt
    assert population_frames.shape[0] == len(time_axis)
    assert coherence_frames.shape[0] == len(time_axis)


def test_simulation_can_skip_response_datasets_by_config(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(
        tmp_path,
        model_path,
        keep_rho_orders=False,
        save_population_dataset=False,
        save_coherence_dataset=False,
    )
    simulation = QXTISimulation.from_file(config_path)

    outputs = simulation.run()

    assert "rho_population_kxky_data" not in outputs
    assert "rho_coherence_kxky_data" not in outputs
    assert "xtp_current_spectrum_data" in outputs
    assert not (tmp_path / "cmd" / "data" / "population_kx_ky_per_band.npz").exists()
    assert not (tmp_path / "cmd" / "data" / "coherence_kx_ky_per_pair.npz").exists()


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


def test_response_graphics_can_run_from_compact_saved_datasets_without_rho(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available in this environment.")

    from qxti.graphics.graphics import plot_response_graphics_from_saved_data

    model_path = write_model_file(tmp_path)
    config_path = write_config_file(tmp_path, model_path, keep_rho_orders=False)
    simulation = QXTISimulation.from_file(config_path)
    simulation.run()

    response_outputs = plot_response_graphics_from_saved_data(config_path)

    assert "rho_population_snapshots" in response_outputs
    assert "rho_coherence_snapshots" in response_outputs
    for path in response_outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_response_graphics_skip_disabled_response_datasets(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available in this environment.")

    from qxti.graphics.graphics import plot_response_graphics_from_saved_data

    model_path = write_model_file(tmp_path)
    config_path = write_config_file(
        tmp_path,
        model_path,
        keep_rho_orders=False,
        save_population_dataset=False,
        save_coherence_dataset=False,
    )
    simulation = QXTISimulation.from_file(config_path)
    simulation.run()

    response_outputs = plot_response_graphics_from_saved_data(config_path)

    assert response_outputs == {}


def test_susceptibility_scan_saves_current_based_conductivity_tensors(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(
        tmp_path,
        model_path,
        xtp_susceptibility_enabled=True,
        xtp_susceptibility_orders="[1, 2]",
        xtp_susceptibility_num_frequencies=2,
    )

    outputs = SusceptibilityScanRunner.from_file(config_path).run()
    dataset = load_dataset_npz(outputs["xtp_susceptibility_data"])

    sigma_order_1 = np.asarray(dataset["sigma_order_1_tensor"], dtype=np.complex128)
    sigma_order_2 = np.asarray(dataset["sigma_order_2_tensor"], dtype=np.complex128)
    sigma_order_1_indices = np.asarray(dataset["sigma_order_1_available_indices"], dtype=int)
    sigma_order_2_indices = np.asarray(dataset["sigma_order_2_available_indices"], dtype=int)

    assert sigma_order_1.shape == (2, 2, 2)
    assert sigma_order_2.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(
        sigma_order_1_indices,
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int),
    )
    np.testing.assert_array_equal(
        sigma_order_2_indices,
        np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=int),
    )
    assert np.any(np.isfinite(sigma_order_1[:, 0, 0]))
    assert np.any(np.isfinite(sigma_order_1[:, 1, 0]))
    assert np.any(np.isfinite(sigma_order_2[:, 0, 0, 0]))
    assert np.any(np.isfinite(sigma_order_2[:, 1, 0, 0]))


def test_susceptibility_graphics_generate_outputs_from_saved_dataset(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available in this environment.")

    from qxti.graphics.graphics import plot_susceptibility_graphics_from_saved_data

    model_path = write_model_file(tmp_path)
    config_path = write_config_file(
        tmp_path,
        model_path,
        xtp_susceptibility_enabled=True,
        xtp_susceptibility_orders="[1]",
        xtp_susceptibility_num_frequencies=2,
        xtp_susceptibility_plot_enabled=True,
    )
    SusceptibilityScanRunner.from_file(config_path).run()

    outputs = plot_susceptibility_graphics_from_saved_data(config_path)

    assert "susceptibility_order_1_grid" in outputs
    assert "susceptibility_order_1_xx" in outputs
    assert "susceptibility_order_1_yx" in outputs
    assert "conductivity_order_1_grid" in outputs
    assert "conductivity_order_1_xx" in outputs
    assert "conductivity_order_1_yx" in outputs
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_simulation_can_skip_xtp_dataset_by_config(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(
        tmp_path,
        model_path,
        keep_rho_orders=False,
        save_xtp_dataset=False,
    )
    simulation = QXTISimulation.from_file(config_path)

    outputs = simulation.run()

    assert "xtp_current_spectrum_data" not in outputs
    assert not (tmp_path / "cmd" / "data" / "current_spectrum.npz").exists()


def test_susceptibility_scan_runner_generates_dataset(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(
        tmp_path,
        model_path,
        xtp_susceptibility_enabled=True,
        xtp_susceptibility_orders="[1, 2]",
        xtp_susceptibility_num_frequencies=2,
    )
    outputs = SusceptibilityScanRunner.from_file(config_path).run()

    assert "xtp_susceptibility_data" in outputs
    assert not (tmp_path / "cmd").exists()
    assert not (tmp_path / "cmd" / "data").exists()
    dataset = load_dataset_npz(outputs["xtp_susceptibility_data"])
    assert tuple(int(order) for order in dataset["orders"]) == (1, 2)
    assert np.asarray(dataset["laser_omega_axis"], dtype=float).shape == (2,)
    tensor1 = np.asarray(dataset["chi_order_1_tensor"], dtype=np.complex128)
    tensor2 = np.asarray(dataset["chi_order_2_tensor"], dtype=np.complex128)
    assert tensor1.shape == (2, 2, 2)
    assert tensor2.shape == (2, 2, 2, 2)
    available1 = np.asarray(dataset["chi_order_1_available_indices"], dtype=int)
    available2 = np.asarray(dataset["chi_order_2_available_indices"], dtype=int)
    np.testing.assert_array_equal(
        available1,
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int),
    )
    np.testing.assert_array_equal(
        available2,
        np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=int),
    )
    assert np.any(np.isfinite(tensor1))
    assert np.any(np.isfinite(tensor2[:, :, 0, 0]))


def test_susceptibility_scan_runner_accepts_special_solver_section_without_cmd(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available in this environment.")

    model_path = write_model_file(tmp_path)
    config_path = tmp_path / "inputParams_susceptibility_only.cfg"
    config_path.write_text(
        dedent(
            f"""
            [hamiltonian]
            source_file = {model_path}
            function_name = H
            basis_size = 2
            dimension = 2
            basis_type = spin

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
            ncycles = 1.0
            envname = gauss
            cep = 0.0
            t0 = 2.0
            phix = 0.0
            thetaz = 0.0
            phiz = 0.0
            blaser = 0.5
            alaser = 0.5

            [xtp]
            bz_mask_enabled = true
            bz_mask_radius_percent = 80.0
            bz_mask_sigma = 0.75
            susceptibility_enabled = true
            susceptibility_output_dir = {tmp_path / "susceptibility"}
            susceptibility_orders = [1]
            susceptibility_omega_values = [0.7, 0.9]
            susceptibility_eps = 1.0e-14
            susceptibility_plot_enabled = true
            susceptibility_plot_conductivity_enabled = true

            [susceptibility_solver]
            max_order = 1
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

            """
        )
    )

    config = QXTIConfig.from_file(config_path)
    assert config.cmd == type(config.cmd)()
    assert config.susceptibility_solver.max_order == 1
    assert config.xtp.susceptibility_enabled is True
    assert config.xtp.susceptibility_plot_enabled is True
    assert config.xtp.susceptibility_plot_conductivity_enabled is True

    outputs = SusceptibilityScanRunner.from_file(config_path).run()

    assert "xtp_susceptibility_data" in outputs
    assert "susceptibility_order_1_overview" in outputs
    assert "conductivity_order_1_overview" in outputs
    assert not (tmp_path / "cmd").exists()
    dataset = load_dataset_npz(outputs["xtp_susceptibility_data"])
    assert tuple(int(order) for order in dataset["orders"]) == (1,)
    assert np.asarray(dataset["laser_omega_axis"], dtype=float).shape == (2,)


def test_simulation_can_skip_all_saved_rho_postprocessing_by_config(tmp_path: Path) -> None:
    model_path = write_model_file(tmp_path)
    config_path = write_config_file(
        tmp_path,
        model_path,
        keep_rho_orders=False,
        save_population_dataset=False,
        save_coherence_dataset=False,
        save_xtp_dataset=False,
    )
    simulation = QXTISimulation.from_file(config_path)

    outputs = simulation.run()

    assert "rho_population_kxky_data" not in outputs
    assert "rho_coherence_kxky_data" not in outputs
    assert "xtp_current_spectrum_data" not in outputs
    assert not (tmp_path / "cmd" / "data").exists()


def test_harmonic_graphics_skip_when_xtp_dataset_disabled_and_missing(tmp_path: Path) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available in this environment.")

    from qxti.graphics.graphics import plot_harmonic_graphics_from_saved_data

    model_path = write_model_file(tmp_path)
    config_path = write_config_file(
        tmp_path,
        model_path,
        keep_rho_orders=False,
        save_xtp_dataset=False,
    )
    simulation = QXTISimulation.from_file(config_path)
    simulation.run()

    harmonic_outputs = plot_harmonic_graphics_from_saved_data(config_path)

    assert harmonic_outputs == {}
