from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.grids import FrequencyGrid, KGrid, TimeGrid
from qxti.data import HarmonicData, ResponseData
from qxti.graphics import HarmonicGraphics, ResponseGraphics
from qxti.physics import Hamiltonian, Laser, LaserSystem, OperatorFactory
from qxti.response import CMD, T1T2Relaxation, XTP, bose_einstein, fermi_dirac, full_occupation, maxwell_boltzmann, t1_t2_relaxation, valence_occupation
from qxti.solvers import RKF45Solver


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


def cycles_for_fwhm(omega: float, fwhm: float) -> float:
    return fwhm * omega / (2.0 * np.pi)


def build_cmd_stack(
    *,
    max_order: int = 2,
    gauge: str = "length",
    distribution: str = "fermi_dirac",
    include_intraband: bool = True,
    include_interband: bool = True,
    kgrid: KGrid | None = None,
) -> tuple[CMD, OperatorFactory]:
    hamiltonian = ToyTwoBandHamiltonian(
        model_name="toy-response",
        basis_size=2,
        dimension=2,
        dk_derivative=1.0e-5,
    )
    operator_factory = OperatorFactory(hamiltonian=hamiltonian, basis="band")
    laser_system = LaserSystem(
        [
            Laser(
                omega=0.8,
                E0=0.05,
                ellipticity=0.0,
                num_cycles=cycles_for_fwhm(0.8, 20.0),
                envelope="constant",
                theta=0.5 * np.pi,
                phi=0.0,
            )
        ]
    )
    if kgrid is None:
        kgrid = KGrid(
            kx_values=np.array([0.05]),
            ky_values=np.array([0.0]),
            kz_values=np.array([0.0]),
            dimension=2,
        )
    timegrid = TimeGrid(0.0, 0.4, 11, zero_padding=True, padding_factor=2)
    solver = RKF45Solver(tolerance=1.0e-6, h_max=timegrid.dt, max_iterations=10000)
    cmd = CMD(
        hamiltonian=hamiltonian,
        laser_system=laser_system,
        kgrid=kgrid,
        timegrid=timegrid,
        operator_factory=operator_factory,
        solver=solver,
        max_order=max_order,
        population_time=50.0,
        coherence_time=20.0,
        temperature=0.02,
        fermi_level=0.0,
        distribution=distribution,
        basis="band",
        gauge=gauge,
        include_intraband=include_intraband,
        include_interband=include_interband,
        include_dephasing=True,
    )
    return cmd, operator_factory


def test_response_package_imports_are_consistent() -> None:
    from qxti.response import CMD as ImportedCMD
    from qxti.response import T1T2Relaxation as ImportedT1T2Relaxation
    from qxti.response import fermi_dirac as imported_fermi_dirac

    assert ImportedCMD is CMD
    assert ImportedT1T2Relaxation is T1T2Relaxation
    assert np.isclose(imported_fermi_dirac(0.0, 0.0, 0.1), 0.5)


def test_distributions_and_operator_factory_behave_consistently() -> None:
    hamiltonian = ToyTwoBandHamiltonian(
        model_name="toy-response",
        basis_size=2,
        dimension=2,
        dk_derivative=1.0e-5,
    )
    factory = OperatorFactory(hamiltonian=hamiltonian, basis="band")

    occupations = np.asarray(fermi_dirac(np.array([-1.0, 1.0]), 0.0, 0.1), dtype=float)
    mb = np.asarray(maxwell_boltzmann(np.array([0.0, 1.0]), 0.0, 0.2), dtype=float)
    be = np.asarray(bose_einstein(np.array([1.0, 2.0]), 0.0, 0.2), dtype=float)
    valence = np.asarray(valence_occupation(np.array([-5.0, -1.0, 0.0, 7.0]), 0.0, 0.2), dtype=float)
    velocity = factory.velocity("x", 0.05, 0.0, 0.0)
    current = factory.current("x", 0.05, 0.0, 0.0)
    dipole = factory.dipole("x", 0.05, 0.0, 0.0)
    position = factory.position("x", 0.05, 0.0, 0.0)
    inverse_mass = factory.inverse_mass("x", "x", 0.05, 0.0, 0.0)
    berry = factory.berry_connection("x", 0.05, 0.0, 0.0)

    assert occupations[0] > occupations[1]
    assert np.all(mb >= 0.0)
    assert np.all(be >= 0.0)
    np.testing.assert_allclose(valence, np.array([1.0, 1.0, 0.0, 0.0]))
    assert velocity.shape == (2, 2)
    assert np.allclose(current, -velocity)
    assert dipole.shape == (2, 2)
    assert np.allclose(dipole, position)
    assert np.allclose(dipole, dipole.conj().T)
    assert inverse_mass.shape == (2, 2)
    assert berry.shape == (2, 2)


def test_t1_t2_relaxation_model_matches_cmd_convention() -> None:
    rho = np.array(
        [
            [0.7 + 0.0j, 0.1 - 0.2j],
            [0.1 + 0.2j, 0.3 + 0.0j],
        ],
        dtype=complex,
    )
    rho_eq = np.array(
        [
            [0.6 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.4 + 0.0j],
        ],
        dtype=complex,
    )
    model = T1T2Relaxation(T1=10.0, T2=20.0)
    derivative = model.term(rho, rho_eq)
    derivative_from_wrapper = t1_t2_relaxation(rho, rho_eq, 10.0, 20.0)

    assert np.allclose(derivative, derivative_from_wrapper)
    assert np.isclose(derivative[0, 0], -(0.7 - 0.6) / 10.0)
    assert np.isclose(derivative[1, 1], -(0.3 - 0.4) / 10.0)
    assert np.isclose(derivative[0, 1], -(0.1 - 0.2j) / 20.0)
    assert np.isclose(derivative[1, 0], -(0.1 + 0.2j) / 20.0)


def test_cmd_dephasing_uses_inverse_population_and_coherence_times() -> None:
    cmd, _ = build_cmd_stack()
    rho = np.array(
        [
            [0.3 + 0.0j, 0.2 - 0.1j],
            [0.2 + 0.1j, -0.4 + 0.0j],
        ],
        dtype=np.complex128,
    )

    derivative = cmd._dephasing_term(rho)

    np.testing.assert_allclose(derivative[0, 0], -(1.0 / 50.0) * rho[0, 0])
    np.testing.assert_allclose(derivative[1, 1], -(1.0 / 50.0) * rho[1, 1])
    np.testing.assert_allclose(derivative[0, 1], -(1.0 / 20.0) * rho[0, 1])
    np.testing.assert_allclose(derivative[1, 0], -(1.0 / 20.0) * rho[1, 0])


def test_cmd_accepts_distribution_selection() -> None:
    cmd, _ = build_cmd_stack(distribution="maxwell_boltzmann")
    rho_eq = cmd.rho_equilibrium(np.array([0.05, 0.0, 0.0], dtype=float))

    assert rho_eq.shape == (2, 2)
    assert np.all(np.isfinite(rho_eq))
    np.testing.assert_allclose(rho_eq, np.diag(np.diag(rho_eq)), atol=1.0e-12)


def test_cmd_valence_occupation_distribution_builds_valence_filled_equilibrium_density() -> None:
    cmd, _ = build_cmd_stack(distribution="valence_occupation")
    rho_eq = cmd.rho_equilibrium(np.array([0.05, 0.0, 0.0], dtype=float))

    np.testing.assert_allclose(rho_eq, np.diag([1.0, 0.0]), atol=1.0e-12)


def test_cmd_full_occupation_alias_matches_valence_occupation() -> None:
    cmd_full, _ = build_cmd_stack(distribution="full_occupation")
    cmd_valence, _ = build_cmd_stack(distribution="valence_occupation")

    rho_full = cmd_full.rho_equilibrium(np.array([0.05, 0.0, 0.0], dtype=float))
    rho_valence = cmd_valence.rho_equilibrium(np.array([0.05, 0.0, 0.0], dtype=float))
    np.testing.assert_allclose(rho_full, rho_valence, atol=1.0e-12)


def test_cmd_time_and_frequency_domain_outputs_have_expected_shapes(tmp_path: Path) -> None:
    cmd, _ = build_cmd_stack(max_order=2)

    rho_eq = cmd.rho_equilibrium(np.array([0.05, 0.0, 0.0], dtype=float))
    rho_orders = cmd.solve_time_domain()
    rho_freq = cmd.solve_frequency_domain()
    cmd.save_density_matrices(str(tmp_path))

    assert rho_eq.shape == (2, 2)
    assert np.isclose(np.trace(rho_eq), 1.0, atol=1.0e-6)
    np.testing.assert_allclose(rho_eq, np.diag(np.diag(rho_eq)), atol=1.0e-12)
    assert np.all(np.real(np.diag(rho_eq)) >= 0.0)
    assert np.all(np.real(np.diag(rho_eq)) <= 1.0)
    assert set(rho_orders) == {0, 1, 2}
    assert rho_orders[0].shape == (1, 11, 2, 2)
    assert rho_orders[1].shape == (1, 11, 2, 2)
    assert rho_orders[2].shape == (1, 11, 2, 2)
    assert np.max(np.abs(rho_orders[1])) > 0.0
    assert np.max(np.abs(rho_orders[2])) > 0.0
    assert rho_freq[0].shape == (1, 22, 2, 2)
    assert rho_freq[1].shape == (1, 22, 2, 2)
    assert rho_freq[2].shape == (1, 22, 2, 2)
    assert (tmp_path / "rho_order_0.npy").exists()
    assert (tmp_path / "rho_order_1.npy").exists()
    assert (tmp_path / "rho_order_2.npy").exists()


def test_response_population_heatmap_data_and_plot(tmp_path: Path) -> None:
    cmd, _ = build_cmd_stack(max_order=3)
    rho_orders = cmd.solve_time_domain()
    response_data = ResponseData(cmd)
    data = response_data.population_heatmap_data(
        orders=(0, 1, 2, 3),
        k_aggregation="mean",
        rho_orders=rho_orders,
    )

    assert data["population_map"].shape == (2, 11)
    assert data["population_traces"].shape == (11, 2)
    assert data["population_frames"].shape == (11, 2, 1)
    assert data["orders"] == (0, 1, 2, 3)
    assert np.all(np.isfinite(data["population_map"]))

    if "matplotlib" not in sys.modules and importlib.util.find_spec("matplotlib") is None:
        return

    output_path = ResponseGraphics.plot_population_heatmap(
        data,
        tmp_path / "population_heatmap.png",
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    coherence_data = response_data.coherence_heatmap_data(
        orders=(0, 1, 2, 3),
        k_aggregation="mean",
        component="magnitude",
        rho_orders=rho_orders,
    )
    assert coherence_data["coherence_map"].shape == (1, 11)
    assert coherence_data["pair_labels"] == ["0-1"]

    coherence_output_path = ResponseGraphics.plot_coherence_heatmap(
        coherence_data,
        tmp_path / "coherence_heatmap.png",
    )
    assert coherence_output_path.exists()
    assert coherence_output_path.stat().st_size > 0


def test_response_population_kxky_animation_data_and_plot(tmp_path: Path) -> None:
    cmd, _ = build_cmd_stack(
        max_order=1,
        kgrid=KGrid(
            kx_values=np.array([-0.10, 0.0, 0.10]),
            ky_values=np.array([-0.10, 0.10]),
            kz_values=np.array([0.0]),
            dimension=2,
        ),
    )
    rho_orders = cmd.solve_time_domain()
    response_data = ResponseData(cmd)
    data = response_data.population_kxky_animation_data(
        orders=(0, 1),
        rho_orders=rho_orders,
    )

    assert data["population_frames"].shape == (11, 2, 2, 3)
    assert np.all(np.isfinite(data["population_frames"]))

    if "matplotlib" not in sys.modules and importlib.util.find_spec("matplotlib") is None:
        return

    from matplotlib import animation

    if not animation.writers.is_available("pillow"):
        return

    output_path = ResponseGraphics.animate_population_kxky_maps(
        data,
        tmp_path / "population_kx_ky.gif",
        fps=8,
        frame_stride=2,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    coherence_data = response_data.coherence_kxky_animation_data(
        orders=(0, 1),
        component="magnitude",
        rho_orders=rho_orders,
    )
    assert coherence_data["coherence_frames"].shape == (11, 1, 2, 3)

    coherence_snapshot_indices = ResponseGraphics.resolve_snapshot_indices(
        np.asarray(coherence_data["time_axis"], dtype=float),
        num_snapshots=3,
    )
    snapshot_output = ResponseGraphics.plot_coherence_snapshots(
        coherence_data,
        tmp_path / "coherence_snapshots.png",
        snapshot_indices=coherence_snapshot_indices,
    )
    assert snapshot_output.exists()
    assert snapshot_output.stat().st_size > 0


def test_cmd_intraband_gradient_source_is_connected() -> None:
    cmd, _ = build_cmd_stack(
        max_order=1,
        include_intraband=True,
        include_interband=False,
        kgrid=KGrid(
            kx_values=np.array([-0.10, 0.0, 0.10]),
            ky_values=np.array([0.0]),
            kz_values=np.array([0.0]),
            dimension=2,
        ),
    )

    rho_orders = cmd.solve_time_domain()

    assert rho_orders[1].shape == (3, 11, 2, 2)
    assert np.max(np.abs(rho_orders[1])) > 0.0


def test_cmd_velocity_gauge_is_rejected_for_recursive_solver() -> None:
    cmd, _ = build_cmd_stack(gauge="velocity")

    try:
        cmd.solve_time_domain()
    except NotImplementedError as exc:
        assert "length gauge" in str(exc)
    else:
        raise AssertionError("CMD should reject velocity gauge for the recursive solver.")


def test_cmd_band_recursive_solver_is_not_limited_by_external_solver_iteration_caps() -> None:
    cmd, _ = build_cmd_stack(max_order=1)
    cmd.solver.max_iterations = 1

    rho_orders = cmd.solve_time_domain()

    assert rho_orders[1].shape[-2:] == (2, 2)
    assert np.max(np.abs(rho_orders[1])) > 0.0


def test_resample_density_trajectory_rejects_incomplete_time_coverage() -> None:
    target_times = np.array([0.0, 0.5, 1.0], dtype=float)
    source_times = np.array([0.0, 0.25], dtype=float)
    states = np.zeros((2, 2, 2), dtype=np.complex128)

    try:
        CMD._resample_density_trajectory(source_times, states, target_times)
    except RuntimeError as exc:
        assert "does not cover" in str(exc)
    else:
        raise AssertionError("Resampling should fail when the solver trajectory is incomplete.")


def test_xtp_basic_observables_run_with_cmd_output() -> None:
    cmd, operator_factory = build_cmd_stack(max_order=1)
    rho_orders = cmd.compute_all_orders()
    xtp = XTP(
        hamiltonian=cmd.hamiltonian,
        rho_orders=rho_orders,
        kgrid=cmd.kgrid,
        timegrid=cmd.timegrid,
        frequencygrid=FrequencyGrid(0.0, 5.0, 32),
        operator_factory=operator_factory,
        directions=["x", "y"],
        orders=[0, 1],
    )

    polarization = xtp.total_polarization()
    current = xtp.total_current()
    susceptibility = xtp.susceptibility(1)

    assert polarization.shape == (len(cmd.timegrid), 3)
    assert current.shape == (len(cmd.timegrid), 3)
    assert susceptibility.shape == (len(cmd.timegrid), 3)
    assert np.all(np.isfinite(polarization))
    assert np.all(np.isfinite(current))


def test_xtp_first_order_polarization_matches_manual_bz_dipole_integral() -> None:
    cmd, operator_factory = build_cmd_stack(
        max_order=1,
        kgrid=KGrid(
            kx_values=np.linspace(-0.5 * np.pi, 0.5 * np.pi, 3),
            ky_values=np.linspace(-0.5 * np.pi, 0.5 * np.pi, 2),
            kz_values=np.array([0.0]),
            dimension=2,
        ),
    )
    rho_orders = cmd.compute_all_orders()
    xtp = XTP(
        hamiltonian=cmd.hamiltonian,
        rho_orders=rho_orders,
        kgrid=cmd.kgrid,
        timegrid=cmd.timegrid,
        frequencygrid=FrequencyGrid(0.0, 5.0, 32),
        operator_factory=operator_factory,
        directions=["x"],
        orders=[1],
    )

    polarization = xtp.polarization(1)
    nkx, nky, nkz = cmd.kgrid.shape
    nt = len(cmd.timegrid)
    point_values = np.zeros((cmd.kgrid.total_points, nt), dtype=np.complex128)

    for ik, (kx, ky, kz) in enumerate(cmd.kgrid.points()):
        dipole = operator_factory.dipole("x", float(kx), float(ky), float(kz))
        point_values[ik] = np.einsum(
            "mn,tnm->t",
            dipole,
            rho_orders[1][ik],
            optimize=True,
        )

    grid_values = point_values.reshape(nkx, nky, nkz, nt)
    manual = np.trapezoid(
        np.trapezoid(grid_values[:, :, 0, :], x=np.asarray(cmd.kgrid.ky_values, dtype=float), axis=1),
        x=np.asarray(cmd.kgrid.kx_values, dtype=float),
        axis=0,
    )

    np.testing.assert_allclose(polarization[:, 0], np.real(manual), atol=1.0e-9)
    np.testing.assert_allclose(polarization[:, 1:], 0.0, atol=1.0e-12)


def test_xtp_bz_mask_reduces_edge_contributions_with_circular_gaussian_cutoff() -> None:
    cmd, operator_factory = build_cmd_stack(
        max_order=1,
        kgrid=KGrid(
            kx_values=np.linspace(-0.5 * np.pi, 0.5 * np.pi, 5),
            ky_values=np.linspace(-0.5 * np.pi, 0.5 * np.pi, 5),
            kz_values=np.array([0.0]),
            dimension=2,
        ),
    )
    rho_orders = cmd.compute_all_orders()
    xtp_plain = XTP(
        hamiltonian=cmd.hamiltonian,
        rho_orders=rho_orders,
        kgrid=cmd.kgrid,
        timegrid=cmd.timegrid,
        frequencygrid=FrequencyGrid(0.0, 5.0, 32),
        operator_factory=operator_factory,
        directions=["x"],
        orders=[0],
    )
    xtp_masked = XTP(
        hamiltonian=cmd.hamiltonian,
        rho_orders=rho_orders,
        kgrid=cmd.kgrid,
        timegrid=cmd.timegrid,
        frequencygrid=FrequencyGrid(0.0, 5.0, 32),
        operator_factory=operator_factory,
        directions=["x"],
        orders=[0],
        bz_mask_enabled=True,
        bz_mask_radius_percent=60.0,
        bz_mask_sigma=0.94,
    )

    point_values = np.ones((cmd.kgrid.total_points, len(cmd.timegrid)), dtype=np.complex128)
    plain = xtp_plain._integrate_over_brillouin_zone(point_values)
    masked = xtp_masked._integrate_over_brillouin_zone(point_values)
    weights = xtp_masked._bz_mask_weights()
    plot_data = xtp_masked.bz_mask_plot_data()

    assert weights.shape == cmd.kgrid.shape
    assert np.count_nonzero(weights == 0.0) > 0
    assert np.max(weights) <= 1.0
    assert np.min(weights) >= 0.0
    assert np.all(np.abs(masked) < np.abs(plain))
    assert xtp_masked.bz_mask_summary()["enabled"] is True
    assert plot_data["integration_region"].shape == (5, 5)
    assert plot_data["mask_weights"].shape == (5, 5)


def test_xtp_current_frequency_domain_matches_manual_fft_and_plot(tmp_path: Path) -> None:
    cmd, operator_factory = build_cmd_stack(max_order=1)
    rho_orders = cmd.compute_all_orders()
    xtp = XTP(
        hamiltonian=cmd.hamiltonian,
        rho_orders=rho_orders,
        kgrid=cmd.kgrid,
        timegrid=cmd.timegrid,
        frequencygrid=FrequencyGrid(0.0, 5.0, 32),
        operator_factory=operator_factory,
        directions=["x", "y"],
        orders=[0, 1],
    )

    omega_axis, current_spectrum = xtp.total_current_frequency_domain()
    current_time = xtp.total_current()
    window = np.asarray(
        cmd.timegrid.apply_window(np.ones(len(cmd.timegrid), dtype=float)),
        dtype=float,
    )
    weighted = current_time * window[:, np.newaxis]
    nfft = len(cmd.timegrid) * cmd.timegrid.padding_factor
    manual_spectrum = cmd.timegrid.dt * np.fft.fft(weighted, n=nfft, axis=0)
    manual_omega = np.asarray(cmd.timegrid.frequency_axis(), dtype=float)

    np.testing.assert_allclose(omega_axis, manual_omega)
    np.testing.assert_allclose(current_spectrum, manual_spectrum, atol=1.0e-10)
    assert current_spectrum.shape == (nfft, 3)
    assert np.max(np.abs(current_spectrum[:, 0])) > 0.0

    harmonic_data = HarmonicData(
        xtp,
        electric_field_time=cmd.laser_system.electric_field(cmd.timegrid.generate()),
    ).current_spectrum_data()
    equilibrium_current = np.column_stack(
        [xtp.current(0, direction) for direction in ("x", "y")]
        + [np.zeros(len(cmd.timegrid), dtype=float)]
    )
    induced_current = current_time - equilibrium_current
    _, induced_spectrum = xtp._fft_time_signal(induced_current)

    assert harmonic_data["orders"] == (1,)
    assert harmonic_data["all_orders"] == (0, 1)
    assert bool(harmonic_data["equilibrium_subtracted"]) is True
    assert harmonic_data["omega_axis"].shape == (nfft,)
    assert harmonic_data["current_spectrum"].shape == (nfft, 3)
    assert harmonic_data["current_magnitude"].shape == (nfft, 3)
    assert harmonic_data["current_time"].shape == (len(cmd.timegrid), 3)
    assert harmonic_data["current_time_total"].shape == (len(cmd.timegrid), 3)
    assert harmonic_data["equilibrium_current_time"].shape == (len(cmd.timegrid), 3)
    assert harmonic_data["electric_field_time"].shape == (len(cmd.timegrid), 3)
    assert harmonic_data["bz_mask"]["enabled"] is False
    assert harmonic_data["kx_grid"].shape == harmonic_data["ky_grid"].shape
    assert harmonic_data["integration_region"].shape == harmonic_data["mask_weights"].shape
    np.testing.assert_allclose(harmonic_data["integration_region"], 1.0)
    np.testing.assert_allclose(harmonic_data["mask_weights"], 1.0)
    np.testing.assert_allclose(harmonic_data["current_time_total"], current_time, atol=1.0e-12)
    np.testing.assert_allclose(harmonic_data["current_time"], induced_current, atol=1.0e-12)
    np.testing.assert_allclose(harmonic_data["current_spectrum"], induced_spectrum, atol=1.0e-10)

    if "matplotlib" not in sys.modules and importlib.util.find_spec("matplotlib") is None:
        return

    current_time_output = HarmonicGraphics.plot_field_current_time_comparison(
        np.asarray(cmd.timegrid.generate(), dtype=float),
        np.asarray(harmonic_data["electric_field_time"], dtype=float),
        np.asarray(harmonic_data["current_time"], dtype=float),
        tmp_path / "field_current_time.png",
        directions=("x", "y"),
        include_total=True,
    )
    assert current_time_output.exists()
    assert current_time_output.stat().st_size > 0

    output_path = HarmonicGraphics.plot_current_magnitude_spectrum(
        np.asarray(harmonic_data["omega_axis"], dtype=float),
        np.asarray(harmonic_data["current_spectrum"], dtype=np.complex128),
        tmp_path / "current_spectrum.png",
        orders=tuple(harmonic_data["orders"]),
        directions=("x", "y"),
        positive_only=True,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    circular_output_path = HarmonicGraphics.plot_circular_current_spectrum(
        np.asarray(harmonic_data["omega_axis"], dtype=float),
        np.asarray(harmonic_data["current_spectrum"], dtype=np.complex128),
        tmp_path / "current_circular_spectrum.png",
        orders=tuple(harmonic_data["orders"]),
        positive_only=True,
    )
    assert circular_output_path.exists()
    assert circular_output_path.stat().st_size > 0
