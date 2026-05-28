from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.grids import FrequencyGrid, KGrid, TimeGrid
from qxti.physics import Hamiltonian, Laser, LaserSystem, OperatorFactory
from qxti.response import CMD, T1T2Relaxation, XTP, bose_einstein, fermi_dirac, maxwell_boltzmann, t1_t2_relaxation
from qxti.solvers import AdamsBashforth2Solver


class ToyTwoBandHamiltonian(Hamiltonian):
    def default_params(self) -> dict[str, float]:
        return {"mass": 0.4, "velocity": 1.2}

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


def build_cmd_stack(*, max_order: int = 2) -> tuple[CMD, OperatorFactory]:
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
                fwhm=20.0,
                envelope="constant",
                theta=0.5 * np.pi,
                phi=0.0,
            )
        ]
    )
    kgrid = KGrid(kx_values=np.array([0.05]), ky_values=np.array([0.0]), kz_values=np.array([0.0]), dimension=2)
    timegrid = TimeGrid(0.0, 0.4, 11, zero_padding=True, padding_factor=2)
    solver = AdamsBashforth2Solver(tolerance=1.0e-8)
    cmd = CMD(
        hamiltonian=hamiltonian,
        laser_system=laser_system,
        kgrid=kgrid,
        timegrid=timegrid,
        operator_factory=operator_factory,
        solver=solver,
        max_order=max_order,
        gamma_population=0.02,
        gamma_coherence=0.05,
        temperature=0.02,
        fermi_level=0.0,
        basis="band",
        gauge="velocity",
        include_intraband=True,
        include_interband=True,
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
    velocity = factory.velocity("x", 0.05, 0.0, 0.0)
    current = factory.current("x", 0.05, 0.0, 0.0)
    dipole = factory.dipole("x", 0.05, 0.0, 0.0)
    position = factory.position("x", 0.05, 0.0, 0.0)
    inverse_mass = factory.inverse_mass("x", "x", 0.05, 0.0, 0.0)
    berry = factory.berry_connection("x", 0.05, 0.0, 0.0)

    assert occupations[0] > occupations[1]
    assert np.all(mb >= 0.0)
    assert np.all(be >= 0.0)
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


def test_cmd_time_and_frequency_domain_outputs_have_expected_shapes(tmp_path: Path) -> None:
    cmd, _ = build_cmd_stack(max_order=2)

    rho_eq = cmd.rho_equilibrium(np.array([0.05, 0.0, 0.0], dtype=float))
    rho_orders = cmd.solve_time_domain()
    rho_freq = cmd.solve_frequency_domain()
    cmd.save_density_matrices(str(tmp_path))

    assert rho_eq.shape == (2, 2)
    assert np.isclose(np.trace(rho_eq), 1.0, atol=1.0e-6)
    assert set(rho_orders) == {0, 1, 2}
    assert rho_orders[0].shape == (1, 11, 2, 2)
    assert rho_orders[1].shape == (1, 11, 2, 2)
    assert np.allclose(rho_orders[2], 0.0)
    assert rho_freq[0].shape == (1, 22, 2, 2)
    assert (tmp_path / "rho_order_0.npy").exists()
    assert (tmp_path / "rho_order_1.npy").exists()
    assert (tmp_path / "rho_order_2.npy").exists()


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
