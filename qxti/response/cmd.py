from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qxti.grids import KGrid, TimeGrid
from qxti.physics import Hamiltonian, LaserSystem, OperatorFactory
from qxti.solvers import Solver

from .distributions import T1T2Relaxation, fermi_dirac


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class CMD:
    """Density-matrix response driver coupled to one Hamiltonian and laser system.

    The class exposes the interface described in the UML diagram and produces
    density-matrix tensors shaped as ``(Nk, Nt, Nb, Nb)``. The zeroth order is
    the thermal equilibrium density matrix. The first order stores the
    time-domain deviation from equilibrium. Higher orders are currently
    initialized to zero unless a future perturbative solver supplies them.
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        laser_system: LaserSystem,
        kgrid: KGrid,
        timegrid: TimeGrid,
        operator_factory: OperatorFactory,
        solver: Solver,
        max_order: int,
        gamma_population: float,
        gamma_coherence: float,
        temperature: float,
        fermi_level: float,
        basis: str,
        gauge: str,
        include_intraband: bool,
        include_interband: bool,
        include_dephasing: bool,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.laser_system = laser_system
        self.kgrid = kgrid
        self.timegrid = timegrid
        self.operator_factory = operator_factory
        self.solver = solver
        self.max_order = int(max_order)
        self.gamma_population = float(gamma_population)
        self.gamma_coherence = float(gamma_coherence)
        self.temperature = float(temperature)
        self.fermi_level = float(fermi_level)
        self.basis = self._normalize_basis(basis)
        self.gauge = self._normalize_gauge(gauge)
        self.include_intraband = bool(include_intraband)
        self.include_interband = bool(include_interband)
        self.include_dephasing = bool(include_dephasing)

        if self.max_order < 0:
            raise ValueError("max_order must be non-negative.")
        if self.gamma_population < 0.0:
            raise ValueError("gamma_population must be non-negative.")
        if self.gamma_coherence < 0.0:
            raise ValueError("gamma_coherence must be non-negative.")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if self.operator_factory.hamiltonian is not self.hamiltonian:
            raise ValueError("operator_factory must be built from the same Hamiltonian instance.")

        self.relaxation_model = T1T2Relaxation.from_rates(
            self.gamma_population,
            self.gamma_coherence,
        )
        self._time_domain_cache: dict[int, ComplexArray] | None = None
        self._frequency_domain_cache: dict[int, ComplexArray] | None = None

    def rho_equilibrium(self, k: NDArray[np.float64]) -> ComplexArray:
        """Return the equilibrium density matrix at one k-point."""

        kx, ky, kz = self._k_components(k)
        energies = self.hamiltonian.eigenvalues(kx, ky, kz)
        occupations = np.asarray(
            fermi_dirac(energies, self.fermi_level, self.temperature),
            dtype=float,
        )
        rho_band = np.diag(occupations).astype(np.complex128)

        if self.basis == "band":
            return self.hamiltonian.validate_matrix(rho_band)
        return self.hamiltonian.transform_from_band_basis(rho_band, kx, ky, kz)

    def compute_rho_order(self, order: int) -> ComplexArray:
        """Return one density-matrix order with shape ``(Nk, Nt, Nb, Nb)``."""

        all_orders = self.compute_all_orders()
        if order not in all_orders:
            raise ValueError(f"Order {order} is not available.")
        return all_orders[order]

    def compute_all_orders(self) -> dict[int, ComplexArray]:
        """Return the full dictionary of density-matrix orders."""

        return self.solve_time_domain()

    def solve_time_domain(self) -> dict[int, ComplexArray]:
        """Solve the density-matrix dynamics on the full k-grid."""

        if self._time_domain_cache is not None:
            return self._time_domain_cache

        target_times = self.timegrid.generate()
        k_points = self.kgrid.points()
        nk = len(k_points)
        nt = len(target_times)
        nb = self.hamiltonian.basis_size

        equilibrium_tensor = np.empty((nk, nt, nb, nb), dtype=np.complex128)
        full_tensor = np.empty_like(equilibrium_tensor)

        for ik, k_point in enumerate(k_points):
            rho0 = self.rho_equilibrium(k_point)
            equilibrium_tensor[ik] = np.broadcast_to(rho0, (nt, nb, nb)).copy()

            times, states = self.solver.solve(
                self._equation_of_motion,
                self.timegrid.t0,
                self.timegrid.tf,
                rho0,
                self.timegrid.initial_h,
                np.asarray(k_point, dtype=float),
            )
            full_tensor[ik] = self._resample_density_trajectory(times, states, target_times)

        rho_orders: dict[int, ComplexArray] = {0: equilibrium_tensor}
        if self.max_order >= 1:
            rho_orders[1] = full_tensor - equilibrium_tensor
            for order in range(2, self.max_order + 1):
                rho_orders[order] = np.zeros_like(full_tensor)

        self._time_domain_cache = rho_orders
        return rho_orders

    def solve_frequency_domain(self) -> dict[int, ComplexArray]:
        """FFT-transform the time-domain density matrices along the time axis."""

        if self._frequency_domain_cache is not None:
            return self._frequency_domain_cache

        rho_orders = self.solve_time_domain()
        window = np.asarray(
            self.timegrid.apply_window(np.ones(self.timegrid.Nt, dtype=float)),
            dtype=float,
        )
        nfft = self.timegrid.Nt * self.timegrid.padding_factor if self.timegrid.zero_padding else self.timegrid.Nt

        transformed: dict[int, ComplexArray] = {}
        for order, tensor in rho_orders.items():
            weighted = tensor * window[np.newaxis, :, np.newaxis, np.newaxis]
            transformed[order] = np.asarray(np.fft.fft(weighted, n=nfft, axis=1), dtype=np.complex128)

        self._frequency_domain_cache = transformed
        return transformed

    def save_density_matrices(self, output_dir: str) -> None:
        """Save all available time-domain density matrices as ``.npy`` files."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for order, tensor in self.solve_time_domain().items():
            np.save(output_path / f"rho_order_{order}.npy", tensor)

    def _equation_of_motion(
        self,
        t: float,
        rho: ComplexArray,
        k_point: NDArray[np.float64],
    ) -> ComplexArray:
        kx, ky, kz = self._k_components(k_point)
        crystal_hamiltonian = self._hamiltonian_in_basis(kx, ky, kz)
        interaction_hamiltonian = self._interaction_hamiltonian(t, kx, ky, kz)
        total_hamiltonian = crystal_hamiltonian + interaction_hamiltonian

        commutator = total_hamiltonian @ rho - rho @ total_hamiltonian
        derivative = -1.0j * commutator

        if self.include_dephasing:
            derivative = derivative + self._dephasing_term(rho, k_point)

        return np.asarray(derivative, dtype=np.complex128)

    def _hamiltonian_in_basis(self, kx: float, ky: float, kz: float) -> ComplexArray:
        return self.operator_factory.hamiltonian_operator(
            kx,
            ky,
            kz,
            basis=self.basis,
        )

    def _interaction_hamiltonian(self, t: float, kx: float, ky: float, kz: float) -> ComplexArray:
        interaction = np.zeros(
            (self.hamiltonian.basis_size, self.hamiltonian.basis_size),
            dtype=np.complex128,
        )

        if self.gauge == "velocity":
            field = self._field_vector(self.laser_system.vector_potential(t))
            for index, direction in enumerate(("x", "y", "z")):
                if index >= self.hamiltonian.dimension:
                    break
                operator = self.operator_factory.velocity_operator(
                    kx,
                    ky,
                    kz,
                    direction,
                    basis="working",
                )
                operator = self._filter_transition_channels(operator, kx, ky, kz)
                interaction += -field[index] * operator
        else:
            field = self._field_vector(self.laser_system.electric_field(t))
            for index, direction in enumerate(("x", "y", "z")):
                if index >= self.hamiltonian.dimension:
                    break
                operator = self.operator_factory.dipole_operator(
                    kx,
                    ky,
                    kz,
                    direction,
                    basis="working",
                )
                operator = self._filter_transition_channels(operator, kx, ky, kz)
                interaction += -field[index] * operator

        if self.basis == "band":
            return self.hamiltonian.transform_to_band_basis(interaction, kx, ky, kz)
        return self.hamiltonian.validate_matrix(interaction)

    def _filter_transition_channels(
        self,
        operator: ComplexArray,
        kx: float,
        ky: float,
        kz: float,
    ) -> ComplexArray:
        if self.include_intraband and self.include_interband:
            return self.hamiltonian.validate_matrix(operator)
        if not self.include_intraband and not self.include_interband:
            return np.zeros_like(operator, dtype=np.complex128)

        operator_band = self.hamiltonian.transform_to_band_basis(operator, kx, ky, kz)
        diagonal = np.diag(np.diag(operator_band)).astype(np.complex128)
        off_diagonal = operator_band - diagonal

        filtered = np.zeros_like(operator_band, dtype=np.complex128)
        if self.include_intraband:
            filtered += diagonal
        if self.include_interband:
            filtered += off_diagonal

        return self.hamiltonian.transform_from_band_basis(filtered, kx, ky, kz)

    def _dephasing_term(
        self,
        rho: ComplexArray,
        k_point: NDArray[np.float64],
    ) -> ComplexArray:
        rho_eq = self.rho_equilibrium(k_point)
        return self.relaxation_model.term(rho, rho_eq)

    @staticmethod
    def _resample_density_trajectory(
        times: list[float] | NDArray[np.float64],
        states: list[ComplexArray] | NDArray[np.complex128],
        target_times: NDArray[np.float64],
    ) -> ComplexArray:
        source_times = np.asarray(times, dtype=float)
        source_states = np.asarray(states, dtype=np.complex128)
        if source_states.shape[0] != source_times.shape[0]:
            raise ValueError("The solver returned inconsistent time/state lengths.")

        if np.allclose(source_times, target_times) and source_states.shape[0] == target_times.shape[0]:
            return np.asarray(source_states, dtype=np.complex128)

        nt = len(target_times)
        nb = source_states.shape[1]
        resampled = np.empty((nt, nb, nb), dtype=np.complex128)

        for row in range(nb):
            for col in range(nb):
                series = source_states[:, row, col]
                real_part = np.interp(target_times, source_times, np.real(series))
                imag_part = np.interp(target_times, source_times, np.imag(series))
                resampled[:, row, col] = real_part + 1.0j * imag_part

        return resampled

    @staticmethod
    def _normalize_basis(basis: str) -> str:
        key = basis.strip().lower()
        if key in {"orbital", "working", "original"}:
            return "working"
        if key in {"band", "bands"}:
            return "band"
        raise ValueError("basis must be 'orbital'/'working' or 'band'.")

    @staticmethod
    def _normalize_gauge(gauge: str) -> str:
        key = gauge.strip().lower()
        if key not in {"velocity", "length"}:
            raise ValueError("gauge must be 'velocity' or 'length'.")
        return key

    @staticmethod
    def _k_components(k_point: NDArray[np.float64]) -> tuple[float, float, float]:
        k_vector = np.asarray(k_point, dtype=float)
        if k_vector.shape != (3,):
            raise ValueError("k must have shape (3,).")
        return float(k_vector[0]), float(k_vector[1]), float(k_vector[2])

    @staticmethod
    def _field_vector(values: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (3,):
            raise ValueError("LaserSystem must return 3 Cartesian components.")
        return vector
