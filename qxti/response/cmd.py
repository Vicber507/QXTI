from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from qxti.grids import KGrid, TimeGrid
from qxti.physics import Hamiltonian, LaserSystem, OperatorFactory
from qxti.solvers import Solver

from .distributions import bose_einstein, fermi_dirac, maxwell_boltzmann


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class CMD:
    """Recursive perturbative density-matrix solver in length gauge.

    The implementation follows the standard length-gauge equation

        d rho^(s) / dt =
            -(i omega + gamma) rho^(s)
            + E(t) · [grad_k rho^(s-1) - i [d, rho^(s-1)]]

    on a discrete k-grid. Zeroth order is the equilibrium density matrix and
    higher orders are built iteratively from the previous one.
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
        distribution: str,
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
        self.distribution_name = self._normalize_distribution(distribution)
        self.distribution = self._resolve_distribution(self.distribution_name)
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

        self._diag_indices = np.diag_indices(self.hamiltonian.basis_size)
        self._offdiag_mask = ~np.eye(self.hamiltonian.basis_size, dtype=bool)
        self._time_domain_cache: dict[int, ComplexArray] | None = None
        self._frequency_domain_cache: dict[int, ComplexArray] | None = None

    def rho_equilibrium(self, k: NDArray[np.float64]) -> ComplexArray:
        """Return the equilibrium density matrix at one k-point."""

        kx, ky, kz = self._k_components(k)
        rho_band = self._rho_equilibrium_band(kx, ky, kz)

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
        """Solve the recursive perturbative density-matrix dynamics."""

        if self.gauge != "length":
            raise NotImplementedError(
                "CMD currently implements the recursive perturbative equation "
                "only in length gauge."
            )
        if self._time_domain_cache is not None:
            return self._time_domain_cache

        target_times = np.asarray(self.timegrid.generate(), dtype=float)
        k_points = np.asarray(self.kgrid.points(), dtype=float)
        orders_band = self._solve_orders_in_band_basis(k_points, target_times)

        if self.basis == "band":
            result = orders_band
        else:
            result = {
                order: self._transform_tensor_from_band_basis(tensor, k_points)
                for order, tensor in orders_band.items()
            }

        self._time_domain_cache = result
        self._frequency_domain_cache = None
        return result

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
        """Save all available time-domain density matrices as ``.npy`` and ``.dat`` files."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for order, tensor in self.solve_time_domain().items():
            np.save(output_path / f"rho_order_{order}.npy", tensor)
            self.save_density_matrix_dat(
                output_path / f"rho_order_{order}.dat",
                tensor,
                order=order,
                domain="time",
            )

    def _solve_orders_in_band_basis(
        self,
        k_points: FloatArray,
        target_times: FloatArray,
    ) -> dict[int, ComplexArray]:
        equilibrium = self._equilibrium_tensor_band(k_points, target_times)
        orders_band: dict[int, ComplexArray] = {0: equilibrium}
        total_order_solves = max(0, self.max_order * len(k_points))
        completed_solves = 0

        self._emit_progress(
            f"CMD starting: {self.max_order} driven orders, {len(k_points)} k-points, "
            f"{total_order_solves} order/k-point solves total."
        )

        for order in range(1, self.max_order + 1):
            self._emit_progress(
                f"CMD order {order}/{self.max_order}: building source terms."
            )
            driving_components = self._build_driving_components(
                orders_band[order - 1],
                k_points,
            )
            orders_band[order], completed_solves = self._solve_single_order_band(
                k_points,
                target_times,
                driving_components,
                order=order,
                completed_solves=completed_solves,
                total_order_solves=total_order_solves,
            )
            self._emit_progress(
                f"CMD order {order}/{self.max_order} completed."
            )

        return orders_band

    def _equilibrium_tensor_band(
        self,
        k_points: FloatArray,
        target_times: FloatArray,
    ) -> ComplexArray:
        nk = len(k_points)
        nt = len(target_times)
        nb = self.hamiltonian.basis_size

        equilibrium_tensor = np.empty((nk, nt, nb, nb), dtype=np.complex128)
        for ik, k_point in enumerate(k_points):
            kx, ky, kz = self._k_components(k_point)
            rho0_band = self._rho_equilibrium_band(kx, ky, kz)
            equilibrium_tensor[ik] = np.broadcast_to(rho0_band, (nt, nb, nb)).copy()
        return equilibrium_tensor

    def _solve_single_order_band(
        self,
        k_points: FloatArray,
        target_times: FloatArray,
        driving_components: ComplexArray,
        *,
        order: int,
        completed_solves: int,
        total_order_solves: int,
    ) -> tuple[ComplexArray, int]:
        nk = len(k_points)
        nt = len(target_times)
        nb = self.hamiltonian.basis_size
        solved = np.empty((nk, nt, nb, nb), dtype=np.complex128)
        initial_state = np.zeros((nb, nb), dtype=np.complex128)

        for ik, k_point in enumerate(k_points):
            kx, ky, kz = self._k_components(k_point)
            omega_matrix = self._omega_matrix(kx, ky, kz)
            times, states = self.solver.solve(
                self._order_equation_of_motion,
                self.timegrid.t0,
                self.timegrid.tf,
                initial_state,
                self.timegrid.initial_h,
                target_times,
                driving_components[ik],
                omega_matrix,
            )
            solved[ik] = self._resample_density_trajectory(times, states, target_times)
            completed_solves += 1
            self._emit_progress(
                f"CMD progress: order {order}/{self.max_order}, "
                f"k-point {ik + 1}/{nk}, "
                f"global {completed_solves}/{total_order_solves}."
            )

        return solved, completed_solves

    def save_density_matrix_dat(
        self,
        output_path: str | Path,
        tensor: ComplexArray,
        *,
        order: int,
        domain: str,
        axis_values: FloatArray | None = None,
        axis_label: str = "time",
    ) -> Path:
        """Save one density-matrix tensor as a flat ``.dat`` table."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        nk, nt, nb, _ = tensor.shape
        k_points = np.asarray(self.kgrid.points(), dtype=float)
        if axis_values is None:
            axis = np.asarray(self.timegrid.generate(), dtype=float)
        else:
            axis = np.asarray(axis_values, dtype=float)
        if nt != len(axis):
            axis = np.arange(nt, dtype=float)

        with path.open("w", encoding="ascii") as handle:
            handle.write(f"# domain={domain} order={order}\n")
            handle.write(f"# shape=({nk},{nt},{nb},{nb})\n")
            handle.write(f"# ik it kx ky kz {axis_label} row col real imag\n")
            for ik in range(nk):
                kx, ky, kz = self._k_components(k_points[ik])
                for it in range(nt):
                    axis_value = float(axis[it])
                    for row in range(nb):
                        for col in range(nb):
                            value = tensor[ik, it, row, col]
                            handle.write(
                                f"{ik} {it} {kx:.16e} {ky:.16e} {kz:.16e} "
                                f"{axis_value:.16e} {row} {col} "
                                f"{value.real:.16e} {value.imag:.16e}\n"
                            )

        return path

    def _order_equation_of_motion(
        self,
        t: float,
        rho: ComplexArray,
        source_times: FloatArray,
        driving_components: ComplexArray,
        omega_matrix: ComplexArray,
    ) -> ComplexArray:
        derivative = -1.0j * omega_matrix * rho

        if self.include_dephasing:
            derivative = derivative + self._dephasing_term(rho)

        field = self._field_vector(self.laser_system.electric_field(t))
        source_components = self._interpolate_complex_tensor(
            t,
            source_times,
            driving_components,
        )
        active_dim = min(self.hamiltonian.dimension, source_components.shape[0])
        derivative = derivative + np.tensordot(
            field[:active_dim],
            source_components[:active_dim],
            axes=(0, 0),
        )

        return np.asarray(derivative, dtype=np.complex128)

    def _build_driving_components(
        self,
        previous_order: ComplexArray,
        k_points: FloatArray,
    ) -> ComplexArray:
        nk, nt, nb, _ = previous_order.shape
        components = np.zeros((nk, nt, 3, nb, nb), dtype=np.complex128)

        if self.include_intraband:
            components += self._k_gradient_components(previous_order)

        if self.include_interband:
            for ik, k_point in enumerate(k_points):
                kx, ky, kz = self._k_components(k_point)
                rho_series = previous_order[ik]
                for axis, direction in enumerate(("x", "y", "z")):
                    if axis >= self.hamiltonian.dimension:
                        break
                    dipole = self.operator_factory.dipole_operator(
                        kx,
                        ky,
                        kz,
                        direction,
                        basis="band",
                    )
                    commutator = np.matmul(dipole[np.newaxis, :, :], rho_series) - np.matmul(
                        rho_series,
                        dipole[np.newaxis, :, :],
                    )
                    components[ik, :, axis] += -1.0j * commutator

        return components

    def _k_gradient_components(self, tensor: ComplexArray) -> ComplexArray:
        nk, nt, nb, _ = tensor.shape
        reshaped = tensor.reshape(*self.kgrid.shape, nt, nb, nb)
        gradients = np.zeros((nk, nt, 3, nb, nb), dtype=np.complex128)

        for axis, grid_values in enumerate(
            (self.kgrid.kx_values, self.kgrid.ky_values, self.kgrid.kz_values)
        ):
            if axis >= self.hamiltonian.dimension:
                break
            if len(grid_values) < 2:
                continue
            edge_order = 2 if len(grid_values) >= 3 else 1
            component = np.gradient(
                reshaped,
                np.asarray(grid_values, dtype=float),
                axis=axis,
                edge_order=edge_order,
            )
            gradients[:, :, axis] = np.asarray(component, dtype=np.complex128).reshape(nk, nt, nb, nb)

        return gradients

    def _rho_equilibrium_band(self, kx: float, ky: float, kz: float) -> ComplexArray:
        energies = self.hamiltonian.eigenvalues(kx, ky, kz)
        occupations = np.asarray(
            self.distribution(energies, self.fermi_level, self.temperature),
            dtype=float,
        )
        return np.diag(occupations).astype(np.complex128)

    def _omega_matrix(self, kx: float, ky: float, kz: float) -> ComplexArray:
        energies = np.asarray(self.hamiltonian.eigenvalues(kx, ky, kz), dtype=float)
        omega = energies[:, np.newaxis] - energies[np.newaxis, :]
        return np.asarray(omega, dtype=np.complex128)

    def _transform_tensor_from_band_basis(
        self,
        tensor: ComplexArray,
        k_points: FloatArray,
    ) -> ComplexArray:
        transformed = np.empty_like(tensor)
        for ik, k_point in enumerate(k_points):
            kx, ky, kz = self._k_components(k_point)
            unitary = self.hamiltonian.eigenvectors(kx, ky, kz)
            transformed[ik] = np.matmul(
                np.matmul(unitary[np.newaxis, :, :], tensor[ik]),
                unitary.conj().T[np.newaxis, :, :],
            )
        return transformed

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

        time_tolerance = max(1.0e-12, 1.0e-10 * np.max(np.abs(target_times)))
        if source_times[0] > target_times[0] + time_tolerance or source_times[-1] < target_times[-1] - time_tolerance:
            raise RuntimeError(
                "The solver trajectory does not cover the requested target-time window. "
                f"Available range: [{source_times[0]:.16e}, {source_times[-1]:.16e}], "
                f"requested range: [{target_times[0]:.16e}, {target_times[-1]:.16e}]."
            )

        if source_states.shape[0] == target_times.shape[0] and np.allclose(source_times, target_times):
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
    def _interpolate_complex_tensor(
        t: float,
        source_times: FloatArray,
        tensor_series: ComplexArray,
    ) -> ComplexArray:
        times = np.asarray(source_times, dtype=float)
        values = np.asarray(tensor_series, dtype=np.complex128)
        if values.shape[0] != times.shape[0]:
            raise ValueError("tensor_series must have the same leading length as source_times.")
        if values.shape[0] == 1:
            return np.asarray(values[0], dtype=np.complex128)

        return np.asarray(
            CMD._linear_interpolate_series(t, times, values),
            dtype=np.complex128,
        )

    def _dephasing_term(self, rho: ComplexArray) -> ComplexArray:
        derivative = np.zeros_like(rho, dtype=np.complex128)
        if self.gamma_population > 0.0:
            derivative[self._diag_indices] = -self.gamma_population * rho[self._diag_indices]
        if self.gamma_coherence > 0.0:
            derivative[self._offdiag_mask] = -self.gamma_coherence * rho[self._offdiag_mask]
        return derivative

    @staticmethod
    def _linear_interpolate_series(
        t: float,
        source_times: FloatArray,
        series: ComplexArray,
    ) -> ComplexArray:
        times = np.asarray(source_times, dtype=float)
        values = np.asarray(series, dtype=np.complex128)
        if values.shape[0] != times.shape[0]:
            raise ValueError("series must have the same leading length as source_times.")
        if values.shape[0] == 1 or t <= times[0]:
            return np.asarray(values[0], dtype=np.complex128)
        if t >= times[-1]:
            return np.asarray(values[-1], dtype=np.complex128)

        upper = int(np.searchsorted(times, t, side="right"))
        lower = upper - 1
        t_lower = float(times[lower])
        t_upper = float(times[upper])
        if t_upper <= t_lower:
            return np.asarray(values[lower], dtype=np.complex128)

        weight = (t - t_lower) / (t_upper - t_lower)
        return np.asarray(
            (1.0 - weight) * values[lower] + weight * values[upper],
            dtype=np.complex128,
        )

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
    def _normalize_distribution(distribution: str) -> str:
        key = distribution.strip().lower()
        aliases = {
            "fermi": "fermi_dirac",
            "fermi_dirac": "fermi_dirac",
            "fd": "fermi_dirac",
            "maxwell": "maxwell_boltzmann",
            "maxwell_boltzmann": "maxwell_boltzmann",
            "mb": "maxwell_boltzmann",
            "bose": "bose_einstein",
            "bose_einstein": "bose_einstein",
            "be": "bose_einstein",
        }
        if key not in aliases:
            raise ValueError("distribution must be fermi_dirac, maxwell_boltzmann, or bose_einstein.")
        return aliases[key]

    @staticmethod
    def _resolve_distribution(distribution: str):
        if distribution == "fermi_dirac":
            return fermi_dirac
        if distribution == "maxwell_boltzmann":
            return maxwell_boltzmann
        if distribution == "bose_einstein":
            return bose_einstein
        raise ValueError(f"Unsupported distribution '{distribution}'.")

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

    @staticmethod
    def _emit_progress(message: str) -> None:
        print(f"[CMD] {message}")
