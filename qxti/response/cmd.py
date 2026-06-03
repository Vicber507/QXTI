from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from qxti.grids import KGrid, TimeGrid
from qxti.physics import BandGaugeFrame, Hamiltonian, LaserSystem, OperatorFactory
from qxti.solvers import Solver

from .distributions import T1T2Relaxation, bose_einstein, fermi_dirac, full_occupation, maxwell_boltzmann, valence_occupation


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


class CMD:
    """Recursive perturbative density-matrix solver in length gauge.

    The implementation follows the standard length-gauge equation

        d rho^(s) / dt =
            -(i omega + gamma) rho^(s)
            + E(t) · [D_k rho^(s-1)]

    where, in the internal band basis,

        D_k rho = grad_k rho - i [A, rho]

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
        population_time: float,
        coherence_time: float,
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
        self.population_time = float(population_time)
        self.coherence_time = float(coherence_time)
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
        if self.population_time <= 0.0 and not np.isinf(self.population_time):
            raise ValueError("population_time must be strictly positive or infinite.")
        if self.coherence_time <= 0.0 and not np.isinf(self.coherence_time):
            raise ValueError("coherence_time must be strictly positive or infinite.")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if self.operator_factory.hamiltonian is not self.hamiltonian:
            raise ValueError("operator_factory must be built from the same Hamiltonian instance.")

        self.gamma_population = 0.0 if np.isinf(self.population_time) else 1.0 / self.population_time
        self.gamma_coherence = 0.0 if np.isinf(self.coherence_time) else 1.0 / self.coherence_time
        self._has_population_relaxation = self.gamma_population > 0.0
        self._has_coherence_relaxation = self.gamma_coherence > 0.0
        self.relaxation_model = T1T2Relaxation(
            T1=self.population_time,
            T2=self.coherence_time,
        )
        self._diag_indices = np.diag_indices(self.hamiltonian.basis_size)
        self._offdiag_mask = ~np.eye(self.hamiltonian.basis_size, dtype=bool)
        self.band_gauge_frame = BandGaugeFrame(
            hamiltonian=self.hamiltonian,
            kgrid=self.kgrid,
        )
        self._time_domain_cache: dict[int, ComplexArray] | None = None
        self._frequency_domain_cache: dict[int, ComplexArray] | None = None

    def rho_equilibrium(self, k: NDArray[np.float64]) -> ComplexArray:
        """Return the equilibrium density matrix at one k-point."""

        rho_band = self._rho_equilibrium_band(self._k_index(k))

        if self.basis == "band":
            return self.hamiltonian.validate_matrix(rho_band)
        return self._transform_one_matrix_from_band_basis(rho_band, self._k_index(k))

    def compute_rho_order(self, order: int) -> ComplexArray:
        """Return one density-matrix order with shape ``(Nk, Nt, Nb, Nb)``."""

        all_orders = self.compute_all_orders()
        if order not in all_orders:
            raise ValueError(f"Order {order} is not available.")
        return all_orders[order]

    def compute_all_orders(self) -> dict[int, ComplexArray]:
        """Return the full dictionary of density-matrix orders."""

        return self.solve_time_domain_in_memory()

    def solve_time_domain_in_memory(self) -> dict[int, ComplexArray]:
        """Solve all density-matrix orders and keep them in memory.

        This is the legacy/convenience path. It is useful for small grids and
        tests, but large ``Nk * Nt * Nb * Nb`` runs should prefer
        :meth:`solve_time_domain`.
        """

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

    def solve_time_domain_legacy(self) -> dict[int, ComplexArray]:
        """Backward-compatible name for :meth:`solve_time_domain_in_memory`."""

        return self.solve_time_domain_in_memory()

    def solve_frequency_domain(self) -> dict[int, ComplexArray]:
        """FFT-transform the time-domain density matrices along the time axis."""

        if self._frequency_domain_cache is not None:
            return self._frequency_domain_cache

        rho_orders = self.solve_time_domain_in_memory()
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

    def solve_time_domain(self, output_dir: str | Path) -> dict[int, Path]:
        """Solve density-matrix orders sequentially and save each one to disk.

        This is the primary low-memory time-domain path. It keeps only the
        previous order in the internal band basis, because that is the only
        tensor needed to build the next perturbative source term. Use
        :meth:`solve_time_domain_in_memory` for the legacy all-orders-in-RAM
        behavior.
        """

        if self.gauge != "length":
            raise NotImplementedError(
                "CMD currently implements the recursive perturbative equation "
                "only in length gauge."
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        target_times = np.asarray(self.timegrid.generate(), dtype=float)
        k_points = np.asarray(self.kgrid.points(), dtype=float)
        field_series = self._field_series(target_times)
        total_order_solves = max(0, self.max_order * len(k_points))
        completed_solves = 0
        saved_paths: dict[int, Path] = {}

        self._emit_progress(
            f"CMD streaming start: {self.max_order} driven orders, {len(k_points)} k-points, "
            f"{total_order_solves} order/k-point solves total."
        )

        previous_order_band = self._equilibrium_tensor_band(k_points, target_times)
        saved_paths[0] = self._save_order_tensor(
            output_path,
            order=0,
            tensor_band=previous_order_band,
        )
        self._emit_progress(f"CMD saved order 0: '{saved_paths[0].name}'.")

        for order in range(1, self.max_order + 1):
            self._emit_progress(
                f"CMD order {order}/{self.max_order}: building source terms."
            )
            driving_components = self._build_driving_components(
                previous_order_band,
                k_points,
            )
            current_order_band, completed_solves = self._solve_single_order_band(
                k_points,
                target_times,
                field_series,
                driving_components,
                order=order,
                completed_solves=completed_solves,
                total_order_solves=total_order_solves,
            )
            saved_paths[order] = self._save_order_tensor(
                output_path,
                order=order,
                tensor_band=current_order_band,
            )
            self._emit_progress(f"CMD saved order {order}: '{saved_paths[order].name}'.")
            previous_order_band = current_order_band

        self._time_domain_cache = None
        self._frequency_domain_cache = None
        return saved_paths

    def save_time_domain_orders(self, output_dir: str | Path) -> dict[int, Path]:
        """Alias for the primary low-memory :meth:`solve_time_domain` path."""

        return self.solve_time_domain(output_dir)

    def solve_time_domain_to_directory(self, output_dir: str | Path) -> dict[int, Path]:
        """Backward-compatible alias for :meth:`solve_time_domain`."""

        return self.solve_time_domain(output_dir)

    def save_density_matrices(self, output_dir: str) -> None:
        """Save all available time-domain density matrices as ``.npy`` files."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if self._time_domain_cache is None:
            self.solve_time_domain(output_path)
            return
        for order, tensor in self.solve_time_domain_in_memory().items():
            np.save(output_path / f"rho_order_{order}.npy", tensor)

    def _solve_orders_in_band_basis(
        self,
        k_points: FloatArray,
        target_times: FloatArray,
    ) -> dict[int, ComplexArray]:
        equilibrium = self._equilibrium_tensor_band(k_points, target_times)
        orders_band: dict[int, ComplexArray] = {0: equilibrium}
        total_order_solves = max(0, self.max_order * len(k_points))
        completed_solves = 0
        field_series = self._field_series(target_times)

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
                field_series,
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
        for ik in range(nk):
            rho0_band = self._rho_equilibrium_band(ik)
            equilibrium_tensor[ik] = np.broadcast_to(rho0_band, (nt, nb, nb)).copy()
        return equilibrium_tensor

    def _solve_single_order_band(
        self,
        k_points: FloatArray,
        target_times: FloatArray,
        field_series: FloatArray,
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

        for ik, k_point in enumerate(k_points):
            omega_matrix = self.band_gauge_frame.omega_matrix(ik)
            source_series = self._field_weighted_source_series(
                field_series,
                driving_components[ik],
            )
            solved[ik] = self._solve_linear_order_band_on_grid(
                target_times,
                source_series,
                omega_matrix,
            )
            completed_solves += 1
            self._emit_progress(
                f"CMD progress: order {order}/{self.max_order}, "
                f"k-point {ik + 1}/{nk}, "
                f"global {completed_solves}/{total_order_solves}."
            )

        return solved, completed_solves

    def _solve_linear_order_band_on_grid(
        self,
        target_times: FloatArray,
        source_series: ComplexArray,
        omega_matrix: ComplexArray,
    ) -> ComplexArray:
        nt = len(target_times)
        nb = self.hamiltonian.basis_size
        rho_series = np.zeros((nt, nb, nb), dtype=np.complex128)
        damping = self._damping_matrix()
        lambda_matrix = damping + 1.0j * omega_matrix

        for it in range(nt - 1):
            dt = float(target_times[it + 1] - target_times[it])
            if dt <= 0.0:
                raise ValueError("target_times must be strictly increasing.")
            propagator = np.exp(-lambda_matrix * dt)
            rho_next = (
                propagator * rho_series[it]
                + 0.5 * dt * (propagator * source_series[it] + source_series[it + 1])
            )
            rho_series[it + 1] = self.hamiltonian.validate_matrix(rho_next)

        return rho_series

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
        gradient_components = self._k_gradient_components(previous_order)
        connection_commutator = self._connection_commutator_components(previous_order)

        if self.include_intraband:
            components += gradient_components

        if self.include_interband:
            components += connection_commutator

        return components

    def _field_series(self, target_times: FloatArray) -> FloatArray:
        field = np.asarray(self.laser_system.electric_field(target_times), dtype=float)
        return np.atleast_2d(field)

    def _field_weighted_source_series(
        self,
        field_series: FloatArray,
        source_components: ComplexArray,
    ) -> ComplexArray:
        active_dim = min(self.hamiltonian.dimension, source_components.shape[1], field_series.shape[1])
        if active_dim <= 0:
            raise ValueError("No active field/source dimensions are available for CMD.")
        return np.asarray(
            np.einsum(
                "ta,tabc->tbc",
                field_series[:, :active_dim],
                source_components[:, :active_dim],
                optimize=True,
            ),
            dtype=np.complex128,
        )

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

    def _connection_commutator_components(self, tensor: ComplexArray) -> ComplexArray:
        nk, nt, nb, _ = tensor.shape
        components = np.zeros((nk, nt, 3, nb, nb), dtype=np.complex128)

        for axis, direction in enumerate(("x", "y", "z")):
            if axis >= self.hamiltonian.dimension:
                break
            connection = self.band_gauge_frame.connection(direction)
            left = np.einsum(
                "kij,ktjl->ktil",
                connection,
                tensor,
                optimize=True,
            )
            right = np.einsum(
                "ktij,kjl->ktil",
                tensor,
                connection,
                optimize=True,
            )
            components[:, :, axis] = -1.0j * (left - right)

        return np.asarray(components, dtype=np.complex128)

    def _rho_equilibrium_band(self, index: int) -> ComplexArray:
        energies = self.band_gauge_frame.energies[index]
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
        del k_points
        return self.band_gauge_frame.transform_from_band_basis(tensor)

    def _transform_one_matrix_from_band_basis(
        self,
        matrix: ComplexArray,
        index: int,
    ) -> ComplexArray:
        unitary = self.band_gauge_frame.eigenvectors[index]
        return np.asarray(unitary @ matrix @ unitary.conj().T, dtype=np.complex128)

    def _save_order_tensor(
        self,
        output_dir: Path,
        *,
        order: int,
        tensor_band: ComplexArray,
    ) -> Path:
        tensor = (
            tensor_band
            if self.basis == "band"
            else self._transform_tensor_from_band_basis(tensor_band, self.kgrid.points())
        )
        path = output_dir / f"rho_order_{order}.npy"
        np.save(path, np.asarray(tensor, dtype=np.complex128))
        return path

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

        if self._has_population_relaxation:
            derivative[self._diag_indices] = -self.gamma_population * rho[self._diag_indices]

        if self._has_coherence_relaxation:
            derivative[self._offdiag_mask] = -self.gamma_coherence * rho[self._offdiag_mask]

        return derivative

    def _damping_matrix(self) -> ComplexArray:
        damping = np.zeros((self.hamiltonian.basis_size, self.hamiltonian.basis_size), dtype=np.complex128)
        if self._has_population_relaxation:
            damping[self._diag_indices] = self.gamma_population
        if self._has_coherence_relaxation:
            damping[self._offdiag_mask] = self.gamma_coherence
        return damping

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
            "valence": "valence_occupation",
            "valence_occupation": "valence_occupation",
            "filled_valence": "valence_occupation",
            "semiconductor": "valence_occupation",
            "full": "full_occupation",
            "full_occupation": "full_occupation",
            "identity": "full_occupation",
        }
        if key not in aliases:
            raise ValueError(
                "distribution must be fermi_dirac, maxwell_boltzmann, "
                "bose_einstein, valence_occupation, or full_occupation."
            )
        return aliases[key]

    @staticmethod
    def _resolve_distribution(distribution: str):
        if distribution == "fermi_dirac":
            return fermi_dirac
        if distribution == "maxwell_boltzmann":
            return maxwell_boltzmann
        if distribution == "bose_einstein":
            return bose_einstein
        if distribution == "valence_occupation":
            return valence_occupation
        if distribution == "full_occupation":
            return full_occupation
        raise ValueError(f"Unsupported distribution '{distribution}'.")

    @staticmethod
    def _k_components(k_point: NDArray[np.float64]) -> tuple[float, float, float]:
        k_vector = np.asarray(k_point, dtype=float)
        if k_vector.shape != (3,):
            raise ValueError("k must have shape (3,).")
        return float(k_vector[0]), float(k_vector[1]), float(k_vector[2])

    def _k_index(self, k_point: NDArray[np.float64]) -> int:
        k_vector = np.asarray(k_point, dtype=float)
        if k_vector.shape != (3,):
            raise ValueError("k must have shape (3,).")
        matches = np.where(np.all(np.isclose(self.kgrid.points(), k_vector[np.newaxis, :], atol=1.0e-12), axis=1))[0]
        if matches.size == 0:
            raise ValueError("The requested k-point is not present in the configured KGrid.")
        return int(matches[0])

    @staticmethod
    def _field_vector(values: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (3,):
            raise ValueError("LaserSystem must return 3 Cartesian components.")
        return vector

    @staticmethod
    def _emit_progress(message: str) -> None:
        print(f"[CMD] {message}")
