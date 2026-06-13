from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from qxti.grids import FrequencyGrid, KGrid, TimeGrid
from qxti.physics import BandGaugeFrame, Hamiltonian, LaserSystem, OperatorFactory


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


class XTP:
    r"""Compute macroscopic response observables from perturbative density matrices.

    The time-domain polarization of order ``s`` is evaluated as

    .. math::

        P_i^{(s)}(t) =
        \int_{\mathrm{BZ}} d\mathbf{k}\,
        \sum_{nm} d^i_{mn}(\mathbf{k})\,\rho^{(s)}_{nm}(\mathbf{k}, t),

    where the Brillouin-zone integral is approximated on the rectangular
    reciprocal-space box associated with the Hamiltonian lattice. When the
    :class:`~qxti.grids.KGrid` is generated from
    :meth:`Hamiltonian.reciprocal_box_bounds`, this corresponds to square/cubic
    boxes ``[-pi / a_i, pi / a_i]`` in atomic units.
    """

    _DIRECTION_TO_AXIS = {"x": 0, "y": 1, "z": 2}

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        rho_orders: dict[int, ComplexArray],
        kgrid: KGrid,
        timegrid: TimeGrid,
        frequencygrid: FrequencyGrid,
        operator_factory: OperatorFactory,
        directions: list[str],
        orders: list[int],
        band_gauge_frame: BandGaugeFrame | None = None,
        laser_system: LaserSystem | None = None,
        bz_mask_enabled: bool = False,
        bz_mask_radius_percent: float = 100.0,
        bz_mask_sigma: float | None = None,
        bz_mask_sigma_percent_legacy: float | None = None,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.rho_orders = {
            int(order): self._as_complex_tensor(tensor)
            for order, tensor in rho_orders.items()
        }
        self.kgrid = kgrid
        self.timegrid = timegrid
        self.frequencygrid = frequencygrid
        self.laser_system = laser_system
        self.operator_factory = operator_factory
        self.directions = [self._normalize_direction(direction) for direction in directions]
        self.orders = sorted({int(order) for order in orders})
        self.band_gauge_frame = band_gauge_frame
        self.bz_mask_enabled = bool(bz_mask_enabled)
        self.bz_mask_radius_percent = float(bz_mask_radius_percent)
        self.bz_mask_sigma = None if bz_mask_sigma is None else float(bz_mask_sigma)
        self.bz_mask_sigma_percent_legacy = (
            None if bz_mask_sigma_percent_legacy is None else float(bz_mask_sigma_percent_legacy)
        )
        if not self.orders:
            raise ValueError("orders must contain at least one perturbative order.")
        if self.bz_mask_radius_percent <= 0.0:
            raise ValueError("bz_mask_radius_percent must be strictly positive.")
        if self.bz_mask_sigma is not None and self.bz_mask_sigma <= 0.0:
            raise ValueError("bz_mask_sigma must be strictly positive when provided.")
        if self.bz_mask_sigma_percent_legacy is not None and self.bz_mask_sigma_percent_legacy <= 0.0:
            raise ValueError("bz_mask_sigma_percent_legacy must be strictly positive when provided.")
        self._validate_rho_orders()

    def polarization(self, order: int) -> RealArray:
        """Return the macroscopic polarization :math:`P^{(s)}(t)` for one order."""

        rho_tensor = self._rho_tensor(order)
        nk = self.kgrid.total_points
        nt = rho_tensor.shape[1]
        polarization = np.zeros((nt, 3), dtype=np.complex128)
        k_points = self.kgrid.points()

        for direction in self.directions:
            axis = self._direction_axis(direction)
            expectation = np.zeros((nk, nt), dtype=np.complex128)
            cached_dipole = None if self.band_gauge_frame is None else self.band_gauge_frame.connection(direction)

            for ik, (kx, ky, kz) in enumerate(k_points):
                dipole = (
                    self.operator_factory.dipole(direction, float(kx), float(ky), float(kz))
                    if cached_dipole is None
                    else cached_dipole[ik]
                )
                expectation[ik] = np.einsum(
                    "mn,tnm->t",
                    dipole,
                    rho_tensor[ik],
                    optimize=True,
                )

            polarization[:, axis] = self._integrate_over_brillouin_zone(expectation)

        return self._coerce_real_matrix(polarization, name=f"polarization(order={order})")

    def current(self, order: int, direction: str) -> RealArray:
        """Return the macroscopic current component for one order and direction."""

        rho_tensor = self._rho_tensor(order)
        direction = self._normalize_direction(direction)
        nk = self.kgrid.total_points
        nt = rho_tensor.shape[1]
        expectation = np.zeros((nk, nt), dtype=np.complex128)
        k_points = self.kgrid.points()
        cached_current = None if self.band_gauge_frame is None else self.band_gauge_frame.current(direction)

        for ik, (kx, ky, kz) in enumerate(k_points):
            current_operator = (
                self.operator_factory.current(direction, float(kx), float(ky), float(kz))
                if cached_current is None
                else cached_current[ik]
            )
            expectation[ik] = np.einsum(
                "mn,tnm->t",
                current_operator,
                rho_tensor[ik],
                optimize=True,
            )

        integrated = self._integrate_over_brillouin_zone(expectation)
        return self._coerce_real_vector(
            integrated,
            name=f"current(order={order}, direction={direction})",
        )

    def has_current_decomposition(self) -> bool:
        """Return whether intraband/interband current separation is available."""

        return self.band_gauge_frame is not None

    def current_decomposition(self, order: int, direction: str) -> dict[str, RealArray]:
        """Return intraband/interband current components for one order and direction."""

        if self.band_gauge_frame is None:
            raise ValueError(
                "Current decomposition requires a band_gauge_frame so operators and rho share the same band basis."
            )

        rho_tensor = self._rho_tensor(order)
        direction = self._normalize_direction(direction)
        nk = self.kgrid.total_points
        nt = rho_tensor.shape[1]
        intraband_expectation = np.zeros((nk, nt), dtype=np.complex128)
        current_operator = self.band_gauge_frame.current(direction)

        for ik in range(nk):
            diagonal_current = np.diag(current_operator[ik])
            band_diagonal = np.diagonal(rho_tensor[ik], axis1=1, axis2=2)
            intraband_expectation[ik] = np.einsum(
                "n,tn->t",
                diagonal_current,
                band_diagonal,
                optimize=True,
            )

        intraband = self._coerce_real_vector(
            self._integrate_over_brillouin_zone(intraband_expectation),
            name=f"intraband_current(order={order}, direction={direction})",
        )
        total = self.current(order, direction)
        interband = np.asarray(total - intraband, dtype=np.float64)
        return {
            "intraband": intraband,
            "interband": interband,
        }

    def polarization_frequency_domain(self, order: int) -> tuple[RealArray, ComplexArray]:
        """Return ``(omega_axis, P(omega))`` for one perturbative order."""

        polarization = self.polarization(order)
        return self._fft_time_signal(polarization)

    def electric_field_frequency_domain(self) -> tuple[RealArray, ComplexArray]:
        """Return ``(omega_axis, E(omega))`` for the total applied electric field."""

        if self.laser_system is None:
            raise ValueError(
                "laser_system is required to compute electric-field spectra and susceptibilities."
            )

        times = self.timegrid.generate()
        electric_field_t = self.laser_system.electric_field(times)
        return self._fft_time_signal(electric_field_t)

    def linear_susceptibility(
        self,
        *,
        input_direction: str,
        eps: float = 1.0e-14,
    ) -> tuple[RealArray, ComplexArray]:
        """Return chi_ij^(1)(omega) for one input direction j.

        Output shape:
            ``(Nomega, dimension)``
        """

        input_direction = self._normalize_direction(input_direction)
        input_axis = self._direction_axis(input_direction)

        omega_axis, polarization_w = self.polarization_frequency_domain(order=1)
        _, electric_field_w = self.electric_field_frequency_domain()

        denominator = electric_field_w[:, input_axis]
        safe_denominator = np.where(
            np.abs(denominator) > eps,
            denominator,
            np.nan + 0.0j,
        )

        active_dimension = self.hamiltonian.dimension
        with np.errstate(divide="ignore", invalid="ignore"):
            chi = polarization_w[:, :active_dimension] / safe_denominator[:, np.newaxis]

        return omega_axis, np.asarray(chi, dtype=np.complex128)

    def effective_susceptibility_spectrum(
        self,
        *,
        order: int,
        input_direction: str = "auto",
        input_omega: float | None = None,
        eps: float = 1.0e-14,
    ) -> tuple[RealArray, ComplexArray, dict[str, Any]]:
        r"""Return one effective ``chi^(order)(omega)`` spectrum for a repeated input axis.

        For ``order > 1`` this is defined as

        .. math::

            \chi^{(s)}_{i j \ldots j}(\omega)
            \equiv
            \frac{P^{(s)}_i(\omega)}{E_j(\omega_\mathrm{in})^s},

        where ``j`` is the selected input direction and ``omega_in`` defaults to
        the lowest positive carrier frequency present in the configured laser
        system.
        """

        if order < 1:
            raise ValueError("order must be >= 1.")

        resolved_direction = self.resolve_input_direction(
            requested_direction=input_direction,
            input_omega=input_omega,
        )
        resolved_input_omega = self.reference_input_omega() if input_omega is None else float(input_omega)
        input_axis = self._direction_axis(resolved_direction)

        omega_axis, polarization_w = self.polarization_frequency_domain(order=order)
        _, electric_field_w = self.electric_field_frequency_domain()
        input_index = self._nearest_frequency_index(
            omega_axis,
            resolved_input_omega,
            prefer_positive=True,
        )

        denominator = electric_field_w[input_index, input_axis]
        if np.abs(denominator) <= eps:
            raise ValueError(
                f"Electric-field component E_{resolved_direction}(omega={resolved_input_omega}) "
                "is too small to normalize the susceptibility spectrum."
            )

        active_dimension = self.hamiltonian.dimension
        chi = polarization_w[:, :active_dimension] / denominator**order
        component_magnitudes = np.abs(electric_field_w[input_index, :active_dimension])

        metadata = {
            "input_direction": resolved_direction,
            "input_axis": input_axis,
            "input_omega": resolved_input_omega,
            "input_frequency_index": input_index,
            "normalization_mode": "fixed_input_frequency_component",
            "single_component_drive": self._is_single_component_drive(component_magnitudes),
            "input_component_magnitudes": np.asarray(component_magnitudes, dtype=np.float64),
            "field_normalization": np.asarray([denominator], dtype=np.complex128),
        }
        return omega_axis, np.asarray(chi, dtype=np.complex128), metadata

    def susceptibility_tensor_spectrum(
        self,
        *,
        order: int,
        input_direction: str = "auto",
        input_omega: float | None = None,
        eps: float = 1.0e-14,
    ) -> dict[str, Any]:
        """Return one sparse susceptibility tensor spectrum for a single XTP run.

        A single simulation can only determine the tensor components aligned with
        the driven Cartesian axis. The returned tensor therefore stores the
        available column for ``order == 1`` and the repeated-input-axis slice
        ``chi_{i j ... j}^{(order)}`` for higher orders, leaving all unavailable
        components as ``NaN``.
        """

        if order < 1:
            raise ValueError("order must be >= 1.")
        if order not in self.rho_orders:
            raise ValueError(f"rho_orders does not contain order {order}.")

        active_dimension = self.hamiltonian.dimension
        resolved_direction = self.resolve_input_direction(
            requested_direction=input_direction,
            input_omega=input_omega,
        )
        input_axis = self._direction_axis(resolved_direction)

        if order == 1:
            omega_axis, chi_column = self.linear_susceptibility(
                input_direction=resolved_direction,
                eps=eps,
            )
            tensor = np.full(
                (len(omega_axis),) + (active_dimension,) * 2,
                np.nan + 1.0j * np.nan,
                dtype=np.complex128,
            )
            tensor[:, :, input_axis] = chi_column[:, :active_dimension]
            reference_omega = self.reference_input_omega()
            input_index = self._nearest_frequency_index(
                omega_axis,
                reference_omega,
                prefer_positive=True,
            )
            _, electric_field_w = self.electric_field_frequency_domain()
            component_magnitudes = np.abs(electric_field_w[input_index, :active_dimension])
            field_normalization = np.empty(0, dtype=np.complex128)
            normalization_mode = "pointwise_field_component"
            resolved_input_omega = reference_omega
        else:
            omega_axis, chi_column, metadata = self.effective_susceptibility_spectrum(
                order=order,
                input_direction=resolved_direction,
                input_omega=input_omega,
                eps=eps,
            )
            tensor = np.full(
                (len(omega_axis),) + (active_dimension,) * (order + 1),
                np.nan + 1.0j * np.nan,
                dtype=np.complex128,
            )
            repeated_input = (input_axis,) * order
            tensor[(slice(None), slice(None)) + repeated_input] = chi_column[:, :active_dimension]
            component_magnitudes = np.asarray(metadata["input_component_magnitudes"], dtype=np.float64)
            field_normalization = np.asarray(metadata["field_normalization"], dtype=np.complex128)
            normalization_mode = str(metadata["normalization_mode"])
            resolved_input_omega = float(metadata["input_omega"])
            input_index = int(metadata["input_frequency_index"])

        available_indices = [
            (output_axis,) + (input_axis,) * order
            for output_axis in range(active_dimension)
        ]
        component_labels = [
            self._tensor_component_label(indices)
            for indices in available_indices
        ]

        return {
            "omega_axis": np.asarray(omega_axis, dtype=np.float64),
            "tensor": np.asarray(tensor, dtype=np.complex128),
            "available_indices": np.asarray(available_indices, dtype=np.int16),
            "component_labels": component_labels,
            "direction_labels": self._direction_labels(active_dimension),
            "order": int(order),
            "input_direction": resolved_direction,
            "input_axis": int(input_axis),
            "input_omega": float(resolved_input_omega),
            "input_frequency_index": int(input_index),
            "normalization_mode": normalization_mode,
            "single_component_drive": self._is_single_component_drive(component_magnitudes),
            "input_component_magnitudes": np.asarray(component_magnitudes, dtype=np.float64),
            "field_normalization": np.asarray(field_normalization, dtype=np.complex128),
        }

    def reference_input_omega(self) -> float:
        """Return the lowest positive drive frequency used to normalize susceptibilities."""

        if self.laser_system is not None and self.laser_system.lasers:
            positive_omegas = [
                float(laser.omega)
                for laser in self.laser_system.lasers
                if float(laser.omega) > 0.0
            ]
            if positive_omegas:
                return float(min(positive_omegas))

        omega_axis, electric_field_w = self.electric_field_frequency_domain()
        active_dimension = self.hamiltonian.dimension
        positive_mask = omega_axis > 0.0
        if np.any(positive_mask):
            positive_indices = np.flatnonzero(positive_mask)
            magnitudes = np.linalg.norm(
                electric_field_w[positive_indices, :active_dimension],
                axis=1,
            )
            return float(omega_axis[positive_indices[int(np.argmax(magnitudes))]])
        return float(omega_axis[int(np.argmax(np.linalg.norm(electric_field_w[:, :active_dimension], axis=1)))])

    def resolve_input_direction(
        self,
        *,
        requested_direction: str = "auto",
        input_omega: float | None = None,
    ) -> str:
        """Resolve the active input direction used for susceptibility normalization."""

        key = requested_direction.strip().lower()
        if key != "auto":
            return self._normalize_direction(key)

        resolved_input_omega = self.reference_input_omega() if input_omega is None else float(input_omega)
        omega_axis, electric_field_w = self.electric_field_frequency_domain()
        input_index = self._nearest_frequency_index(
            omega_axis,
            resolved_input_omega,
            prefer_positive=True,
        )
        active_dimension = self.hamiltonian.dimension
        component_magnitudes = np.abs(electric_field_w[input_index, :active_dimension])
        axis = int(np.argmax(component_magnitudes))
        return self._direction_labels(active_dimension)[axis]

    def susceptibility(self, order: int) -> RealArray:
        """Backward-compatible alias for the order-resolved polarization response."""

        return self.polarization(order)

    def current_frequency_domain(
        self,
        order: int,
        direction: str,
    ) -> tuple[RealArray, ComplexArray]:
        """Return ``(omega_axis, J(omega))`` for one order and one direction."""

        current = self.current(order, direction)
        omega_axis, spectrum = self._fft_time_signal(current)
        return omega_axis, np.asarray(spectrum, dtype=np.complex128)

    def harmonic_spectrum(self, signal: RealArray | ComplexArray) -> ComplexArray:
        """Return the FFT of one time-domain signal."""

        return np.asarray(np.fft.fft(np.asarray(signal), axis=0), dtype=np.complex128)

    def total_polarization(self) -> RealArray:
        """Return the sum of the polarization over the configured orders."""

        nt = len(self.timegrid)
        total = np.zeros((nt, 3), dtype=np.float64)
        for order in self.orders:
            total += self.polarization(order)
        return total

    def total_current(self) -> RealArray:
        """Return the sum of current contributions over orders and directions."""

        nt = len(self.timegrid)
        total = np.zeros((nt, 3), dtype=np.float64)
        for direction in self.directions:
            axis = self._direction_axis(direction)
            for order in self.orders:
                total[:, axis] += self.current(order, direction)
        return total

    def total_current_decomposition(self) -> dict[str, RealArray]:
        """Return intraband/interband current vectors summed over configured orders."""

        if self.band_gauge_frame is None:
            raise ValueError("total_current_decomposition requires band-basis data.")

        nt = len(self.timegrid)
        intraband = np.zeros((nt, 3), dtype=np.float64)
        interband = np.zeros((nt, 3), dtype=np.float64)
        for direction in self.directions:
            axis = self._direction_axis(direction)
            for order in self.orders:
                parts = self.current_decomposition(order, direction)
                intraband[:, axis] += parts["intraband"]
                interband[:, axis] += parts["interband"]
        return {
            "intraband": intraband,
            "interband": interband,
        }

    def total_current_frequency_domain(self) -> tuple[RealArray, ComplexArray]:
        """Return ``(omega_axis, J_total(omega))`` for all configured directions."""

        return self._fft_time_signal(self.total_current())

    def compute_all(self) -> dict[str, Any]:
        """Return all observables currently implemented by XTP."""

        omega_axis, current_spectrum = self.total_current_frequency_domain()
        outputs: dict[str, Any] = {
        "polarization": self.total_polarization(),
        "current": self.total_current(),
        "omega_axis": omega_axis,
        "current_spectrum": current_spectrum,
        "polarization_frequency_domain": {
            order: self.polarization_frequency_domain(order)[1]
            for order in self.orders
        },
    }

        if self.laser_system is not None and 1 in self.orders:
            outputs["linear_susceptibility"] = {
            direction: self.linear_susceptibility(input_direction=direction)[1]
            for direction in self.directions
        }

        return outputs

    def bz_mask_summary(self) -> dict[str, Any]:
        """Return one serializable description of the Brillouin-zone mask."""

        bounds = self.brillouin_zone_bounds()
        reference_radius = self._mask_reference_radius(bounds)
        radius = self._mask_radius(bounds)
        sigma = self._mask_sigma(bounds)
        return {
            "enabled": self.bz_mask_enabled,
            "shape": "radial" if self.kgrid.dimension == 1 else ("circular" if self.kgrid.dimension == 2 else "spherical"),
            "radius_percent": self.bz_mask_radius_percent,
            "sigma_input": self.bz_mask_sigma,
            "reference_radius": reference_radius,
            "radius": radius,
            "sigma": sigma,
        }

    def bz_mask_plot_data(self) -> dict[str, Any]:
        """Return serializable 2D data describing the integration region and mask."""

        if self.kgrid.dimension != 2:
            raise ValueError("bz_mask_plot_data currently supports only 2D k-grids.")

        kx_mesh, ky_mesh, _ = self.kgrid.mesh(indexing="ij")
        region = np.ones(self.kgrid.shape[:2], dtype=np.float64)
        weights = self._bz_mask_weights()[:, :, 0]
        return {
            "kx_grid": np.asarray(kx_mesh[:, :, 0], dtype=np.float64),
            "ky_grid": np.asarray(ky_mesh[:, :, 0], dtype=np.float64),
            "integration_region": region,
            "mask_weights": np.asarray(weights, dtype=np.float64),
            "mask_metadata": self.bz_mask_summary(),
        }

    def brillouin_zone_bounds(self) -> tuple[tuple[float, float], ...]:
        """Return the rectangular reciprocal box used for BZ integrations."""

        try:
            return tuple(self.hamiltonian.reciprocal_box_bounds())
        except ValueError:
            arrays = (
                np.asarray(self.kgrid.kx_values, dtype=float),
                np.asarray(self.kgrid.ky_values, dtype=float),
                np.asarray(self.kgrid.kz_values, dtype=float),
            )
            bounds: list[tuple[float, float]] = []
            for axis in range(self.kgrid.dimension):
                values = arrays[axis]
                bounds.append((float(values[0]), float(values[-1])))
            return tuple(bounds)

    def _integrate_over_brillouin_zone(self, point_values: ComplexArray) -> ComplexArray:
        values = np.asarray(point_values, dtype=np.complex128)
        if values.ndim < 1:
            raise ValueError("point_values must have at least one dimension.")
        if values.shape[0] != self.kgrid.total_points:
            raise ValueError(
                "point_values first dimension must match the number of k-points in the grid."
            )

        grid = values.reshape(*self.kgrid.shape, *values.shape[1:])
        if self.bz_mask_enabled:
            grid = grid * self._bz_mask_weights().reshape(*self.kgrid.shape, *([1] * (grid.ndim - 3)))
        bounds = self.brillouin_zone_bounds()
        result = grid

        result = self._integrate_axis(
            result,
            axis=2,
            coordinates=np.asarray(self.kgrid.kz_values, dtype=float),
            bounds=bounds[2] if self.kgrid.dimension >= 3 else (0.0, 0.0),
            active=self.kgrid.dimension >= 3,
        )
        result = self._integrate_axis(
            result,
            axis=1,
            coordinates=np.asarray(self.kgrid.ky_values, dtype=float),
            bounds=bounds[1] if self.kgrid.dimension >= 2 else (0.0, 0.0),
            active=self.kgrid.dimension >= 2,
        )
        result = self._integrate_axis(
            result,
            axis=0,
            coordinates=np.asarray(self.kgrid.kx_values, dtype=float),
            bounds=bounds[0],
            active=True,
        )
        return np.asarray(result, dtype=np.complex128)

    def _bz_mask_weights(self) -> RealArray:
        if not self.bz_mask_enabled:
            return np.ones(self.kgrid.shape, dtype=np.float64)

        mesh = self.kgrid.mesh(indexing="ij")
        active_coordinates = [np.asarray(mesh[axis], dtype=float) for axis in range(self.kgrid.dimension)]
        radial_distance = np.sqrt(np.sum([coordinate**2 for coordinate in active_coordinates], axis=0))

        bounds = self.brillouin_zone_bounds()
        radius = self._mask_radius(bounds)
        sigma = self._mask_sigma(bounds)

        weights = np.exp(-0.5 * (radial_distance / sigma) ** 2)
        weights = np.where(radial_distance <= radius, weights, 0.0)
        return np.asarray(weights, dtype=np.float64)

    def _mask_reference_radius(self, bounds: tuple[tuple[float, float], ...]) -> float:
        active_bounds = bounds[: self.kgrid.dimension]
        return float(min(max(abs(lower), abs(upper)) for lower, upper in active_bounds))

    def _mask_radius(self, bounds: tuple[tuple[float, float], ...]) -> float:
        return 0.01 * self.bz_mask_radius_percent * self._mask_reference_radius(bounds)

    def _mask_sigma(self, bounds: tuple[tuple[float, float], ...]) -> float:
        if self.bz_mask_sigma is not None:
            return float(max(self.bz_mask_sigma, 1.0e-15))

        radius = self._mask_radius(bounds)
        sigma_percent = 100.0 if self.bz_mask_sigma_percent_legacy is None else self.bz_mask_sigma_percent_legacy
        sigma = 0.01 * sigma_percent * radius
        return float(max(sigma, 1.0e-15))

    @staticmethod
    def _integrate_axis(
        values: ComplexArray,
        *,
        axis: int,
        coordinates: NDArray[np.float64],
        bounds: tuple[float, float],
        active: bool,
    ) -> ComplexArray:
        if not active:
            return np.asarray(np.take(values, indices=0, axis=axis), dtype=np.complex128)

        if coordinates.size > 1:
            return np.asarray(np.trapezoid(values, x=coordinates, axis=axis), dtype=np.complex128)

        width = float(bounds[1] - bounds[0])
        collapsed = np.asarray(np.take(values, indices=0, axis=axis), dtype=np.complex128)
        return width * collapsed

    def _rho_tensor(self, order: int) -> ComplexArray:
        try:
            return self.rho_orders[int(order)]
        except KeyError as exc:
            raise ValueError(f"rho_orders does not contain order {order}.") from exc

    @staticmethod
    def _nearest_frequency_index(
        omega_axis: RealArray,
        target_omega: float,
        *,
        prefer_positive: bool = False,
    ) -> int:
        omega = np.asarray(omega_axis, dtype=np.float64)
        if not prefer_positive:
            return int(np.argmin(np.abs(omega - float(target_omega))))

        positive_indices = np.flatnonzero(omega >= 0.0)
        if positive_indices.size == 0:
            return int(np.argmin(np.abs(omega - float(target_omega))))
        local_index = int(np.argmin(np.abs(omega[positive_indices] - float(target_omega))))
        return int(positive_indices[local_index])

    @staticmethod
    def _as_complex_tensor(tensor: ComplexArray) -> ComplexArray:
        if isinstance(tensor, np.ndarray) and np.issubdtype(tensor.dtype, np.complexfloating):
            return tensor
        return np.asarray(tensor, dtype=np.complex128)

    @classmethod
    def _direction_labels(cls, dimension: int) -> tuple[str, ...]:
        labels = tuple(cls._DIRECTION_TO_AXIS.keys())
        return labels[:dimension]

    @classmethod
    def _tensor_component_label(cls, indices: tuple[int, ...]) -> str:
        labels = cls._direction_labels(max(indices) + 1 if indices else 1)
        return "".join(labels[index] for index in indices)

    @staticmethod
    def _is_single_component_drive(
        component_magnitudes: NDArray[np.float64],
        *,
        relative_threshold: float = 1.0e-3,
    ) -> bool:
        magnitudes = np.asarray(component_magnitudes, dtype=np.float64)
        if magnitudes.size == 0:
            return True
        reference = float(np.max(magnitudes))
        if reference <= 1.0e-30:
            return True
        active_components = np.count_nonzero(magnitudes >= relative_threshold * reference)
        return bool(active_components <= 1)

    def _fft_time_signal(
        self,
        signal: RealArray | ComplexArray,
    ) -> tuple[RealArray, ComplexArray]:
        values = np.asarray(signal, dtype=np.complex128)
        if values.ndim not in {1, 2}:
            raise ValueError("signal must be a 1D or 2D time-domain array.")
        if values.shape[0] != len(self.timegrid):
            raise ValueError(
                f"signal first dimension must match Nt={len(self.timegrid)}."
            )

        window = np.asarray(
            self.timegrid.apply_window(np.ones(len(self.timegrid), dtype=float)),
            dtype=np.float64,
        )
        reshape = (len(window),) + (1,) * (values.ndim - 1)
        weighted = values * window.reshape(reshape)

        nfft = len(self.timegrid)
        if self.timegrid.zero_padding:
            nfft *= self.timegrid.padding_factor

        spectrum = self.timegrid.dt * np.fft.fft(weighted, n=nfft, axis=0)
        omega_axis = np.asarray(self.timegrid.frequency_axis(), dtype=np.float64)
        return omega_axis, np.asarray(spectrum, dtype=np.complex128)

    def _validate_rho_orders(self) -> None:
        reference_shape = None
        expected_matrix_shape = (self.hamiltonian.basis_size, self.hamiltonian.basis_size)

        for order, tensor in self.rho_orders.items():
            if tensor.ndim != 4:
                raise ValueError(
                    f"rho_orders[{order}] must have shape (Nk, Nt, Nb, Nb); got {tensor.shape}."
                )
            if tensor.shape[0] != self.kgrid.total_points:
                raise ValueError(
                    f"rho_orders[{order}] has Nk={tensor.shape[0]}, expected {self.kgrid.total_points}."
                )
            if tensor.shape[1] != len(self.timegrid):
                raise ValueError(
                    f"rho_orders[{order}] has Nt={tensor.shape[1]}, expected {len(self.timegrid)}."
                )
            if tensor.shape[2:] != expected_matrix_shape:
                raise ValueError(
                    f"rho_orders[{order}] has matrix shape {tensor.shape[2:]}, "
                    f"expected {expected_matrix_shape}."
                )
            if reference_shape is None:
                reference_shape = tensor.shape
            elif tensor.shape != reference_shape:
                raise ValueError("All rho_orders tensors must share the same shape.")

    def _direction_axis(self, direction: str) -> int:
        axis = self._DIRECTION_TO_AXIS[self._normalize_direction(direction)]
        if axis >= self.hamiltonian.dimension:
            raise ValueError(
                f"Direction '{direction}' is outside Hamiltonian dimension {self.hamiltonian.dimension}."
            )
        return axis

    @classmethod
    def _normalize_direction(cls, direction: str) -> str:
        key = direction.strip().lower()
        if key not in cls._DIRECTION_TO_AXIS:
            raise ValueError("direction must be one of 'x', 'y', or 'z'.")
        return key

    @staticmethod
    def _coerce_real_vector(values: ComplexArray, *, name: str, atol: float = 1.0e-9) -> RealArray:
        imag_max = float(np.max(np.abs(np.imag(values))))
        if imag_max > atol:
            raise ValueError(f"{name} contains a non-negligible imaginary part ({imag_max:.3e}).")
        return np.asarray(np.real(values), dtype=np.float64)


    @classmethod
    def _coerce_real_matrix(cls, values: ComplexArray, *, name: str, atol: float = 1.0e-9) -> RealArray:
        imag_max = float(np.max(np.abs(np.imag(values))))
        if imag_max > atol:
            raise ValueError(f"{name} contains a non-negligible imaginary part ({imag_max:.3e}).")
        return np.asarray(np.real(values), dtype=np.float64)
