from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from qxti.grids import FrequencyGrid, KGrid, TimeGrid
from qxti.physics import BandGaugeFrame, Hamiltonian, LaserSystem, OperatorFactory


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
ProgressCallback = Callable[[int, str], None]


class XTP:
    r"""Compute macroscopic response observables from perturbative density matrices.

    **X-To-Polarization** (XTP) integrates precomputed perturbative density
    matrices ``rho^(s)(k, t)`` over the Brillouin zone to obtain macroscopic
    time-domain observables.

    The *polarization* of order ``s`` along Cartesian direction ``i`` is

    .. math::

        P_i^{(s)}(t) =
        \int_{\mathrm{BZ}} d\mathbf{k}\,
        \sum_{nm} d^i_{mn}(\mathbf{k})\,\rho^{(s)}_{nm}(\mathbf{k}, t),

    and the *current* is

    .. math::

        J_i^{(s)}(t) =
        -\int_{\mathrm{BZ}} d\mathbf{k}\,
        \sum_{nm} v^i_{mn}(\mathbf{k})\,\rho^{(s)}_{nm}(\mathbf{k}, t).

    The Brillouin-zone integral is approximated on the rectangular
    reciprocal-space box inferred from the Hamiltonian lattice constants.
    When the :class:`~qxti.grids.KGrid` is generated from
    :meth:`Hamiltonian.reciprocal_box_bounds`, this corresponds to square/cubic
    boxes ``[-pi/a_i, pi/a_i]`` (atomic units) with trapezoidal weights.

    **Frequency-domain spectra** are obtained by windowed FFT of the
    time-domain signals.  The harmonic spectrum of the driven current is the
    primary output for high-harmonic generation (HHG) simulations.

    **Susceptibility tensors** ``chi^(s)(omega)`` are computed by normalising
    the frequency-domain polarization by the driving-field amplitude raised to
    the ``s``-th power.

    **BZ masking.** An optional radial Gaussian mask can suppress edge
    contributions near the zone boundary (useful when the k-grid does not cover
    a full BZ with periodic boundary conditions).

    Parameters
    ----------
    hamiltonian:
        Tight-binding Hamiltonian — used for dimension, basis size, and BZ
        bounds.
    rho_orders:
        Dictionary mapping order index ``s`` to a ``(Nk, Nt, Nb, Nb)``
        complex density-matrix tensor.
    kgrid:
        Reciprocal-space grid matching the first axis of every rho tensor.
    timegrid:
        Time discretisation, provides ``dt`` and FFT window settings.
    frequencygrid:
        Frequency axis metadata (currently informational; FFT axis is derived
        from ``timegrid``).
    operator_factory:
        Builds dipole, velocity, and current operators on demand.
    directions:
        Cartesian directions to include in observable calculations (subset of
        ``["x", "y", "z"]``).
    orders:
        Perturbative orders to compute; must be present in ``rho_orders``.
    band_gauge_frame:
        Pre-built Berry-connection and current operators in the band basis.
        When provided, the chunked einsum path is used for efficiency instead
        of calling ``operator_factory`` per k-point.
    laser_system:
        Required for frequency-domain susceptibility and conductivity
        calculations (provides ``E(omega)`` for normalisation).
    bz_mask_enabled:
        Apply a radial Gaussian mask to the BZ integration weights.
    bz_mask_radius_percent:
        Radius of the flat-top mask region as a percentage of the shortest
        reciprocal-box half-width.
    bz_mask_sigma:
        Gaussian decay width of the mask edge in reciprocal-length units.
    bz_mask_sigma_percent_legacy:
        Legacy percentage-of-radius sigma parameter (overridden by
        ``bz_mask_sigma``).
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
        self._k_points = np.asarray(self.kgrid.points(), dtype=np.float64)
        self._integration_weights_cache: RealArray | None = None
        self._current_cache: dict[tuple[int, str], RealArray] = {}
        self._polarization_cache: dict[int, RealArray] = {}
        self._current_decomposition_cache: dict[tuple[int, str], dict[str, RealArray]] = {}
        if not self.orders:
            raise ValueError("orders must contain at least one perturbative order.")
        if self.bz_mask_radius_percent <= 0.0:
            raise ValueError("bz_mask_radius_percent must be strictly positive.")
        if self.bz_mask_sigma is not None and self.bz_mask_sigma <= 0.0:
            raise ValueError("bz_mask_sigma must be strictly positive when provided.")
        if self.bz_mask_sigma_percent_legacy is not None and self.bz_mask_sigma_percent_legacy <= 0.0:
            raise ValueError("bz_mask_sigma_percent_legacy must be strictly positive when provided.")
        self._validate_rho_orders()

    def polarization(
        self,
        order: int,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> RealArray:
        """Return the macroscopic polarization :math:`P^{(s)}(t)` for one order."""

        cache_key = int(order)
        cached = self._polarization_cache.get(cache_key)
        if cached is not None:
            return np.asarray(cached, dtype=np.float64)

        rho_tensor = self._rho_tensor(order)
        nt = rho_tensor.shape[1]
        polarization = np.zeros((nt, 3), dtype=np.complex128)

        for direction in self.directions:
            axis = self._direction_axis(direction)
            cached_dipole = None if self.band_gauge_frame is None else self.band_gauge_frame.connection(direction)
            polarization[:, axis] = self._weighted_observable_over_k(
                rho_tensor,
                operator_direction=direction,
                cached_operator=cached_dipole,
                operator_builder=self.operator_factory.dipole,
                progress_callback=progress_callback,
                progress_message=f"polarization(order={order}, direction={direction})",
            )

        result = self._coerce_real_matrix(polarization, name=f"polarization(order={order})")
        self._polarization_cache[cache_key] = np.asarray(result, dtype=np.float64)
        return np.asarray(result, dtype=np.float64)

    def current(
        self,
        order: int,
        direction: str,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> RealArray:
        """Return the macroscopic current component for one order and direction."""

        direction = self._normalize_direction(direction)
        cache_key = (int(order), direction)
        cached = self._current_cache.get(cache_key)
        if cached is not None:
            return np.asarray(cached, dtype=np.float64)

        rho_tensor = self._rho_tensor(order)
        cached_current = None if self.band_gauge_frame is None else self.band_gauge_frame.current(direction)
        integrated = self._weighted_observable_over_k(
            rho_tensor,
            operator_direction=direction,
            cached_operator=cached_current,
            operator_builder=self.operator_factory.current,
            progress_callback=progress_callback,
            progress_message=f"current(order={order}, direction={direction})",
        )
        result = self._coerce_real_vector(
            integrated,
            name=f"current(order={order}, direction={direction})",
        )
        self._current_cache[cache_key] = np.asarray(result, dtype=np.float64)
        return np.asarray(result, dtype=np.float64)

    def has_current_decomposition(self) -> bool:
        """Return whether intraband/interband current separation is available."""

        return self.band_gauge_frame is not None

    def current_decomposition(
        self,
        order: int,
        direction: str,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, RealArray]:
        """Return intraband/interband current components for one order and direction."""

        if self.band_gauge_frame is None:
            raise ValueError(
                "Current decomposition requires a band_gauge_frame so operators and rho share the same band basis."
            )

        direction = self._normalize_direction(direction)
        cache_key = (int(order), direction)
        cached_parts = self._current_decomposition_cache.get(cache_key)
        if cached_parts is not None:
            return {
                "intraband": np.asarray(cached_parts["intraband"], dtype=np.float64),
                "interband": np.asarray(cached_parts["interband"], dtype=np.float64),
            }

        rho_tensor = self._rho_tensor(order)
        current_operator = self.band_gauge_frame.current(direction)
        intraband = self._coerce_real_vector(
            self._weighted_intraband_current_over_k(
                rho_tensor,
                current_operator=current_operator,
                progress_callback=progress_callback,
                progress_message=f"intraband_current(order={order}, direction={direction})",
            ),
            name=f"intraband_current(order={order}, direction={direction})",
        )
        total = self.current(order, direction)
        interband = np.asarray(total - intraband, dtype=np.float64)
        parts = {
            "intraband": intraband,
            "interband": interband,
        }
        self._current_decomposition_cache[cache_key] = {
            "intraband": np.asarray(intraband, dtype=np.float64),
            "interband": np.asarray(interband, dtype=np.float64),
        }
        return {
            "intraband": np.asarray(parts["intraband"], dtype=np.float64),
            "interband": np.asarray(parts["interband"], dtype=np.float64),
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

    def order_current_frequency_domain(
        self,
        order: int,
    ) -> tuple[RealArray, ComplexArray]:
        """Return ``(omega_axis, J^(order)(omega))`` for all active Cartesian outputs."""

        active_dimension = self.hamiltonian.dimension
        direction_labels = self._direction_labels(active_dimension)
        reference_omega_axis: RealArray | None = None
        spectra: ComplexArray | None = None

        for axis, direction in enumerate(direction_labels):
            omega_axis, spectrum = self.current_frequency_domain(order, direction)
            if reference_omega_axis is None:
                reference_omega_axis = np.asarray(omega_axis, dtype=np.float64)
                spectra = np.zeros((len(omega_axis), active_dimension), dtype=np.complex128)
            elif (
                reference_omega_axis.shape != omega_axis.shape
                or not np.allclose(reference_omega_axis, omega_axis)
            ):
                raise ValueError("Current spectra for different output directions must share the same frequency axis.")

            spectra[:, axis] = np.asarray(spectrum, dtype=np.complex128)

        if reference_omega_axis is None or spectra is None:
            raise ValueError("No active Cartesian current directions are available.")

        return reference_omega_axis, np.asarray(spectra, dtype=np.complex128)

    def linear_conductivity(
        self,
        *,
        input_direction: str,
        eps: float = 1.0e-14,
    ) -> tuple[RealArray, ComplexArray]:
        """Return ``sigma_ij^(1)(omega) = J_i^(1)(omega) / E_j(omega)`` for one input direction."""

        input_direction = self._normalize_direction(input_direction)
        input_axis = self._direction_axis(input_direction)

        omega_axis, current_w = self.order_current_frequency_domain(order=1)
        _, electric_field_w = self.electric_field_frequency_domain()

        denominator = electric_field_w[:, input_axis]
        safe_denominator = np.where(
            np.abs(denominator) > eps,
            denominator,
            np.nan + 0.0j,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = current_w / safe_denominator[:, np.newaxis]

        return omega_axis, np.asarray(sigma, dtype=np.complex128)

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

    def effective_conductivity_spectrum(
        self,
        *,
        order: int,
        input_direction: str = "auto",
        input_omega: float | None = None,
        eps: float = 1.0e-14,
    ) -> tuple[RealArray, ComplexArray, dict[str, Any]]:
        r"""Return one effective ``sigma^(order)(omega)`` spectrum for a repeated input axis.

        For ``order > 1`` this is defined as

        .. math::

            \sigma^{(s)}_{i j \ldots j}(\omega)
            \equiv
            \frac{J^{(s)}_i(\omega)}{E_j(\omega_\mathrm{in})^s},

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

        omega_axis, current_w = self.order_current_frequency_domain(order=order)
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
                "is too small to normalize the conductivity spectrum."
            )

        active_dimension = self.hamiltonian.dimension
        sigma = current_w[:, :active_dimension] / denominator**order
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
        return omega_axis, np.asarray(sigma, dtype=np.complex128), metadata

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

    def conductivity_tensor_spectrum(
        self,
        *,
        order: int,
        input_direction: str = "auto",
        input_omega: float | None = None,
        eps: float = 1.0e-14,
    ) -> dict[str, Any]:
        """Return one sparse conductivity tensor spectrum for a single XTP run."""

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
            omega_axis, sigma_column = self.linear_conductivity(
                input_direction=resolved_direction,
                eps=eps,
            )
            tensor = np.full(
                (len(omega_axis),) + (active_dimension,) * 2,
                np.nan + 1.0j * np.nan,
                dtype=np.complex128,
            )
            tensor[:, :, input_axis] = sigma_column[:, :active_dimension]
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
            omega_axis, sigma_column, metadata = self.effective_conductivity_spectrum(
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
            tensor[(slice(None), slice(None)) + repeated_input] = sigma_column[:, :active_dimension]
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

    def total_polarization(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> RealArray:
        """Return the sum of the polarization over the configured orders."""

        nt = len(self.timegrid)
        total = np.zeros((nt, 3), dtype=np.float64)
        for order in self.orders:
            total += self.polarization(order, progress_callback=progress_callback)
        return total

    def total_current(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> RealArray:
        """Return the sum of current contributions over orders and directions."""

        nt = len(self.timegrid)
        total = np.zeros((nt, 3), dtype=np.float64)
        for direction in self.directions:
            axis = self._direction_axis(direction)
            for order in self.orders:
                total[:, axis] += self.current(order, direction, progress_callback=progress_callback)
        return total

    def total_current_decomposition(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, RealArray]:
        """Return intraband/interband current vectors summed over configured orders."""

        if self.band_gauge_frame is None:
            raise ValueError("total_current_decomposition requires band-basis data.")

        nt = len(self.timegrid)
        intraband = np.zeros((nt, 3), dtype=np.float64)
        interband = np.zeros((nt, 3), dtype=np.float64)
        for direction in self.directions:
            axis = self._direction_axis(direction)
            for order in self.orders:
                parts = self.current_decomposition(order, direction, progress_callback=progress_callback)
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
        weights = self._integration_weights()
        flat = values.reshape(self.kgrid.total_points, -1)
        integrated = np.einsum("k,km->m", weights, flat, optimize=True)
        return np.asarray(integrated.reshape(values.shape[1:]), dtype=np.complex128)

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

    def _integration_weights(self) -> RealArray:
        if self._integration_weights_cache is not None:
            return np.asarray(self._integration_weights_cache, dtype=np.float64)

        bounds = self.brillouin_zone_bounds()
        # Shifted (Monkhorst-Pack) grids sample a periodic function without
        # touching the BZ edges, so the correct quadrature is uniform midpoint
        # weights (dk per point) rather than the trapezoidal/Simpson rule.
        periodic = bool(getattr(self.kgrid, "shifted", False))
        weight_x = self._axis_integration_weights(
            np.asarray(self.kgrid.kx_values, dtype=np.float64),
            bounds[0],
            active=True,
            periodic=periodic,
        )
        weight_y = self._axis_integration_weights(
            np.asarray(self.kgrid.ky_values, dtype=np.float64),
            bounds[1] if self.kgrid.dimension >= 2 else (0.0, 0.0),
            active=self.kgrid.dimension >= 2,
            periodic=periodic,
        )
        weight_z = self._axis_integration_weights(
            np.asarray(self.kgrid.kz_values, dtype=np.float64),
            bounds[2] if self.kgrid.dimension >= 3 else (0.0, 0.0),
            active=self.kgrid.dimension >= 3,
            periodic=periodic,
        )
        weights = (
            weight_x[:, np.newaxis, np.newaxis]
            * weight_y[np.newaxis, :, np.newaxis]
            * weight_z[np.newaxis, np.newaxis, :]
        )
        if self.bz_mask_enabled:
            weights = weights * self._bz_mask_weights()
        self._integration_weights_cache = np.asarray(weights.reshape(-1), dtype=np.float64)
        return np.asarray(self._integration_weights_cache, dtype=np.float64)

    @staticmethod
    def _axis_integration_weights(
        coordinates: NDArray[np.float64],
        bounds: tuple[float, float],
        *,
        active: bool,
        periodic: bool = False,
    ) -> RealArray:
        """Return 1-D numerical integration weights for one k-axis.

        **Periodic (shifted / Monkhorst-Pack) grids** use uniform midpoint
        weights ``(b_hi - b_lo) / N`` per point.  For a smooth periodic
        integrand this is the correct rule and converges exponentially; it also
        keeps the weights symmetric under ``k -> -k`` so even-harmonic
        cancellation is preserved.

        **Edge-inclusive grids** use the **composite Simpson's 1/3 rule** when
        the grid is uniform with an **odd** number of points (O(h⁴)), and the
        **trapezoidal rule** otherwise (non-uniform, even count, or < 3 points).

        **Recommendation:** prefer ``shifted = true`` in ``[kgrid]`` to avoid
        band degeneracies; otherwise use an *odd* k-count per axis to activate
        Simpson's rule.
        """
        if not active:
            return np.ones(1, dtype=np.float64)

        coords = np.asarray(coordinates, dtype=np.float64)
        n = coords.size

        if n == 1:
            return np.asarray([float(bounds[1] - bounds[0])], dtype=np.float64)

        if periodic:
            # Uniform midpoint weights: total weight = full BZ width.
            width = float(bounds[1] - bounds[0])
            return np.full(n, width / n, dtype=np.float64)

        diffs = np.diff(coords)

        # Composite Simpson's 1/3 rule: requires uniform spacing and odd n.
        is_uniform = bool(np.allclose(diffs, diffs[0], rtol=1e-10, atol=0.0))
        if is_uniform and n >= 3 and n % 2 == 1:
            h = float(diffs[0])
            # Pattern: h/3 * [1, 4, 2, 4, 2, …, 4, 1]
            weights = np.full(n, 2.0 * h / 3.0, dtype=np.float64)
            weights[0] = h / 3.0
            weights[-1] = h / 3.0
            weights[1:-1:2] = 4.0 * h / 3.0  # positions 1, 3, 5, …, n-2
            return weights

        # Trapezoidal rule (non-uniform spacing, even point count, or n == 2).
        weights = np.empty_like(coords)
        weights[0] = 0.5 * float(diffs[0])
        weights[-1] = 0.5 * float(diffs[-1])
        if n > 2:
            weights[1:-1] = 0.5 * (coords[2:] - coords[:-2])
        return np.asarray(weights, dtype=np.float64)

    def _weighted_observable_over_k(
        self,
        rho_tensor: ComplexArray,
        *,
        operator_direction: str,
        cached_operator: ComplexArray | None,
        operator_builder,
        progress_callback: ProgressCallback | None,
        progress_message: str,
    ) -> ComplexArray:
        weights = self._integration_weights()
        nt = rho_tensor.shape[1]
        integrated = np.zeros(nt, dtype=np.complex128)
        nk = self.kgrid.total_points

        if cached_operator is not None:
            chunk_size = self._recommended_k_chunk_size(
                nt=nt,
                rho_dtype=rho_tensor.dtype,
                matrix_elements=self.hamiltonian.basis_size * self.hamiltonian.basis_size,
                result_itemsize=np.dtype(np.complex128).itemsize,
            )
            for start in range(0, nk, chunk_size):
                stop = min(start + chunk_size, nk)
                expectation_chunk = np.einsum(
                    "kmn,ktnm->kt",
                    cached_operator[start:stop],
                    rho_tensor[start:stop],
                    optimize=True,
                )
                integrated += np.einsum(
                    "k,kt->t",
                    weights[start:stop],
                    expectation_chunk,
                    optimize=True,
                )
                self._advance_progress(progress_callback, stop - start, progress_message)
            return np.asarray(integrated, dtype=np.complex128)

        for ik, (kx, ky, kz) in enumerate(self._k_points):
            operator = operator_builder(operator_direction, float(kx), float(ky), float(kz))
            integrated += weights[ik] * np.einsum(
                "mn,tnm->t",
                operator,
                rho_tensor[ik],
                optimize=True,
            )
            self._advance_progress(progress_callback, 1, progress_message)
        return np.asarray(integrated, dtype=np.complex128)

    def _weighted_intraband_current_over_k(
        self,
        rho_tensor: ComplexArray,
        *,
        current_operator: ComplexArray,
        progress_callback: ProgressCallback | None,
        progress_message: str,
    ) -> ComplexArray:
        weights = self._integration_weights()
        nt = rho_tensor.shape[1]
        integrated = np.zeros(nt, dtype=np.complex128)
        nk = self.kgrid.total_points
        chunk_size = self._recommended_k_chunk_size(
            nt=nt,
            rho_dtype=rho_tensor.dtype,
            matrix_elements=self.hamiltonian.basis_size,
            result_itemsize=np.dtype(np.complex128).itemsize,
        )

        for start in range(0, nk, chunk_size):
            stop = min(start + chunk_size, nk)
            diagonal_current = np.diagonal(current_operator[start:stop], axis1=1, axis2=2)
            band_diagonal = np.diagonal(rho_tensor[start:stop], axis1=2, axis2=3)
            expectation_chunk = np.einsum(
                "kn,ktn->kt",
                diagonal_current,
                band_diagonal,
                optimize=True,
            )
            integrated += np.einsum(
                "k,kt->t",
                weights[start:stop],
                expectation_chunk,
                optimize=True,
            )
            self._advance_progress(progress_callback, stop - start, progress_message)

        return np.asarray(integrated, dtype=np.complex128)

    @staticmethod
    def _recommended_k_chunk_size(
        *,
        nt: int,
        rho_dtype: np.dtype,
        matrix_elements: int,
        result_itemsize: int,
        target_bytes: int = 64 * 1024 * 1024,
        max_chunk_size: int = 256,
    ) -> int:
        rho_itemsize = np.dtype(rho_dtype).itemsize
        bytes_per_k = max(1, nt) * max(1, int(matrix_elements) * rho_itemsize + result_itemsize)
        chunk_size = max(1, min(max_chunk_size, target_bytes // max(bytes_per_k, 1)))
        return int(max(1, chunk_size))

    @staticmethod
    def _advance_progress(
        progress_callback: ProgressCallback | None,
        steps: int,
        message: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(int(steps), str(message))

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
