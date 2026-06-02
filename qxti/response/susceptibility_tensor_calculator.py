from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qxti.response.xtp import XTP


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(slots=True)
class SusceptibilityTensorCalculator:
    """Reconstruct susceptibility tensors from one or more XTP calculations."""

    xtp_by_label: dict[str, XTP]
    eps: float = 1.0e-14

    _DIRECTION_TO_AXIS = {"x": 0, "y": 1, "z": 2}

    def chi1(self) -> tuple[RealArray, ComplexArray]:
        """
        Build the full linear tensor chi_ij^(1)(omega).

        The tensor dimension is inferred from the first XTP Hamiltonian.

        Returns
        -------
        omega_axis:
            Shape (Nomega,)

        chi:
            Shape (Nomega, dimension, dimension)
            chi[:, i, j] = chi_ij^(1)(omega)
        """

        if not self.xtp_by_label:
            raise ValueError("At least one XTP calculation is required.")

        first_xtp = next(iter(self.xtp_by_label.values()))
        dimension = int(first_xtp.hamiltonian.dimension)

        if dimension not in {1, 2, 3}:
            raise ValueError("Hamiltonian dimension must be 1, 2, or 3.")

        directions = tuple(self._DIRECTION_TO_AXIS.keys())[:dimension]

        chi: ComplexArray | None = None
        reference_omega: RealArray | None = None

        for input_axis, direction in enumerate(directions):
            xtp = self._get_xtp(direction)
            omega_axis, chi_column = xtp.linear_susceptibility(
                input_direction=direction,
                eps=self.eps,
            )

            if chi is None:
                reference_omega = omega_axis
                chi = np.zeros(
                    (len(omega_axis), dimension, dimension),
                    dtype=np.complex128,
                )
            else:
                self._validate_same_frequency_axis(reference_omega, omega_axis)

            if chi_column.shape[1] < dimension:
                raise ValueError(
                    f"chi column for direction {direction!r} has shape {chi_column.shape}, "
                    f"expected at least {dimension} output components."
                )

            chi[:, :, input_axis] = chi_column[:, :dimension]

        if chi is None or reference_omega is None:
            raise ValueError("No XTP calculations were provided.")

        return reference_omega, chi

    def chi2_at_frequency(
        self,
        *,
        input_omega: float,
        output_omega: float | None = None,
    ) -> ComplexArray:
        """
        Reconstruct chi_ijk^(2) at one output frequency.

        Uses:
            P_i^(2)(omega_out) = sum_jk chi_ijk E_j(omega_in) E_k(omega_in)

        Recommended labels:
            "x", "y", "z", "xy", "xz", "yz", "xyz"

        Returns
        -------
        chi2:
            Shape (3, 3, 3)
            chi2[i, j, k] = chi_ijk^(2)
        """

        if output_omega is None:
            output_omega = 2.0 * input_omega

        labels = list(self.xtp_by_label)
        if len(labels) < 6:
            raise ValueError(
                "chi2 reconstruction needs several field combinations. "
                "Use at least 6-9 XTP calculations."
            )

        rows: list[ComplexArray] = []
        polarization_values: list[ComplexArray] = []

        reference_omega: RealArray | None = None

        for label in labels:
            xtp = self.xtp_by_label[label]

            omega_axis, polarization_w = xtp.polarization_frequency_domain(order=2)
            _, electric_field_w = xtp.electric_field_frequency_domain()

            if reference_omega is None:
                reference_omega = omega_axis
            else:
                self._validate_same_frequency_axis(reference_omega, omega_axis)

            input_index = self._nearest_frequency_index(omega_axis, input_omega)
            output_index = self._nearest_frequency_index(omega_axis, output_omega)

            field_vector = electric_field_w[input_index, :]
            polarization_vector = polarization_w[output_index, :]

            rows.append(self._second_order_design_row(field_vector))
            polarization_values.append(np.asarray(polarization_vector, dtype=np.complex128))

        design = np.asarray(rows, dtype=np.complex128)
        response = np.asarray(polarization_values, dtype=np.complex128)

        chi2 = np.zeros((3, 3, 3), dtype=np.complex128)

        for output_axis in range(3):
            solution, *_ = np.linalg.lstsq(
                design,
                response[:, output_axis],
                rcond=None,
            )
            chi2[output_axis] = solution.reshape(3, 3)

        return chi2

    def effective_chi(
        self,
        label: str,
        *,
        order: int,
        input_omega: float,
        output_omega: float | None = None,
    ) -> ComplexArray:
        """
        Compute an effective susceptibility for one simulation.

        This does not reconstruct the full tensor. It computes:

            P_i^(order)(omega_out) / |E(omega_in)|^order

        Returns shape:
            (3,)
        """

        if order < 1:
            raise ValueError("order must be >= 1.")

        if output_omega is None:
            output_omega = order * input_omega

        xtp = self._get_xtp(label)
        omega_axis, polarization_w = xtp.polarization_frequency_domain(order)
        _, electric_field_w = xtp.electric_field_frequency_domain()

        input_index = self._nearest_frequency_index(omega_axis, input_omega)
        output_index = self._nearest_frequency_index(omega_axis, output_omega)

        field_amplitude = np.linalg.norm(electric_field_w[input_index, :])

        if field_amplitude <= self.eps:
            raise ValueError(
                f"Electric field is too small near omega={input_omega}."
            )

        return np.asarray(
            polarization_w[output_index, :] / field_amplitude**order,
            dtype=np.complex128,
        )

    def _get_xtp(self, label: str) -> XTP:
        try:
            return self.xtp_by_label[label]
        except KeyError as exc:
            raise ValueError(f"Missing XTP calculation for label {label!r}.") from exc

    @staticmethod
    def _nearest_frequency_index(omega_axis: RealArray, omega: float) -> int:
        return int(np.argmin(np.abs(omega_axis - omega)))

    @staticmethod
    def _second_order_design_row(field_vector: ComplexArray) -> ComplexArray:
        ex, ey, ez = np.asarray(field_vector, dtype=np.complex128)

        return np.asarray(
            [
                ex * ex,
                ex * ey,
                ex * ez,
                ey * ex,
                ey * ey,
                ey * ez,
                ez * ex,
                ez * ey,
                ez * ez,
            ],
            dtype=np.complex128,
        )

    @staticmethod
    def _validate_same_frequency_axis(
        reference: RealArray,
        candidate: RealArray,
    ) -> None:
        if reference.shape != candidate.shape or not np.allclose(reference, candidate):
            raise ValueError("All XTP calculations must share the same frequency axis.")
