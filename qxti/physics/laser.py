from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class Laser:
    """Single laser pulse model.

    The pulse is defined by its amplitude, carrier, envelope, ellipticity,
    and a simple angular orientation. The sign of ``ellipticity`` sets the
    rotation sense: ``+1`` is right-circular, ``-1`` is left-circular, and
    ``0`` is linear.
    """

    omega: float
    E0: float
    phase: float = 0.0
    ellipticity: float = 0.0
    fwhm: float = 1.0
    envelope: str = "gaussian"
    t0: float = 0.0
    theta: float = math.pi / 2.0
    phi: float = 0.0

    _SUPPORTED_ENVELOPES = {"gaussian", "sech", "cos2", "constant"}

    def __post_init__(self) -> None:
        self.envelope = self.envelope.lower()

        if self.omega <= 0.0:
            raise ValueError("omega must be strictly positive.")
        if self.E0 < 0.0:
            raise ValueError("E0 must be non-negative.")
        if not -1.0 <= self.ellipticity <= 1.0:
            raise ValueError("ellipticity must be between -1 and 1.")
        if self.fwhm <= 0.0:
            raise ValueError("fwhm must be strictly positive.")
        if self.envelope not in self._SUPPORTED_ENVELOPES:
            raise ValueError(
                f"Unsupported envelope '{self.envelope}'. "
                f"Choose from {sorted(self._SUPPORTED_ENVELOPES)}."
            )

    def envelope_function(self, t: ArrayLike) -> float | FloatArray:
        """Return the pulse envelope evaluated at time t."""

        time = self._time_array(t)
        tau = time - self.t0

        if self.envelope == "gaussian":
            values = np.exp(-4.0 * math.log(2.0) * (tau / self.fwhm) ** 2)
        elif self.envelope == "sech":
            scale = 2.0 * math.acosh(2.0) / self.fwhm
            values = 1.0 / np.cosh(scale * tau)
        elif self.envelope == "cos2":
            arg = math.pi * tau / self.fwhm
            values = np.where(np.abs(tau) <= self.fwhm / 2.0, np.cos(arg) ** 2, 0.0)
        else:
            values = np.ones_like(time, dtype=float)

        return self._scalar_or_array(values)

    def carrier(self, t: ArrayLike) -> float | FloatArray:
        """Return the carrier cos(omega * (t - t0) + phase)."""

        values = np.cos(self._phase_argument(t))
        return self._scalar_or_array(values)

    def polarization_vector(self) -> FloatArray:
        """Return the major-axis polarization vector in the lab frame.

        The angular convention is spherical:
        - ``theta = pi/2, phi = 0`` gives the ``x`` direction
        - ``theta = pi/2, phi = pi/2`` gives the ``y`` direction
        - ``theta = 0`` gives the ``z`` direction
        """

        return np.array(
            [
                math.sin(self.theta) * math.cos(self.phi),
                math.sin(self.theta) * math.sin(self.phi),
                math.cos(self.theta),
            ],
            dtype=float,
        )

    def rotation_matrix(self) -> FloatArray:
        """Return the local orthonormal basis attached to the pulse.

        The columns are:
        1. major-axis polarization
        2. minor-axis polarization
        3. normal to the polarization plane
        """

        major_axis, minor_axis, normal_axis = self._field_basis()
        return np.column_stack((major_axis, minor_axis, normal_axis))

    def electric_field(self, t: ArrayLike) -> FloatArray:
        """Return the electric field E(t) in Cartesian components."""

        time = self._time_array(t)
        phase = self._phase_argument(time)
        envelope = np.asarray(self.envelope_function(time), dtype=float)
        major_axis, minor_axis, _ = self._field_basis()

        return (
            self.E0
            * envelope[..., np.newaxis]
            * (
                np.cos(phase)[..., np.newaxis] * major_axis
                + self.ellipticity * np.sin(phase)[..., np.newaxis] * minor_axis
            )
        )

    def vector_potential(self, t: ArrayLike) -> FloatArray:
        """Return the vector potential A(t) in Cartesian components."""

        time = self._time_array(t)
        phase = self._phase_argument(time)
        envelope = np.asarray(self.envelope_function(time), dtype=float)
        major_axis, minor_axis, _ = self._field_basis()

        return (
            -(self.E0 / self.omega)
            * envelope[..., np.newaxis]
            * (
                np.sin(phase)[..., np.newaxis] * major_axis
                - self.ellipticity * np.cos(phase)[..., np.newaxis] * minor_axis
            )
        )

    def intensity(self) -> float:
        """Return a simple cycle-averaged intensity in atomic units."""

        return 0.5 * self.E0**2 * (1.0 + self.ellipticity**2)

    def summary(self) -> dict[str, Any]:
        """Return a serializable summary of the pulse configuration."""

        data = asdict(self)
        data["polarization_vector"] = self.polarization_vector().tolist()
        data["rotation_matrix"] = self.rotation_matrix().tolist()
        data["intensity"] = self.intensity()
        return data

    def temporal_bounds(self) -> tuple[float, float]:
        """Return a finite time window that fully contains the pulse in atomic units."""

        if self.envelope == "gaussian":
            half_width = 4.0 * self.fwhm
        elif self.envelope == "sech":
            half_width = 6.0 * self.fwhm
        elif self.envelope == "cos2":
            half_width = 0.5 * self.fwhm
        else:
            raise ValueError(
                "A finite automatic time window cannot be inferred for a constant envelope. "
                "Use a pulsed envelope such as gaussian, sech, or cos2."
            )

        return float(self.t0 - half_width), float(self.t0 + half_width)

    def _field_basis(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        major_axis = self.polarization_vector()
        major_axis /= np.linalg.norm(major_axis)

        reference = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(major_axis, reference)) > 0.999:
            reference = np.array([0.0, 1.0, 0.0], dtype=float)

        minor_axis = np.cross(reference, major_axis)
        minor_axis /= np.linalg.norm(minor_axis)
        normal_axis = np.cross(major_axis, minor_axis)
        normal_axis /= np.linalg.norm(normal_axis)
        return major_axis, minor_axis, normal_axis

    def _phase_argument(self, t: ArrayLike) -> FloatArray:
        time = self._time_array(t)
        return self.omega * (time - self.t0) + self.phase

    @staticmethod
    def _time_array(t: ArrayLike) -> FloatArray:
        return np.asarray(t, dtype=float)

    @staticmethod
    def _scalar_or_array(values: FloatArray) -> float | FloatArray:
        if values.ndim == 0:
            return float(values)
        return values
