from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

GAUSSIAN_WIDTH_MULTIPLY = 2.0
INTENSITY_AU_TO_W_CM2 = 3.5e16


class Laser:
    """Single pulse following the Pulse convention used in the reference code.

    All numerical parameters are interpreted in atomic units except:
    - ``cep``, ``phix``, ``thetaz``, ``phiz``: radians
    - ``ellip``: dimensionless ellipticity
    - ``ncycles`` and ``ncycles_clipped``: dimensionless cycle counts

    Backward-compatible aliases are also accepted:
    - ``phase`` -> ``cep``
    - ``ellipticity`` -> ``ellip``
    - ``num_cycles`` -> ``ncycles``
    - ``envelope`` -> ``envname``
    - ``theta``/``phi`` -> legacy direct polarization-vector angles
    """

    _ENVELOPE_ALIASES = {
        "gauss": "gauss",
        "gaussian": "gauss",
        "gauss_clipped": "gauss_clipped",
        "gaussian_clipped": "gauss_clipped",
        "cos4": "cos4",
        "cos4_clipped": "cos4_clipped",
        "constant": "constant",
    }

    def __init__(
        self,
        *,
        omega: float,
        E0: float,
        ellip: float | None = None,
        ellipticity: float | None = None,
        ncycles: float | None = None,
        num_cycles: float | None = None,
        cep: float | None = None,
        phase: float | None = None,
        t0: float = 0.0,
        phix: float | None = None,
        thetaz: float | None = None,
        phiz: float | None = None,
        theta: float | None = None,
        phi: float | None = None,
        envname: str | None = None,
        envelope: str | None = None,
        ncycles_clipped: float = 0.0,
    ) -> None:
        self.omega = float(omega)
        self.E0 = float(E0)
        resolved_ellip = 0.0 if ellip is None and ellipticity is None else (ellipticity if ellip is None else ellip)
        resolved_ncycles = 1.0 if ncycles is None and num_cycles is None else (num_cycles if ncycles is None else ncycles)
        resolved_cep = 0.0 if cep is None and phase is None else (phase if cep is None else cep)
        self.ellip = float(resolved_ellip)
        self.ncycles = float(resolved_ncycles)
        self.cep = float(resolved_cep)
        self.t0 = float(t0)
        self.envname = self._normalize_envelope_name(envname if envname is not None else envelope)
        self.ncycles_clipped = float(ncycles_clipped)

        self._legacy_theta = None if theta is None else float(theta)
        self._legacy_phi = None if phi is None else float(phi)
        self.thetaz = 0.0 if thetaz is None else float(thetaz)
        self.phiz = 0.0 if phiz is None else float(phiz)
        self.phix = 0.0 if phix is None else float(phix)

        self._validate()
        self._set_internal_quantities()
        self._build_field_basis()

    @property
    def w0(self) -> float:
        """Alias for the central angular frequency in atomic units."""

        return self.omega

    @property
    def phase(self) -> float:
        """Backward-compatible alias for ``cep`` in radians."""

        return self.cep

    @property
    def ellipticity(self) -> float:
        """Backward-compatible alias for the dimensionless ellipticity."""

        return self.ellip

    @property
    def num_cycles(self) -> float:
        """Backward-compatible alias for the dimensionless cycle count."""

        return self.ncycles

    @property
    def envelope(self) -> str:
        """Backward-compatible alias for the envelope name."""

        return self.envname

    @property
    def period(self) -> float:
        """Optical period in atomic units."""

        return self.period0

    @property
    def fwhm(self) -> float:
        """Gaussian FWHM scale inferred from ``ncycles`` in atomic units."""

        return self.ncycles * self.period0

    @property
    def theta(self) -> float:
        """Legacy spherical polar angle of the major polarization axis."""

        if self._legacy_theta is not None:
            return self._legacy_theta
        return float(math.acos(np.clip(self.xdir[2], -1.0, 1.0)))

    @property
    def phi(self) -> float:
        """Legacy spherical azimuth of the major polarization axis."""

        if self._legacy_phi is not None:
            return self._legacy_phi
        return float(math.atan2(self.xdir[1], self.xdir[0]))

    def envelope_function(self, t: ArrayLike) -> float | FloatArray:
        """Return the envelope evaluated at absolute time ``t``."""

        values = self.f_envelope(self._time_array(t) - self.t0, is_dt=False)
        return self._scalar_or_array(np.asarray(values, dtype=float))

    def envelope_derivative(self, t: ArrayLike) -> float | FloatArray:
        """Return the time derivative of the envelope at absolute time ``t``."""

        values = self.f_envelope(self._time_array(t) - self.t0, is_dt=True)
        return self._scalar_or_array(np.asarray(values, dtype=float))

    def carrier(self, t: ArrayLike) -> float | FloatArray:
        """Return ``cos(omega * (t - t0) - cep)``."""

        values = np.cos(self._phase_argument(t))
        return self._scalar_or_array(np.asarray(values, dtype=float))

    def f_envelope(self, t: ArrayLike, is_dt: bool) -> float | FloatArray:
        """Envelope used by the reference implementation.

        The input time must already be centered around ``t0``.
        """

        tau = self._time_array(t)

        if self.envname == "cos4":
            values = self._cos4_envelope(tau, is_dt=is_dt)
        elif self.envname == "cos4_clipped":
            values = self._cos4_clipped_envelope(tau, is_dt=is_dt)
        elif self.envname == "gauss":
            values = self._gaussian_envelope(tau, is_dt=is_dt)
        elif self.envname == "gauss_clipped":
            values = self._gaussian_clipped_envelope(tau, is_dt=is_dt)
        else:
            values = self._constant_envelope(tau, is_dt=is_dt)

        return self._scalar_or_array(np.asarray(values, dtype=float))

    def avlaser_2d(self, t: ArrayLike) -> complex | ComplexArray:
        """Return the in-plane vector potential as one complex signal."""

        tau = self._time_array(t) - self.t0
        phase = self._phase_argument(t)
        envelope = np.asarray(self.f_envelope(tau, is_dt=False), dtype=float)
        values = (
            -self.A0x * np.sin(phase) * envelope
            + 1j * self.A0y * np.cos(phase) * envelope
        )
        return self._scalar_or_array_complex(np.asarray(values, dtype=complex))

    def elaser_2d(self, t: ArrayLike) -> complex | ComplexArray:
        """Return the in-plane electric field as one complex signal."""

        tau = self._time_array(t) - self.t0
        phase = self._phase_argument(t)
        envelope = np.asarray(self.f_envelope(tau, is_dt=False), dtype=float)
        envelope_dt = np.asarray(self.f_envelope(tau, is_dt=True), dtype=float)
        values = (
            self.E0x * np.cos(phase) * envelope
            - self.A0x * np.sin(phase) * envelope_dt
            + 1j
            * (
                self.E0y * np.sin(phase) * envelope
                + self.A0y * np.cos(phase) * envelope_dt
            )
        )
        return self._scalar_or_array_complex(np.asarray(values, dtype=complex))

    def vector_potential(self, t: ArrayLike) -> FloatArray:
        """Return the vector potential in Cartesian components (a.u.)."""

        return self._embed_plane_complex(self.avlaser_2d(t))

    def electric_field(self, t: ArrayLike) -> FloatArray:
        """Return the electric field in Cartesian components (a.u.)."""

        return self._embed_plane_complex(self.elaser_2d(t))

    def polarization_vector(self) -> FloatArray:
        """Return the major-axis polarization direction in the lab frame."""

        return np.array(self.xdir, dtype=float, copy=True)

    def rotation_matrix(self) -> FloatArray:
        """Return the local basis ``[xdir, ydir, zdir]`` as one 3x3 matrix."""

        return np.column_stack((self.xdir, self.ydir, self.zdir))

    def intensity(self) -> float:
        """Peak intensity estimate in W/cm^2, matching the reference code."""

        return self.I0

    def temporal_bounds(self) -> tuple[float, float]:
        """Finite pulse bounds in atomic units."""

        return float(self.t0 - self.twidth), float(self.t0 + self.twidth)

    def summary(self) -> dict[str, Any]:
        """Return a serializable description of the pulse."""

        return {
            "omega": self.omega,
            "w0": self.w0,
            "E0": self.E0,
            "ellip": self.ellip,
            "ellipticity": self.ellipticity,
            "ncycles": self.ncycles,
            "num_cycles": self.num_cycles,
            "cep": self.cep,
            "phase": self.phase,
            "t0": self.t0,
            "phix": self.phix,
            "thetaz": self.thetaz,
            "phiz": self.phiz,
            "theta": self.theta,
            "phi": self.phi,
            "envname": self.envname,
            "envelope": self.envelope,
            "ncycles_clipped": self.ncycles_clipped,
            "period": self.period0,
            "period0": self.period0,
            "fwhm": self.fwhm,
            "twidth": self.twidth,
            "twidth_clipped": self.twidth_clipped,
            "E0x": self.E0x,
            "E0y": self.E0y,
            "A0x": self.A0x,
            "A0y": self.A0y,
            "I0_W_cm2": self.I0,
            "intensity": self.intensity(),
            "polarization_vector": self.polarization_vector().tolist(),
            "rotation_matrix": self.rotation_matrix().tolist(),
        }

    def _validate(self) -> None:
        if self.omega <= 0.0:
            raise ValueError("omega must be strictly positive.")
        if self.E0 < 0.0:
            raise ValueError("E0 must be non-negative.")
        if not -1.0 <= self.ellip <= 1.0:
            raise ValueError("ellip/ellipticity must be between -1 and 1.")
        if self.ncycles <= 0.0:
            raise ValueError("ncycles/num_cycles must be strictly positive.")
        if self.ncycles_clipped < 0.0:
            raise ValueError("ncycles_clipped must be non-negative.")

    def _set_internal_quantities(self) -> None:
        self.I0 = self.E0 * self.E0 * INTENSITY_AU_TO_W_CM2
        self.period0 = 2.0 * math.pi / self.omega
        self.inverse_w0 = 1.0 / self.omega if self.omega > 1.0e-30 else 0.0
        self.twidth_clipped = 0.0

        if self.envname == "gauss":
            self.twidth = GAUSSIAN_WIDTH_MULTIPLY * self.ncycles * self.period0
            self.envfactor = -4.0 * math.log(2.0) / (self.fwhm * self.fwhm)
        elif self.envname == "gauss_clipped":
            self.twidth = (
                GAUSSIAN_WIDTH_MULTIPLY * self.ncycles * self.period0
                + 0.5 * self.ncycles_clipped * self.period0
            )
            self.twidth_clipped = 0.5 * self.ncycles_clipped * self.period0
            self.envfactor = -4.0 * math.log(2.0) / (self.fwhm * self.fwhm)
        elif self.envname == "cos4":
            self.twidth = 0.5 * self.ncycles * self.period0
            self.envfactor = self.omega / (2.0 * self.ncycles)
        elif self.envname == "cos4_clipped":
            self.twidth = 0.5 * (self.ncycles + self.ncycles_clipped) * self.period0
            self.twidth_clipped = 0.5 * self.ncycles_clipped * self.period0
            self.envfactor = self.omega / (2.0 * self.ncycles)
        else:
            self.twidth = 0.5 * self.ncycles * self.period0
            self.envfactor = 0.0

        factor = math.sqrt(1.0 / (1.0 + self.ellip * self.ellip))
        self.E0x = factor * self.E0
        self.E0y = self.ellip * factor * self.E0
        self.A0x = self.E0x * self.inverse_w0
        self.A0y = self.E0y * self.inverse_w0

    def _build_field_basis(self) -> None:
        if self._legacy_theta is not None or self._legacy_phi is not None:
            theta = 0.5 * math.pi if self._legacy_theta is None else self._legacy_theta
            phi = 0.0 if self._legacy_phi is None else self._legacy_phi
            xdir = np.array(
                [
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                ],
                dtype=float,
            )
            xdir /= np.linalg.norm(xdir)
            reference = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(xdir, reference))) > 0.999:
                reference = np.array([0.0, 1.0, 0.0], dtype=float)
            ydir = np.cross(reference, xdir)
            ydir /= np.linalg.norm(ydir)
            zdir = np.cross(xdir, ydir)
            zdir /= np.linalg.norm(zdir)
        else:
            trans_x0 = np.array(
                [
                    math.cos(self.thetaz) * math.cos(self.phiz),
                    math.cos(self.thetaz) * math.sin(self.phiz),
                    -math.sin(self.thetaz),
                ],
                dtype=float,
            )
            trans_y0 = np.array(
                [
                    -math.sin(self.phiz),
                    math.cos(self.phiz),
                    0.0,
                ],
                dtype=float,
            )
            xdir = math.cos(self.phix) * trans_x0 + math.sin(self.phix) * trans_y0
            ydir = -math.sin(self.phix) * trans_x0 + math.cos(self.phix) * trans_y0
            xdir /= np.linalg.norm(xdir)
            ydir /= np.linalg.norm(ydir)
            zdir = np.cross(xdir, ydir)
            zdir /= np.linalg.norm(zdir)

        self.xdir = xdir
        self.ydir = ydir
        self.zdir = zdir

    def _phase_argument(self, t: ArrayLike) -> FloatArray:
        time = self._time_array(t)
        return self.omega * (time - self.t0) - self.cep

    def _constant_envelope(self, tau: FloatArray, *, is_dt: bool) -> FloatArray:
        values = np.where(np.abs(tau) <= self.twidth, 1.0, 0.0)
        if is_dt:
            return np.zeros_like(values, dtype=float)
        return np.asarray(values, dtype=float)

    def _gaussian_envelope(self, tau: FloatArray, *, is_dt: bool) -> FloatArray:
        values = np.exp(self.envfactor * tau * tau)
        if is_dt:
            return 2.0 * tau * self.envfactor * values
        return values

    def _gaussian_clipped_envelope(self, tau: FloatArray, *, is_dt: bool) -> FloatArray:
        values = np.ones_like(tau, dtype=float)
        derivatives = np.zeros_like(tau, dtype=float)

        upper_mask = tau > self.twidth_clipped
        lower_mask = tau < -self.twidth_clipped

        tau_upper = tau[upper_mask] - self.twidth_clipped
        tau_lower = tau[lower_mask] + self.twidth_clipped

        upper_values = np.exp(self.envfactor * tau_upper * tau_upper)
        lower_values = np.exp(self.envfactor * tau_lower * tau_lower)

        values[upper_mask] = upper_values
        values[lower_mask] = lower_values
        derivatives[upper_mask] = 2.0 * tau_upper * self.envfactor * upper_values
        derivatives[lower_mask] = 2.0 * tau_lower * self.envfactor * lower_values

        return derivatives if is_dt else values

    def _cos4_envelope(self, tau: FloatArray, *, is_dt: bool) -> FloatArray:
        values = np.zeros_like(tau, dtype=float)
        inside = np.abs(tau) <= self.twidth
        argument = tau[inside] * self.envfactor
        cosine = np.cos(argument)
        if is_dt:
            values[inside] = -4.0 * self.envfactor * np.sin(argument) * cosine**3
        else:
            values[inside] = cosine**4
        return values

    def _cos4_clipped_envelope(self, tau: FloatArray, *, is_dt: bool) -> FloatArray:
        values = np.zeros_like(tau, dtype=float)
        inside = np.abs(tau) <= self.twidth
        values[inside] = 1.0

        upper_mask = tau > self.twidth_clipped
        lower_mask = tau < -self.twidth_clipped
        plateau_mask = (~upper_mask) & (~lower_mask) & inside

        tau_upper = tau[upper_mask] - self.twidth_clipped
        tau_lower = tau[lower_mask] + self.twidth_clipped

        if is_dt:
            values[:] = 0.0
            arg_upper = tau_upper * self.envfactor
            arg_lower = tau_lower * self.envfactor
            values[upper_mask] = -4.0 * self.envfactor * np.sin(arg_upper) * np.cos(arg_upper) ** 3
            values[lower_mask] = -4.0 * self.envfactor * np.sin(arg_lower) * np.cos(arg_lower) ** 3
            values[plateau_mask] = 0.0
        else:
            arg_upper = tau_upper * self.envfactor
            arg_lower = tau_lower * self.envfactor
            values[upper_mask] = np.cos(arg_upper) ** 4
            values[lower_mask] = np.cos(arg_lower) ** 4
            values[plateau_mask] = 1.0

        outside = np.abs(tau) > self.twidth
        values[outside] = 0.0
        return values

    def _embed_plane_complex(self, plane_values: complex | ComplexArray) -> FloatArray:
        values = np.asarray(plane_values, dtype=complex)
        real_part = np.real(values)[..., np.newaxis]
        imag_part = np.imag(values)[..., np.newaxis]
        field = real_part * self.xdir + imag_part * self.ydir
        return np.asarray(field, dtype=float)

    @classmethod
    def _normalize_envelope_name(cls, name: str | None) -> str:
        normalized = "gauss" if name is None else name.strip().lower()
        try:
            return cls._ENVELOPE_ALIASES[normalized]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported envelope '{name}'. Choose from {sorted(cls._ENVELOPE_ALIASES)}."
            ) from exc

    @staticmethod
    def _time_array(t: ArrayLike) -> FloatArray:
        return np.asarray(t, dtype=float)

    @staticmethod
    def _scalar_or_array(values: FloatArray) -> float | FloatArray:
        if values.ndim == 0:
            return float(values)
        return values

    @staticmethod
    def _scalar_or_array_complex(values: ComplexArray) -> complex | ComplexArray:
        if values.ndim == 0:
            return complex(values)
        return values
