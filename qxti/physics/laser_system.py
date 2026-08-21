from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .laser import Laser


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class LaserSystem:
    """Collection of pulses with the same time-window convention as the reference code."""

    lasers: list[Laser] = field(default_factory=list)
    blaser: float = 0.0
    alaser: float = 0.0
    minus0: float = field(init=False, default=0.0)
    major0: float = field(init=False, default=0.0)
    atmin: float = field(init=False, default=0.0)
    atmax: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.blaser = float(self.blaser)
        self.alaser = float(self.alaser)
        for laser in self.lasers:
            self._validate_laser(laser)
        self._refresh_window()

    def add_laser(self, laser: Laser) -> None:
        """Append one pulse to the system."""

        self._validate_laser(laser)
        self.lasers.append(laser)
        self._refresh_window()

    def remove_laser(self, index: int) -> None:
        """Remove one pulse by list index."""

        del self.lasers[index]
        self._refresh_window()

    def electric_field(self, t: ArrayLike) -> FloatArray:
        """Return the total electric field: the PURE analytic sum of the pulses.

        No constant offset is subtracted.  The analytic pulse already has ~zero DC
        and ~zero net area; subtracting ``E(atmin)`` (as the reference code did to
        force the field to start at exactly zero) does NOT clean anything — it shifts
        the whole signal by a constant, i.e. it INJECTS a spurious DC offset (~1e-5 of
        the peak, the value of the Gaussian tail at the window edge).  That DC beats
        against the carrier and produces spurious even harmonics (pfddm envelope) and
        a DC ramp when integrated (tddm).  See docs/INTEGRATORS.md.
        """

        return np.asarray(self._sum_raw_vectors("electric_field", t), dtype=float)

    def vector_potential(self, t: ArrayLike) -> FloatArray:
        """Return the total vector potential: the PURE analytic sum of the pulses.

        No constant offset is subtracted (see :meth:`electric_field`)."""

        return np.asarray(self._sum_raw_vectors("vector_potential", t), dtype=float)

    def get_electric_field(self, t: ArrayLike) -> FloatArray:
        """Backward-compatible alias for :meth:`electric_field`."""

        return self.electric_field(t)

    def get_vector_potential(self, t: ArrayLike) -> FloatArray:
        """Backward-compatible alias for :meth:`vector_potential`."""

        return self.vector_potential(t)

    def total_intensity(self) -> float:
        """Return the sum of pulse peak intensities in W/cm^2."""

        return float(sum(laser.intensity() for laser in self.lasers))

    def number_of_lasers(self) -> int:
        """Return the number of stored pulses."""

        return len(self.lasers)

    def temporal_bounds(self) -> tuple[float, float]:
        """Return the full time window including the extra pre/post offsets, in a.u."""

        if not self.lasers:
            raise ValueError("Cannot infer temporal bounds from an empty LaserSystem.")
        return float(self.atmin), float(self.atmax)

    def _refresh_window(self) -> None:
        if not self.lasers:
            self.minus0 = 0.0
            self.major0 = 0.0
            self.atmin = 0.0
            self.atmax = 0.0
            return

        start_time = [laser.t0 - laser.twidth for laser in self.lasers]
        end_time = [laser.t0 + laser.twidth for laser in self.lasers]
        self.minus0 = float(min(start_time))
        self.major0 = float(max(end_time))
        self.atmin = self.minus0 - self.blaser
        self.atmax = self.major0 + self.alaser

    def _sum_raw_vectors(self, method_name: str, t: ArrayLike) -> FloatArray:
        time = np.asarray(t, dtype=float)

        if not self.lasers:
            if time.ndim == 0:
                return np.zeros(3, dtype=float)
            return np.zeros(time.shape + (3,), dtype=float)

        if len(self.lasers) == 1:
            return np.asarray(getattr(self.lasers[0], method_name)(time), dtype=float)

        contributions = [
            np.asarray(getattr(laser, method_name)(time), dtype=float)
            for laser in self.lasers
        ]
        return np.sum(contributions, axis=0, dtype=float)

    @staticmethod
    def _validate_laser(laser: Laser) -> None:
        if not isinstance(laser, Laser):
            raise TypeError("LaserSystem only accepts instances of Laser.")
