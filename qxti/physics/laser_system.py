from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .laser import Laser


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class LaserSystem:
    """Composition of one or more Laser objects.

    The system acts as a thin container that returns the total external field
    produced by the sum of all stored pulses.
    """

    lasers: list[Laser] = field(default_factory=list)

    def __post_init__(self) -> None:
        for laser in self.lasers:
            self._validate_laser(laser)

    def add_laser(self, laser: Laser) -> None:
        """Append one Laser to the system."""

        self._validate_laser(laser)
        self.lasers.append(laser)

    def remove_laser(self, index: int) -> None:
        """Remove one Laser by list index."""

        del self.lasers[index]

    def electric_field(self, t: ArrayLike) -> FloatArray:
        """Return the total electric field from all pulses."""

        return self._sum_vectors("electric_field", t)

    def vector_potential(self, t: ArrayLike) -> FloatArray:
        """Return the total vector potential from all pulses."""

        return self._sum_vectors("vector_potential", t)

    def get_electric_field(self, t: ArrayLike) -> FloatArray:
        """Backward-compatible alias for :meth:`electric_field`."""

        return self.electric_field(t)

    def get_vector_potential(self, t: ArrayLike) -> FloatArray:
        """Backward-compatible alias for :meth:`vector_potential`."""

        return self.vector_potential(t)

    def total_intensity(self) -> float:
        """Return the sum of the individual pulse intensities."""

        return float(sum(laser.intensity() for laser in self.lasers))

    def number_of_lasers(self) -> int:
        """Return the number of stored Laser objects."""

        return len(self.lasers)

    def _sum_vectors(self, method_name: str, t: ArrayLike) -> FloatArray:
        time = np.asarray(t, dtype=float)

        if not self.lasers:
            if time.ndim == 0:
                return np.zeros(3, dtype=float)
            return np.zeros(time.shape + (3,), dtype=float)

        contributions = [
            np.asarray(getattr(laser, method_name)(time), dtype=float)
            for laser in self.lasers
        ]
        return np.sum(contributions, axis=0, dtype=float)

    @staticmethod
    def _validate_laser(laser: Laser) -> None:
        if not isinstance(laser, Laser):
            raise TypeError("LaserSystem only accepts instances of Laser.")
