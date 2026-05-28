from __future__ import annotations

from typing import Any

import numpy as np

from qxti.response import XTP


class HarmonicData:
    """Build reusable spectral datasets from XTP observables."""

    def __init__(
        self,
        xtp: XTP,
        *,
        electric_field_time: np.ndarray | None = None,
    ) -> None:
        self.xtp = xtp
        self.electric_field_time = None if electric_field_time is None else np.asarray(electric_field_time, dtype=float)

    def current_spectrum_data(self) -> dict[str, Any]:
        """Return one serializable dataset for the total current spectrum."""

        omega_axis, current_spectrum = self.xtp.total_current_frequency_domain()
        data = {
            "omega_axis": np.asarray(omega_axis, dtype=float),
            "current_spectrum": np.asarray(current_spectrum, dtype=np.complex128),
            "current_magnitude": np.asarray(np.abs(current_spectrum), dtype=float),
            "current_time": np.asarray(self.xtp.total_current(), dtype=float),
            "polarization_time": np.asarray(self.xtp.total_polarization(), dtype=float),
            "time_axis": np.asarray(self.xtp.timegrid.generate(), dtype=float),
            "directions": tuple(self.xtp.directions),
            "orders": tuple(self.xtp.orders),
        }
        if self.electric_field_time is not None:
            data["electric_field_time"] = np.asarray(self.electric_field_time, dtype=float)
        return data
