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
        """Return one serializable dataset for the induced current spectrum.

        Harmonic plots use only the radiative orders ``rho^(1) + rho^(2) + rho^(3)``
        from whatever subset is available in ``xtp.orders``. This keeps the
        exported HHG observables aligned with the usual perturbative
        interpretation and avoids contamination from the equilibrium
        contribution ``rho^(0)``.
        """

        selected_orders = tuple(order for order in self.xtp.orders if 1 <= int(order) <= 3)
        if not selected_orders:
            selected_orders = tuple(order for order in self.xtp.orders if int(order) > 0)

        current_time_total = np.asarray(self.xtp.total_current(), dtype=float)
        polarization_time_total = np.asarray(self.xtp.total_polarization(), dtype=float)
        equilibrium_current_time = self._equilibrium_current_time()
        equilibrium_polarization_time = self._equilibrium_polarization_time()
        driven_current_time = self._current_time_for_orders(selected_orders)
        driven_polarization_time = self._polarization_time_for_orders(selected_orders)
        omega_axis, current_spectrum = self._fft_time_signal(driven_current_time)
        _, total_current_spectrum = self._fft_time_signal(current_time_total)
        data = {
            "omega_axis": np.asarray(omega_axis, dtype=float),
            "current_spectrum": np.asarray(current_spectrum, dtype=np.complex128),
            "current_magnitude": np.asarray(np.abs(current_spectrum), dtype=float),
            "current_time": np.asarray(driven_current_time, dtype=float),
            "polarization_time": np.asarray(driven_polarization_time, dtype=float),
            "current_time_total": np.asarray(current_time_total, dtype=float),
            "current_spectrum_total": np.asarray(total_current_spectrum, dtype=np.complex128),
            "equilibrium_current_time": np.asarray(equilibrium_current_time, dtype=float),
            "equilibrium_polarization_time": np.asarray(equilibrium_polarization_time, dtype=float),
            "time_axis": np.asarray(self.xtp.timegrid.generate(), dtype=float),
            "directions": tuple(self.xtp.directions),
            "orders": selected_orders,
            "all_orders": tuple(self.xtp.orders),
            "equilibrium_subtracted": bool(0 in self.xtp.orders),
            "bz_mask": self.xtp.bz_mask_summary(),
        }
        if self.xtp.kgrid.dimension == 2:
            data.update(self.xtp.bz_mask_plot_data())
        if self.electric_field_time is not None:
            data["electric_field_time"] = np.asarray(self.electric_field_time, dtype=float)
        return data

    def _equilibrium_current_time(self) -> np.ndarray:
        if 0 not in self.xtp.orders or len(self.xtp.orders) <= 1:
            return np.zeros((len(self.xtp.timegrid), 3), dtype=float)

        equilibrium = np.zeros((len(self.xtp.timegrid), 3), dtype=float)
        for direction in self.xtp.directions:
            axis = self.xtp._direction_axis(direction)
            equilibrium[:, axis] = self.xtp.current(0, direction)
        return equilibrium

    def _equilibrium_polarization_time(self) -> np.ndarray:
        if 0 not in self.xtp.orders or len(self.xtp.orders) <= 1:
            return np.zeros((len(self.xtp.timegrid), 3), dtype=float)
        return np.asarray(self.xtp.polarization(0), dtype=float)

    def _current_time_for_orders(self, orders: tuple[int, ...]) -> np.ndarray:
        current = np.zeros((len(self.xtp.timegrid), 3), dtype=float)
        for order in orders:
            for direction in self.xtp.directions:
                axis = self.xtp._direction_axis(direction)
                current[:, axis] += self.xtp.current(order, direction)
        return current

    def _polarization_time_for_orders(self, orders: tuple[int, ...]) -> np.ndarray:
        polarization = np.zeros((len(self.xtp.timegrid), 3), dtype=float)
        for order in orders:
            polarization += np.asarray(self.xtp.polarization(order), dtype=float)
        return polarization

    def _fft_time_signal(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(signal, dtype=np.complex128)
        window = np.asarray(
            self.xtp.timegrid.apply_window(np.ones(len(self.xtp.timegrid), dtype=float)),
            dtype=np.float64,
        )
        reshape = (len(window),) + (1,) * (values.ndim - 1)
        weighted = values * window.reshape(reshape)
        nfft = len(self.xtp.timegrid)
        if self.xtp.timegrid.zero_padding:
            nfft *= self.xtp.timegrid.padding_factor
        spectrum = self.xtp.timegrid.dt * np.fft.fft(weighted, n=nfft, axis=0)
        omega_axis = np.asarray(self.xtp.timegrid.frequency_axis(), dtype=np.float64)
        return omega_axis, np.asarray(spectrum, dtype=np.complex128)
