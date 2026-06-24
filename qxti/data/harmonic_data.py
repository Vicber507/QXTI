from __future__ import annotations

from typing import Any, Callable

import numpy as np

from qxti.response import XTP

ProgressCallback = Callable[[int, str], None]


class HarmonicData:
    """Build reusable high-harmonic generation (HHG) spectral datasets.

    Wraps an :class:`~qxti.response.XTP` instance and assembles a single
    serialisable dictionary with all time- and frequency-domain observables
    needed for HHG analysis and plotting.

    **Harmonic spectrum.**  The induced (driven) current is defined as the
    sum of all positive perturbative orders:

        J_driven(t) = sum_{s=1}^{max_order} J^(s)(t)

    Its FFT magnitude ``|J_driven(omega)|`` constitutes the HHG spectrum.
    The equilibrium (s = 0) contribution is reported separately so that
    callers can inspect or subtract it.

    **Intraband/interband decomposition.**  When ``band_gauge_frame`` is
    available in the XTP instance, the current is further decomposed into
    intraband (diagonal in band basis, Bloch oscillation) and interband
    (off-diagonal, optical transition) contributions.

    Parameters
    ----------
    xtp:
        Fully configured :class:`~qxti.response.XTP` instance whose
        ``rho_orders`` contain at least one positive perturbative order.
    electric_field_time:
        Optional ``(Nt, 3)`` electric-field time series to include verbatim
        in the output dataset (useful for normalisation or plotting).
    """

    def __init__(
        self,
        xtp: XTP,
        *,
        electric_field_time: np.ndarray | None = None,
    ) -> None:
        self.xtp = xtp
        self.electric_field_time = None if electric_field_time is None else np.asarray(electric_field_time, dtype=float)

    def current_spectrum_work_units(self) -> int:
        """Return one coarse estimate of the heavy k-space work in HHG dataset construction."""

        selected_orders = tuple(order for order in self.xtp.orders if int(order) > 0)
        if not selected_orders:
            return 0

        nk = int(self.xtp.kgrid.total_points)
        num_directions = len(self.xtp.directions)
        current_units = nk * sum(
            1
            for order in self.xtp.orders
            for direction in self.xtp.directions
            if (int(order), self.xtp._normalize_direction(direction)) not in self.xtp._current_cache
        )
        polarization_units = nk * num_directions * sum(
            1 for order in self.xtp.orders if int(order) not in self.xtp._polarization_cache
        )
        decomposition_units = 0
        if self.xtp.has_current_decomposition():
            decomposition_units = nk * sum(
                1
                for order in selected_orders
                for direction in self.xtp.directions
                if (int(order), self.xtp._normalize_direction(direction)) not in self.xtp._current_decomposition_cache
            )
        return int(current_units + polarization_units + decomposition_units)

    def current_spectrum_data(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Return one serializable dataset for the induced current spectrum.

        Harmonic plots use the sum of every radiative order available in
        ``xtp.orders`` except the equilibrium contribution ``rho^(0)``.
        This keeps the exported HHG observables aligned with the induced
        response stored in the saved ``rho_order_*.npy`` files without
        hard-coding a maximum perturbative order.
        """

        selected_orders = tuple(order for order in self.xtp.orders if int(order) > 0)
        if not selected_orders:
            raise ValueError("Harmonic spectra require at least one positive perturbative order.")

        self._notify_progress(progress_callback, 0, "integrating total current")
        current_time_total = np.asarray(
            self.xtp.total_current(progress_callback=progress_callback),
            dtype=float,
        )
        self._notify_progress(progress_callback, 0, "collecting equilibrium current")
        equilibrium_current_time = self._equilibrium_current_time(progress_callback=progress_callback)
        self._notify_progress(progress_callback, 0, "collecting equilibrium polarization")
        equilibrium_polarization_time = self._equilibrium_polarization_time(progress_callback=progress_callback)
        self._notify_progress(progress_callback, 0, "collecting driven current")
        driven_current_time = self._current_time_for_orders(selected_orders, progress_callback=progress_callback)
        self._notify_progress(progress_callback, 0, "collecting driven polarization")
        driven_polarization_time = self._polarization_time_for_orders(selected_orders, progress_callback=progress_callback)
        self._notify_progress(progress_callback, 0, "computing driven-current FFT")
        omega_axis, current_spectrum = self._fft_time_signal(driven_current_time)
        self._notify_progress(progress_callback, 0, "computing total-current FFT")
        _, total_current_spectrum = self._fft_time_signal(current_time_total)
        current_total_magnitude = np.sqrt(np.sum(np.abs(current_spectrum) ** 2, axis=1))
        data = {
            "omega_axis": np.asarray(omega_axis, dtype=float),
            "current_spectrum": np.asarray(current_spectrum, dtype=np.complex128),
            "current_magnitude": np.asarray(np.abs(current_spectrum), dtype=float),
            "current_total_magnitude": np.asarray(current_total_magnitude, dtype=float),
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
        if self.xtp.has_current_decomposition():
            self._notify_progress(progress_callback, 0, "integrating intraband/interband current")
            intraband_time, interband_time = self._current_decomposition_for_orders(
                selected_orders,
                progress_callback=progress_callback,
            )
            self._notify_progress(progress_callback, 0, "computing intraband FFT")
            _, intraband_spectrum = self._fft_time_signal(intraband_time)
            self._notify_progress(progress_callback, 0, "computing interband FFT")
            _, interband_spectrum = self._fft_time_signal(interband_time)
            data.update(
                {
                    "current_decomposition_available": True,
                    "current_time_intraband": intraband_time,
                    "current_time_interband": interband_time,
                    "current_spectrum_intraband": np.asarray(intraband_spectrum, dtype=np.complex128),
                    "current_spectrum_interband": np.asarray(interband_spectrum, dtype=np.complex128),
                    "current_total_magnitude_intraband": np.asarray(
                        np.sqrt(np.sum(np.abs(intraband_spectrum) ** 2, axis=1)),
                        dtype=float,
                    ),
                    "current_total_magnitude_interband": np.asarray(
                        np.sqrt(np.sum(np.abs(interband_spectrum) ** 2, axis=1)),
                        dtype=float,
                    ),
                }
            )
        else:
            data["current_decomposition_available"] = False
        if self.xtp.kgrid.dimension == 2:
            self._notify_progress(progress_callback, 0, "packing Brillouin-zone mask data")
            data.update(self.xtp.bz_mask_plot_data())
        if self.electric_field_time is not None:
            data["electric_field_time"] = np.asarray(self.electric_field_time, dtype=float)
        self._notify_progress(progress_callback, 0, "packing current-spectrum dataset")
        return data

    def _equilibrium_current_time(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> np.ndarray:
        if 0 not in self.xtp.orders or len(self.xtp.orders) <= 1:
            return np.zeros((len(self.xtp.timegrid), 3), dtype=float)

        equilibrium = np.zeros((len(self.xtp.timegrid), 3), dtype=float)
        for direction in self.xtp.directions:
            axis = self.xtp._direction_axis(direction)
            equilibrium[:, axis] = self.xtp.current(0, direction, progress_callback=progress_callback)
        return equilibrium

    def _equilibrium_polarization_time(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> np.ndarray:
        if 0 not in self.xtp.orders or len(self.xtp.orders) <= 1:
            return np.zeros((len(self.xtp.timegrid), 3), dtype=float)
        return np.asarray(self.xtp.polarization(0, progress_callback=progress_callback), dtype=float)

    def _current_time_for_orders(
        self,
        orders: tuple[int, ...],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> np.ndarray:
        current = np.zeros((len(self.xtp.timegrid), 3), dtype=float)
        for order in orders:
            for direction in self.xtp.directions:
                axis = self.xtp._direction_axis(direction)
                current[:, axis] += self.xtp.current(order, direction, progress_callback=progress_callback)
        return current

    def _polarization_time_for_orders(
        self,
        orders: tuple[int, ...],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> np.ndarray:
        polarization = np.zeros((len(self.xtp.timegrid), 3), dtype=float)
        for order in orders:
            polarization += np.asarray(
                self.xtp.polarization(order, progress_callback=progress_callback),
                dtype=float,
            )
        return polarization

    def _current_decomposition_for_orders(
        self,
        orders: tuple[int, ...],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        intraband = np.zeros((len(self.xtp.timegrid), 3), dtype=float)
        interband = np.zeros((len(self.xtp.timegrid), 3), dtype=float)
        for order in orders:
            for direction in self.xtp.directions:
                axis = self.xtp._direction_axis(direction)
                parts = self.xtp.current_decomposition(
                    order,
                    direction,
                    progress_callback=progress_callback,
                )
                intraband[:, axis] += np.asarray(parts["intraband"], dtype=float)
                interband[:, axis] += np.asarray(parts["interband"], dtype=float)
        return intraband, interband

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

    @staticmethod
    def _notify_progress(
        progress_callback: ProgressCallback | None,
        steps: int,
        message: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(int(steps), str(message))
