from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

RealArray = NDArray[np.float64]


class TimeGrid:
    """
    Defines temporal grid and FFT utilities in atomic units.
    """

    def __init__(
        self,
        t_min: float,
        t_max: float,
        Nt: int,
        fft_window: str = "hann",
        zero_padding: bool = False,
        padding_factor: int = 2
    ) -> None:
        if Nt < 2:
            raise ValueError("Nt must be at least 2.")
        if t_max <= t_min:
            raise ValueError("t_max must be strictly larger than t_min.")
        if padding_factor <= 0:
            raise ValueError("padding_factor must be strictly positive.")

        self.t_min: float = t_min
        self.t_max: float = t_max
        self.Nt: int = Nt

        self.fft_window: str = fft_window

        self.zero_padding: bool = zero_padding

        self.padding_factor: int = padding_factor

        self.dt: float = (
            (self.t_max - self.t_min)
            / (self.Nt - 1)
        )

    @classmethod
    def from_dt(
        cls,
        *,
        t_min: float,
        t_max: float,
        dt: float,
        fft_window: str = "hann",
        zero_padding: bool = False,
        padding_factor: int = 2,
    ) -> TimeGrid:
        """Build one grid from ``dt`` so that the interval covers ``[t_min, t_max]``."""

        if dt <= 0.0:
            raise ValueError("dt must be strictly positive.")
        if t_max <= t_min:
            raise ValueError("t_max must be strictly larger than t_min.")

        duration = float(t_max) - float(t_min)
        Nt = int(np.ceil(duration / float(dt))) + 1
        effective_t_max = float(t_min) + (Nt - 1) * float(dt)
        return cls(
            t_min=float(t_min),
            t_max=effective_t_max,
            Nt=Nt,
            fft_window=fft_window,
            zero_padding=zero_padding,
            padding_factor=padding_factor,
        )

    @property
    def t0(self) -> float:
        """Backward-compatible alias for the initial time."""

        return self.t_min

    @property
    def tf(self) -> float:
        """Backward-compatible alias for the final time."""

        return self.t_max

    @property
    def initial_h(self) -> float:
        """Backward-compatible alias for the nominal time step."""

        return self.dt

    # =====================================================
    # Generate Time Axis
    # =====================================================

    def generate(self) -> RealArray:
        """
        Generates temporal grid.
        """

        return np.linspace(
            self.t_min,
            self.t_max,
            self.Nt
        )

    # =====================================================
    # Time Step
    # =====================================================

    def dt_value(self) -> float:
        """
        Returns time step dt.
        """

        return self.dt

    def __len__(self) -> int:
        return self.Nt

    # =====================================================
    # Window Function
    # =====================================================

    def apply_window(
        self,
        signal: RealArray
    ) -> RealArray:
        """
        Applies FFT window.
        """

        if self.fft_window.lower() == "hann":

            window = np.hanning(len(signal))

        elif self.fft_window.lower() == "hamming":

            window = np.hamming(len(signal))

        elif self.fft_window.lower() == "blackman":

            window = np.blackman(len(signal))

        else:

            window = np.ones(len(signal))
        """Estos métodos son para evitar errores debido 
        al intervalo finito en el que se trabaja
        lo que puede hacer que no se cumpla f(0) = f(T)"""
        return signal * window

    # =====================================================
    # Zero Padding
    # =====================================================

    def padded_signal(
        self,
        signal: RealArray
    ) -> RealArray:
        """
        Applies zero padding.
        """

        if not self.zero_padding:

            return signal

        padded_size = (
            self.padding_factor
            * len(signal)
        )

        padded = np.zeros(
            padded_size,
            dtype=signal.dtype
        )

        padded[:len(signal)] = signal

        return padded

    # =====================================================
    # Frequency Axis
    # =====================================================

    def frequency_axis(self) -> RealArray:
        """
        Returns FFT angular-frequency axis in atomic units.
        """

        N = self.Nt

        if self.zero_padding:

            N *= self.padding_factor

        freq = np.fft.fftfreq(
            N,
            d=self.dt
        )
        omega = 2.0 * np.pi * freq
        return np.asarray(omega, dtype=np.float64)
