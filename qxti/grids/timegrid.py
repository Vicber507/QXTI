from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

RealArray = NDArray[np.float64]


class TimeGrid:
    """
    Defines temporal grid and FFT utilities.
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
        Returns FFT frequency axis.
        """

        N = self.Nt

        if self.zero_padding:

            N *= self.padding_factor

        freq = np.fft.fftfreq(
            N,
            d=self.dt
        )
        # si se necesita frecuencia angular es necesario editar
        # omega = 2*np.pi*freq
        return freq
