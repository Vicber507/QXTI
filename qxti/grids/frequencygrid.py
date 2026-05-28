from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class FrequencyGrid:
    """
    Defines frequency and harmonic axes for spectral analysis.
    Used to map FFT outputs to physical frequencies and identify harmonics.
    """

    omega_min: float
    omega_max: float
    Nomega: int
    harmonic_min: int = 1
    harmonic_max: int = 9

    def __post_init__(self) -> None:
        """Validaciones estrictas al inicializar, estilo KGrid."""
        if self.omega_min >= self.omega_max:
            raise ValueError("omega_min must be strictly less than omega_max.")
        if self.Nomega <= 0:
            raise ValueError("Nomega must be strictly positive.")
        if self.harmonic_min > self.harmonic_max:
            raise ValueError("harmonic_min must be less than or equal to harmonic_max.")

    def generate(self) -> FloatArray:
        """
        Generates the linear frequency axis array.
        
        Returns:
            ndarray[Nomega]: Array of frequencies from omega_min to omega_max.
        """
        return np.linspace(self.omega_min, self.omega_max, self.Nomega)

    def harmonic_axis(self, fundamental_omega: float) -> FloatArray:
        """
        Calculates the frequencies corresponding to specific harmonic orders
        based on a fundamental frequency (e.g., laser frequency).

        Input:
            fundamental_omega (float): The base frequency (omega_0).
        
        Output:
            ndarray: Array of harmonic frequencies [n * omega_0].
        """
        if fundamental_omega <= 0:
            raise ValueError("fundamental_omega must be positive.")

        # Genera los enteros de los armónicos: [harmonic_min, ..., harmonic_max]
        harmonics = np.arange(self.harmonic_min, self.harmonic_max + 1)
        
        # Retorna las frecuencias multiplicadas por el fundamental
        return np.asarray(harmonics * fundamental_omega, dtype=float)

    def get_frequency_index(self, omega: float) -> int:
        """
        Finds the index of the grid point closest to a specific frequency.
        Useful for extracting values at specific harmonic peaks.
        
        Input:
            omega (float): The target frequency.
        
        Output:
            int: The index in the generated grid.
        """
        if omega < self.omega_min or omega > self.omega_max:
            raise ValueError(
                f"Frequency {omega} is out of bounds "
                f"[{self.omega_min}, {self.omega_max}]."
            )
        
        # Genera el eje y busca el índice con la distancia mínima
        axis = self.generate()
        return int(np.argmin(np.abs(axis - omega)))

    def __len__(self) -> int:
        """Returns the number of frequency points."""
        return self.Nomega
    
    def __repr__(self) -> str:
        return (
            f"FrequencyGrid(omega_min={self.omega_min}, "
            f"omega_max={self.omega_max}, Nomega={self.Nomega}, "
            f"harmonic_range=[{self.harmonic_min}, {self.harmonic_max}])"
        )
