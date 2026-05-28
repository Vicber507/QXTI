from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def fermi_dirac(
    energy: ArrayLike,
    chemical_potential: float,
    temperature: float,
    *,
    boltzmann_constant: float = 1.0,
) -> float | FloatArray:
    """Return the Fermi-Dirac occupation."""

    return _scalar_or_array(
        _fermi_dirac_array(
            energy,
            chemical_potential,
            temperature,
            boltzmann_constant=boltzmann_constant,
        )
    )


def maxwell_boltzmann(
    energy: ArrayLike,
    chemical_potential: float,
    temperature: float,
    *,
    boltzmann_constant: float = 1.0,
) -> float | FloatArray:
    """Return the Maxwell-Boltzmann occupation."""

    energies = np.asarray(energy, dtype=float)
    thermal_scale = _thermal_scale(temperature, boltzmann_constant)
    if thermal_scale == 0.0:
        values = np.zeros_like(energies, dtype=float)
        values[energies <= chemical_potential] = 1.0
        return _scalar_or_array(values)

    argument = np.clip((energies - chemical_potential) / thermal_scale, -700.0, 700.0)
    return _scalar_or_array(np.exp(-argument))


def bose_einstein(
    energy: ArrayLike,
    chemical_potential: float,
    temperature: float,
    *,
    boltzmann_constant: float = 1.0,
) -> float | FloatArray:
    """Return the Bose-Einstein occupation."""

    energies = np.asarray(energy, dtype=float)
    thermal_scale = _thermal_scale(temperature, boltzmann_constant)
    if thermal_scale == 0.0:
        return _scalar_or_array(np.zeros_like(energies, dtype=float))

    argument = np.clip((energies - chemical_potential) / thermal_scale, -700.0, 700.0)
    denominator = np.expm1(argument)
    values = np.where(np.abs(denominator) > 1.0e-14, 1.0 / denominator, np.inf)
    return _scalar_or_array(np.asarray(values, dtype=float))


def valence_occupation(
    energy: ArrayLike,
    chemical_potential: float,
    temperature: float,
    *,
    boltzmann_constant: float = 1.0,
) -> float | FloatArray:
    """Return unit occupation for valence bands and zero for conduction bands.

    The occupations are assigned by energy ordering at each ``k`` point:

    - the lowest ``Nb // 2`` bands are treated as valence bands and get
      occupation ``1``
    - the remaining bands are treated as conduction bands and get occupation
      ``0``

    Since :class:`CMD` builds ``rho^(0)`` as ``diag(f_n)`` in the band basis,
    this yields a diagonal equilibrium density matrix with zero coherences.
    """

    del chemical_potential, temperature, boltzmann_constant
    energies = np.asarray(energy, dtype=float)
    if energies.ndim == 0:
        return 1.0

    flat = energies.reshape(-1)
    occupations = np.zeros_like(flat, dtype=float)
    occupied_count = flat.size // 2
    if occupied_count > 0:
        occupied_indices = np.argsort(flat, kind="stable")[:occupied_count]
        occupations[occupied_indices] = 1.0
    return _scalar_or_array(occupations.reshape(energies.shape))


def full_occupation(
    energy: ArrayLike,
    chemical_potential: float,
    temperature: float,
    *,
    boltzmann_constant: float = 1.0,
) -> float | FloatArray:
    """Backward-compatible alias for :func:`valence_occupation`."""

    return valence_occupation(
        energy,
        chemical_potential,
        temperature,
        boltzmann_constant=boltzmann_constant,
    )


@dataclass(slots=True)
class T1T2Relaxation:
    """Phenomenological relaxation model with population and coherence times.

    ``T1`` controls diagonal relaxation toward equilibrium populations.
    ``T2`` controls off-diagonal coherence decay toward zero.
    """

    T1: float = np.inf
    T2: float = np.inf

    def __post_init__(self) -> None:
        self.T1 = _validate_relaxation_time(self.T1, name="T1")
        self.T2 = _validate_relaxation_time(self.T2, name="T2")

    @classmethod
    def from_rates(
        cls,
        gamma_population: float,
        gamma_coherence: float,
    ) -> T1T2Relaxation:
        """Build the model from decay rates ``gamma = 1/T``."""

        if gamma_population < 0.0:
            raise ValueError("gamma_population must be non-negative.")
        if gamma_coherence < 0.0:
            raise ValueError("gamma_coherence must be non-negative.")

        T1 = np.inf if gamma_population == 0.0 else 1.0 / float(gamma_population)
        T2 = np.inf if gamma_coherence == 0.0 else 1.0 / float(gamma_coherence)
        return cls(T1=T1, T2=T2)

    def term(
        self,
        rho: ArrayLike,
        rho_equilibrium: ArrayLike,
    ) -> ComplexArray:
        """Return the dissipative contribution ``d rho / dt``."""

        rho_matrix = _validate_complex_square(rho, name="rho")
        rho_eq_matrix = _validate_complex_square(rho_equilibrium, name="rho_equilibrium")
        if rho_matrix.shape != rho_eq_matrix.shape:
            raise ValueError("rho and rho_equilibrium must have the same shape.")

        derivative = np.zeros_like(rho_matrix, dtype=np.complex128)
        diagonal_mask = np.eye(rho_matrix.shape[0], dtype=bool)

        if np.isfinite(self.T1):
            derivative[diagonal_mask] = -(
                np.diag(rho_matrix) - np.diag(rho_eq_matrix)
            ) / self.T1

        if np.isfinite(self.T2):
            derivative[~diagonal_mask] = -rho_matrix[~diagonal_mask] / self.T2

        return derivative


def t1_t2_relaxation(
    rho: ArrayLike,
    rho_equilibrium: ArrayLike,
    T1: float,
    T2: float,
) -> ComplexArray:
    """Convenience wrapper for the ``T1/T2`` relaxation model."""

    return T1T2Relaxation(T1=T1, T2=T2).term(rho, rho_equilibrium)


def _fermi_dirac_array(
    energy: ArrayLike,
    chemical_potential: float,
    temperature: float,
    *,
    boltzmann_constant: float,
) -> FloatArray:
    energies = np.asarray(energy, dtype=float)
    thermal_scale = _thermal_scale(temperature, boltzmann_constant)

    if thermal_scale == 0.0:
        values = np.zeros_like(energies, dtype=float)
        values[energies < chemical_potential] = 1.0
        values[np.isclose(energies, chemical_potential)] = 0.5
        return np.asarray(values, dtype=float)

    argument = np.clip((energies - chemical_potential) / thermal_scale, -700.0, 700.0)
    return np.asarray(1.0 / (np.exp(argument) + 1.0), dtype=float)


def _thermal_scale(temperature: float, boltzmann_constant: float) -> float:
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    if boltzmann_constant <= 0.0:
        raise ValueError("boltzmann_constant must be strictly positive.")
    return float(temperature) * float(boltzmann_constant)


def _validate_relaxation_time(value: float, *, name: str) -> float:
    time = float(value)
    if np.isinf(time):
        return np.inf
    if time <= 0.0:
        raise ValueError(f"{name} must be strictly positive or infinite.")
    return time


def _validate_complex_square(values: ArrayLike, *, name: str) -> ComplexArray:
    matrix = np.asarray(values, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square 2D array.")
    return np.asarray(matrix, dtype=np.complex128)


def _scalar_or_array(values: FloatArray) -> float | FloatArray:
    if values.ndim == 0:
        return float(values)
    return values
