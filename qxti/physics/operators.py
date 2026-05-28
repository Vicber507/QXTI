from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hamiltonian import ComplexArray, Hamiltonian


@dataclass(slots=True)
class OperatorFactory:
    """Build physical operators directly from one Hamiltonian ``H(k)``.

    The public interface mirrors the UML diagram used by CMD, XTP and future
    observable calculators:

    - ``velocity(dir, kx, ky, kz)``
    - ``current(dir, kx, ky, kz)``
    - ``dipole(dir, kx, ky, kz)``
    - ``position(dir, kx, ky, kz)``
    - ``inverse_mass(dir1, dir2, kx, ky, kz)``
    - ``berry_connection(dir, kx, ky, kz)``
    """

    hamiltonian: Hamiltonian
    basis: str = "working"
    dipole_regularization: float = 1.0e-12

    def __post_init__(self) -> None:
        self.basis = self._normalize_basis(self.basis)
        if self.dipole_regularization <= 0.0:
            raise ValueError("dipole_regularization must be strictly positive.")

    def hamiltonian_operator(
        self,
        kx: float,
        ky: float,
        kz: float,
        *,
        basis: str | None = None,
    ) -> ComplexArray:
        """Return the Hamiltonian matrix in the requested basis."""

        matrix = self.hamiltonian.H(kx, ky, kz)
        return self._maybe_transform(matrix, kx, ky, kz, basis=basis)

    def velocity(
        self,
        direction: str,
        kx: float,
        ky: float,
        kz: float,
        *,
        basis: str | None = None,
    ) -> ComplexArray:
        """Return the velocity operator."""

        matrix = self.hamiltonian.velocity_operator(kx, ky, kz, direction)
        return self._maybe_transform(matrix, kx, ky, kz, basis=basis)

    def current(
        self,
        direction: str,
        kx: float,
        ky: float,
        kz: float,
        *,
        basis: str | None = None,
    ) -> ComplexArray:
        """Return the current operator.

        In atomic units for electrons, ``j = -v``.
        """

        return self.hamiltonian.validate_matrix(
            -self.velocity(direction, kx, ky, kz, basis=basis)
        )

    def dipole(
        self,
        direction: str,
        kx: float,
        ky: float,
        kz: float,
        *,
        basis: str | None = None,
    ) -> ComplexArray:
        """Return the dipole operator."""

        return self.position(direction, kx, ky, kz, basis=basis)

    def position(
        self,
        direction: str,
        kx: float,
        ky: float,
        kz: float,
        *,
        basis: str | None = None,
    ) -> ComplexArray:
        """Return the position operator."""

        return self.berry_connection(direction, kx, ky, kz, basis=basis)

    def inverse_mass(
        self,
        direction1: str,
        direction2: str,
        kx: float,
        ky: float,
        kz: float,
        *,
        basis: str | None = None,
    ) -> ComplexArray:
        """Return one inverse-mass tensor element."""

        matrix = self.hamiltonian.inverse_mass_operator(kx, ky, kz, direction1, direction2)
        return self._maybe_transform(matrix, kx, ky, kz, basis=basis)

    def berry_connection(
        self,
        direction: str,
        kx: float,
        ky: float,
        kz: float,
        *,
        basis: str | None = None,
    ) -> ComplexArray:
        """Return the Berry-connection matrix.

        For non-degenerate bands we use

            A_nm = i v_nm / (E_m - E_n),  n != m

        and set the diagonal gauge part to zero.
        """

        connection_band = self._berry_connection_band(direction, kx, ky, kz)
        if self._normalize_basis(basis or self.basis) == "band":
            return self.hamiltonian.validate_matrix(connection_band)
        return self.hamiltonian.transform_from_band_basis(connection_band, kx, ky, kz)

    # Backward-compatible aliases used by the rest of the codebase.
    def velocity_operator(
        self,
        kx: float,
        ky: float,
        kz: float,
        direction: str,
        *,
        basis: str | None = None,
    ) -> ComplexArray:
        return self.velocity(direction, kx, ky, kz, basis=basis)

    def inverse_mass_operator(
        self,
        kx: float,
        ky: float,
        kz: float,
        direction1: str,
        direction2: str,
        *,
        basis: str | None = None,
    ) -> ComplexArray:
        return self.inverse_mass(direction1, direction2, kx, ky, kz, basis=basis)

    def position_operator(
        self,
        kx: float,
        ky: float,
        kz: float,
        direction: str,
        *,
        basis: str | None = None,
    ) -> ComplexArray:
        return self.position(direction, kx, ky, kz, basis=basis)

    def dipole_operator(
        self,
        kx: float,
        ky: float,
        kz: float,
        direction: str,
        *,
        basis: str | None = None,
    ) -> ComplexArray:
        return self.dipole(direction, kx, ky, kz, basis=basis)

    def summary(self) -> dict[str, Any]:
        return {
            "factory": self.__class__.__name__,
            "hamiltonian": self.hamiltonian.model_name,
            "basis": self.basis,
            "dipole_regularization": self.dipole_regularization,
        }

    def _berry_connection_band(
        self,
        direction: str,
        kx: float,
        ky: float,
        kz: float,
    ) -> ComplexArray:
        velocity_band = self.hamiltonian.transform_to_band_basis(
            self.hamiltonian.velocity_operator(kx, ky, kz, direction),
            kx,
            ky,
            kz,
        )
        energies = self.hamiltonian.eigenvalues(kx, ky, kz)
        connection_band = np.zeros_like(velocity_band, dtype=np.complex128)

        for row in range(self.hamiltonian.basis_size):
            for col in range(self.hamiltonian.basis_size):
                if row == col:
                    continue
                delta = float(energies[col] - energies[row])
                if abs(delta) <= self.dipole_regularization:
                    continue
                connection_band[row, col] = 1.0j * velocity_band[row, col] / delta

        return self.hamiltonian.validate_matrix(connection_band)

    def _maybe_transform(
        self,
        matrix: ComplexArray,
        kx: float,
        ky: float,
        kz: float,
        basis: str | None = None,
    ) -> ComplexArray:
        target_basis = self._normalize_basis(basis or self.basis)
        if target_basis == "band":
            return self.hamiltonian.transform_to_band_basis(matrix, kx, ky, kz)
        return self.hamiltonian.validate_matrix(matrix)

    @staticmethod
    def _normalize_basis(basis: str) -> str:
        key = basis.strip().lower()
        aliases = {
            "orbital": "working",
            "working": "working",
            "original": "working",
            "band": "band",
            "bands": "band",
        }
        if key not in aliases:
            raise ValueError("basis must be 'working'/'orbital' or 'band'.")
        return aliases[key]
