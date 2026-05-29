from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from qxti.grids import KGrid

from .hamiltonian import ComplexArray, Hamiltonian, RealArray


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class BandGaugeFrame:
    """Smooth band-basis data defined on one regular :class:`KGrid`.

    The frame stores one phase-aligned set of eigenvectors on the grid, the
    full Berry connection obtained from finite differences of those
    eigenvectors, and operator transforms that are consistent with the same
    basis. This keeps CMD and XTP in the same band gauge.
    """

    hamiltonian: Hamiltonian
    kgrid: KGrid
    hermiticity_atol: float = 1.0e-10
    overlap_tolerance: float = 1.0e-14
    _points: FloatArray = field(init=False, repr=False)
    _energies: RealArray = field(init=False, repr=False)
    _vectors: ComplexArray = field(init=False, repr=False)
    _connections: dict[str, ComplexArray] = field(init=False, repr=False, default_factory=dict)
    _currents: dict[str, ComplexArray] = field(init=False, repr=False, default_factory=dict)
    _velocities: dict[str, ComplexArray] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._points = np.asarray(self.kgrid.points(), dtype=float)
        self._energies, raw_vectors = self._diagonalize_grid()
        self._vectors = self._smooth_vectors(raw_vectors)
        self._connections = self._build_connection_cache()

    @property
    def energies(self) -> RealArray:
        return np.asarray(self._energies, dtype=np.float64)

    @property
    def eigenvectors(self) -> ComplexArray:
        return np.asarray(self._vectors, dtype=np.complex128)

    def omega_matrix(self, index: int) -> ComplexArray:
        energies = self._energies[index]
        omega = energies[:, np.newaxis] - energies[np.newaxis, :]
        return np.asarray(omega, dtype=np.complex128)

    def connection(self, direction: str) -> ComplexArray:
        key = self._normalize_direction(direction)
        return np.asarray(self._connections[key], dtype=np.complex128)

    def velocity(self, direction: str) -> ComplexArray:
        key = self._normalize_direction(direction)
        if key not in self._velocities:
            self._velocities[key] = self._transform_operator_over_grid(
                lambda kx, ky, kz: self.hamiltonian.velocity_operator(kx, ky, kz, key)
            )
        return np.asarray(self._velocities[key], dtype=np.complex128)

    def current(self, direction: str) -> ComplexArray:
        key = self._normalize_direction(direction)
        if key not in self._currents:
            self._currents[key] = np.asarray(-self.velocity(key), dtype=np.complex128)
        return np.asarray(self._currents[key], dtype=np.complex128)

    def transform_from_band_basis(self, tensor: ComplexArray) -> ComplexArray:
        values = np.asarray(tensor, dtype=np.complex128)
        if values.shape[:1] != (self.kgrid.total_points,):
            raise ValueError("tensor first dimension must match the number of k-points.")
        transformed = np.empty_like(values)
        for ik, unitary in enumerate(self._vectors):
            transformed[ik] = np.matmul(
                np.matmul(unitary[np.newaxis, :, :], values[ik]),
                unitary.conj().T[np.newaxis, :, :],
            )
        return np.asarray(transformed, dtype=np.complex128)

    def summary(self) -> dict[str, object]:
        return {
            "frame": self.__class__.__name__,
            "hamiltonian": self.hamiltonian.model_name,
            "kgrid_shape": self.kgrid.shape,
            "total_points": self.kgrid.total_points,
            "dimension": self.kgrid.dimension,
        }

    def _diagonalize_grid(self) -> tuple[RealArray, ComplexArray]:
        nk = self.kgrid.total_points
        nb = self.hamiltonian.basis_size
        energies = np.empty((nk, nb), dtype=np.float64)
        vectors = np.empty((nk, nb, nb), dtype=np.complex128)

        for ik, (kx, ky, kz) in enumerate(self._points):
            self.hamiltonian.require_hermitian(float(kx), float(ky), float(kz), atol=self.hermiticity_atol)
            values, unitary = self.hamiltonian.diagonalize(float(kx), float(ky), float(kz))
            energies[ik] = values
            vectors[ik] = unitary
        return energies, vectors

    def _smooth_vectors(self, raw_vectors: ComplexArray) -> ComplexArray:
        nb = self.hamiltonian.basis_size
        vectors = np.asarray(raw_vectors, dtype=np.complex128).reshape(*self.kgrid.shape, nb, nb).copy()
        self._fix_reference_phase(vectors[(0, 0, 0)])

        for axis, axis_size in enumerate(self.kgrid.shape[: self.kgrid.dimension]):
            if axis_size <= 1:
                continue
            other_ranges = [range(size) for index, size in enumerate(self.kgrid.shape) if index != axis]
            for other_indices in np.ndindex(*(len(rng) for rng in other_ranges)):
                fixed = [0, 0, 0]
                cursor = 0
                for candidate_axis in range(3):
                    if candidate_axis == axis:
                        continue
                    fixed[candidate_axis] = other_ranges[cursor][other_indices[cursor]]
                    cursor += 1
                for step in range(1, axis_size):
                    current = list(fixed)
                    previous = list(fixed)
                    current[axis] = step
                    previous[axis] = step - 1
                    self._align_columns(vectors[tuple(previous)], vectors[tuple(current)])

        return np.asarray(vectors.reshape(self.kgrid.total_points, nb, nb), dtype=np.complex128)

    def _build_connection_cache(self) -> dict[str, ComplexArray]:
        nb = self.hamiltonian.basis_size
        vectors_mesh = self._vectors.reshape(*self.kgrid.shape, nb, nb)
        connections: dict[str, ComplexArray] = {}
        coordinates = (
            np.asarray(self.kgrid.kx_values, dtype=float),
            np.asarray(self.kgrid.ky_values, dtype=float),
            np.asarray(self.kgrid.kz_values, dtype=float),
        )

        for axis, direction in enumerate(("x", "y", "z")):
            if axis >= self.hamiltonian.dimension:
                connections[direction] = np.zeros((self.kgrid.total_points, nb, nb), dtype=np.complex128)
                continue
            if len(coordinates[axis]) < 2:
                connections[direction] = self._fallback_connection(direction)
                continue

            edge_order = 2 if len(coordinates[axis]) >= 3 else 1
            derivative = np.gradient(
                vectors_mesh,
                coordinates[axis],
                axis=axis,
                edge_order=edge_order,
            )
            connection_mesh = 1.0j * np.einsum(
                "...in,...im->...nm",
                np.conjugate(vectors_mesh),
                derivative,
                optimize=True,
            )
            connection_mesh = 0.5 * (connection_mesh + np.conjugate(np.swapaxes(connection_mesh, -1, -2)))
            connections[direction] = np.asarray(
                connection_mesh.reshape(self.kgrid.total_points, nb, nb),
                dtype=np.complex128,
            )

        return connections

    def _fallback_connection(self, direction: str) -> ComplexArray:
        velocity = self.velocity(direction)
        nk, nb, _ = velocity.shape
        connection = np.zeros((nk, nb, nb), dtype=np.complex128)

        for ik in range(nk):
            energies = self._energies[ik]
            for row in range(nb):
                for col in range(nb):
                    if row == col:
                        continue
                    delta = float(energies[col] - energies[row])
                    if abs(delta) <= self.overlap_tolerance:
                        continue
                    connection[ik, row, col] = 1.0j * velocity[ik, row, col] / delta

        return np.asarray(connection, dtype=np.complex128)

    def _transform_operator_over_grid(
        self,
        builder,
    ) -> ComplexArray:
        nk = self.kgrid.total_points
        nb = self.hamiltonian.basis_size
        transformed = np.empty((nk, nb, nb), dtype=np.complex128)

        for ik, (kx, ky, kz) in enumerate(self._points):
            operator = np.asarray(builder(float(kx), float(ky), float(kz)), dtype=np.complex128)
            unitary = self._vectors[ik]
            transformed[ik] = unitary.conj().T @ operator @ unitary

        return np.asarray(transformed, dtype=np.complex128)

    def _fix_reference_phase(self, unitary: ComplexArray) -> None:
        for band in range(unitary.shape[1]):
            column = unitary[:, band]
            pivot = int(np.argmax(np.abs(column)))
            amplitude = column[pivot]
            if abs(amplitude) <= self.overlap_tolerance:
                continue
            unitary[:, band] /= amplitude / abs(amplitude)

    def _align_columns(self, reference: ComplexArray, current: ComplexArray) -> None:
        for band in range(reference.shape[1]):
            overlap = np.vdot(reference[:, band], current[:, band])
            if abs(overlap) <= self.overlap_tolerance:
                continue
            current[:, band] /= overlap / abs(overlap)

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        key = direction.strip().lower()
        if key not in {"x", "y", "z"}:
            raise ValueError("direction must be 'x', 'y', or 'z'.")
        return key
