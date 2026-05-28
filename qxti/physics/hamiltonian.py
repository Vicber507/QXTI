from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(slots=True)
class Hamiltonian(ABC):
    """Abstract base class for tight-binding Hamiltonians in atomic units.

    The class is intentionally restricted to k-space matrix mechanics. It
    stores model metadata, validates matrix structure, provides numerical
    derivatives with respect to reciprocal-space coordinates, and exposes
    derived operators used by downstream response calculations.
    """

    DEFAULT_LATTICE: ClassVar[dict[str, Any]] = {}

    model_name: str
    params: dict[str, Any] = field(default_factory=dict)
    lattice: dict[str, Any] = field(default_factory=dict)
    basis_size: int = 1
    dimension: int = 3
    basis_type: str = "orbital"
    is_periodic: bool = True
    dk_derivative: float = 1.0e-5

    _VALID_DIRECTIONS = {"x": 0, "y": 1, "z": 2}

    def __post_init__(self) -> None:
        """Validate constructor inputs and merge default model parameters."""

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if self.basis_size <= 0:
            raise ValueError("basis_size must be strictly positive.")
        if self.dimension not in {1, 2, 3}:
            raise ValueError("dimension must be one of 1, 2, or 3.")
        if not isinstance(self.basis_type, str) or not self.basis_type.strip():
            raise ValueError("basis_type must be a non-empty string.")
        if self.dk_derivative <= 0.0:
            raise ValueError("dk_derivative must be strictly positive.")

        initial_params = dict(self.params)
        initial_lattice = dict(self.lattice)
        self.params = {}
        self.lattice = {}
        self.set_params(initial_params)
        self.set_lattice(initial_lattice)

    def default_params(self) -> dict[str, Any]:
        """Return default model parameters for the concrete Hamiltonian."""

        return {}

    def default_lattice(self) -> dict[str, Any]:
        """Return default lattice information for the concrete Hamiltonian."""

        lattice = getattr(type(self), "DEFAULT_LATTICE", {})
        if not isinstance(lattice, dict):
            raise TypeError("DEFAULT_LATTICE must be a dict.")
        return deepcopy(lattice)

    def set_params(self, params: dict[str, Any]) -> None:
        """Update model parameters without discarding default values.

        The update order is:
        1. concrete-model defaults from :meth:`default_params`
        2. currently stored parameters
        3. the user-provided ``params`` mapping
        """

        updated = dict(self.default_params())
        updated.update(self.params)
        updated.update(dict(params))
        self.params = updated

    def set_lattice(self, lattice: dict[str, Any]) -> None:
        """Update lattice information without discarding default values."""

        updated = dict(self.default_lattice())
        updated.update(self.lattice)
        updated.update(dict(lattice))
        self.lattice = updated

    @abstractmethod
    def H(self, kx: float, ky: float, kz: float) -> np.ndarray:
        """Return the Hamiltonian matrix :math:`H(\\mathbf{k})`."""

    def validate_matrix(self, matrix: ArrayLike) -> ComplexArray:
        """Return one validated complex Hamiltonian matrix.

        The matrix is converted to ``complex`` and must be two-dimensional,
        square, and of shape ``(basis_size, basis_size)``.
        """

        values = np.asarray(matrix, dtype=complex)
        if values.ndim != 2:
            raise ValueError("Hamiltonian matrix must be a 2D array.")
        if values.shape[0] != values.shape[1]:
            raise ValueError("Hamiltonian matrix must be square.")

        expected_shape = (self.basis_size, self.basis_size)
        if values.shape != expected_shape:
            raise ValueError(
                f"Hamiltonian matrix must have shape {expected_shape}, "
                f"got {values.shape}."
            )
        return np.asarray(values, dtype=np.complex128)

    def validate_hermiticity(
        self,
        kx: float,
        ky: float,
        kz: float,
        atol: float = 1.0e-10,
    ) -> bool:
        """Return whether :math:`H(\\mathbf{k})` is Hermitian within tolerance."""

        matrix = self._matrix_at(kx, ky, kz)
        return bool(np.allclose(matrix, matrix.conj().T, atol=atol))

    def require_hermitian(
        self,
        kx: float,
        ky: float,
        kz: float,
        atol: float = 1.0e-10,
    ) -> None:
        """Raise ``ValueError`` when :math:`H(\\mathbf{k})` is not Hermitian."""

        if not self.validate_hermiticity(kx, ky, kz, atol=atol):
            raise ValueError(
                f"Hamiltonian '{self.model_name}' is not Hermitian at "
                f"(kx, ky, kz)=({kx}, {ky}, {kz})."
            )

    def dH_dk(
        self,
        kx: float,
        ky: float,
        kz: float,
        direction: str,
    ) -> ComplexArray:
        """Return the first finite-difference derivative of :math:`H`.

        A centered finite difference is used with step ``dk_derivative``.
        """

        k_plus, k_minus = self._shifted_k_points(kx, ky, kz, direction)
        step = self.dk_derivative
        return (self._matrix_at(*k_plus) - self._matrix_at(*k_minus)) / (2.0 * step)

    def d2H_dk2(
        self,
        kx: float,
        ky: float,
        kz: float,
        direction1: str,
        direction2: str,
    ) -> ComplexArray:
        """Return the second finite-difference derivative of :math:`H`.

        The method supports both pure and mixed derivatives:

        .. math::

            \\partial_i^2 H \\approx \\frac{H(k+h\\hat{i}) - 2H(k) + H(k-h\\hat{i})}{h^2}

        and

        .. math::

            \\partial_i\\partial_j H \\approx
            \\frac{H_{++} - H_{+-} - H_{-+} + H_{--}}{4 h^2}.
        """

        step = self.dk_derivative
        k_vector = np.array([kx, ky, kz], dtype=float)
        axis1 = self._direction_axis(direction1)
        axis2 = self._direction_axis(direction2)

        if axis1 == axis2:
            k_plus = k_vector.copy()
            k_minus = k_vector.copy()
            k_plus[axis1] += step
            k_minus[axis1] -= step
            return (
                self._matrix_at(*k_plus)
                - 2.0 * self._matrix_at(*k_vector)
                + self._matrix_at(*k_minus)
            ) / step**2

        k_pp = k_vector.copy()
        k_pm = k_vector.copy()
        k_mp = k_vector.copy()
        k_mm = k_vector.copy()

        k_pp[axis1] += step
        k_pp[axis2] += step
        k_pm[axis1] += step
        k_pm[axis2] -= step
        k_mp[axis1] -= step
        k_mp[axis2] += step
        k_mm[axis1] -= step
        k_mm[axis2] -= step

        return (
            self._matrix_at(*k_pp)
            - self._matrix_at(*k_pm)
            - self._matrix_at(*k_mp)
            + self._matrix_at(*k_mm)
        ) / (4.0 * step**2)

    def velocity_operator(
        self,
        kx: float,
        ky: float,
        kz: float,
        direction: str,
    ) -> ComplexArray:
        """Return the velocity operator in atomic units."""

        return self.dH_dk(kx, ky, kz, direction)

    def inverse_mass_operator(
        self,
        kx: float,
        ky: float,
        kz: float,
        direction1: str,
        direction2: str,
    ) -> ComplexArray:
        """Return the inverse-mass tensor element in atomic units."""

        return self.d2H_dk2(kx, ky, kz, direction1, direction2)

    def diagonalize(
        self,
        kx: float,
        ky: float,
        kz: float,
    ) -> tuple[RealArray, ComplexArray]:
        """Diagonalize :math:`H(\\mathbf{k})` using ``numpy.linalg.eigh``."""

        self.require_hermitian(kx, ky, kz)
        eigenvalues, eigenvectors = np.linalg.eigh(self._matrix_at(kx, ky, kz))
        return np.asarray(eigenvalues, dtype=float), np.asarray(eigenvectors, dtype=np.complex128)

    def eigenvalues(self, kx: float, ky: float, kz: float) -> RealArray:
        """Return the band energies of :math:`H(\\mathbf{k})`."""

        values, _ = self.diagonalize(kx, ky, kz)
        return values

    def eigenvectors(self, kx: float, ky: float, kz: float) -> ComplexArray:
        """Return the eigenvectors of :math:`H(\\mathbf{k})`."""

        _, vectors = self.diagonalize(kx, ky, kz)
        return vectors

    def transform_to_band_basis(
        self,
        operator: ArrayLike,
        kx: float,
        ky: float,
        kz: float,
    ) -> ComplexArray:
        """Transform one operator from the working basis to the band basis."""

        validated_operator = self.validate_matrix(operator)
        vectors = self.eigenvectors(kx, ky, kz)
        return vectors.conj().T @ validated_operator @ vectors

    def transform_from_band_basis(
        self,
        operator: ArrayLike,
        kx: float,
        ky: float,
        kz: float,
    ) -> ComplexArray:
        """Transform one operator from the band basis back to the working basis."""

        validated_operator = self.validate_matrix(operator)
        vectors = self.eigenvectors(kx, ky, kz)
        return vectors @ validated_operator @ vectors.conj().T

    def gap(
        self,
        kx: float,
        ky: float,
        kz: float,
        occupied_bands: int | Iterable[int] | None = None,
    ) -> float:
        """Return the direct band gap at one k-point.

        When ``occupied_bands`` is ``None``, half filling is assumed through
        ``basis_size // 2``.
        """

        values = self.eigenvalues(kx, ky, kz)
        if values.size < 2:
            return 0.0

        occupied_count = self._occupied_band_count(values.size, occupied_bands)
        return float(values[occupied_count] - values[occupied_count - 1])

    def band_projector(
        self,
        kx: float,
        ky: float,
        kz: float,
        band_index: int,
    ) -> ComplexArray:
        """Return the projector onto one band eigenstate."""

        _, vectors = self.diagonalize(kx, ky, kz)
        if not 0 <= band_index < self.basis_size:
            raise ValueError(
                f"band_index must be between 0 and {self.basis_size - 1}, got {band_index}."
            )

        state = vectors[:, band_index]
        return self.validate_matrix(np.outer(state, state.conj()))

    def occupied_projector(
        self,
        kx: float,
        ky: float,
        kz: float,
        fermi_level: float,
    ) -> ComplexArray:
        """Return the projector onto the subspace below ``fermi_level``."""

        values, vectors = self.diagonalize(kx, ky, kz)
        projector = np.zeros((self.basis_size, self.basis_size), dtype=complex)
        for band_index, energy in enumerate(values):
            if energy <= fermi_level:
                state = vectors[:, band_index]
                projector += np.outer(state, state.conj())
        return self.validate_matrix(projector)

    def summary(self) -> dict[str, Any]:
        """Return a serializable summary of the Hamiltonian configuration."""

        return {
            "model_name": self.model_name,
            "params": dict(self.params),
            "lattice": deepcopy(self.lattice),
            "basis_size": self.basis_size,
            "dimension": self.dimension,
            "basis_type": self.basis_type,
            "is_periodic": self.is_periodic,
            "dk_derivative": self.dk_derivative,
        }

    def real_space_axis_lengths(self) -> tuple[float, ...]:
        """Return effective real-space lattice lengths in atomic units.

        The result is ordered as ``(x,)``, ``(x, y)``, or ``(x, y, z)``
        depending on ``dimension``. When explicit real-space vectors are
        available they are preferred; otherwise common lattice-constant keys
        are used as fallbacks.
        """

        lengths: list[float] = []
        for axis in range(self.dimension):
            vector_length = self._real_space_vector_length(axis)
            if vector_length is not None:
                lengths.append(vector_length)
                continue
            lengths.append(self._fallback_lattice_constant(axis))
        return tuple(lengths)

    def reciprocal_box_bounds(self) -> tuple[tuple[float, float], ...]:
        """Return a square/cubic reciprocal box in atomic units.

        Each active direction uses the interval ``[-pi / a_i, pi / a_i]``,
        where ``a_i`` is the corresponding real-space lattice length.
        """

        bounds: list[tuple[float, float]] = []
        for length in self.real_space_axis_lengths():
            half_width = float(np.pi / length)
            bounds.append((-half_width, half_width))
        return tuple(bounds)

    def _matrix_at(self, kx: float, ky: float, kz: float) -> ComplexArray:
        return self.validate_matrix(self.H(float(kx), float(ky), float(kz)))

    def _shifted_k_points(
        self,
        kx: float,
        ky: float,
        kz: float,
        direction: str,
    ) -> tuple[RealArray, RealArray]:
        axis = self._direction_axis(direction)
        step = self.dk_derivative
        k_plus = np.array([kx, ky, kz], dtype=float)
        k_minus = np.array([kx, ky, kz], dtype=float)
        k_plus[axis] += step
        k_minus[axis] -= step
        return k_plus, k_minus

    def _direction_axis(self, direction: str) -> int:
        try:
            axis = self._VALID_DIRECTIONS[direction.lower()]
        except KeyError as exc:
            raise ValueError("direction must be one of 'x', 'y', or 'z'.") from exc

        if axis >= self.dimension:
            raise ValueError(
                f"Direction '{direction}' is outside dimension {self.dimension}."
            )
        return axis

    def _occupied_band_count(
        self,
        num_bands: int,
        occupied_bands: int | Iterable[int] | None,
    ) -> int:
        if occupied_bands is None:
            count = num_bands // 2
        elif isinstance(occupied_bands, int):
            count = occupied_bands
        else:
            indices = sorted({int(index) for index in occupied_bands})
            if not indices:
                raise ValueError("occupied_bands cannot be empty.")
            if indices[0] < 0 or indices[-1] >= num_bands:
                raise ValueError("occupied_bands contains out-of-range band indices.")
            count = indices[-1] + 1

        if not 0 < count < num_bands:
            raise ValueError(
                "Gap requires at least one occupied band and one unoccupied band."
            )
        return count

    # Backward-compatible alias for older code paths.
    def _validate_matrix(self, matrix: ArrayLike) -> ComplexArray:
        return self.validate_matrix(matrix)

    def _real_space_vector_length(self, axis: int) -> float | None:
        vectors = self.lattice.get("real_space_vectors", {})
        if not isinstance(vectors, dict):
            return None

        key = f"a{axis + 1}"
        raw_vector = vectors.get(key)
        if raw_vector is None:
            return None

        vector = np.asarray(raw_vector, dtype=float)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(f"real_space_vectors['{key}'] must be a non-empty 1D vector.")

        length = float(np.linalg.norm(vector))
        if length <= 0.0:
            raise ValueError(f"real_space_vectors['{key}'] must have strictly positive norm.")
        return length

    def _fallback_lattice_constant(self, axis: int) -> float:
        constants = self.lattice.get("lattice_constants", {})
        if not isinstance(constants, dict):
            constants = {}

        per_axis_keys = {
            0: ("ax", "a", "a1_length", "a_length", "a0"),
            1: ("ay", "b", "a2_length", "b_length", "a0"),
            2: ("az", "c", "a3_length", "c_length", "a0"),
        }
        for key in per_axis_keys[axis]:
            if key in constants:
                length = float(constants[key])
                if length <= 0.0:
                    raise ValueError(f"lattice_constants['{key}'] must be strictly positive.")
                return length

        raise ValueError(
            "Could not infer the lattice constant for one active direction. "
            "Define lattice_constants or real_space_vectors in atomic units."
        )
