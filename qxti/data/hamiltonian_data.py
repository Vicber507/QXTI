from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qxti.physics import Hamiltonian


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def degeneracy_resolved_diagonal(
    eigenvalues: FloatArray,
    operator_band_basis: ComplexArray,
    *,
    rel_tol: float = 1.0e-6,
) -> FloatArray:
    r"""Well-defined band-diagonal of an operator, robust to degeneracies.

    The naive band velocity :math:`\langle n|O|n\rangle` is ill-defined wherever
    bands are degenerate: ``eigh`` returns an arbitrary basis inside a degenerate
    subspace, so the diagonal jumps around (numerical noise). The physically
    meaningful values are the eigenvalues of the operator RESTRICTED to each
    degenerate subspace (degenerate perturbation theory / Hellmann-Feynman); those
    are gauge invariant and smooth.

    ``eigenvalues`` must be sorted ascending (as returned by ``eigh``).
    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    op = np.asarray(operator_band_basis, dtype=np.complex128)
    nb = ev.size
    scale = max(float(ev[-1] - ev[0]), 1.0) if nb else 1.0
    tol = float(rel_tol) * scale
    out = np.empty(nb, dtype=np.float64)
    i = 0
    while i < nb:
        j = i + 1
        while j < nb and (ev[j] - ev[j - 1]) <= tol:
            j += 1
        if j - i == 1:
            out[i] = float(np.real(op[i, i]))
        else:  # diagonalize O within the degenerate block -> unambiguous velocities
            block = op[i:j, i:j]
            block = 0.5 * (block + np.conj(block.T))
            out[i:j] = np.sort(np.linalg.eigvalsh(block))
        i = j
    return out


@dataclass(slots=True)
class HamiltonianData:
    """Numerical data products derived from one equilibrium Hamiltonian."""

    hamiltonian: Hamiltonian

    _VALID_PATH_TYPES = {
        "kx",
        "ky",
        "kz",
        "diagonal_kx_ky",
        "diagonal_kx_kz",
        "diagonal_ky_kz",
        "manual",
    }
    _VALID_PLANES = {"kx_ky", "kx_kz", "ky_kz"}

    def band_structure_2d_data(
        self,
        *,
        path_type: str = "kx",
        k_min: float = -0.1,
        k_max: float = 0.1,
        num_points: int = 201,
        fixed_kx: float = 0.0,
        fixed_ky: float = 0.0,
        fixed_kz: float = 0.0,
        manual_path: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Return 1D band data for direct validation plots."""

        path_coordinate, k_points = self._build_path(
            path_type=path_type,
            k_min=k_min,
            k_max=k_max,
            num_points=num_points,
            fixed_kx=fixed_kx,
            fixed_ky=fixed_ky,
            fixed_kz=fixed_kz,
            manual_path=manual_path,
        )

        bands = np.empty((len(k_points), self.hamiltonian.basis_size), dtype=float)
        for index, (kx, ky, kz) in enumerate(k_points):
            bands[index], _ = self.hamiltonian.diagonalize(kx, ky, kz)

        return {
            "path_type": path_type,
            "path_coordinate": path_coordinate,
            "k_points": k_points,
            "bands": bands,
            "basis_size": self.hamiltonian.basis_size,
        }

    def band_surface_3d_data(
        self,
        *,
        plane: str = "kx_ky",
        band_index: int | None = None,
        band_indices: tuple[int, ...] | list[int] | None = None,
        k1_min: float = -0.1,
        k1_max: float = 0.1,
        k2_min: float = -0.1,
        k2_max: float = 0.1,
        num_points_1: int = 121,
        num_points_2: int = 121,
        fixed_kx: float = 0.0,
        fixed_ky: float = 0.0,
        fixed_kz: float = 0.0,
    ) -> dict[str, Any]:
        """Return one or more band energy surfaces over a reciprocal-space plane."""

        resolved_band_indices = self._resolve_band_indices(
            band_index=band_index,
            band_indices=band_indices,
        )
        axis_labels = self._validate_plane(plane)

        axis1_values = self._build_uniform_axis(k1_min, k1_max, num_points_1, name="num_points_1")
        axis2_values = self._build_uniform_axis(k2_min, k2_max, num_points_2, name="num_points_2")
        axis1_grid, axis2_grid = np.meshgrid(axis1_values, axis2_values, indexing="xy")
        energy_surfaces = np.empty(
            (len(resolved_band_indices), *axis1_grid.shape),
            dtype=float,
        )

        for row in range(axis1_grid.shape[0]):
            for col in range(axis1_grid.shape[1]):
                kx, ky, kz = self._plane_point(
                    plane=plane,
                    axis1=float(axis1_grid[row, col]),
                    axis2=float(axis2_grid[row, col]),
                    fixed_kx=fixed_kx,
                    fixed_ky=fixed_ky,
                    fixed_kz=fixed_kz,
                )
                eigenvalues, _ = self.hamiltonian.diagonalize(kx, ky, kz)
                for band_position, band_id in enumerate(resolved_band_indices):
                    energy_surfaces[band_position, row, col] = float(eigenvalues[band_id])

        return {
            "plane": plane,
            "axis_labels": axis_labels,
            "axis1_grid": axis1_grid,
            "axis2_grid": axis2_grid,
            "energy_surfaces": energy_surfaces,
            "band_indices": resolved_band_indices,
        }

    def velocity_2d_data(
        self,
        *,
        path_type: str = "kx",
        k_min: float = -0.1,
        k_max: float = 0.1,
        num_points: int = 201,
        fixed_kx: float = 0.0,
        fixed_ky: float = 0.0,
        fixed_kz: float = 0.0,
        manual_path: ArrayLike | None = None,
    ) -> dict[str, Any]:
        """Return band velocities along one direct reciprocal-space path."""

        band_data = self.band_structure_2d_data(
            path_type=path_type,
            k_min=k_min,
            k_max=k_max,
            num_points=num_points,
            fixed_kx=fixed_kx,
            fixed_ky=fixed_ky,
            fixed_kz=fixed_kz,
            manual_path=manual_path,
        )

        velocities = {
            "vx": np.empty_like(band_data["bands"]),
            "vy": np.empty_like(band_data["bands"]),
            "vz": np.empty_like(band_data["bands"]),
        }

        for index, (kx, ky, kz) in enumerate(band_data["k_points"]):
            eigenvalues, eigenvectors = self.hamiltonian.diagonalize(kx, ky, kz)
            for label, direction in (("vx", "x"), ("vy", "y"), ("vz", "z")):
                operator = self._velocity_operator_or_zero(kx, ky, kz, direction)
                band_basis = eigenvectors.conj().T @ operator @ eigenvectors
                # Degeneracy-robust: eigenvalues of v within each degenerate block,
                # instead of the gauge-ambiguous naive diagonal.
                velocities[label][index] = degeneracy_resolved_diagonal(
                    eigenvalues, band_basis
                )

        return {
            **band_data,
            "vx": velocities["vx"],
            "vy": velocities["vy"],
            "vz": velocities["vz"],
            "active_velocity_components": self._active_velocity_components(),
        }

    def velocity_field_3d_data(
        self,
        *,
        plane: str = "kx_ky",
        band_index: int | None = None,
        band_indices: tuple[int, ...] | list[int] | None = None,
        k1_min: float = -0.1,
        k1_max: float = 0.1,
        k2_min: float = -0.1,
        k2_max: float = 0.1,
        num_points_1: int = 61,
        num_points_2: int = 61,
        fixed_kx: float = 0.0,
        fixed_ky: float = 0.0,
        fixed_kz: float = 0.0,
    ) -> dict[str, Any]:
        """Return a band-resolved velocity field on one reciprocal-space plane."""

        resolved_band_indices = self._resolve_band_indices(
            band_index=band_index,
            band_indices=band_indices,
        )
        axis_labels = self._validate_plane(plane)

        axis1_values = self._build_uniform_axis(k1_min, k1_max, num_points_1, name="num_points_1")
        axis2_values = self._build_uniform_axis(k2_min, k2_max, num_points_2, name="num_points_2")
        axis1_grid, axis2_grid = np.meshgrid(axis1_values, axis2_values, indexing="xy")

        field_shape = (len(resolved_band_indices), *axis1_grid.shape)
        vx = np.empty(field_shape, dtype=float)
        vy = np.empty(field_shape, dtype=float)
        vz = np.empty(field_shape, dtype=float)
        component_1 = np.empty(field_shape, dtype=float)
        component_2 = np.empty(field_shape, dtype=float)

        direction_1, direction_2 = axis_labels
        direction_map = {
            "kx": "vx",
            "ky": "vy",
            "kz": "vz",
        }

        for row in range(axis1_grid.shape[0]):
            for col in range(axis1_grid.shape[1]):
                kx, ky, kz = self._plane_point(
                    plane=plane,
                    axis1=float(axis1_grid[row, col]),
                    axis2=float(axis2_grid[row, col]),
                    fixed_kx=fixed_kx,
                    fixed_ky=fixed_ky,
                    fixed_kz=fixed_kz,
                )
                for band_position, band_id in enumerate(resolved_band_indices):
                    full_velocity = self._band_velocity_components(kx, ky, kz, band_id)
                    vx[band_position, row, col] = full_velocity["vx"]
                    vy[band_position, row, col] = full_velocity["vy"]
                    vz[band_position, row, col] = full_velocity["vz"]
                    component_1[band_position, row, col] = full_velocity[direction_map[direction_1]]
                    component_2[band_position, row, col] = full_velocity[direction_map[direction_2]]

        magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
        return {
            "plane": plane,
            "axis_labels": axis_labels,
            "axis1_grid": axis1_grid,
            "axis2_grid": axis2_grid,
            "band_indices": resolved_band_indices,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "plane_component_1": component_1,
            "plane_component_2": component_2,
            "magnitude": magnitude,
        }

    def velocity_magnitude_data(self, **kwargs: Any) -> dict[str, Any]:
        """Return plane-resolved maps of ``|v_n(k)|`` for one or more bands."""

        velocity_field = self.velocity_field_3d_data(**kwargs)
        return {
            key: value
            for key, value in velocity_field.items()
            if key in {"plane", "axis_labels", "axis1_grid", "axis2_grid", "band_indices", "magnitude"}
        }

    def _build_path(
        self,
        *,
        path_type: str,
        k_min: float,
        k_max: float,
        num_points: int,
        fixed_kx: float,
        fixed_ky: float,
        fixed_kz: float,
        manual_path: ArrayLike | None,
    ) -> tuple[FloatArray, FloatArray]:
        path_key = path_type.strip().lower()
        if path_key not in self._VALID_PATH_TYPES:
            raise ValueError(
                f"path_type must be one of {sorted(self._VALID_PATH_TYPES)}, got {path_type!r}."
            )

        if path_key == "manual":
            if manual_path is None:
                raise ValueError("manual_path must be provided when path_type='manual'.")
            k_points = np.asarray(manual_path, dtype=float)
            if k_points.ndim != 2 or k_points.shape[1] != 3:
                raise ValueError("manual_path must have shape (N, 3).")
            deltas = np.diff(k_points, axis=0)
            distances = np.linalg.norm(deltas, axis=1)
            path_coordinate = np.concatenate([[0.0], np.cumsum(distances)])
            return np.asarray(path_coordinate, dtype=np.float64), np.asarray(k_points, dtype=np.float64)

        path_coordinate = self._build_uniform_axis(k_min, k_max, num_points, name="num_points")
        k_points = np.column_stack(
            [
                np.full_like(path_coordinate, fill_value=float(fixed_kx)),
                np.full_like(path_coordinate, fill_value=float(fixed_ky)),
                np.full_like(path_coordinate, fill_value=float(fixed_kz)),
            ]
        )

        if path_key == "kx":
            self._require_direction("x")
            k_points[:, 0] = path_coordinate
        elif path_key == "ky":
            self._require_direction("y")
            k_points[:, 1] = path_coordinate
        elif path_key == "kz":
            self._require_direction("z")
            k_points[:, 2] = path_coordinate
        elif path_key == "diagonal_kx_ky":
            self._require_direction("y")
            k_points[:, 0] = path_coordinate
            k_points[:, 1] = path_coordinate
        elif path_key == "diagonal_kx_kz":
            self._require_direction("z")
            k_points[:, 0] = path_coordinate
            k_points[:, 2] = path_coordinate
        elif path_key == "diagonal_ky_kz":
            self._require_direction("z")
            k_points[:, 1] = path_coordinate
            k_points[:, 2] = path_coordinate

        return path_coordinate, np.asarray(k_points, dtype=np.float64)

    def _validate_plane(self, plane: str) -> tuple[str, str]:
        plane_key = plane.strip().lower()
        if plane_key not in self._VALID_PLANES:
            raise ValueError(
                f"plane must be one of {sorted(self._VALID_PLANES)}, got {plane!r}."
            )

        if plane_key == "kx_ky":
            self._require_direction("y")
            return ("kx", "ky")
        if plane_key == "kx_kz":
            self._require_direction("z")
            return ("kx", "kz")

        self._require_direction("z")
        return ("ky", "kz")

    def _plane_point(
        self,
        *,
        plane: str,
        axis1: float,
        axis2: float,
        fixed_kx: float,
        fixed_ky: float,
        fixed_kz: float,
    ) -> tuple[float, float, float]:
        plane_key = plane.strip().lower()
        if plane_key == "kx_ky":
            return float(axis1), float(axis2), float(fixed_kz)
        if plane_key == "kx_kz":
            return float(axis1), float(fixed_ky), float(axis2)
        if plane_key == "ky_kz":
            return float(fixed_kx), float(axis1), float(axis2)
        raise ValueError(f"Unsupported plane {plane!r}.")

    def _band_velocity_components(
        self,
        kx: float,
        ky: float,
        kz: float,
        band_index: int,
    ) -> dict[str, float]:
        eigenvalues, eigenvectors = self.hamiltonian.diagonalize(kx, ky, kz)
        values: dict[str, float] = {}
        for label, direction in (("vx", "x"), ("vy", "y"), ("vz", "z")):
            operator = self._velocity_operator_or_zero(kx, ky, kz, direction)
            projected = eigenvectors.conj().T @ operator @ eigenvectors
            resolved = degeneracy_resolved_diagonal(eigenvalues, projected)
            values[label] = float(resolved[band_index])
        return values

    def _velocity_operator_or_zero(
        self,
        kx: float,
        ky: float,
        kz: float,
        direction: str,
    ) -> ComplexArray:
        axis_map = {"x": 1, "y": 2, "z": 3}
        if self.hamiltonian.dimension < axis_map[direction]:
            return np.zeros((self.hamiltonian.basis_size, self.hamiltonian.basis_size), dtype=complex)
        return self.hamiltonian.velocity_operator(kx, ky, kz, direction)

    def _active_velocity_components(self) -> tuple[str, ...]:
        labels = ("vx", "vy", "vz")
        return labels[: self.hamiltonian.dimension]

    def _require_direction(self, direction: str) -> None:
        _ = self.hamiltonian._direction_axis(direction)

    def _validate_band_index(self, band_index: int) -> None:
        if not 0 <= int(band_index) < self.hamiltonian.basis_size:
            raise ValueError(
                f"band_index must be between 0 and {self.hamiltonian.basis_size - 1}."
            )

    def _resolve_band_indices(
        self,
        *,
        band_index: int | None,
        band_indices: tuple[int, ...] | list[int] | None,
    ) -> tuple[int, ...]:
        if band_indices is not None:
            resolved = tuple(int(index) for index in band_indices)
            if not resolved:
                raise ValueError("band_indices cannot be empty.")
        elif band_index is not None:
            resolved = (int(band_index),)
        else:
            resolved = tuple(range(self.hamiltonian.basis_size))

        for index in resolved:
            self._validate_band_index(index)
        return resolved

    @staticmethod
    def _build_uniform_axis(k_min: float, k_max: float, num_points: int, *, name: str) -> FloatArray:
        if num_points <= 1:
            raise ValueError(f"{name} must be greater than one.")
        return np.linspace(float(k_min), float(k_max), int(num_points), dtype=np.float64)
