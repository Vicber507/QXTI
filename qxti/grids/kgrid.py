from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_1d_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one point.")
    return np.asarray(array, dtype=np.float64)


@dataclass(slots=True)
class KGrid:
    """Uniform or user-provided k-space grid compatible with Hamiltonian scans.

    When ``shifted`` is ``True`` the grid is a symmetric Monkhorst-Pack grid:
    the points are offset by half a step so they never land on the Brillouin
    zone edges or center.  This avoids placing a sample exactly on a band
    degeneracy (e.g. the Dirac points of graphene), which would otherwise
    inject a singular Berry-connection contribution and spuriously break
    even-harmonic selection rules.  A shifted grid represents a *periodic*
    sampling, so Brillouin-zone integrals use uniform (midpoint) weights
    instead of the trapezoidal/Simpson rule used for edge-inclusive grids.
    """

    kx_values: ArrayLike = field(default_factory=lambda: np.array([0.0], dtype=float))
    ky_values: ArrayLike = field(default_factory=lambda: np.array([0.0], dtype=float))
    kz_values: ArrayLike = field(default_factory=lambda: np.array([0.0], dtype=float))
    dimension: int = 3
    shifted: bool = False

    def __post_init__(self) -> None:
        if self.dimension not in {1, 2, 3}:
            raise ValueError("dimension must be one of 1, 2, or 3.")

        self.kx_values = _as_1d_float_array(self.kx_values, name="kx_values")
        self.ky_values = _as_1d_float_array(self.ky_values, name="ky_values")
        self.kz_values = _as_1d_float_array(self.kz_values, name="kz_values")

        if self.dimension < 2 and self.ky_values.size != 1:
            raise ValueError("ky_values must contain exactly one point for dimension 1.")
        if self.dimension < 3 and self.kz_values.size != 1:
            raise ValueError("kz_values must contain exactly one point for dimension 1 or 2.")

    @classmethod
    def uniform(
        cls,
        *,
        dimension: int,
        k_min: float = -0.1,
        k_max: float = 0.1,
        num_points: int = 41,
        num_points_kx: int | None = None,
        num_points_ky: int | None = None,
        num_points_kz: int | None = None,
    ) -> KGrid:
        """Build one uniform grid with independent resolutions per active axis."""

        if num_points <= 0:
            raise ValueError("num_points must be strictly positive.")

        if dimension not in {1, 2, 3}:
            raise ValueError("dimension must be one of 1, 2, or 3.")

        nkx = num_points if num_points_kx is None else int(num_points_kx)
        nky = num_points if num_points_ky is None else int(num_points_ky)
        nkz = num_points if num_points_kz is None else int(num_points_kz)

        if nkx <= 0 or nky <= 0 or nkz <= 0:
            raise ValueError("All k-grid point counts must be strictly positive.")

        kx_values = np.linspace(k_min, k_max, nkx, dtype=float)
        ky_values = np.linspace(k_min, k_max, nky, dtype=float) if dimension >= 2 else np.array([0.0])
        kz_values = np.linspace(k_min, k_max, nkz, dtype=float) if dimension >= 3 else np.array([0.0])
        return cls(
            kx_values=kx_values,
            ky_values=ky_values,
            kz_values=kz_values,
            dimension=dimension,
        )

    @staticmethod
    def shifted_axis(lower: float, upper: float, num_points: int) -> FloatArray:
        """Return a symmetric Monkhorst-Pack axis on ``[lower, upper]``.

        The points are ``k_i = c + (2i - N + 1) / N * L`` where ``c`` is the
        interval center and ``L`` its half-width.  They are symmetric under
        ``k -> -k`` about the center, uniformly spaced by ``2L/N``, and never
        touch the endpoints ``lower``/``upper`` (so they never land on a BZ
        edge).  For ``N`` odd this also skips the exact center.
        """
        n = int(num_points)
        if n <= 0:
            raise ValueError("num_points must be strictly positive.")
        center = 0.5 * (float(lower) + float(upper))
        half_width = 0.5 * (float(upper) - float(lower))
        i = np.arange(n, dtype=float)
        return np.asarray(center + (2.0 * i - n + 1.0) / n * half_width, dtype=np.float64)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.kx_values), len(self.ky_values), len(self.kz_values))

    @property
    def total_points(self) -> int:
        return int(len(self.kx_values) * len(self.ky_values) * len(self.kz_values))

    def mesh(self, *, indexing: str = "xy") -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the full k-space mesh for plotting or batched evaluation."""

        return np.meshgrid(
            self.kx_values,
            self.ky_values,
            self.kz_values,
            indexing=indexing,
        )

    def points(self) -> FloatArray:
        """Return the flattened list of k-points with shape ``(Nk, 3)``."""

        mesh_kx, mesh_ky, mesh_kz = self.mesh(indexing="ij")
        stacked = np.stack(
            [mesh_kx.reshape(-1), mesh_ky.reshape(-1), mesh_kz.reshape(-1)],
            axis=1,
        )
        return np.asarray(stacked, dtype=np.float64)

    def get_all_points(self) -> FloatArray:
        """Backward-compatible alias for :meth:`points`."""

        return self.points()

    def summary(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "shape": self.shape,
            "total_points": self.total_points,
            "kx_range": (float(self.kx_values[0]), float(self.kx_values[-1])),
            "ky_range": (float(self.ky_values[0]), float(self.ky_values[-1])),
            "kz_range": (float(self.kz_values[0]), float(self.kz_values[-1])),
        }
