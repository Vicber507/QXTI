from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np


HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
else:
    plt = None


class HamiltonianGraphics:
    """Plot-only helpers for intrinsic tight-binding Hamiltonian properties."""

    @staticmethod
    def plot_band_structure_2d(data: dict[str, Any], output_path: Path) -> Path:
        figure, axis = HamiltonianGraphics._new_figure()
        bands = np.asarray(data["bands"], dtype=float)
        path_coordinate = np.asarray(data["path_coordinate"], dtype=float)

        for band_index in range(bands.shape[1]):
            axis.plot(path_coordinate, bands[:, band_index], linewidth=1.8, label=f"band {band_index}")

        axis.set_title(f"Band structure ({data['path_type']})")
        axis.set_xlabel("path coordinate")
        axis.set_ylabel("energy")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8)
        return HamiltonianGraphics._save(figure, output_path)

    @staticmethod
    def plot_band_surface_3d(
        data: dict[str, Any],
        output_path: Path,
        *,
        style: str = "surface",
    ) -> Path:
        style_key = style.strip().lower()
        axis1_grid = np.asarray(data["axis1_grid"], dtype=float)
        axis2_grid = np.asarray(data["axis2_grid"], dtype=float)
        energy_surfaces = np.asarray(data["energy_surfaces"], dtype=float)
        band_indices = tuple(int(index) for index in data["band_indices"])
        label_x, label_y = data["axis_labels"]
        figure, axes = HamiltonianGraphics._band_axes(
            num_bands=len(band_indices),
            projection="3d" if style_key == "surface" else None,
        )

        for axis, band_id, energy_surface in zip(axes, band_indices, energy_surfaces, strict=False):
            if style_key == "surface":
                surface = axis.plot_surface(axis1_grid, axis2_grid, energy_surface, cmap="viridis")
                axis.set_zlabel("energy")
                figure.colorbar(surface, ax=axis, shrink=0.78, pad=0.08)
            elif style_key == "contour":
                contour = axis.contourf(axis1_grid, axis2_grid, energy_surface, levels=40, cmap="viridis")
                figure.colorbar(contour, ax=axis, shrink=0.85)
            elif style_key == "colormap":
                image = axis.pcolormesh(axis1_grid, axis2_grid, energy_surface, shading="auto", cmap="viridis")
                figure.colorbar(image, ax=axis, shrink=0.85)
            else:
                raise ValueError("style must be 'surface', 'contour', or 'colormap'.")

            axis.set_title(f"Band {band_id} on {data['plane']}")
            axis.set_xlabel(label_x)
            axis.set_ylabel(label_y)
        return HamiltonianGraphics._save(figure, output_path)

    @staticmethod
    def plot_velocity_2d(data: dict[str, Any], output_path: Path) -> Path:
        figure, axis = HamiltonianGraphics._new_figure()
        path_coordinate = np.asarray(data["path_coordinate"], dtype=float)

        for component in ("vx", "vy", "vz"):
            component_values = np.asarray(data[component], dtype=float)
            for band_index in range(component_values.shape[1]):
                axis.plot(
                    path_coordinate,
                    component_values[:, band_index],
                    linewidth=1.5,
                    label=f"{component} band {band_index}",
                )

        axis.set_title(f"Band velocities ({data['path_type']})")
        axis.set_xlabel("path coordinate")
        axis.set_ylabel("velocity expectation value")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=7, ncol=2)
        return HamiltonianGraphics._save(figure, output_path)

    @staticmethod
    def plot_velocity_field_3d(
        data: dict[str, Any],
        output_path: Path,
        *,
        stride: int = 4,
    ) -> Path:
        stride = max(int(stride), 1)
        axis1_grid = np.asarray(data["axis1_grid"], dtype=float)
        axis2_grid = np.asarray(data["axis2_grid"], dtype=float)
        component_1 = np.asarray(data["plane_component_1"], dtype=float)
        component_2 = np.asarray(data["plane_component_2"], dtype=float)
        magnitude = np.asarray(data["magnitude"], dtype=float)
        band_indices = tuple(int(index) for index in data["band_indices"])
        label_x, label_y = data["axis_labels"]
        figure, axes = HamiltonianGraphics._band_axes(num_bands=len(band_indices))

        for axis, band_id, band_component_1, band_component_2, band_magnitude in zip(
            axes,
            band_indices,
            component_1,
            component_2,
            magnitude,
            strict=False,
        ):
            image = axis.pcolormesh(
                axis1_grid,
                axis2_grid,
                band_magnitude,
                shading="auto",
                cmap="magma",
                alpha=0.70,
            )
            axis.quiver(
                axis1_grid[::stride, ::stride],
                axis2_grid[::stride, ::stride],
                band_component_1[::stride, ::stride],
                band_component_2[::stride, ::stride],
                color="white",
                pivot="mid",
                linewidth=0.6,
            )
            axis.set_title(f"Velocity field for band {band_id} on {data['plane']}")
            axis.set_xlabel(label_x)
            axis.set_ylabel(label_y)
            figure.colorbar(image, ax=axis, label="|v|", shrink=0.85)
        return HamiltonianGraphics._save(figure, output_path)

    @staticmethod
    def plot_velocity_magnitude(data: dict[str, Any], output_path: Path) -> Path:
        axis1_grid = np.asarray(data["axis1_grid"], dtype=float)
        axis2_grid = np.asarray(data["axis2_grid"], dtype=float)
        magnitude = np.asarray(data["magnitude"], dtype=float)
        band_indices = tuple(int(index) for index in data["band_indices"])
        label_x, label_y = data["axis_labels"]
        figure, axes = HamiltonianGraphics._band_axes(num_bands=len(band_indices))

        for axis, band_id, band_magnitude in zip(axes, band_indices, magnitude, strict=False):
            image = axis.pcolormesh(axis1_grid, axis2_grid, band_magnitude, shading="auto", cmap="magma")
            axis.set_title(f"|v| for band {band_id} on {data['plane']}")
            axis.set_xlabel(label_x)
            axis.set_ylabel(label_y)
            figure.colorbar(image, ax=axis, label="|v|", shrink=0.85)
        return HamiltonianGraphics._save(figure, output_path)

    @staticmethod
    def _new_figure() -> tuple[Any, Any]:
        pyplot = HamiltonianGraphics._require_matplotlib()
        figure, axis = pyplot.subplots(figsize=(8, 5))
        return figure, axis

    @staticmethod
    def _band_axes(num_bands: int, projection: str | None = None) -> tuple[Any, list[Any]]:
        pyplot = HamiltonianGraphics._require_matplotlib()
        if num_bands <= 0:
            raise ValueError("num_bands must be strictly positive.")

        ncols = min(2, num_bands)
        nrows = int(math.ceil(num_bands / ncols))
        figure = pyplot.figure(figsize=(7.0 * ncols, 5.4 * nrows))
        axes: list[Any] = []

        for index in range(num_bands):
            subplot_index = index + 1
            if projection is None:
                axis = figure.add_subplot(nrows, ncols, subplot_index)
            else:
                axis = figure.add_subplot(nrows, ncols, subplot_index, projection=projection)
            axes.append(axis)
        return figure, axes

    @staticmethod
    def _save(figure: Any, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        return output_path

    @staticmethod
    def _require_matplotlib() -> Any:
        if not HAS_MATPLOTLIB:
            raise RuntimeError(
                "matplotlib is not installed in this environment. "
                "Install it to generate Hamiltonian plots."
            )
        return plt
