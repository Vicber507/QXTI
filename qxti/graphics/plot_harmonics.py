from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")


try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - environment dependent
    plt = None
    HAS_MATPLOTLIB = False


DEFAULT_HARMONIC_PLOT_CONFIG = {
    "field_current_time": {
        "enabled": True,
        "dataset_file": "current_spectrum.npz",
        "output_file": "field_current_time.png",
        "directions": ("x", "y", "z"),
        "include_total": True,
    },
    "current_spectrum": {
        "enabled": True,
        "dataset_file": "current_spectrum.npz",
        "output_file": "current_spectrum.png",
        "directions": ("x", "y", "z"),
        "positive_only": True,
        "omega_min": None,
        "omega_max": None,
        "use_harmonic_order": True,
        "max_harmonic_order": 4.0,
        "log_scale": True,
    },
    "current_circular_spectrum": {
        "enabled": True,
        "dataset_file": "current_spectrum.npz",
        "output_file": "current_circular_spectrum.png",
        "positive_only": True,
        "omega_min": None,
        "omega_max": None,
        "use_harmonic_order": True,
        "max_harmonic_order": 4.0,
        "log_scale": True,
    },
}

# Edit this dictionary directly if you want to change the plotting behavior.
HARMONIC_PLOT_CONFIG = copy.deepcopy(DEFAULT_HARMONIC_PLOT_CONFIG)


class HarmonicGraphics:
    """Plot spectral observables derived from XTP currents/polarizations."""

    @staticmethod
    def plot_field_current_time_comparison(
        time_axis: np.ndarray,
        electric_field_time: np.ndarray,
        current_time: np.ndarray,
        output_path: str | Path,
        *,
        directions: tuple[str, ...] = ("x", "y", "z"),
        include_total: bool = True,
    ) -> Path:
        pyplot = HarmonicGraphics._require_matplotlib()
        time = np.asarray(time_axis, dtype=float)
        field = np.asarray(electric_field_time, dtype=float)
        current = np.asarray(current_time, dtype=float)

        if field.ndim != 2 or field.shape[1] != 3:
            raise ValueError("electric_field_time must have shape (Nt, 3).")
        if current.ndim != 2 or current.shape[1] != 3:
            raise ValueError("current_time must have shape (Nt, 3).")
        if field.shape[0] != time.size or current.shape[0] != time.size:
            raise ValueError("time_axis must match the first dimension of electric_field_time and current_time.")

        rows = len(directions) + (1 if include_total else 0)
        figure, axes = pyplot.subplots(rows, 1, figsize=(10.0, 3.0 * rows), sharex=True, squeeze=False)
        field_colors = {"x": "#c23b22", "y": "#1f6aa5", "z": "#2c8c4a", "total": "#8f3fb0"}
        current_colors = {"x": "#ff8c69", "y": "#58a5f0", "z": "#65c18c", "total": "#c98df0"}

        for row, direction in enumerate(directions):
            idir = HarmonicGraphics._direction_axis(direction)
            axis = axes[row, 0]
            twin = axis.twinx()
            field_line = axis.plot(
                time,
                field[:, idir],
                linewidth=1.6,
                color=field_colors.get(direction, None),
                label=f"E{direction}(t)",
            )[0]
            current_line = twin.plot(
                time,
                current[:, idir],
                linewidth=1.4,
                color=current_colors.get(direction, None),
                label=f"J{direction}(t)",
            )[0]
            axis.set_ylabel(f"E{direction}(t)")
            twin.set_ylabel(f"J{direction}(t)")
            axis.set_title(f"{direction}-component")
            axis.grid(alpha=0.25)
            axis.legend([field_line, current_line], [field_line.get_label(), current_line.get_label()], loc="upper right")

        if include_total:
            axis = axes[len(directions), 0]
            twin = axis.twinx()
            field_total = np.linalg.norm(field, axis=1)
            current_total = np.linalg.norm(current, axis=1)
            field_line = axis.plot(
                time,
                field_total,
                linewidth=1.8,
                color=field_colors["total"],
                label="|E(t)|",
            )[0]
            current_line = twin.plot(
                time,
                current_total,
                linewidth=1.6,
                color=current_colors["total"],
                label="|J(t)|",
            )[0]
            axis.set_ylabel("|E(t)|")
            twin.set_ylabel("|J(t)|")
            axis.set_title("total")
            axis.grid(alpha=0.25)
            axis.legend([field_line, current_line], [field_line.get_label(), current_line.get_label()], loc="upper right")

        axes[-1, 0].set_xlabel("time (a.u.)")
        figure.suptitle("Electric field and current in time", fontsize=14)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        pyplot.close(figure)
        return output

    @staticmethod
    def plot_current_magnitude_spectrum(
        omega_axis: np.ndarray,
        current_spectrum: np.ndarray,
        output_path: str | Path,
        *,
        orders: tuple[int, ...] | None = None,
        directions: tuple[str, ...] = ("x", "y", "z"),
        positive_only: bool = True,
        omega_min: float | None = None,
        omega_max: float | None = None,
        fundamental_omega: float | None = None,
        use_harmonic_order: bool = False,
        max_harmonic_order: float | None = None,
        log_scale: bool = False,
    ) -> Path:
        pyplot = HarmonicGraphics._require_matplotlib()
        omega = np.asarray(omega_axis, dtype=float)
        spectrum = np.asarray(current_spectrum, dtype=np.complex128)

        if spectrum.ndim != 2 or spectrum.shape[1] != 3:
            raise ValueError("current_spectrum must have shape (Nomega, 3).")
        if spectrum.shape[0] != omega.size:
            raise ValueError("omega_axis and current_spectrum must share the same first dimension.")

        mask = np.ones_like(omega, dtype=bool)
        if positive_only:
            mask &= omega >= 0.0
        if omega_min is not None:
            mask &= omega >= float(omega_min)
        if omega_max is not None:
            mask &= omega <= float(omega_max)
        if not np.any(mask):
            raise ValueError("No frequency points remain after applying the requested filters.")

        figure, axis = pyplot.subplots(figsize=(9.2, 5.4))
        axis.set_facecolor("#fbfaf7")
        colors = {"x": "#c23b22", "y": "#1f6aa5", "z": "#2c8c4a"}
        x_values = omega[mask]
        xlabel = "ω (a.u.)"
        if use_harmonic_order:
            if fundamental_omega is None or fundamental_omega <= 0.0:
                raise ValueError("fundamental_omega must be positive when use_harmonic_order=True.")
            x_values = omega[mask] / float(fundamental_omega)
            xlabel = "harmonic order"
        for direction in directions:
            idir = HarmonicGraphics._direction_axis(direction)
            magnitude = np.abs(spectrum[mask, idir])
            if not np.any(magnitude > 0.0):
                continue
            axis.plot(
                x_values,
                magnitude,
                linewidth=2.0,
                color=colors.get(direction, None),
                label=f"|J_{direction}(ω)|",
            )

        axis.set_xlabel(xlabel)
        axis.set_ylabel("|J(ω)|")
        if orders:
            orders_text = ", ".join(str(order) for order in orders)
            axis.set_title(f"Current spectrum from sum of rho^(s), s = {orders_text}")
        else:
            axis.set_title("Current spectrum")
        if use_harmonic_order and max_harmonic_order is not None:
            axis.set_xlim(0.0, float(max_harmonic_order))
            tick_max = int(np.floor(float(max_harmonic_order)))
            axis.set_xticks(np.arange(0, tick_max + 1, 1, dtype=int))
            for ho in range(1, tick_max + 1):
                axis.axvline(float(ho), color="#d7d1c6", linewidth=0.9, linestyle="--", zorder=0)
        if log_scale:
            axis.set_yscale("log")
        axis.grid(alpha=0.24, color="#b9b2a6")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if axis.has_data():
            axis.legend()
        else:
            raise ValueError("Selected directions do not contain any non-zero spectral data.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        pyplot.close(figure)
        return output

    @staticmethod
    def plot_circular_current_spectrum(
        omega_axis: np.ndarray,
        current_spectrum: np.ndarray,
        output_path: str | Path,
        *,
        orders: tuple[int, ...] | None = None,
        positive_only: bool = True,
        omega_min: float | None = None,
        omega_max: float | None = None,
        fundamental_omega: float | None = None,
        use_harmonic_order: bool = False,
        max_harmonic_order: float | None = None,
        log_scale: bool = False,
    ) -> Path:
        pyplot = HarmonicGraphics._require_matplotlib()
        omega = np.asarray(omega_axis, dtype=float)
        spectrum = np.asarray(current_spectrum, dtype=np.complex128)

        if spectrum.ndim != 2 or spectrum.shape[1] != 3:
            raise ValueError("current_spectrum must have shape (Nomega, 3).")
        if spectrum.shape[0] != omega.size:
            raise ValueError("omega_axis and current_spectrum must share the same first dimension.")

        mask = np.ones_like(omega, dtype=bool)
        if positive_only:
            mask &= omega >= 0.0
        if omega_min is not None:
            mask &= omega >= float(omega_min)
        if omega_max is not None:
            mask &= omega <= float(omega_max)
        if not np.any(mask):
            raise ValueError("No frequency points remain after applying the requested filters.")

        jx = spectrum[:, 0]
        jy = spectrum[:, 1]
        current_right = (jx - 1.0j * jy) / np.sqrt(2.0)
        current_left = (jx + 1.0j * jy) / np.sqrt(2.0)

        figure, axis = pyplot.subplots(figsize=(9.2, 5.4))
        axis.set_facecolor("#fbfaf7")
        right_mag = np.abs(current_right[mask])
        left_mag = np.abs(current_left[mask])
        x_values = omega[mask]
        xlabel = "ω (a.u.)"
        if use_harmonic_order:
            if fundamental_omega is None or fundamental_omega <= 0.0:
                raise ValueError("fundamental_omega must be positive when use_harmonic_order=True.")
            x_values = omega[mask] / float(fundamental_omega)
            xlabel = "harmonic order"
        if np.any(right_mag > 0.0):
            axis.plot(
                x_values,
                right_mag,
                linewidth=2.1,
                color="#2f6db2",
                label="|J_R(ω)|",
            )
        if np.any(left_mag > 0.0):
            axis.plot(
                x_values,
                left_mag,
                linewidth=2.1,
                color="#c83a3a",
                label="|J_L(ω)|",
            )
        if not axis.has_data():
            raise ValueError("The circularly polarized currents are zero for the selected range.")

        axis.set_xlabel(xlabel)
        axis.set_ylabel("|J_{R/L}(ω)|")
        if orders:
            orders_text = ", ".join(str(order) for order in orders)
            axis.set_title(f"Circular current spectrum from sum of rho^(s), s = {orders_text}")
        else:
            axis.set_title("Circular current spectrum")
        if use_harmonic_order and max_harmonic_order is not None:
            axis.set_xlim(0.0, float(max_harmonic_order))
            tick_max = int(np.floor(float(max_harmonic_order)))
            axis.set_xticks(np.arange(0, tick_max + 1, 1, dtype=int))
            for ho in range(1, tick_max + 1):
                axis.axvline(float(ho), color="#d7d1c6", linewidth=0.9, linestyle="--", zorder=0)
        if log_scale:
            axis.set_yscale("log")
        axis.grid(alpha=0.24, color="#b9b2a6")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        pyplot.close(figure)
        return output

    @staticmethod
    def plot_bz_mask(
        kx_grid: np.ndarray,
        ky_grid: np.ndarray,
        integration_region: np.ndarray,
        mask_weights: np.ndarray,
        output_path: str | Path,
        *,
        cmap: str = "inferno",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        pyplot = HarmonicGraphics._require_matplotlib()
        from matplotlib.patches import Circle

        kx = np.asarray(kx_grid, dtype=float)
        ky = np.asarray(ky_grid, dtype=float)
        region = np.asarray(integration_region, dtype=float)
        weights = np.asarray(mask_weights, dtype=float)

        if kx.shape != ky.shape or kx.shape != region.shape or kx.shape != weights.shape:
            raise ValueError("kx_grid, ky_grid, integration_region and mask_weights must share the same shape.")
        if kx.ndim != 2:
            raise ValueError("BZ mask plotting currently expects 2D grids.")

        figure, axes = pyplot.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
        vmin, vmax = 0.0, 1.0
        extent = (
            float(np.min(kx)),
            float(np.max(kx)),
            float(np.min(ky)),
            float(np.max(ky)),
        )
        if np.isclose(extent[0], extent[1]):
            pad = 0.5 if np.isclose(extent[0], 0.0) else 0.05 * max(1.0, abs(extent[0]))
            extent = (extent[0] - pad, extent[1] + pad, extent[2], extent[3])
        if np.isclose(extent[2], extent[3]):
            pad = 0.5 if np.isclose(extent[2], 0.0) else 0.05 * max(1.0, abs(extent[2]))
            extent = (extent[0], extent[1], extent[2] - pad, extent[3] + pad)

        panels = (
            (axes[0], region, "full integration region"),
            (axes[1], weights, "BZ mask weights"),
        )
        image = None
        for axis, values, title in panels:
            image = axis.imshow(
                values.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axis.set_title(title)
            axis.set_xlabel(r"$k_x$ (a.u.)")
            axis.set_ylabel(r"$k_y$ (a.u.)")
            axis.grid(alpha=0.15, color="#d9d4c8", linewidth=0.8)

        if metadata and bool(metadata.get("enabled", False)):
            radius = float(metadata.get("radius", 0.0))
            if radius > 0.0:
                for axis in axes:
                    axis.add_patch(
                        Circle(
                            (0.0, 0.0),
                            radius,
                            fill=False,
                            linewidth=1.6,
                            linestyle="--",
                            edgecolor="white",
                            alpha=0.95,
                        )
                    )

        if metadata and bool(metadata.get("enabled", False)):
            radius_percent = float(metadata.get("radius_percent", 100.0))
            sigma = float(metadata.get("sigma", 0.0))
            axes[1].text(
                0.03,
                0.97,
                f"radius = {radius_percent:.1f}%\nsigma = {sigma:.4f} a.u.",
                transform=axes[1].transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="white",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
            )
        elif metadata:
            axes[1].text(
                0.03,
                0.97,
                "mask disabled",
                transform=axes[1].transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="white",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
            )

        colorbar = figure.colorbar(image, ax=axes, shrink=0.92, pad=0.02)
        colorbar.set_label("weight")
        colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        pyplot.close(figure)
        return output

    @staticmethod
    def _direction_axis(direction: str) -> int:
        mapping = {"x": 0, "y": 1, "z": 2}
        try:
            return mapping[direction.strip().lower()]
        except KeyError as exc:
            raise ValueError("direction must be one of 'x', 'y', or 'z'.") from exc

    @staticmethod
    def _require_matplotlib() -> Any:
        if not HAS_MATPLOTLIB:
            raise RuntimeError(
                "matplotlib is not installed in this environment. "
                "Install it to generate harmonic plots."
            )
        return plt


def resolve_harmonic_plot_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(HARMONIC_PLOT_CONFIG)
    if overrides:
        _deep_update(config, overrides)
    return config


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate current-spectrum plots from saved data using the plot dictionary in plot_harmonics.py."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="inputParams.cfg",
        help="Path to the configuration file. Defaults to inputParams.cfg.",
    )
    return parser


def _main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from qxti.graphics.graphics import plot_harmonic_graphics_from_saved_data

    parser = _build_parser()
    args = parser.parse_args()
    outputs = plot_harmonic_graphics_from_saved_data(
        args.config,
        plot_config=resolve_harmonic_plot_config(),
    )
    print(f"Generated {len(outputs)} harmonic graphics from saved data in {args.config}:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
