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

        figure, axis = pyplot.subplots(figsize=(8.5, 5.0))
        colors = {"x": "#c23b22", "y": "#1f6aa5", "z": "#2c8c4a"}
        for direction in directions:
            idir = HarmonicGraphics._direction_axis(direction)
            magnitude = np.abs(spectrum[mask, idir])
            if not np.any(magnitude > 0.0):
                continue
            axis.plot(
                omega[mask],
                magnitude,
                linewidth=1.8,
                color=colors.get(direction, None),
                label=f"|J_{direction}(ω)|",
            )

        axis.set_xlabel("ω (a.u.)")
        axis.set_ylabel("|J(ω)|")
        if orders:
            orders_text = ", ".join(str(order) for order in orders)
            axis.set_title(f"Current spectrum from sum of rho^(s), s = {orders_text}")
        else:
            axis.set_title("Current spectrum")
        if log_scale:
            axis.set_yscale("log")
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
