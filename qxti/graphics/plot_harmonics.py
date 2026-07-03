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
    import matplotlib.patheffects as pe
    from matplotlib import pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.ticker import LogLocator, NullFormatter

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - environment dependent
    pe = None
    plt = None
    mcolors = None
    LogLocator = None
    NullFormatter = None
    HAS_MATPLOTLIB = False


AU_TO_EV = 27.211386245988
DEFAULT_MAX_HARMONIC_ORDER = 3.5


DEFAULT_HARMONIC_PLOT_CONFIG = {
    "field_current_time": {
        "enabled": True,
        "dataset_file": "current_spectrum.npz",
        "output_file": "field_current_time.png",
        "directions": ("x", "y", "z"),
        "include_total": False,
        "combine_planar": True,
    },
    "current_total_spectrum": {
        "enabled": True,
        "dataset_file": "current_spectrum.npz",
        "output_file": "current_total_spectrum.png",
        "positive_only": True,
        "omega_min": None,
        "omega_max": None,
        "use_harmonic_order": True,
        "max_harmonic_order": DEFAULT_MAX_HARMONIC_ORDER,
        "log_scale": True,
    },
    "current_components_spectrum": {
        "enabled": True,
        "dataset_file": "current_spectrum.npz",
        "output_file": "current_components_spectrum.png",
        "directions": ("x", "y", "z"),
        "positive_only": True,
        "omega_min": None,
        "omega_max": None,
        "use_harmonic_order": True,
        "max_harmonic_order": DEFAULT_MAX_HARMONIC_ORDER,
        "log_scale": True,
    },
    "current_inter_intra_spectrum": {
        "enabled": True,
        "dataset_file": "current_spectrum.npz",
        "output_file": "current_inter_intra_spectrum.png",
        "positive_only": True,
        "omega_min": None,
        "omega_max": None,
        "use_harmonic_order": True,
        "max_harmonic_order": DEFAULT_MAX_HARMONIC_ORDER,
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
        "max_harmonic_order": DEFAULT_MAX_HARMONIC_ORDER,
        "log_scale": True,
    },
    "current_overview_spectrum": {
        "enabled": True,
        "dataset_file": "current_spectrum.npz",
        "output_file": "current_overview_spectrum.png",
        "directions": ("x", "y", "z"),
        "positive_only": True,
        "omega_min": None,
        "omega_max": None,
        "use_harmonic_order": True,
        "max_harmonic_order": DEFAULT_MAX_HARMONIC_ORDER,
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
        combine_planar: bool = False,
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

        panel_specs: list[tuple[str, tuple[str, ...]]] = []
        normalized_directions = tuple(direction.strip().lower() for direction in directions)
        if combine_planar and "x" in normalized_directions and "y" in normalized_directions:
            panel_specs.append(("xy-plane", ("x", "y")))
            normalized_directions = tuple(direction for direction in normalized_directions if direction not in {"x", "y"})
        for direction in normalized_directions:
            panel_specs.append((f"{direction}-component", (direction,)))
        if include_total:
            panel_specs.append(("total", ("total",)))

        rows = len(panel_specs)
        figure, axes = pyplot.subplots(rows, 1, figsize=(10.0, 3.0 * rows), sharex=True, squeeze=False)
        field_colors = {"x": "#E69F00", "y": "#56B4E9", "z": "#009E73", "total": "#000000"}
        current_colors = {"x": "#D55E00", "y": "#0072B2", "z": "#117733", "total": "#CC79A7"}

        for row, (title, panel_directions) in enumerate(panel_specs):
            axis = axes[row, 0]
            twin = axis.twinx()
            axis.set_facecolor("white")
            field_lines = []
            current_lines = []
            if panel_directions == ("total",):
                field_total = np.linalg.norm(field, axis=1)
                current_total = np.linalg.norm(current, axis=1)
                axis.fill_between(
                    time,
                    0.0,
                    field_total,
                    color=field_colors["total"],
                    alpha=0.10,
                    linewidth=0.0,
                )
                field_lines.append(
                    axis.plot(
                        time,
                        field_total,
                        linewidth=1.6,
                        color=field_colors["total"],
                        label=r"$|E(t)|$",
                    )[0]
                )
                current_lines.append(
                    twin.plot(
                        time,
                        current_total,
                        linewidth=1.4,
                        color=current_colors["total"],
                        label=r"$|J(t)|$",
                    )[0]
                )
                axis.set_ylabel(r"$|E(t)|$")
                twin.set_ylabel(r"$|J(t)|$")
            else:
                for direction in panel_directions:
                    idir = HarmonicGraphics._direction_axis(direction)
                    axis.fill_between(
                        time,
                        0.0,
                        field[:, idir],
                        color=field_colors.get(direction, "#000000"),
                        alpha=0.08,
                        linewidth=0.0,
                    )
                    field_lines.append(
                        axis.plot(
                            time,
                            field[:, idir],
                            linewidth=1.5,
                            color=field_colors.get(direction, None),
                            label=rf"$E_{{{direction}}}(t)$",
                        )[0]
                    )
                    current_lines.append(
                        twin.plot(
                            time,
                            current[:, idir],
                            linewidth=1.35,
                            linestyle="--",
                            color=current_colors.get(direction, None),
                            label=rf"$J_{{{direction}}}(t)$",
                        )[0]
                    )
                field_label = ", ".join(rf"$E_{{{direction}}}(t)$" for direction in panel_directions)
                current_label = ", ".join(rf"$J_{{{direction}}}(t)$" for direction in panel_directions)
                axis.set_ylabel(field_label)
                twin.set_ylabel(current_label)
            axis.axhline(0.0, color="#cfcfcf", linewidth=0.7, zorder=0)
            axis.set_title(title, fontsize=9, color="black")
            axis.grid(False)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_linewidth(0.5)
            axis.spines["bottom"].set_linewidth(0.5)
            axis.xaxis.set_ticks_position("bottom")
            axis.tick_params(
                axis="both",
                which="major",
                labelsize=7,
                width=0.5,
                length=3,
                direction="out",
                top=False,
                labeltop=False,
                right=False,
                labelright=False,
            )
            twin.spines["top"].set_visible(False)
            twin.spines["left"].set_visible(False)
            twin.spines["bottom"].set_visible(False)
            twin.spines["right"].set_linewidth(0.5)
            twin.tick_params(
                axis="x",
                which="both",
                top=False,
                labeltop=False,
                bottom=False,
                labelbottom=False,
            )
            twin.tick_params(axis="y", which="major", labelsize=7, width=0.5, length=3, direction="out")
            legend_lines = field_lines + current_lines
            axis.legend(
                legend_lines,
                [line.get_label() for line in legend_lines],
                loc="upper right",
                ncol=2,
                frameon=False,
                fontsize=7,
            )

        axes[-1, 0].set_xlabel(r"$t\;(\mathrm{a.u.})$")
        figure.patch.set_facecolor("white")
        figure.suptitle(r"Driving field and induced current in time", fontsize=10, color="black")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=300, facecolor=figure.get_facecolor())
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
        omega = np.asarray(omega_axis, dtype=float)
        spectrum = np.asarray(current_spectrum, dtype=np.complex128)

        if spectrum.ndim != 2 or spectrum.shape[1] != 3:
            raise ValueError("current_spectrum must have shape (Nomega, 3).")
        if spectrum.shape[0] != omega.size:
            raise ValueError("omega_axis and current_spectrum must share the same first dimension.")

        mask = HarmonicGraphics._build_frequency_mask(
            omega,
            positive_only=positive_only,
            omega_min=omega_min,
            omega_max=omega_max,
        )
        x_values, xlabel = HarmonicGraphics._build_spectral_xaxis(
            omega[mask],
            fundamental_omega=fundamental_omega,
            use_harmonic_order=use_harmonic_order,
        )
        colors = {"x": "#E69F00", "y": "#56B4E9", "z": "#009E73"}
        series: list[tuple[str, np.ndarray, str]] = []
        for direction in directions:
            axis_index = HarmonicGraphics._direction_axis(direction)
            series.append(
                (
                    rf"$|J_{{{direction}}}(\omega)|$",
                    np.abs(spectrum[mask, axis_index]),
                    colors.get(direction, "#444444"),
                )
            )

        return HarmonicGraphics._plot_multi_series_spectrum(
            x_values=x_values,
            xlabel=xlabel,
            series=series,
            output_path=output_path,
            title="Current Spectrum",
            ylabel=r"$|J_i(\omega)|$",
            use_harmonic_order=use_harmonic_order,
            max_harmonic_order=max_harmonic_order,
            log_scale=log_scale,
            top_energy_scale_ev=None if fundamental_omega is None else float(fundamental_omega) * AU_TO_EV,
            panel_tag="Cartesian components",
            context_note=HarmonicGraphics._format_orders_badge(orders),
        )

    @staticmethod
    def plot_total_current_spectrum(
        omega_axis: np.ndarray,
        total_magnitude: np.ndarray,
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
        omega = np.asarray(omega_axis, dtype=float)
        magnitude = np.asarray(total_magnitude, dtype=float)
        if magnitude.ndim != 1 or magnitude.shape[0] != omega.size:
            raise ValueError("total_magnitude must have shape (Nomega,).")

        mask = HarmonicGraphics._build_frequency_mask(
            omega,
            positive_only=positive_only,
            omega_min=omega_min,
            omega_max=omega_max,
        )
        x_values, xlabel = HarmonicGraphics._build_spectral_xaxis(
            omega[mask],
            fundamental_omega=fundamental_omega,
            use_harmonic_order=use_harmonic_order,
        )
        return HarmonicGraphics._plot_multi_series_spectrum(
            x_values=x_values,
            xlabel=xlabel,
            series=[(r"$|J(\omega)|$", magnitude[mask], "#000000")],
            output_path=output_path,
            title="Current Spectrum",
            ylabel=r"$|J(\omega)|$",
            use_harmonic_order=use_harmonic_order,
            max_harmonic_order=max_harmonic_order,
            log_scale=log_scale,
            top_energy_scale_ev=None if fundamental_omega is None else float(fundamental_omega) * AU_TO_EV,
            panel_tag="Total magnitude",
            context_note=HarmonicGraphics._format_orders_badge(orders),
        )

    @staticmethod
    def plot_inter_intra_current_spectrum(
        omega_axis: np.ndarray,
        intraband_magnitude: np.ndarray,
        interband_magnitude: np.ndarray,
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
        omega = np.asarray(omega_axis, dtype=float)
        intraband = np.asarray(intraband_magnitude, dtype=float)
        interband = np.asarray(interband_magnitude, dtype=float)
        if intraband.ndim != 1 or interband.ndim != 1:
            raise ValueError("intraband_magnitude and interband_magnitude must be 1D arrays.")
        if intraband.shape[0] != omega.size or interband.shape[0] != omega.size:
            raise ValueError("Magnitude arrays must match omega_axis.")

        mask = HarmonicGraphics._build_frequency_mask(
            omega,
            positive_only=positive_only,
            omega_min=omega_min,
            omega_max=omega_max,
        )
        x_values, xlabel = HarmonicGraphics._build_spectral_xaxis(
            omega[mask],
            fundamental_omega=fundamental_omega,
            use_harmonic_order=use_harmonic_order,
        )
        return HarmonicGraphics._plot_multi_series_spectrum(
            x_values=x_values,
            xlabel=xlabel,
            series=[
                (r"$|J_{\mathrm{intra}}(\omega)|$", intraband[mask], "#0072B2"),
                (r"$|J_{\mathrm{inter}}(\omega)|$", interband[mask], "#D55E00"),
            ],
            output_path=output_path,
            title="Current Spectrum",
            ylabel=r"$|J(\omega)|$",
            use_harmonic_order=use_harmonic_order,
            max_harmonic_order=max_harmonic_order,
            log_scale=log_scale,
            top_energy_scale_ev=None if fundamental_omega is None else float(fundamental_omega) * AU_TO_EV,
            panel_tag="Interband and intraband",
            context_note=HarmonicGraphics._format_orders_badge(orders),
        )

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
        omega = np.asarray(omega_axis, dtype=float)
        spectrum = np.asarray(current_spectrum, dtype=np.complex128)

        if spectrum.ndim != 2 or spectrum.shape[1] != 3:
            raise ValueError("current_spectrum must have shape (Nomega, 3).")
        if spectrum.shape[0] != omega.size:
            raise ValueError("omega_axis and current_spectrum must share the same first dimension.")

        mask = HarmonicGraphics._build_frequency_mask(
            omega,
            positive_only=positive_only,
            omega_min=omega_min,
            omega_max=omega_max,
        )

        jx = spectrum[:, 0]
        jy = spectrum[:, 1]
        current_right = (jx - 1.0j * jy) / np.sqrt(2.0)
        current_left = (jx + 1.0j * jy) / np.sqrt(2.0)
        right_mag = np.abs(current_right[mask])
        left_mag = np.abs(current_left[mask])
        x_values, xlabel = HarmonicGraphics._build_spectral_xaxis(
            omega[mask],
            fundamental_omega=fundamental_omega,
            use_harmonic_order=use_harmonic_order,
        )
        return HarmonicGraphics._plot_multi_series_spectrum(
            x_values=x_values,
            xlabel=xlabel,
            series=[
                (r"$|J_{\mathrm{R}}(\omega)|$", right_mag, "#0072B2"),
                (r"$|J_{\mathrm{L}}(\omega)|$", left_mag, "#CC79A7"),
            ],
            output_path=output_path,
            title="Current Spectrum",
            ylabel=r"$|J_{\mathrm{R/L}}(\omega)|$",
            use_harmonic_order=use_harmonic_order,
            max_harmonic_order=max_harmonic_order,
            log_scale=log_scale,
            top_energy_scale_ev=None if fundamental_omega is None else float(fundamental_omega) * AU_TO_EV,
            panel_tag="Circular basis",
            context_note=HarmonicGraphics._format_orders_badge(orders),
        )

    @staticmethod
    def plot_current_overview_spectrum(
        omega_axis: np.ndarray,
        current_spectrum: np.ndarray,
        total_magnitude: np.ndarray,
        output_path: str | Path,
        *,
        orders: tuple[int, ...] | None = None,
        directions: tuple[str, ...] = ("x", "y", "z"),
        intraband_magnitude: np.ndarray | None = None,
        interband_magnitude: np.ndarray | None = None,
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
        magnitude = np.asarray(total_magnitude, dtype=float)

        if spectrum.ndim != 2 or spectrum.shape[1] != 3:
            raise ValueError("current_spectrum must have shape (Nomega, 3).")
        if spectrum.shape[0] != omega.size or magnitude.shape != omega.shape:
            raise ValueError("omega_axis, current_spectrum and total_magnitude must match.")

        mask = HarmonicGraphics._build_frequency_mask(
            omega,
            positive_only=positive_only,
            omega_min=omega_min,
            omega_max=omega_max,
        )
        x_values, xlabel = HarmonicGraphics._build_spectral_xaxis(
            omega[mask],
            fundamental_omega=fundamental_omega,
            use_harmonic_order=use_harmonic_order,
        )

        component_colors = {"x": "#E69F00", "y": "#56B4E9", "z": "#009E73"}
        components_series = [
            (
                rf"$|J_{{{direction}}}(\omega)|$",
                np.abs(spectrum[mask, HarmonicGraphics._direction_axis(direction)]),
                component_colors.get(direction, "#444444"),
            )
            for direction in directions
        ]

        jx = spectrum[:, 0]
        jy = spectrum[:, 1]
        circular_series = [
            (r"$|J_{\mathrm{R}}(\omega)|$", np.abs(((jx - 1.0j * jy) / np.sqrt(2.0))[mask]), "#0072B2"),
            (r"$|J_{\mathrm{L}}(\omega)|$", np.abs(((jx + 1.0j * jy) / np.sqrt(2.0))[mask]), "#CC79A7"),
        ]

        panels: list[tuple[str, str, list[tuple[str, np.ndarray, str]], str]] = [
            ("A", "Total magnitude", [(r"$|J(\omega)|$", magnitude[mask], "#111827")], r"$|J(\omega)|$"),
            ("B", "Cartesian components", components_series, r"$|J_i(\omega)|$"),
            ("C", "Circular basis", circular_series, r"$|J_{\mathrm{R/L}}(\omega)|$"),
        ]

        if intraband_magnitude is not None and interband_magnitude is not None:
            intraband = np.asarray(intraband_magnitude, dtype=float)
            interband = np.asarray(interband_magnitude, dtype=float)
            if intraband.shape != omega.shape or interband.shape != omega.shape:
                raise ValueError("Decomposition magnitudes must match omega_axis.")
            panels.insert(
                2,
                (
                    "C",
                    "Interband and intraband",
                    [
                        (r"$|J_{\mathrm{intra}}(\omega)|$", intraband[mask], "#0072B2"),
                        (r"$|J_{\mathrm{inter}}(\omega)|$", interband[mask], "#D55E00"),
                    ],
                    r"$|J(\omega)|$",
                ),
            )
            panels[-1] = ("D", panels[-1][1], panels[-1][2], panels[-1][3])

        figure, axes = pyplot.subplots(2, 2, figsize=(12.8, 8.6), sharex=True)
        figure.patch.set_facecolor("white")
        flat_axes = list(axes.flat)
        top_energy_scale_ev = None if fundamental_omega is None else float(fundamental_omega) * AU_TO_EV

        for index, (axis, (panel_label, panel_tag, panel_series, ylabel)) in enumerate(zip(flat_axes, panels)):
            HarmonicGraphics._draw_multi_series_spectrum(
                axis,
                x_values=x_values,
                xlabel=xlabel,
                series=panel_series,
                title="Current Spectrum",
                ylabel=ylabel,
                use_harmonic_order=use_harmonic_order,
                max_harmonic_order=max_harmonic_order,
                log_scale=log_scale,
                top_energy_scale_ev=top_energy_scale_ev,
                panel_tag=None,
                context_note=None,
                show_top_axis=index < 2,
                legend_ncol=1,
            )

        for axis in flat_axes[len(panels) :]:
            axis.set_visible(False)

        figure.tight_layout()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=320, facecolor=figure.get_facecolor())
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
    def _build_frequency_mask(
        omega: np.ndarray,
        *,
        positive_only: bool,
        omega_min: float | None,
        omega_max: float | None,
    ) -> np.ndarray:
        mask = np.ones_like(omega, dtype=bool)
        if positive_only:
            mask &= omega >= 0.0
        if omega_min is not None:
            mask &= omega >= float(omega_min)
        if omega_max is not None:
            mask &= omega <= float(omega_max)
        if not np.any(mask):
            raise ValueError("No frequency points remain after applying the requested filters.")
        return mask

    @staticmethod
    def _build_spectral_xaxis(
        omega: np.ndarray,
        *,
        fundamental_omega: float | None,
        use_harmonic_order: bool,
    ) -> tuple[np.ndarray, str]:
        if not use_harmonic_order:
            return np.asarray(omega, dtype=float), r"$\omega\;(\mathrm{a.u.})$"
        if fundamental_omega is None or fundamental_omega <= 0.0:
            raise ValueError("fundamental_omega must be positive when use_harmonic_order=True.")
        return np.asarray(omega / float(fundamental_omega), dtype=float), "Harmonic order"

    @staticmethod
    def _plot_multi_series_spectrum(
        *,
        x_values: np.ndarray,
        xlabel: str,
        series: list[tuple[str, np.ndarray, str]],
        output_path: str | Path,
        title: str,
        ylabel: str,
        use_harmonic_order: bool,
        max_harmonic_order: float | None,
        log_scale: bool,
        top_energy_scale_ev: float | None = None,
        panel_tag: str | None = None,
        context_note: str | None = None,
    ) -> Path:
        pyplot = HarmonicGraphics._require_matplotlib()
        figure, axis = pyplot.subplots(figsize=(7.3, 4.9))
        figure.patch.set_facecolor("white")
        HarmonicGraphics._draw_multi_series_spectrum(
            axis,
            x_values=x_values,
            xlabel=xlabel,
            series=series,
            title=title,
            ylabel=ylabel,
            use_harmonic_order=use_harmonic_order,
            max_harmonic_order=max_harmonic_order,
            log_scale=log_scale,
            top_energy_scale_ev=top_energy_scale_ev,
            panel_tag=panel_tag,
            context_note=context_note,
            show_top_axis=True,
            legend_ncol=min(2, max(1, len(series))),
        )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=320, facecolor=figure.get_facecolor())
        pyplot.close(figure)
        return output

    @staticmethod
    def _spectrum_edge_floor(
        *,
        x_values: np.ndarray,
        valid_series: list[tuple[str, np.ndarray, str]],
        use_harmonic_order: bool,
        max_harmonic_order: float | None,
    ) -> float | None:
        """Return the minimum positive spectrum value at the visible edge.

        The HHG panels are explicitly cropped at ``max_harmonic_order``. Their
        logarithmic y-axis should therefore be set by the values that remain
        visible, especially by the minimum positive value at the last displayed
        harmonic order (currently H=3.5), not by peaks outside the cropped range.
        """
        if not use_harmonic_order or max_harmonic_order is None:
            return None
        x_arr = np.asarray(x_values, dtype=float)
        limit = float(max_harmonic_order)
        window = np.isfinite(x_arr) & (x_arr >= 0.5) & (x_arr <= limit)
        if not np.any(window):
            return None
        visible_indices = np.flatnonzero(window)
        edge_index = int(visible_indices[np.argmin(np.abs(x_arr[visible_indices] - limit))])
        edge_values: list[float] = []
        for _label, y_values, _color in valid_series:
            value = float(np.asarray(y_values, dtype=float)[edge_index])
            if np.isfinite(value) and value > 0.0:
                edge_values.append(value)
        if edge_values:
            return min(edge_values)

        # If the closest FFT bin to the edge happens to be exactly zero, fall
        # back to the minimum positive value in the final visible interval.
        lower = max(0.5, limit - 0.25)
        edge_window = window & (x_arr >= lower)
        fallback_values: list[float] = []
        for _label, y_values, _color in valid_series:
            segment = np.asarray(y_values, dtype=float)[edge_window]
            segment = segment[np.isfinite(segment) & (segment > 0.0)]
            if segment.size:
                fallback_values.append(float(np.min(segment)))
        if not fallback_values:
            return None
        return min(fallback_values)

    def _draw_multi_series_spectrum(
        axis: Any,
        *,
        x_values: np.ndarray,
        xlabel: str,
        series: list[tuple[str, np.ndarray, str]],
        title: str,
        ylabel: str,
        use_harmonic_order: bool,
        max_harmonic_order: float | None,
        log_scale: bool,
        top_energy_scale_ev: float | None,
        panel_tag: str | None,
        context_note: str | None,
        show_top_axis: bool,
        legend_ncol: int,
    ) -> None:
        axis.set_facecolor("white")

        x_arr = np.asarray(x_values, dtype=float)
        visible_mask = np.ones_like(x_arr, dtype=bool)
        if use_harmonic_order and max_harmonic_order is not None:
            visible_mask &= x_arr >= 0.5
            visible_mask &= x_arr <= float(max_harmonic_order)

        valid_series: list[tuple[str, np.ndarray, str]] = []
        positive_minima: list[float] = []
        y_peak = 0.0
        for label, values, color in series:
            y_values = np.asarray(values, dtype=float)
            if y_values.shape != x_values.shape:
                raise ValueError("Each spectral series must match the x-axis shape.")
            visible_values = y_values[visible_mask]
            visible_positives = visible_values[np.isfinite(visible_values) & (visible_values > 0.0)]
            if not visible_positives.size:
                continue
            valid_series.append((label, y_values, color))
            positive_minima.append(float(np.min(visible_positives)))
            y_peak = max(y_peak, float(np.max(visible_positives)))

        if not valid_series:
            raise ValueError("The selected spectrum is zero over the requested range.")

        baseline = 0.0
        if log_scale:
            axis.set_yscale("log")
            # Choose the y-axis floor from the minimum positive value at the
            # highest visible harmonic (x ~ max_harmonic_order), instead of
            # from the global spectrum or from a peak in the cropped-out region.
            edge_floor = HarmonicGraphics._spectrum_edge_floor(
                x_values=x_values,
                valid_series=valid_series,
                use_harmonic_order=use_harmonic_order,
                max_harmonic_order=max_harmonic_order,
            )
            if edge_floor is not None and edge_floor > 0.0:
                baseline = max(edge_floor * 0.8, 1.0e-16)  # small headroom below the H_max floor
            else:
                min_positive = min(positive_minima) if positive_minima else 1.0e-12
                baseline = max(min_positive * 0.35, 1.0e-16)
            if y_peak > 0.0:
                axis.set_ylim(bottom=baseline, top=y_peak * 1.8)  # a little headroom above the peak
            else:
                axis.set_ylim(bottom=baseline)
        else:
            if y_peak > 0.0:
                axis.set_ylim(bottom=baseline, top=y_peak * 1.12)
            else:
                axis.set_ylim(bottom=baseline)

        if use_harmonic_order and max_harmonic_order is not None:
            limit = float(max_harmonic_order)
            axis.set_xlim(0.5, limit)
            tick_max = int(np.floor(limit))
            axis.set_xticks(np.arange(1, tick_max + 1, 1, dtype=int))
        else:
            axis.set_xlim(float(np.min(x_values)), float(np.max(x_values)))

        axis.yaxis.grid(True, which="major", color="#e5e7eb", linewidth=0.9, alpha=1.0)
        axis.xaxis.grid(True, which="major", color="#d9dee7", linewidth=0.75, linestyle=(0, (2, 4)), alpha=0.95)
        if log_scale and LogLocator is not None and NullFormatter is not None:
            axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.yaxis.grid(True, which="minor", color="#f1f5f9", linewidth=0.7, alpha=1.0)

        for label, y_values, color in valid_series:
            HarmonicGraphics._add_gradient_fill(
                axis,
                x_values,
                y_values,
                color=color,
                baseline=baseline,
                log_scale=log_scale,
            )
            HarmonicGraphics._plot_series_line(axis, x_values, y_values, color=color, label=label)

        axis.set_title(title, fontsize=12.5, fontweight="semibold", color="#111827", pad=12)
        axis.set_xlabel(xlabel, fontsize=11.0, color="#111827")
        axis.set_ylabel(ylabel, fontsize=11.0, color="#111827")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#6b7280")
        axis.spines["bottom"].set_color("#6b7280")
        axis.spines["left"].set_linewidth(0.9)
        axis.spines["bottom"].set_linewidth(0.9)
        # The global paper style used elsewhere in QXTI enables top ticks on
        # every axis. HHG panels add a dedicated secondary top x-axis
        # (harmonic-energy scale), so the primary axis must explicitly disable
        # its own top ticks/labels to avoid duplicated/superposed top axes.
        axis.tick_params(
            axis="both",
            which="major",
            labelsize=9,
            width=0.8,
            length=4,
            direction="out",
            colors="#111827",
            top=False,
            labeltop=False,
            right=False,
            labelright=False,
        )
        axis.tick_params(
            axis="both",
            which="minor",
            width=0.6,
            length=2.5,
            direction="out",
            colors="#111827",
            top=False,
            right=False,
        )
        axis.margins(x=0.01)

        legend = axis.legend(
            frameon=True,
            fancybox=True,
            framealpha=0.94,
            fontsize=8,
            ncol=max(1, legend_ncol),
            loc="upper right",
            borderpad=0.45,
            labelspacing=0.35,
            handlelength=2.6,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#e5e7eb")
        legend.get_frame().set_linewidth(0.8)

        if show_top_axis and use_harmonic_order and top_energy_scale_ev is not None and top_energy_scale_ev > 0.0:
            axis.xaxis.set_ticks_position("bottom")
            top_axis = axis.secondary_xaxis(
                "top",
                functions=(
                    lambda harmonic_order: harmonic_order * top_energy_scale_ev,
                    lambda energy_ev: energy_ev / top_energy_scale_ev,
                ),
            )
            top_axis.set_xlabel(r"$\hbar\omega\;(\mathrm{eV})$", fontsize=10.5, color="#111827", labelpad=10)
            top_axis.tick_params(
                axis="x",
                which="major",
                labelsize=8,
                width=0.8,
                length=4,
                direction="out",
                colors="#111827",
                bottom=False,
                labelbottom=False,
                top=True,
                labeltop=True,
            )
            top_axis.spines["top"].set_linewidth(0.9)
            top_axis.spines["top"].set_color("#6b7280")

    @staticmethod
    def _plot_series_line(
        axis: Any,
        x_values: np.ndarray,
        y_values: np.ndarray,
        *,
        color: str,
        label: str,
    ) -> None:
        line = axis.plot(
            x_values,
            y_values,
            linewidth=2.3,
            color=color,
            label=label,
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=3,
        )[0]
        if pe is not None:
            line.set_path_effects(
                [
                    pe.Stroke(linewidth=4.4, foreground=(*mcolors.to_rgb(color), 0.10)),
                    pe.Normal(),
                ]
            )

    @staticmethod
    def _add_gradient_fill(
        axis: Any,
        x_values: np.ndarray,
        y_values: np.ndarray,
        *,
        color: str,
        baseline: float,
        log_scale: bool,
        steps: int = 32,
        max_alpha: float = 0.5,
    ) -> None:
        rgba = mcolors.to_rgba(color)
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        if log_scale:
            valid = np.isfinite(y_values) & (y_values > baseline)
            if not np.any(valid):
                return
            ratio = np.ones_like(y_values, dtype=float)
            ratio[valid] = np.maximum(y_values[valid] / baseline, 1.0)
            for index in range(steps):
                lower_frac = index / steps
                upper_frac = (index + 1) / steps
                lower = np.full_like(y_values, baseline)
                upper = np.full_like(y_values, baseline)
                lower[valid] = baseline * np.power(ratio[valid], lower_frac)
                upper[valid] = baseline * np.power(ratio[valid], upper_frac)
                axis.fill_between(
                    x_values,
                    lower,
                    upper,
                    color=rgba,
                    where=valid,
                    alpha=max_alpha * (upper_frac ** 1.25),
                    linewidth=0.0,
                    zorder=1,
                )
            return

        valid = np.isfinite(y_values)
        if not np.any(valid):
            return
        for index in range(steps):
            lower_frac = index / steps
            upper_frac = (index + 1) / steps
            lower = baseline + (y_values - baseline) * lower_frac
            upper = baseline + (y_values - baseline) * upper_frac
            axis.fill_between(
                x_values,
                lower,
                upper,
                color=rgba,
                where=valid,
                alpha=max_alpha * (upper_frac ** 1.25),
                linewidth=0.0,
                zorder=1,
            )

    @staticmethod
    def _format_orders_badge(orders: tuple[int, ...] | None) -> str | None:
        if not orders:
            return None
        orders_text = ", ".join(str(order) for order in orders)
        return rf"$\sum_s \rho^{{(s)}}$, $s={orders_text}$"

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
