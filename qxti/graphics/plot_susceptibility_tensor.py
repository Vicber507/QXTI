from __future__ import annotations

import argparse
import copy
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")


DEFAULT_SUSCEPTIBILITY_PLOT_CONFIG = {
    "dataset_file": "xtp_susceptibility.npz",
    "orders": "all",
    "positive_only": False,
    "omega_min": None,
    "omega_max": None,
    "dpi": 240,
    "include_ev_axis": True,
    "overview": {
        "enabled": True,
        "output_file_template": "chi{order}_overview.png",
    },
    "grid": {
        "enabled": True,
        "output_file_template": "chi{order}_grid.png",
    },
    "components": {
        "enabled": True,
        "output_file_template": "chi{order}_{label}.png",
    },
}

# Edit this dictionary directly if you want to change the plotting behavior.
SUSCEPTIBILITY_PLOT_CONFIG = copy.deepcopy(DEFAULT_SUSCEPTIBILITY_PLOT_CONFIG)


HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
else:  # pragma: no cover - environment dependent
    plt = None


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
AU_TO_EV = 27.211386245988
OKABE_ITO = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)


class SusceptibilityTensorPlotter:
    """Plot real and imaginary parts of susceptibility tensor components."""

    def __init__(
        self,
        *,
        x_axis: RealArray,
        tensor: ComplexArray,
        output_dir: str | Path = "outputs/susceptibility",
        x_label: str = r"$\omega\;(\mathrm{a.u.})$",
        argument_label: str = r"\omega",
        tensor_name: str = "chi",
        direction_labels: tuple[str, ...] = ("x", "y", "z"),
        available_components: Iterable[tuple[int, ...]] | None = None,
        dpi: int = 240,
        include_ev_axis: bool = True,
    ) -> None:
        self.x_axis = np.asarray(x_axis, dtype=np.float64)
        self.tensor = np.asarray(tensor, dtype=np.complex128)
        self.output_dir = Path(output_dir)
        self.x_label = str(x_label)
        self.argument_label = str(argument_label)
        self.tensor_name = str(tensor_name)
        self.direction_labels = tuple(direction_labels)
        self.dpi = int(dpi)
        self.include_ev_axis = bool(include_ev_axis)

        if self.tensor.ndim < 3:
            raise ValueError("tensor must have shape (N, dim, dim, ...) with at least 3 dimensions.")
        if self.tensor.shape[0] != self.x_axis.shape[0]:
            raise ValueError("x_axis length must match tensor first dimension.")

        self.dimension = int(self.tensor.shape[1])
        self.order = int(self.tensor.ndim - 2)
        expected_shape = (self.x_axis.shape[0],) + (self.dimension,) * (self.order + 1)
        if self.tensor.shape != expected_shape:
            raise ValueError(f"Expected tensor shape {expected_shape}, got {self.tensor.shape}.")
        if len(self.direction_labels) < self.dimension:
            raise ValueError("direction_labels must contain at least dimension labels.")

        if available_components is None:
            self.available_components = None
        else:
            self.available_components = tuple(tuple(int(index) for index in item) for item in available_components)

    @classmethod
    def from_npy(
        cls,
        *,
        x_axis_path: str | Path,
        tensor_path: str | Path,
        output_dir: str | Path = "outputs/susceptibility",
        x_label: str = r"$\omega\;(\mathrm{a.u.})$",
        argument_label: str = r"\omega",
        tensor_name: str = "chi",
        direction_labels: tuple[str, ...] = ("x", "y", "z"),
        available_components: Iterable[tuple[int, ...]] | None = None,
        dpi: int = 240,
        include_ev_axis: bool = True,
    ) -> SusceptibilityTensorPlotter:
        return cls(
            x_axis=np.load(x_axis_path),
            tensor=np.load(tensor_path),
            output_dir=output_dir,
            x_label=x_label,
            argument_label=argument_label,
            tensor_name=tensor_name,
            direction_labels=direction_labels,
            available_components=available_components,
            dpi=dpi,
            include_ev_axis=include_ev_axis,
        )

    def plot_component(
        self,
        indices: tuple[int, ...],
        *,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> Path:
        pyplot = self._require_matplotlib()
        self._validate_component_indices(indices)

        values = self.tensor[(slice(None),) + indices]
        label = self._component_label(indices)

        figure, axis = pyplot.subplots(figsize=(7.2, 4.6))
        axis.plot(
            self.x_axis,
            np.real(values),
            color=OKABE_ITO[0],
            linewidth=2.0,
            label=rf"$\Re\,{self.tensor_name}_{{{label}}}$",
        )
        axis.plot(
            self.x_axis,
            np.imag(values),
            color=OKABE_ITO[1],
            linewidth=1.9,
            linestyle="--",
            label=rf"$\Im\,{self.tensor_name}_{{{label}}}$",
        )
        axis.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
        axis.set_xlabel(self.x_label)
        axis.set_ylabel(rf"${self.tensor_name}_{{{label}}}$")
        axis.set_title(rf"${self.tensor_name}_{{{label}}}({self.argument_label})$")
        axis.grid(True, alpha=0.18, linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axis.legend(frameon=False, fontsize=9)
        if self.include_ev_axis:
            top_axis = axis.secondary_xaxis(
                "top",
                functions=(lambda values: values * AU_TO_EV, lambda values: values / AU_TO_EV),
            )
            top_axis.set_xlabel(r"$\omega_\mathrm{laser}\;(\mathrm{eV})$")

        output = self.output_dir / f"{self.tensor_name}_{label}.png" if output_path is None else Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=self.dpi, facecolor="white")

        if show:
            pyplot.show()
        else:
            pyplot.close(figure)
        return output

    def plot_all_components(
        self,
        *,
        output_file_template: str | None = None,
        show: bool = False,
    ) -> list[Path]:
        paths: list[Path] = []
        for indices in self.component_indices():
            label = self._component_label(indices)
            output_path = None
            if output_file_template is not None:
                output_path = self.output_dir / output_file_template.format(
                    order=self.order,
                    label=label,
                )
            paths.append(self.plot_component(indices, output_path=output_path, show=show))
        return paths

    def plot_grid(
        self,
        *,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> Path:
        pyplot = self._require_matplotlib()
        indices = list(self.component_indices())
        if not indices:
            raise ValueError("No susceptibility components are available to plot.")

        n_components = len(indices)
        ncols = min(max(self.dimension, 1), 3)
        nrows = int(np.ceil(n_components / ncols))
        figure, axes = pyplot.subplots(
            nrows,
            ncols,
            figsize=(4.9 * ncols, 3.5 * nrows),
            squeeze=False,
            sharex=True,
        )

        panel_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for ipanel, (axis, component_indices) in enumerate(zip(axes.flat, indices)):
            values = self.tensor[(slice(None),) + component_indices]
            label = self._component_label(component_indices)
            axis.plot(self.x_axis, np.real(values), color=OKABE_ITO[0], linewidth=1.6, label="Re")
            axis.plot(self.x_axis, np.imag(values), color=OKABE_ITO[1], linewidth=1.6, linestyle="--", label="Im")
            axis.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
            axis.set_title(rf"{panel_labels[ipanel]}. ${self.tensor_name}_{{{label}}}$", fontsize=10, loc="left")
            axis.grid(True, alpha=0.18, linewidth=0.6)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

        for axis in axes.flat[n_components:]:
            axis.axis("off")

        for axis in axes[-1, :]:
            axis.set_xlabel(self.x_label)

        axes[0, 0].legend(frameon=False, fontsize=8)
        figure.suptitle(
            rf"{self.tensor_name}$^{{({self.order})}}$ tensor components vs. $\omega_\mathrm{{laser}}$",
            fontsize=12.5,
        )

        output = self.output_dir / f"{self.tensor_name}_grid.png" if output_path is None else Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=self.dpi, facecolor="white")

        if show:
            pyplot.show()
        else:
            pyplot.close(figure)
        return output

    def plot_overview(
        self,
        *,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> Path:
        pyplot = self._require_matplotlib()
        component_indices = list(self.component_indices())
        if not component_indices:
            raise ValueError("No susceptibility components are available to plot.")

        figure, axes = pyplot.subplots(2, 1, figsize=(8.2, 6.2), sharex=True, squeeze=False)
        real_axis = axes[0, 0]
        imag_axis = axes[1, 0]

        for index, component in enumerate(component_indices):
            label = self._component_label(component)
            color = OKABE_ITO[index % len(OKABE_ITO)]
            values = self.tensor[(slice(None),) + component]
            real_axis.plot(
                self.x_axis,
                np.real(values),
                color=color,
                linewidth=1.8,
                label=rf"${self.tensor_name}_{{{label}}}$",
            )
            imag_axis.plot(
                self.x_axis,
                np.imag(values),
                color=color,
                linewidth=1.8,
                label=rf"${self.tensor_name}_{{{label}}}$",
            )

        for axis, title in ((real_axis, "Real part"), (imag_axis, "Imaginary part")):
            axis.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
            axis.grid(True, alpha=0.18, linewidth=0.6)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            axis.set_title(title, fontsize=10, loc="left")

        imag_axis.set_xlabel(self.x_label)
        real_axis.set_ylabel(rf"$\Re\,{self.tensor_name}$")
        imag_axis.set_ylabel(rf"$\Im\,{self.tensor_name}$")
        real_axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.30),
            ncol=min(4, len(component_indices)),
            frameon=False,
            fontsize=8,
        )
        if self.include_ev_axis:
            top_axis = real_axis.secondary_xaxis(
                "top",
                functions=(lambda values: values * AU_TO_EV, lambda values: values / AU_TO_EV),
            )
            top_axis.set_xlabel(r"$\omega_\mathrm{laser}\;(\mathrm{eV})$")

        figure.suptitle(
            rf"Overview of {self.tensor_name}$^{{({self.order})}}$ components",
            fontsize=12.5,
        )
        output = self.output_dir / f"{self.tensor_name}_overview.png" if output_path is None else Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=self.dpi, facecolor="white")

        if show:
            pyplot.show()
        else:
            pyplot.close(figure)
        return output

    def component_indices(self) -> Iterable[tuple[int, ...]]:
        if self.available_components is not None:
            return self.available_components
        return tuple(self._all_component_indices())

    def _all_component_indices(self) -> Iterable[tuple[int, ...]]:
        yield from np.ndindex((self.dimension,) * (self.order + 1))

    def _validate_component_indices(self, indices: tuple[int, ...]) -> None:
        expected = self.order + 1
        if len(indices) != expected:
            raise ValueError(f"indices must contain {expected} entries.")
        for index in indices:
            if index < 0 or index >= self.dimension:
                raise ValueError(f"component index {index} is outside dimension {self.dimension}.")

    def _component_label(self, indices: tuple[int, ...]) -> str:
        labels = self.direction_labels[: self.dimension]
        return "".join(labels[index] for index in indices)

    @staticmethod
    def _require_matplotlib() -> Any:
        if not HAS_MATPLOTLIB:
            raise RuntimeError(
                "matplotlib is not installed in this environment. "
                "Install it to generate susceptibility plots."
            )
        return plt


def resolve_susceptibility_plot_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(SUSCEPTIBILITY_PLOT_CONFIG)
    if overrides:
        _deep_update(config, overrides)
    return config


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate susceptibility tensor plots from a dedicated laser-frequency "
            "sweep dataset using the plot settings stored in [xtp]."
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="inputParams.susceptibility.cfg",
        help="Path to the dedicated susceptibility-sweep config file.",
    )
    return parser


def _main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from qxti.graphics.graphics import plot_susceptibility_graphics_from_saved_data

    parser = _build_parser()
    args = parser.parse_args()
    outputs = plot_susceptibility_graphics_from_saved_data(
        args.config,
        plot_config=resolve_susceptibility_plot_config(),
    )
    print(f"Generated {len(outputs)} susceptibility graphics from saved data in {args.config}:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
