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
    "conductivity": {
        "enabled": True,
        "overview": {
            "enabled": True,
            "output_file_template": "sigma{order}_overview.png",
        },
        "grid": {
            "enabled": True,
            "output_file_template": "sigma{order}_grid.png",
        },
        "components": {
            "enabled": True,
            "output_file_template": "sigma{order}_{label}.png",
        },
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


_PAPER_STYLE_APPLIED = False


def apply_paper_style() -> None:
    """Apply a professional, publication-quality Matplotlib style.

    Uses LaTeX for all text when a LaTeX installation is available, otherwise
    falls back to Matplotlib's Computer Modern math fonts (the LaTeX look without
    needing a LaTeX install). Inward ticks on all four sides, minor ticks, thin
    spines and serif fonts give a clean "paper" appearance.
    """
    global _PAPER_STYLE_APPLIED
    if not HAS_MATPLOTLIB or _PAPER_STYLE_APPLIED:
        return
    rc = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": True,
        "font.size": 12,
        "axes.labelsize": 15,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 5.5,
        "ytick.major.size": 5.5,
        "xtick.minor.size": 3.0,
        "ytick.minor.size": 3.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
    # Use Matplotlib's mathtext with the Computer Modern font set: it renders
    # LaTeX-syntax math in the genuine LaTeX font WITHOUT invoking a real LaTeX
    # subprocess, so it is robust (never breaks on labels other plots use) and
    # needs no LaTeX installation.
    try:
        plt.rcParams.update(rc)
        _PAPER_STYLE_APPLIED = True
    except Exception:  # pragma: no cover
        _PAPER_STYLE_APPLIED = True


if HAS_MATPLOTLIB:
    apply_paper_style()


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

# A 12-colour categorical palette so all 9 tensor components are distinguishable
# in the overview (a 3x3 tensor has 9 curves; an 8-colour palette repeats).
TENSOR_PALETTE = (
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
    "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f", "#393b79", "#637939",
)
# Cyclic line styles, used together with colour to reinforce distinguishability.
LINE_STYLES = ("-", "--", "-.", ":")


def to_helicity_basis(
    tensor: ComplexArray,
    dimension: int,
) -> tuple[ComplexArray, tuple[str, ...]]:
    r"""Rotate a cartesian response tensor to the helicity (circular) basis.

    The in-plane ``(x, y)`` block is expressed in the circular basis
    ``e_\pm = (x \pm i y)/\sqrt{2}`` (left/right rotating), while ``z`` (if
    present) is kept. For a rank-2 response ``J_i = T_{ij} E_j`` the transformed
    tensor is ``T' = U^\dagger T U`` with ``U`` the columns ``[e_+, e_-, (z)]``.

    This is the tensor analogue of forming the circular current ``J_\pm =
    J_x \pm i J_y``. The chiral/Hall part shows up as ``T_{++} \neq T_{--}``
    (their difference is the circular dichroism, proportional to the
    antisymmetric ``T_{xy} - T_{yx}``).

    Parameters
    ----------
    tensor:
        Cartesian tensor of shape ``(Nomega, dim, dim)``.
    dimension:
        Spatial dimension (1, 2, or 3).

    Returns
    -------
    (tensor_helicity, labels):
        The rotated tensor (same shape) and the component labels, e.g.
        ``("+", "-", "z")`` for 3D.
    """
    values = np.asarray(tensor, dtype=np.complex128)
    if dimension <= 1:
        # No in-plane rotation is possible in 1D.
        return values, ("x",)

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    if dimension == 2:
        unitary = np.array(
            [[inv_sqrt2, inv_sqrt2], [1j * inv_sqrt2, -1j * inv_sqrt2]],
            dtype=np.complex128,
        )
        labels: tuple[str, ...] = ("+", "-")
    else:
        unitary = np.array(
            [
                [inv_sqrt2, inv_sqrt2, 0.0],
                [1j * inv_sqrt2, -1j * inv_sqrt2, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.complex128,
        )
        labels = ("+", "-", "z")

    # T'_{ab} = sum_ij (U^dagger)_{ai} T_{ij} U_{jb}, applied for every omega.
    transformed = np.einsum(
        "ai,wij,jb->wab",
        unitary.conj().T,
        values,
        unitary,
        optimize=True,
    )
    return np.asarray(transformed, dtype=np.complex128), labels


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

    @property
    def _tex_name(self) -> str:
        """Proper LaTeX symbol for the tensor, e.g. ``chi2`` -> ``\\chi^{(2)}``."""
        base = self.tensor_name.rstrip("0123456789")
        greek = {"chi": r"\chi", "sigma": r"\sigma"}.get(base, rf"\mathrm{{{base}}}")
        return rf"{greek}^{{({self.order})}}"

    def plot_component_modulus(
        self,
        indices: tuple[int, ...],
        *,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> Path:
        """Paper-style plot of ONLY the modulus of one component, y-axis from 0."""
        pyplot = self._require_matplotlib()
        self._validate_component_indices(indices)

        values = self.tensor[(slice(None),) + indices]
        label = self._component_label(indices)
        modulus = np.abs(values)
        sym = rf"{self._tex_name}_{{\mathrm{{{label}}}}}"

        figure, axis = pyplot.subplots(figsize=(6.6, 4.4))
        line_color = "#0B3D91"  # deep "publication" blue
        axis.plot(self.x_axis, modulus, color=line_color, linewidth=2.3, solid_capstyle="round")
        axis.fill_between(self.x_axis, 0.0, modulus, color=line_color, alpha=0.13, linewidth=0.0)

        # y-axis starts exactly at 0.
        top = float(np.nanmax(modulus)) if modulus.size and np.isfinite(modulus).any() else 1.0
        axis.set_ylim(0.0, 1.05 * top if top > 0 else 1.0)
        axis.set_xlim(float(self.x_axis.min()), float(self.x_axis.max()))
        axis.margins(x=0.0)

        axis.set_xlabel(self.x_label)
        axis.set_ylabel(rf"$\left|{sym}\right|$")
        axis.set_title(rf"$\left|{sym}({self.argument_label})\right|$")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        if self.include_ev_axis:
            top_axis = axis.secondary_xaxis(
                "top",
                functions=(lambda v: v * AU_TO_EV, lambda v: v / AU_TO_EV),
            )
            top_axis.set_xlabel(r"$\hbar\omega_\mathrm{laser}\;(\mathrm{eV})$")

        output = (
            self.output_dir / f"{self.tensor_name}_{label}_modulus.png"
            if output_path is None else Path(output_path)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=self.dpi)
        pyplot.show() if show else pyplot.close(figure)
        return output

    def plot_all_components_modulus(
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
                output_path = self.output_dir / output_file_template.format(order=self.order, label=label)
            paths.append(self.plot_component_modulus(indices, output_path=output_path, show=show))
        return paths

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
        sym = rf"{self._tex_name}_{{\mathrm{{{label}}}}}"

        figure, axis = pyplot.subplots(figsize=(7.0, 4.6))
        axis.plot(
            self.x_axis,
            np.real(values),
            color=OKABE_ITO[0],
            linewidth=2.0,
            label=rf"$\Re\,{sym}$",
        )
        axis.plot(
            self.x_axis,
            np.imag(values),
            color=OKABE_ITO[1],
            linewidth=1.9,
            linestyle="--",
            label=rf"$\Im\,{sym}$",
        )
        # Modulus |chi| (the envelope sqrt(Re^2+Im^2)); a faint filled band makes
        # it easy to read off the magnitude alongside Re and Im.
        modulus = np.abs(values)
        axis.plot(
            self.x_axis,
            modulus,
            color=OKABE_ITO[2],
            linewidth=2.2,
            linestyle="-",
            alpha=0.9,
            label=rf"$\left|{sym}\right|$",
        )
        axis.fill_between(self.x_axis, 0.0, modulus, color=OKABE_ITO[2], alpha=0.07, linewidth=0.0)
        axis.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
        axis.set_xlim(float(self.x_axis.min()), float(self.x_axis.max()))
        axis.set_xlabel(self.x_label)
        axis.set_ylabel(rf"${sym}$")
        axis.set_title(rf"${sym}({self.argument_label})$")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axis.legend(frameon=False, fontsize=10.5)
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
            modulus = np.abs(values)
            axis.plot(self.x_axis, np.real(values), color=OKABE_ITO[0], linewidth=1.6, label="Re")
            axis.plot(self.x_axis, np.imag(values), color=OKABE_ITO[1], linewidth=1.6, linestyle="--", label="Im")
            axis.plot(self.x_axis, modulus, color=OKABE_ITO[2], linewidth=1.8, label=r"$|\,\cdot\,|$")
            axis.fill_between(self.x_axis, 0.0, modulus, color=OKABE_ITO[2], alpha=0.07, linewidth=0.0)
            axis.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
            sym = rf"{self._tex_name}_{{\mathrm{{{label}}}}}"
            axis.set_title(rf"({panel_labels[ipanel].lower()}) ${sym}$", fontsize=11, loc="left")
            axis.set_xlim(float(self.x_axis.min()), float(self.x_axis.max()))
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

        for axis in axes.flat[n_components:]:
            axis.axis("off")

        for axis in axes[-1, :]:
            axis.set_xlabel(self.x_label)

        axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
        figure.suptitle(
            rf"${self._tex_name}$ tensor components vs.\ $\omega_\mathrm{{laser}}$"
            if plt.rcParams.get("text.usetex", False)
            else rf"${self._tex_name}$ tensor components",
            fontsize=14,
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

        # Three stacked panels: real part, imaginary part, and modulus |.|.
        figure, axes = pyplot.subplots(3, 1, figsize=(9.0, 9.5), sharex=True, squeeze=False)
        real_axis = axes[0, 0]
        imag_axis = axes[1, 0]
        modulus_axis = axes[2, 0]

        for index, component in enumerate(component_indices):
            label = self._component_label(component)
            # Distinct colour AND line style per component so all 9 curves of a
            # 3x3 tensor are distinguishable. Diagonal components (e.g. xx) are
            # drawn thicker to stand out from the off-diagonal ones.
            color = TENSOR_PALETTE[index % len(TENSOR_PALETTE)]
            linestyle = LINE_STYLES[index % len(LINE_STYLES)]
            is_diagonal = len(set(component)) == 1
            linewidth = 2.4 if is_diagonal else 1.5
            values = self.tensor[(slice(None),) + component]
            sym = rf"{self._tex_name}_{{\mathrm{{{label}}}}}"
            real_axis.plot(
                self.x_axis, np.real(values), color=color,
                linewidth=linewidth, linestyle=linestyle, label=rf"${sym}$",
            )
            imag_axis.plot(
                self.x_axis, np.imag(values), color=color,
                linewidth=linewidth, linestyle=linestyle, label=rf"${sym}$",
            )
            modulus_axis.plot(
                self.x_axis, np.abs(values), color=color,
                linewidth=linewidth, linestyle=linestyle, label=rf"${sym}$",
            )

        panels = (
            (real_axis, "(a) Real part"),
            (imag_axis, "(b) Imaginary part"),
            (modulus_axis, "(c) Modulus"),
        )
        for axis, title in panels:
            axis.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
            axis.set_xlim(float(self.x_axis.min()), float(self.x_axis.max()))
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            axis.set_title(title, fontsize=12, loc="left")
        # modulus panel: y-axis from 0
        modulus_axis.set_ylim(bottom=0.0)

        modulus_axis.set_xlabel(self.x_label)
        real_axis.set_ylabel(rf"$\Re\,{self._tex_name}$")
        imag_axis.set_ylabel(rf"$\Im\,{self._tex_name}$")
        modulus_axis.set_ylabel(rf"$\left|{self._tex_name}\right|$")
        if self.include_ev_axis:
            top_axis = real_axis.secondary_xaxis(
                "top",
                functions=(lambda values: values * AU_TO_EV, lambda values: values / AU_TO_EV),
            )
            top_axis.set_xlabel(r"$\omega_\mathrm{laser}\;(\mathrm{eV})$")

        # Legend to the RIGHT of the panels (one column) so it never overlaps the
        # eV top axis and all 9 components stay readable.
        handles, leg_labels = real_axis.get_legend_handles_labels()
        figure.legend(
            handles,
            leg_labels,
            loc="center left",
            bbox_to_anchor=(0.84, 0.5),
            ncol=1,
            frameon=False,
            fontsize=9,
        )
        figure.suptitle(
            rf"Overview of ${self._tex_name}$ tensor components",
            fontsize=14,
        )
        output = self.output_dir / f"{self.tensor_name}_overview.png" if output_path is None else Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the right margin for the legend and the top for the suptitle.
        figure.tight_layout(rect=(0.0, 0.0, 0.83, 0.95))
        figure.savefig(output, dpi=self.dpi, facecolor="white", bbox_inches="tight")

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
            "Generate susceptibility and conductivity tensor plots from a dedicated "
            "laser-frequency sweep dataset using the plot settings stored in [xtp]."
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
