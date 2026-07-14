#!/usr/bin/env python3
r"""Paper-quality susceptibility / conductivity MODULUS maps for a QXTI model.

For a given config it computes, over a photon-energy axis, the analytic
    sigma^(1)_{ij}(omega)        -- linear conductivity
    chi^(1)_{ij}(omega)          -- linear susceptibility from sigma^(1)/(i omega)
    chi^(2)_{i,jk}(2 omega)      -- second-order (SHG) susceptibility (FULL tensor)
and renders two grids of the MODULUS |.| only. Every panel carries its own x and
y axis (independent ticks and labels). All labels in LaTeX.

Layout
------
* sigma^(1): 2x2 grid = {xx, xy, yx, yy}  (in-plane block, 4 components).
* chi^(2)  : FULL tensor, rows = output i, columns = all (j,k) pairs.
             2D model -> 2x4 = 8 components ; 3D model -> 3x9 = 27 components.

Results are cached to ``<out>/<label>_data.npz`` so re-plotting is instant.

Usage:
    plot_susceptibility_maps.py <config> <label> [--grid N] [--emin eV] [--emax eV]
                                [--nw N] [--pretty NAME] [--recompute]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qxti.analytics.theory_response import (
    build_k_integration_weights,
    compute_linear_response_spectrum,
)
from qxti.core.config import QXTIConfig
from qxti.core.simulation import QXTISimulation
from _shg_tensor_gridbased import order2_full_tensor_spectrum

_EV = 27.211386245988
XYZ = ("x", "y", "z")

FILL = "#9DC3E6"    # modulus fill
LINE = "#0B3C5D"    # modulus line
PLOT_PALETTE = (
    "#59819E",  # RGB(89,129,158)
    "#7C7AA2",  # RGB(124,122,162)
    "#F67E7D",  # RGB(246,126,125)
    "#FFC0A7",  # RGB(255,192,167)
    "#6CC2BD",  # RGB(108,194,189)
    "#4A4A4A",
)
CHI1_COMPONENTS = ("xx", "yy", "zz")
SELECTED_COMPONENTS = ("zzz", "zxx", "zyy", "xzx", "yzy")
CHI1_COLORS = dict(zip(CHI1_COMPONENTS, PLOT_PALETTE, strict=False))
SELECTED_COLORS = {
    key: color for key, color in zip(SELECTED_COMPONENTS, PLOT_PALETTE, strict=False)
}


def _grid_override(cfg, grid, dim):
    if grid <= 0:
        return cfg
    return replace(cfg, kgrid=replace(cfg.kgrid, k_points=[grid] * dim))


def _panel(ax, e_ev, z, title, ylabel):
    mod = np.abs(z)
    ax.fill_between(e_ev, mod, color=FILL, alpha=0.42, lw=0, zorder=2)
    ax.plot(e_ev, mod, color=LINE, lw=2.0, zorder=3)
    ax.set_title(title, fontsize=11, pad=3)
    ax.set_xlim(e_ev[0], e_ev[-1]); ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$\hbar\omega\ (\mathrm{eV})$", fontsize=8, labelpad=1.5)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=1.5)
    ax.tick_params(labelsize=7.5)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    ax.yaxis.get_offset_text().set_fontsize(7)


def _component_to_indices(component: str, dim: int) -> tuple[int, int, int]:
    key = str(component).strip().lower()
    if len(key) != 3 or any(axis not in XYZ for axis in key):
        raise ValueError(f"Invalid chi^(2) component {component!r}; use labels like zzz or xzx.")
    idx = tuple(XYZ.index(axis) for axis in key)
    if any(i >= dim for i in idx):
        raise ValueError(f"Component {component!r} is not available for a {dim}D model.")
    return idx  # type: ignore[return-value]


def _sigma_component_to_indices(component: str, dim: int) -> tuple[int, int]:
    key = str(component).strip().lower()
    if len(key) != 2 or any(axis not in XYZ for axis in key):
        raise ValueError(f"Invalid rank-2 tensor component {component!r}; use labels like xx or yz.")
    idx = tuple(XYZ.index(axis) for axis in key)
    if any(i >= dim for i in idx):
        raise ValueError(f"Component {component!r} is not available for a {dim}D model.")
    return idx  # type: ignore[return-value]


def _component_label(component: str) -> str:
    return rf"$|\chi^{{(2)}}_{{{component}}}|$"


def _chi1_component_label(component: str) -> str:
    return rf"$|\chi^{{(1)}}_{{{component}}}|$"


def _set_energy_ticks(ax, e_ev: np.ndarray) -> None:
    lo = float(np.nanmin(e_ev))
    hi = float(np.nanmax(e_ev))
    tick_start = np.ceil(lo * 2.0) / 2.0
    tick_stop = np.floor(hi * 2.0) / 2.0
    ticks = np.arange(tick_start, tick_stop + 0.25, 0.5)
    ticks = ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in ticks])


def _selected_component_data(chi2: np.ndarray, components: tuple[str, ...], dim: int) -> dict[str, np.ndarray]:
    selected: dict[str, np.ndarray] = {}
    for comp in components:
        i, j, k = _component_to_indices(comp, dim)
        selected[comp] = np.abs(chi2[:, i, j, k])
    return selected


def _rank2_component_data(tensor: np.ndarray, components: tuple[str, ...], dim: int) -> dict[str, np.ndarray]:
    selected: dict[str, np.ndarray] = {}
    for comp in components:
        i, j = _sigma_component_to_indices(comp, dim)
        selected[comp] = np.abs(tensor[:, i, j])
    return selected


def _available_chi1_components(dim: int) -> tuple[str, ...]:
    return tuple(comp for comp in CHI1_COMPONENTS if all(XYZ.index(axis) < dim for axis in comp))


def _plot_selected_stack(
    e_ev: np.ndarray,
    selected: dict[str, np.ndarray],
    output_path: Path,
    *,
    colors: dict[str, str],
    labeler: Callable[[str], str],
) -> Path:
    import matplotlib.pyplot as plt

    nrows = len(selected)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10.4, max(10.4, 7.8 * nrows)),
        squeeze=False,
    )
    for ax, (comp, values) in zip(axes.flat, selected.items()):
        color = colors.get(comp, LINE)
        ax.fill_between(e_ev, values, color=color, alpha=0.34, lw=0, zorder=2)
        ax.plot(e_ev, values, color=color, lw=3.8, zorder=3)
        ax.set_xlim(float(e_ev[0]), float(e_ev[-1]))
        ax.set_ylim(bottom=0.0)
        ax.set_box_aspect(1.0)
        _set_energy_ticks(ax, e_ev)
        ax.set_xlabel(r"$\hbar\omega\;(\mathrm{eV})$", fontsize=34, labelpad=12)
        ax.set_ylabel(labeler(comp), fontsize=34, labelpad=14)
        ax.tick_params(axis="both", which="major", labelsize=31, length=10, width=1.4)
        ax.tick_params(axis="both", which="minor", length=5, width=1.1)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.yaxis.get_offset_text().set_fontsize(28)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.04, top=0.99, hspace=0.58)
    fig.savefig(output_path, dpi=320, facecolor="white")
    plt.close(fig)
    return output_path


def _plot_selected_single_panel(
    e_ev: np.ndarray,
    selected: dict[str, np.ndarray],
    output_path: Path,
    *,
    colors: dict[str, str],
    labeler: Callable[[str], str],
    ylabel: str,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.4, 10.4))
    ymax = 0.0
    for comp, values in selected.items():
        ymax = max(ymax, float(np.nanmax(values)))
        color = colors.get(comp, LINE)
        ax.fill_between(e_ev, values, color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(e_ev, values, color=color, lw=4.2, label=labeler(comp), zorder=3)
    ax.set_xlim(float(e_ev[0]), float(e_ev[-1]))
    ax.set_ylim(0.0, ymax * 1.06 if ymax > 0 else 1.0)
    ax.set_box_aspect(1.0)
    _set_energy_ticks(ax, e_ev)
    ax.set_xlabel(r"$\hbar\omega\;(\mathrm{eV})$", fontsize=36, labelpad=13)
    ax.set_ylabel(ylabel, fontsize=36, labelpad=15)
    ax.tick_params(axis="both", which="major", labelsize=32, length=10, width=1.4)
    ax.tick_params(axis="both", which="minor", length=5, width=1.1)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    ax.yaxis.get_offset_text().set_fontsize(29)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.legend(
        frameon=False,
        fontsize=32,
        handlelength=1.9,
        labelspacing=0.65,
        loc="upper right",
    )
    fig.subplots_adjust(left=0.21, right=0.96, bottom=0.15, top=0.96)
    fig.savefig(output_path, dpi=320, facecolor="white")
    plt.close(fig)
    return output_path


def _plot_selected_separate(
    e_ev: np.ndarray,
    selected: dict[str, np.ndarray],
    output_dir: Path,
    *,
    colors: dict[str, str],
    labeler: Callable[[str], str],
    file_prefix: str,
) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob(f"{file_prefix}_*.png"):
        stale_path.unlink()
    paths: list[Path] = []
    for comp, values in selected.items():
        color = colors.get(comp, LINE)
        fig, ax = plt.subplots(figsize=(10.4, 10.4))
        ax.fill_between(e_ev, values, color=color, alpha=0.34, lw=0, zorder=2)
        ax.plot(e_ev, values, color=color, lw=3.8, zorder=3)
        ax.set_xlim(float(e_ev[0]), float(e_ev[-1]))
        ax.set_ylim(bottom=0.0)
        ax.set_box_aspect(1.0)
        _set_energy_ticks(ax, e_ev)
        ax.set_xlabel(r"$\hbar\omega\;(\mathrm{eV})$", fontsize=34, labelpad=12)
        ax.set_ylabel(labeler(comp), fontsize=34, labelpad=14)
        ax.tick_params(axis="both", which="major", labelsize=31, length=10, width=1.4)
        ax.tick_params(axis="both", which="minor", length=5, width=1.1)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.yaxis.get_offset_text().set_fontsize(28)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)
        fig.subplots_adjust(left=0.25, right=0.96, bottom=0.17, top=0.96)
        path = output_dir / f"{file_prefix}_{comp}.png"
        fig.savefig(path, dpi=320, facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths


def _parse_orders(raw: str) -> set[int]:
    text = str(raw).strip().lower()
    if text in {"all", "both"}:
        return {1, 2}
    orders = {
        int(part.strip())
        for part in text.replace(";", ",").split(",")
        if part.strip()
    }
    invalid = sorted(order for order in orders if order not in {1, 2})
    if invalid:
        raise ValueError(f"Unsupported order(s): {invalid}. This tool supports orders 1 and 2.")
    if not orders:
        raise ValueError("At least one order must be requested.")
    return orders


def _compute_sigma1(cfg, omega):
    lin = compute_linear_response_spectrum(cfg, omega, progress=False)
    return np.asarray(lin["sigma"])                       # (nw, dim, dim)


def _compute_sigma2(cfg, omega):
    sim = QXTISimulation(config=cfg); ham = sim.build_hamiltonian(); kg = sim.build_kgrid(ham)
    wts = build_k_integration_weights(cfg, hamiltonian=ham, kgrid=kg)
    return order2_full_tensor_spectrum(ham, kg, omega, wts, cfg.susceptibility_solver,
                                       progress=True)     # (nw, dim, dim, dim)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("label")
    ap.add_argument("--grid", type=int, default=0)
    ap.add_argument("--emin", type=float, default=0.1)
    ap.add_argument("--emax", type=float, default=6.0)
    ap.add_argument("--nw", type=int, default=240)
    ap.add_argument("--pretty", default=None)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument(
        "--orders",
        default="1,2",
        help="Response orders to compute/plot: 1, 2, 1,2, or all. Use --orders 1 for chi^(1) only.",
    )
    ap.add_argument(
        "--selected-components",
        default=",".join(SELECTED_COMPONENTS),
        help="Comma-separated chi^(2) components for the extra stacked/separate plots.",
    )
    args = ap.parse_args()
    orders = _parse_orders(args.orders)

    cfg0 = QXTIConfig.from_file(args.config)
    dim = int(QXTISimulation(config=cfg0).build_hamiltonian().dimension)
    cfg = _grid_override(cfg0, args.grid, dim)
    pretty = args.pretty or args.label

    omega = np.linspace(args.emin / _EV, args.emax / _EV, args.nw)
    e_ev = omega * _EV
    out = Path("outputs/susceptibility_maps") / args.label
    out.mkdir(parents=True, exist_ok=True)
    cache_tag = "data" if orders == {1, 2} else "chi1_data" if orders == {1} else "chi2_data"
    cache = out / f"{args.label}_{cache_tag}.npz"

    tag = np.array([dim, args.grid, args.nw], dtype=float)
    erange = np.array([args.emin, args.emax], dtype=float)
    sigma1 = sigma2 = None
    if cache.exists() and not args.recompute:
        d = np.load(cache)
        if (d["tag"].shape == tag.shape and np.allclose(d["tag"], tag)
                and np.allclose(d["erange"], erange)):
            sigma1 = d["sigma1"] if "sigma1" in d.files else None
            sigma2 = d["sigma2"] if "sigma2" in d.files else None
            print(f"[{args.label}] loaded cache {cache.name}")
    if (1 in orders and sigma1 is None) or (2 in orders and sigma2 is None):
        print(f"[{args.label}] dim={dim}, grid={cfg.kgrid.k_points}, "
              f"{args.nw} energies {args.emin}-{args.emax} eV, orders={sorted(orders)} -> computing")
        if 1 in orders and sigma1 is None:
            sigma1 = _compute_sigma1(cfg, omega)
        if 2 in orders and sigma2 is None:
            sigma2 = _compute_sigma2(cfg, omega)
        payload = {"e_ev": e_ev, "tag": tag, "erange": erange}
        if sigma1 is not None:
            payload["sigma1"] = sigma1
        if sigma2 is not None:
            payload["sigma2"] = sigma2
        np.savez_compressed(cache, **payload)
        print(f"[{args.label}] cached -> {cache}")

    chi1 = None if sigma1 is None else sigma1 / (1j * omega[:, None, None])
    chi2 = None if sigma2 is None else sigma2 / (2j * omega[:, None, None, None])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    if 1 in orders:
        # ================= sigma^(1): 2x2 in-plane, modulus =================
        if orders == {1, 2}:
            inplane = [(0, 0), (0, 1), (1, 0), (1, 1)]
            fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.4))
            for ax, (i, j) in zip(axes.flat, inplane):
                _panel(ax, e_ev, sigma1[:, i, j],
                       rf"$|\sigma^{{(1)}}_{{{XYZ[i]}{XYZ[j]}}}|$",
                       r"$|\sigma^{(1)}|\ (\mathrm{arb.})$")
            fig.suptitle(rf"$\mathrm{{{pretty}}}$: $\ |\sigma^{{(1)}}_{{ij}}(\omega)|\ "
                         r"\mathrm{linear\ conductivity}$", fontsize=15)
            fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6, w_pad=1.4)
            f1 = out / f"{args.label}_sigma1_grid.png"
            fig.savefig(f1, dpi=230, facecolor="white"); plt.close(fig)
            print(f"wrote {f1}")

        # ================= chi^(1): diagonal susceptibility components =================
        for stale_path in (
            out / f"{args.label}_sigma1_unique_components_stack.png",
            out / f"{args.label}_sigma1_unique_components_single_panel.png",
        ):
            if stale_path.exists():
                stale_path.unlink()
        stale_sigma1_dir = out / "selected_sigma1_components"
        if stale_sigma1_dir.exists():
            for stale_path in stale_sigma1_dir.glob("sigma1_*.png"):
                stale_path.unlink()

        chi1_components = _available_chi1_components(dim)
        chi1_selected = _rank2_component_data(chi1, chi1_components, dim)
        f1_stack = out / f"{args.label}_chi1_diagonal_components_stack.png"
        _plot_selected_stack(
            e_ev,
            chi1_selected,
            f1_stack,
            colors=CHI1_COLORS,
            labeler=_chi1_component_label,
        )
        print(f"wrote {f1_stack}")
        f1_single = out / f"{args.label}_chi1_diagonal_components_single_panel.png"
        _plot_selected_single_panel(
            e_ev,
            chi1_selected,
            f1_single,
            colors=CHI1_COLORS,
            labeler=_chi1_component_label,
            ylabel=r"$|\chi^{(1)}|\;(\mathrm{arb.})$",
        )
        print(f"wrote {f1_single}")
        chi1_dir = out / "selected_chi1_components"
        for path in _plot_selected_separate(
            e_ev,
            chi1_selected,
            chi1_dir,
            colors=CHI1_COLORS,
            labeler=_chi1_component_label,
            file_prefix="chi1",
        ):
            print(f"wrote {path}")

    if 2 in orders:
        # ================= chi^(2): FULL tensor, modulus =================
        allpairs = [(j, k) for j in range(dim) for k in range(dim)]
        ncols, nrows = len(allpairs), dim            # dim^2 x dim
        fig, axes = plt.subplots(nrows, ncols, figsize=(max(8.0, ncols * 2.45), nrows * 3.05),
                                 squeeze=False)
        for i in range(dim):
            for c, (j, k) in enumerate(allpairs):
                _panel(axes[i][c], e_ev, chi2[:, i, j, k],
                       rf"$|\chi^{{(2)}}_{{{XYZ[i]}{XYZ[j]}{XYZ[k]}}}|$", r"$|\chi^{(2)}|$")
        ncomp = dim ** 3
        fig.suptitle(rf"$\mathrm{{{pretty}}}$: $\ |\chi^{{(2)}}_{{ijk}}(2\omega)|\ "
                     rf"\mathrm{{second\text{{-}}order\ susceptibility}}\ "
                     rf"({ncomp}\ \mathrm{{components}})$", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.6, w_pad=1.2)
        f2 = out / f"{args.label}_chi2_grid.png"
        fig.savefig(f2, dpi=230, facecolor="white"); plt.close(fig)
        print(f"wrote {f2}")

        # ================= selected chi^(2) components: stacked + separate panels =================
        selected_components = tuple(
            comp.strip().lower()
            for comp in str(args.selected_components).replace(";", ",").split(",")
            if comp.strip()
        )
        if selected_components:
            selected = _selected_component_data(chi2, selected_components, dim)
            legacy_overlay = out / f"{args.label}_chi2_selected_components_overlay.png"
            if legacy_overlay.exists():
                legacy_overlay.unlink()
            f3 = out / f"{args.label}_chi2_selected_components_stack.png"
            _plot_selected_stack(
                e_ev,
                selected,
                f3,
                colors=SELECTED_COLORS,
                labeler=_component_label,
            )
            print(f"wrote {f3}")
            f4 = out / f"{args.label}_chi2_selected_components_single_panel.png"
            _plot_selected_single_panel(
                e_ev,
                selected,
                f4,
                colors=SELECTED_COLORS,
                labeler=_component_label,
                ylabel=r"$|\chi^{(2)}|\;(\mathrm{arb.})$",
            )
            print(f"wrote {f4}")
            selected_dir = out / "selected_components"
            for path in _plot_selected_separate(
                e_ev,
                selected,
                selected_dir,
                colors=SELECTED_COLORS,
                labeler=_component_label,
                file_prefix="chi2",
            ):
                print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
