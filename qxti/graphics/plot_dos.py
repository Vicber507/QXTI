"""Plots for the density-of-states branch (``--family ldos``).

Reads the ``ldos.npz`` dataset written by :class:`qxti.core.LDOSRunner` and
produces publication-style figures:

  * ``dos_total``       total DOS g(E).
  * ``dos_surface_bulk`` comparación bulk vs surface en el mismo eje.
  * ``dos_projected``   orbital/sublattice-projected bulk PDOS (stacked + total).
  * ``dos_projected_surface`` orbital/sublattice-projected surface PDOS.
  * ``spectral_map``    momentum-resolved spectral function A(k, E) heatmap.
  * ``spectral_plane``  constant-energy A(kx, ky; E0) map.
  * ``finite_spectrum`` finite-state spectrum colored by edge localization.
  * ``finite_ldos_map`` site-resolved LDOS(r, E0) for a finite 2-D plaque.

Datasets store everything in **atomic units**, but the figures are ALWAYS drawn
with **energy in electron-volts** and **momentum in inverse angstrom (1/Ang)**.
The Fermi level is marked; all text uses the shared paper style.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qxti.graphics.plot_susceptibility_tensor import apply_paper_style

_AU_TO_EV = 27.211386245988
_BOHR_TO_ANG = 0.529177210903
# Reciprocal-length conversion: 1/Bohr (atomic units) -> 1/Angstrom.
_AU_K_TO_ANG_INV = 1.0 / _BOHR_TO_ANG
_DOS_FIGSIZE = (4.4, 7.6)
_DOS_BOX_ASPECT = 1.55


def _energy_axis(data: dict[str, Any]) -> tuple[np.ndarray, str, float]:
    """Energies in eV (always), with the axis label and the a.u.->eV scale."""
    energies = np.asarray(data["energies"], dtype=float)
    scale = float(data.get("au_to_ev", _AU_TO_EV))
    return energies * scale, r"$E\;(\mathrm{eV})$", scale


def _display_dos_curve(data: dict[str, Any], *, key: str = "dos") -> np.ndarray:
    """Return the DOS-like curve to display."""
    return np.asarray(data[key], dtype=float)


def _save_fixed_canvas(fig, output_path: str | Path, *, dpi: int) -> Path:
    """Save DOS-family figures with a fixed canvas ratio.

    Avoid ``bbox_inches='tight'`` here: it can slightly change the final pixel
    ratio depending on legends/text, which is exactly what makes neighboring DOS
    plots look mismatched even when they share the same figsize.
    """
    out = Path(output_path)
    fig.tight_layout(pad=0.8)
    fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches=None, pad_inches=0.0)
    return out


def _set_dos_axes_limits(ax, energies: np.ndarray, values: np.ndarray) -> None:
    ax.set_ylim(float(energies[0]), float(energies[-1]))
    vmax = float(np.nanmax(values)) if values.size else 0.0
    right = 1.04 * vmax if vmax > 0.0 else 1.0
    ax.set_xlim(0.0, right)


def _set_dos_box_aspect(ax) -> None:
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(_DOS_BOX_ASPECT)


def plot_dos_total(data: dict[str, Any], output_path: str | Path, *, dpi: int = 300) -> Path:
    import matplotlib.pyplot as plt

    apply_paper_style()
    energies, x_label, scale = _energy_axis(data)
    dos = _display_dos_curve(data)
    fermi = float(data.get("fermi_level", 0.0)) * scale
    # Convert the per-(a.u. energy) density to per-eV.
    dos_axis = dos / scale

    method = str(data.get("method", ""))
    surface = method == "surface" and "surface_dos" not in data
    finite = method == "finite"
    glabel = (
        r"$g_\mathrm{surf}(E)$" if surface else
        r"$g_\mathrm{finite}(E)$" if finite else
        r"$g(E)$"
    )
    ylabel = (
        r"$g_\mathrm{surf}(E)\;(\mathrm{arb.})$" if surface else
        r"$g_\mathrm{finite}(E)\;(\mathrm{states}/\mathrm{eV})$" if finite else
        r"$g(E)\;(\mathrm{states}/\mathrm{eV}/\mathrm{cell})$"
    )

    # Tall layout: energy on the long (y) axis, DOS on the short (x) axis.
    fig, ax = plt.subplots(figsize=_DOS_FIGSIZE)
    ax.plot(dos_axis, energies, color="#1f4e8c", label=glabel)
    ax.fill_betweenx(energies, 0.0, dos_axis, color="#1f4e8c", alpha=0.12)
    ax.axhline(fermi, color="0.4", linestyle="--", linewidth=1.0, label=r"$E_F$")
    ax.set_xlabel(ylabel)
    ax.set_ylabel(x_label)
    _set_dos_axes_limits(ax, energies, dos_axis)
    _set_dos_box_aspect(ax)

    ax.legend(loc="upper right")
    out = _save_fixed_canvas(fig, output_path, dpi=dpi)
    plt.close(fig)
    return out


def plot_dos_surface_bulk(data: dict[str, Any], output_path: str | Path, *, dpi: int = 300) -> Path | None:
    """Compare bulk DOS and finite-slab surface contribution in one panel."""
    if "surface_dos" not in data:
        return None
    import matplotlib.pyplot as plt

    apply_paper_style()
    energies, x_label, scale = _energy_axis(data)
    bulk_dos = _display_dos_curve(data, key="dos") / scale
    fermi = float(data.get("fermi_level", 0.0)) * scale
    surface_layers = max(int(data.get("surface_compare_layers", 1) or 1), 1)
    surface_dos = _display_dos_curve(data, key="surface_dos") / scale
    surface_contribution = surface_dos / float(surface_layers)

    fig, ax = plt.subplots(figsize=_DOS_FIGSIZE)
    ax.plot(bulk_dos, energies, color="#1f4e8c", linewidth=1.6, label="bulk")
    ax.fill_betweenx(energies, 0.0, bulk_dos, color="#1f4e8c", alpha=0.10)
    ax.plot(
        surface_contribution,
        energies,
        color="#c96b4b",
        linewidth=1.6,
        label=rf"surface / {surface_layers}",
    )
    ax.fill_betweenx(energies, 0.0, surface_contribution, color="#c96b4b", alpha=0.08)
    compare_values = np.maximum(bulk_dos, surface_contribution)
    ax.axhline(fermi, color="0.4", linestyle="--", linewidth=1.0, label=r"$E_F$")
    ax.set_xlabel(r"$g_\mathrm{bulk}(E),\,g_\mathrm{surf}(E)/N_\mathrm{layers}\;(\mathrm{states}/\mathrm{eV})$")
    ax.set_ylabel(x_label)
    _set_dos_axes_limits(ax, energies, compare_values)
    _set_dos_box_aspect(ax)
    ax.legend(loc="upper right")

    out = _save_fixed_canvas(fig, output_path, dpi=dpi)
    plt.close(fig)
    return out


def plot_dos_projected(
    data: dict[str, Any],
    output_path: str | Path,
    *,
    dpi: int = 300,
    pdos_key: str = "pdos",
    total_key: str = "dos",
    total_label: str | None = None,
    x_label_override: str | None = None,
) -> Path | None:
    if not bool(data.get("projected", False)) or pdos_key not in data or total_key not in data:
        return None
    import matplotlib.pyplot as plt

    apply_paper_style()
    energies, x_label, scale = _energy_axis(data)
    pdos = np.asarray(data[pdos_key], dtype=float) / scale
    dos_axis = _display_dos_curve(data, key=total_key) / scale
    labels = [str(x) for x in data.get("orbital_labels", [f"orb {i}" for i in range(pdos.shape[0])])]
    fermi = float(data.get("fermi_level", 0.0)) * scale
    finite = str(data.get("method", "")) == "finite"
    if total_label is None:
        total_label = r"total $g(E)$" if pdos_key == "pdos" else r"total $g_\mathrm{surf}(E)$"

    fig, ax = plt.subplots(figsize=_DOS_FIGSIZE)
    cmap = plt.get_cmap("viridis")
    nb = pdos.shape[0]
    base = np.zeros_like(dos_axis)
    for i in range(nb):
        color = cmap(i / max(nb - 1, 1))
        ax.fill_betweenx(energies, base, base + pdos[i], color=color, alpha=0.65,
                         label=labels[i] if nb <= 12 else None)
        base = base + pdos[i]
    ax.plot(dos_axis, energies, color="black", linewidth=1.2, label=total_label)
    ax.axhline(fermi, color="0.4", linestyle="--", linewidth=1.0, label=r"$E_F$")
    ax.set_ylabel(x_label)
    if x_label_override is not None:
        ax.set_xlabel(x_label_override)
    else:
        ax.set_xlabel(
            r"$g_\alpha(E)\;(\mathrm{states}/\mathrm{eV})$"
            if finite or pdos_key != "pdos"
            else r"$g_\alpha(E)\;(\mathrm{states}/\mathrm{eV}/\mathrm{cell})$"
        )
    _set_dos_axes_limits(ax, energies, dos_axis)
    _set_dos_box_aspect(ax)
    if nb <= 12:
        ax.legend(loc="upper right")
    out = _save_fixed_canvas(fig, output_path, dpi=dpi)
    plt.close(fig)
    return out


def _plot_surface_side_pdos(
    data: dict[str, Any],
    output_path: str | Path,
    *,
    side: str,
    dpi: int = 300,
) -> Path | None:
    key = f"surface_{side}_pdos"
    if str(data.get("method", "")) != "surface" or key not in data:
        return None
    import matplotlib.pyplot as plt

    apply_paper_style()
    energies, x_label, scale = _energy_axis(data)
    fermi = float(data.get("fermi_level", 0.0)) * scale
    pdos = np.asarray(data[key], dtype=float) / scale
    total_dos = np.asarray(pdos.sum(axis=0), dtype=float)
    labels = [str(x) for x in data.get("orbital_labels", [f"orb {i}" for i in range(pdos.shape[0])])]

    fig, ax = plt.subplots(figsize=_DOS_FIGSIZE)
    cmap = plt.get_cmap("viridis")
    for i, label in enumerate(labels):
        color = cmap(i / max(len(labels) - 1, 1))
        ax.fill_betweenx(energies, 0.0, pdos[i], color=color, alpha=0.18)
        ax.plot(pdos[i], energies, color=color, linewidth=1.3, label=label)
    ax.plot(total_dos, energies, color="black", linewidth=1.2, label="total")
    ax.axhline(fermi, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_ylabel(x_label)
    ax.set_xlabel(r"$g_\alpha(E)\;(\mathrm{states}/\mathrm{eV})$")
    _set_dos_axes_limits(ax, energies, total_dos)
    _set_dos_box_aspect(ax)
    ax.text(
        0.015,
        0.98,
        f"{side.capitalize()} Surface Orbital Projection",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.legend(loc="upper right")

    out = _save_fixed_canvas(fig, output_path, dpi=dpi)
    plt.close(fig)
    return out


def plot_surface_side_dos(data: dict[str, Any], output_path: str | Path, *, dpi: int = 300) -> Path | None:
    if str(data.get("method", "")) != "surface" or "surface_bottom_dos" not in data:
        return None
    import matplotlib.pyplot as plt

    apply_paper_style()
    energies, x_label, scale = _energy_axis(data)
    fermi = float(data.get("fermi_level", 0.0)) * scale
    bottom_dos = np.asarray(data["surface_bottom_dos"], dtype=float) / scale
    top_dos = np.asarray(data["surface_top_dos"], dtype=float) / scale
    total_dos = _display_dos_curve(data) / scale

    fig, ax = plt.subplots(figsize=_DOS_FIGSIZE)
    ax.plot(bottom_dos, energies, color="#1f4e8c", linewidth=1.5, label="bottom")
    ax.plot(top_dos, energies, color="#d97904", linewidth=1.5, label="top")
    ax.plot(total_dos, energies, color="black", linewidth=1.1, alpha=0.85, label="both")
    ax.fill_betweenx(energies, 0.0, bottom_dos, color="#1f4e8c", alpha=0.10)
    ax.fill_betweenx(energies, 0.0, top_dos, color="#d97904", alpha=0.10)
    ax.axhline(fermi, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_ylabel(x_label)
    ax.set_xlabel(r"$g_\mathrm{surf}(E)\;(\mathrm{states}/\mathrm{eV})$")
    _set_dos_axes_limits(ax, energies, total_dos)
    _set_dos_box_aspect(ax)
    ax.text(0.015, 0.98, "Bottom / Top / Both", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.legend(loc="upper right")
    out = _save_fixed_canvas(fig, output_path, dpi=dpi)
    plt.close(fig)
    return out


def plot_finite_spectrum(data: dict[str, Any], output_path: str | Path, *, dpi: int = 300) -> Path | None:
    if str(data.get("method", "")) != "finite" or "finite_eigenvalues" not in data:
        return None
    import matplotlib.pyplot as plt

    apply_paper_style()
    scale = float(data.get("au_to_ev", _AU_TO_EV))
    eigenvalues = np.asarray(data["finite_eigenvalues"], dtype=float) * scale
    edge_weight = np.asarray(data.get("finite_edge_weight", np.zeros_like(eigenvalues)), dtype=float)
    fermi = float(data.get("fermi_level", 0.0)) * scale

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.arange(eigenvalues.size, dtype=float)
    sc = ax.scatter(
        x,
        eigenvalues,
        c=edge_weight,
        cmap="magma",
        s=18,
        linewidths=0.0,
        alpha=0.95,
    )
    ax.axhline(fermi, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlabel("State Index")
    ax.set_ylabel(r"$E\;(\mathrm{eV})$")
    ax.set_xlim(-1.0, float(max(eigenvalues.size - 1, 1)))
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Edge Weight")
    out = _save_fixed_canvas(fig, output_path, dpi=dpi)
    plt.close(fig)
    return out


def plot_finite_ldos_map(data: dict[str, Any], output_path: str | Path, *, dpi: int = 300) -> Path | None:
    if str(data.get("method", "")) != "finite" or "finite_positions" not in data:
        return None
    import matplotlib.pyplot as plt

    apply_paper_style()
    scale = float(data.get("au_to_ev", _AU_TO_EV))
    positions = np.asarray(data["finite_positions"], dtype=float) * _BOHR_TO_ANG
    site_ldos = np.asarray(data["finite_site_ldos"], dtype=float) / scale
    boundary = np.asarray(data.get("finite_boundary_mask", np.zeros(site_ldos.size, dtype=bool)), dtype=bool)
    energy_ev = float(data.get("finite_ldos_energy", 0.0)) * scale

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=46,
        c="#d9dce6",
        alpha=0.40,
        linewidths=0.0,
        marker="h",
        zorder=1,
    )
    sc = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=62,
        c=site_ldos,
        cmap="magma",
        linewidths=0.0,
        marker="h",
        zorder=2,
    )
    if np.any(boundary):
        ax.scatter(
            positions[boundary, 0],
            positions[boundary, 1],
            s=68,
            facecolors="none",
            edgecolors="black",
            linewidths=0.25,
            marker="h",
            zorder=3,
        )
    ax.set_xlabel(r"$x\;(\mathrm{\AA})$")
    ax.set_ylabel(r"$y\;(\mathrm{\AA})$")
    ax.set_aspect("equal")
    ax.text(
        0.02,
        0.98,
        rf"$E={energy_ev:.3f}\,\mathrm{{eV}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.5},
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\mathrm{LDOS}(\mathbf{r},E)\;(\mathrm{states}/\mathrm{eV})$")
    out = _save_fixed_canvas(fig, output_path, dpi=dpi)
    plt.close(fig)
    return out


# Colorbar label for the spectral maps: the quantity we plot, A(k,E) = -Im Tr G/pi.
_SPECTRAL_CBAR_LABEL = r"$-\mathrm{Im}\,\mathrm{Tr}\,G(\mathbf{k},\omega+i\eta)/\pi$"


def _single_path_direction(data: dict[str, Any]) -> int | None:
    """Return the axis index (0/1/2) when the k-path varies in exactly one
    Cartesian direction, else None (multi-segment / high-symmetry path)."""
    path_k = np.asarray(data.get("spectral_path_k", np.empty((0, 3))), dtype=float)
    if path_k.ndim == 2 and path_k.shape[0] > 1:
        spans = path_k.max(axis=0) - path_k.min(axis=0)
        varying = np.where(spans > 1e-9 * (np.abs(path_k).max() + 1e-12))[0]
        if varying.size == 1:
            return int(varying[0])
    return None


def _build_alpha_colormap(name: str = "Blues", alpha_min: float = 0.0, alpha_max: float = 1.0, n: int = 256):
    """Colormap with linear transparency (min -> alpha_min, max -> alpha_max).

    Copied from the original surface-Green example so the spectral maps share its
    exact look (default ``Blues`` fading in from transparent).
    """
    import matplotlib.colors
    import matplotlib.pyplot as plt

    base = plt.get_cmap(name)
    colors = base(np.linspace(0.0, 1.0, int(n)))
    a0 = float(np.clip(alpha_min, 0.0, 1.0))
    a1 = float(np.clip(alpha_max, 0.0, 1.0))
    if a1 < a0:
        a0, a1 = a1, a0
    colors[:, -1] = a0 + (a1 - a0) * np.linspace(0.0, 1.0, int(n))
    return matplotlib.colors.ListedColormap(colors, name=f"{name}_alpha")


def _spectral_style(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve the example's spectral plot style from the dataset metadata."""
    return {
        "cmap": str(data.get("spectral_cmap", "Blues")),
        "log_scale": bool(data.get("spectral_log_scale", True)),
        "log_vmin": data.get("spectral_log_vmin", -1.5),
        "log_vmax": data.get("spectral_log_vmax", 3.3),
        "alpha_min": float(data.get("spectral_alpha_min", 0.0)),
        "alpha_max": float(data.get("spectral_alpha_max", 1.0)),
    }


def _scaled_imshow_values(values: np.ndarray, style: dict[str, Any]) -> tuple[np.ndarray, float | None, float | None]:
    """Apply log10 (with a small floor) and resolve the fixed color range."""
    data = np.asarray(values, dtype=float)
    if style["log_scale"]:
        eps = 1e-14
        data = np.log10(np.maximum(data, eps))
        vmin = -1.5 if style["log_vmin"] is None else float(style["log_vmin"])
        vmax = 3.3 if style["log_vmax"] is None else float(style["log_vmax"])
    else:
        vmin = None if style["log_vmin"] is None else float(style["log_vmin"])
        vmax = None if style["log_vmax"] is None else float(style["log_vmax"])
    if vmin is not None and vmax is not None and vmax <= vmin:
        vmax = vmin + 1.0
    return data, vmin, vmax


def _spectral_map_x_axis(data: dict[str, Any], spectral: np.ndarray) -> dict[str, Any]:
    """Resolve the displayed x-axis for A(k,E) in inverse angstroms."""
    path_k = np.asarray(data.get("spectral_path_k", np.empty((0, 3))), dtype=float)
    ticks = np.asarray(data.get("spectral_path_ticks", []), dtype=float) * _AU_K_TO_ANG_INV
    tick_labels = [str(x) for x in data.get("spectral_path_tick_labels", [])]
    direction = _single_path_direction(data)

    if direction is not None and path_k.shape[0] == spectral.shape[0]:
        return {
            "x": path_k[:, direction] * _AU_K_TO_ANG_INV,
            "bottom_label": rf"$k_{'xyz'[direction]}\;(\mathrm{{\AA}}^{{-1}})$",
            "top_label": None,
            "ticks": ticks,
            "tick_labels": tick_labels,
            "use_symmetry_ticks": False,
        }

    x = np.asarray(data.get("spectral_path_s", np.arange(spectral.shape[0])), dtype=float) * _AU_K_TO_ANG_INV
    symmetry_ticks = bool(ticks.size and len(tick_labels) == ticks.size)
    return {
        "x": x,
        "bottom_label": "" if symmetry_ticks else r"$s_{\mathbf{k}}\;(\mathrm{\AA}^{-1})$",
        "top_label": (r"$s_{\mathbf{k}}\;(\mathrm{\AA}^{-1})$" if symmetry_ticks else None),
        "ticks": ticks,
        "tick_labels": tick_labels,
        "use_symmetry_ticks": symmetry_ticks,
    }


def plot_spectral_map(data: dict[str, Any], output_path: str | Path, *, dpi: int = 300) -> Path | None:
    """E (eV) vs k (1/Ang) spectral map A(k, E) along the path (mode 1)."""
    if "spectral" not in data:
        return None
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    apply_paper_style()
    energies, y_label, scale = _energy_axis(data)
    spectral = np.asarray(data["spectral"], dtype=float)          # (num_k, NE)
    fermi = float(data.get("fermi_level", 0.0)) * scale

    style = _spectral_style(data)
    cmap = _build_alpha_colormap(style["cmap"], style["alpha_min"], style["alpha_max"])
    z, vmin, vmax = _scaled_imshow_values(spectral.T, style)
    axis_meta = _spectral_map_x_axis(data, spectral)
    x = np.asarray(axis_meta["x"], dtype=float)
    ticks = np.asarray(axis_meta["ticks"], dtype=float)
    tick_labels = list(axis_meta["tick_labels"])
    symmetry_ticks = bool(axis_meta["use_symmetry_ticks"])

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    extent = (float(x[0]), float(x[-1]), float(energies[0]), float(energies[-1]))
    im = ax.imshow(z, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axhline(fermi, color="0.25", linestyle="--", linewidth=1.0, alpha=0.8)

    if symmetry_ticks:
        for tpos in ticks[1:-1]:
            ax.axvline(float(tpos), color="0.35", linewidth=0.8, alpha=0.7)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        top = ax.secondary_xaxis("top")
        top.set_xlabel(str(axis_meta["top_label"]))
        top.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel(str(axis_meta["bottom_label"]))
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylabel(y_label)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(_SPECTRAL_CBAR_LABEL)
    fig.tight_layout()
    out = Path(output_path)
    fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def plot_spectral_plane(data: dict[str, Any], output_path: str | Path, *, dpi: int = 300) -> Path | None:
    """kx vs ky (1/Ang) constant-energy spectral map (mode 2)."""
    if "spectral_plane" not in data:
        return None
    import matplotlib.pyplot as plt

    apply_paper_style()
    plane = np.asarray(data["spectral_plane"], dtype=float)       # (nky, nkx)
    kx = np.asarray(data["spectral_plane_kx"], dtype=float) * _AU_K_TO_ANG_INV
    ky = np.asarray(data["spectral_plane_ky"], dtype=float) * _AU_K_TO_ANG_INV
    e0_ev = float(data.get("spectral_plane_energy", 0.0)) * float(data.get("au_to_ev", _AU_TO_EV))

    style = _spectral_style(data)
    cmap = _build_alpha_colormap(style["cmap"], style["alpha_min"], style["alpha_max"])
    z, vmin, vmax = _scaled_imshow_values(plane, style)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    extent = (float(kx[0]), float(kx[-1]), float(ky[0]), float(ky[-1]))
    im = ax.imshow(z, origin="lower", aspect="equal", extent=extent, cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_xlabel(r"$k_x\;(\mathrm{\AA}^{-1})$")
    ax.set_ylabel(r"$k_y\;(\mathrm{\AA}^{-1})$")
    ax.set_title(rf"$E = {e0_ev:.3g}\;\mathrm{{eV}}$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(_SPECTRAL_CBAR_LABEL)
    fig.tight_layout()
    out = Path(output_path)
    fig.savefig(out, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out
