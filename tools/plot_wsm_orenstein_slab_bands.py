#!/usr/bin/env python3
"""Slab (ribbon) band structure of the Wu/Orenstein Weyl semimetal.

Instead of the surface Green function / spectral map (which broadens the states
into a colour density), this tool **diagonalizes a finite slab** and plots the
real eigen-energies E_n(k) as points/lines, coloured by how much each state is
localized on the surface (edge). Surface / Fermi-arc states therefore appear as
sharp in-gap bands highlighted by the colour, without any spectral broadening.

Geometry
--------
Break the crystal along one direction (``--normal``, default ``y``): the model is
made **finite in that direction** (a slab of ``--layers`` principal layers) and
kept **periodic in the other two**. One in-plane momentum is swept (the plot
axis) and the other is held fixed (``--k-fixed``, e.g. kz = 0). So for
``--normal y`` you get the standard **E vs kx at kz = 0** ribbon spectrum.

How it works
------------
1. Build H(kx,ky,kz) for the model from a QXTI config (params come from there).
2. Fourier-transform H along the normal into principal-layer blocks H00, H01
   (nearest-layer approximation; the tool checks it is valid for this model).
3. Assemble the block-tridiagonal slab Hamiltonian of ``--layers`` layers and
   diagonalize it at each swept momentum -> eigen-energies + eigenvectors.
4. Colour every (k, E_n) point by its weight on the outer ``--edge-layers`` on
   both terminations (0 = bulk, 1 = fully edge-localized).

Axes are always drawn in **eV** (energy) and **1/Angstrom** (momentum), matching
the LDOS plots.

Examples
--------
    python tools/plot_wsm_orenstein_slab_bands.py
    python tools/plot_wsm_orenstein_slab_bands.py --normal y --k-fixed 0.0 --layers 60
    python tools/plot_wsm_orenstein_slab_bands.py --emin -0.06 --emax 0.06
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep matplotlib caches writable for local runs.
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

from qxti.core import QXTIConfig
from qxti.core.simulation import QXTISimulation
from qxti.analytics.dos import _principal_layer_blocks

_AU_TO_EV = 27.211386245988
_BOHR_TO_ANG = 0.529177210903
_AU_K_TO_ANG_INV = 1.0 / _BOHR_TO_ANG          # 1/Bohr -> 1/Angstrom
_AXES = {"x": 0, "y": 1, "z": 2}

DEFAULT_CONFIG = "inputs/inputParams.wsm_orenstein.cfg"
DEFAULT_OUTPUT = "outputs/wsm_orenstein/ldos/slab_bands.png"


def _build_slab(H00: np.ndarray, H01: np.ndarray, nlayers: int) -> np.ndarray:
    """Block-tridiagonal finite-slab Hamiltonian from principal-layer blocks."""
    nb = int(H00.shape[0])
    nlayers = max(int(nlayers), 2)
    H = np.zeros((nlayers * nb, nlayers * nb), dtype=np.complex128)
    for layer in range(nlayers):
        s = slice(layer * nb, (layer + 1) * nb)
        H[s, s] = H00
        if layer + 1 < nlayers:
            r = slice((layer + 1) * nb, (layer + 2) * nb)
            H[s, r] = H01
            H[r, s] = H01.conj().T
    return H


def _edge_mask(nb: int, nlayers: int, edge_layers: int) -> np.ndarray:
    """Boolean mask selecting the outer ``edge_layers`` on both terminations."""
    edge_layers = min(max(int(edge_layers), 1), nlayers // 2)
    mask = np.zeros(nlayers * nb, dtype=bool)
    mask[: edge_layers * nb] = True
    mask[(nlayers - edge_layers) * nb :] = True
    return mask


def _check_nearest_layer(Hf, kpar: np.ndarray, normal: int, c_n: float, n_kn: int) -> float:
    """Return the norm of the r=2 hopping (should be ~0 for nearest-layer models)."""
    kn = np.linspace(-np.pi / c_n, np.pi / c_n, int(n_kn), endpoint=False)
    k = np.array(kpar, dtype=np.float64)
    mats = []
    for knj in kn:
        k[normal] = knj
        mats.append(np.asarray(Hf(float(k[0]), float(k[1]), float(k[2])), dtype=np.complex128))
    Hs = np.array(mats)
    H2 = (Hs * np.exp(-2j * kn * c_n)[:, None, None]).mean(axis=0)
    return float(np.max(np.abs(H2)))


def _parse_path(text: str) -> np.ndarray:
    """Parse 'kx,ky,kz ; kx,ky,kz ; ...' into an array of vertices (M,3)."""
    verts = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [float(x) for x in chunk.replace("[", "").replace("]", "").split(",") if x.strip()]
        v = np.zeros(3, dtype=np.float64)
        v[: min(3, len(vals))] = vals[: min(3, len(vals))]
        verts.append(v)
    return np.array(verts, dtype=np.float64)


def _build_kpath(vertices: np.ndarray, num_k: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Piecewise-linear path through vertices. Returns (k_points (N,3), arclength
    s (N,), tick arclengths at each vertex)."""
    seg_len = [float(np.linalg.norm(vertices[i + 1] - vertices[i])) for i in range(len(vertices) - 1)]
    total = sum(seg_len) or 1.0
    pts, ticks, s_acc = [], [0.0], 0.0
    for i, L in enumerate(seg_len):
        n = max(2, int(round(num_k * L / total)))
        t = np.linspace(0.0, 1.0, n)
        seg = (1 - t)[:, None] * vertices[i][None, :] + t[:, None] * vertices[i + 1][None, :]
        if i > 0:
            seg = seg[1:]
        pts.append(seg)
        s_acc += L
        ticks.append(s_acc)
    k_points = np.concatenate(pts, axis=0)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(k_points, axis=0), axis=1))))
    # snap ticks onto the assembled arclength
    ticks, cum = [0.0], 0
    for seg in pts:
        cum += len(seg)
        ticks.append(float(s[cum - 1]))
    return k_points, s, ticks


def compute_slab_bands(args) -> dict:
    """Diagonalize the slab along the momentum path and return bands + edge weight."""
    config = QXTIConfig.from_file(Path(args.config).expanduser())
    hamiltonian = QXTISimulation(config=config).build_hamiltonian()
    Hf = hamiltonian._matrix_at
    nb = int(hamiltonian.basis_size)
    bounds = hamiltonian.reciprocal_box_bounds()

    normal = _AXES[args.normal]
    c_n = float(np.pi / max(abs(bounds[normal][1]), 1e-12))
    inplane = [a for a in range(3) if a != normal]
    path_axis = inplane[0]
    fixed_axis = inplane[1] if len(inplane) > 1 else None

    # Build the momentum list: a multi-segment path (--path) or a straight sweep.
    if args.path:
        vertices = _parse_path(args.path)
        k_points, x_coord, ticks = _build_kpath(vertices, int(args.nk))
        tick_labels = [s.strip() for s in args.path_labels.split(",")] if args.path_labels else []
    else:
        k_min = float(args.k_min) if args.k_min is not None else float(bounds[path_axis][0])
        k_max = float(args.k_max) if args.k_max is not None else float(bounds[path_axis][1])
        sweep = np.linspace(k_min, k_max, int(args.nk))
        k_points = np.zeros((sweep.size, 3), dtype=np.float64)
        k_points[:, path_axis] = sweep
        if fixed_axis is not None:
            k_points[:, fixed_axis] = float(args.k_fixed)
        x_coord = sweep
        ticks, tick_labels = [], []

    # Validity of the nearest-layer (principal-layer) truncation.
    r2 = _check_nearest_layer(Hf, k_points[len(k_points) // 2], normal, c_n, args.fourier_points)
    if r2 > 1e-6:
        print(f"[slab] WARNING: non-nearest-layer hopping along {args.normal} "
              f"(||H_r=2|| = {r2:.2e}); the slab is an approximation.", flush=True)

    nlayers = int(args.layers)
    edge_mask = _edge_mask(nb, nlayers, args.edge_layers)

    n = k_points.shape[0]
    energies = np.empty((n, nb * nlayers), dtype=np.float64)
    edge_weight = np.empty((n, nb * nlayers), dtype=np.float64)

    for i in range(n):
        H00, H01 = _principal_layer_blocks(Hf, k_points[i], normal, c_n, int(args.fourier_points))
        H = _build_slab(H00, H01, nlayers)
        ev, U = np.linalg.eigh(H)
        prob = (U.conj() * U).real                      # (dim, state)
        energies[i] = ev
        edge_weight[i] = prob[edge_mask].sum(axis=0)     # per state
        if args.progress and (i % max(1, n // 10) == 0 or i == n - 1):
            print(f"[slab] {i + 1}/{n} k-points", flush=True)

    return {
        "x_coord": np.asarray(x_coord, dtype=np.float64),
        "energies": energies,
        "edge_weight": edge_weight,
        "path_axis": path_axis,
        "fixed_axis": fixed_axis,
        "normal": normal,
        "nlayers": nlayers,
        "k_fixed": float(args.k_fixed),
        "is_path": bool(args.path),
        "ticks": ticks,
        "tick_labels": tick_labels,
        "model_name": str(getattr(hamiltonian, "model_name", "") or "model"),
    }


def plot_slab_bands(data: dict, args) -> Path:
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    k = np.asarray(data["k_path"], dtype=float) * _AU_K_TO_ANG_INV      # 1/Ang
    E = np.asarray(data["energies"], dtype=float) * _AU_TO_EV           # eV
    w = np.asarray(data["edge_weight"], dtype=float)                   # 0..1
    axis_name = "xyz"[data["path_axis"]]
    normal_name = "xyz"[data["normal"]]
    fixed_name = "xyz"[data["fixed_axis"]] if data["fixed_axis"] is not None else None

    # One scatter point per (k, band). Broadcast k across bands.
    K = np.repeat(k[:, None], E.shape[1], axis=1)

    emin = args.emin * _AU_TO_EV if args.emin is not None else float(np.min(E))
    emax = args.emax * _AU_TO_EV if args.emax is not None else float(np.max(E))
    sel = (E >= emin) & (E <= emax)

    # Draw bulk-like states first (low edge weight), surface states on top so they
    # are not hidden. Colour encodes surface localization.
    order = np.argsort(w[sel])
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    sc = ax.scatter(
        K[sel][order], E[sel][order],
        c=w[sel][order], cmap=args.cmap, s=float(args.point_size),
        vmin=0.0, vmax=float(np.clip(w[sel].max(), 1e-6, 1.0)),
        linewidths=0.0, rasterized=True,
    )
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(rf"$k_{axis_name}\;(\mathrm{{\AA}}^{{-1}})$")
    ax.set_ylabel(r"$E\;(\mathrm{eV})$")
    ax.set_xlim(float(k.min()), float(k.max()))
    ax.set_ylim(emin, emax)
    title = rf"Slab abierto en ${normal_name}$ ({data['nlayers']} capas)"
    if fixed_name is not None:
        title += rf",  $k_{fixed_name}={data['k_fixed'] * _AU_K_TO_ANG_INV:.3g}\;\mathrm{{\AA}}^{{-1}}$"
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("surface (edge) localization")

    fig.tight_layout()
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Slab (ribbon) band structure coloured by surface localization.")
    p.add_argument("config", nargs="?", default=DEFAULT_CONFIG, help="QXTI config that defines the model/params.")
    p.add_argument("--normal", choices=("x", "y", "z"), default="y", help="Direction made finite (the slab normal).")
    p.add_argument("--k-fixed", type=float, default=0.0, help="Fixed value of the OTHER in-plane momentum (a.u.), e.g. kz.")
    p.add_argument("--layers", type=int, default=40, help="Number of principal layers in the slab.")
    p.add_argument("--edge-layers", type=int, default=2, help="Outer layers per side used for the surface-localization colour.")
    p.add_argument("--nk", type=int, default=400, help="Number of swept-momentum points.")
    p.add_argument("--k-min", type=float, default=None, help="Path lower limit (a.u.); default = -pi/a.")
    p.add_argument("--k-max", type=float, default=None, help="Path upper limit (a.u.); default = +pi/a.")
    p.add_argument("--emin", type=float, default=None, help="Energy window lower limit (a.u.); default = auto.")
    p.add_argument("--emax", type=float, default=None, help="Energy window upper limit (a.u.); default = auto.")
    p.add_argument("--fourier-points", type=int, default=200, help="k-points to Fourier H along the normal into H00,H01.")
    p.add_argument("--cmap", default="plasma", help="Matplotlib colormap for the surface-localization colour.")
    p.add_argument("--point-size", type=float, default=5.0, help="Scatter point size.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG path.")
    p.add_argument("--no-progress", dest="progress", action="store_false", help="Silence progress prints.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    print(
        f"[slab] model from {args.config}: open along {args.normal}, {args.layers} layers, "
        f"sweeping k with the other in-plane momentum fixed at {args.k_fixed}.",
        flush=True,
    )
    data = compute_slab_bands(args)
    out = plot_slab_bands(data, args)
    print(f"[slab] saved: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
