#!/usr/bin/env python3
r"""Pseudo-3D 'waterfall' of HHG spectra vs the laser ellipse angle phi_x.

Each completed CMD run in a phi_x sweep provides one 2D HHG spectrum |J(omega)|.
This stacks them into depth along a phi_x axis: x = harmonic order (omega/omega0),
z = log10 |J| (harmonic modulus), y (into the page) = phi_x. Every spectrum is a
filled curve coloured by phi_x, occluding the ones behind it (classic waterfall).

Data source: tools/run_parameter_sweep.py output, one folder per phi_x with
``cmd/data/current_spectrum.npz`` (keys omega_axis, current_total_magnitude) and
``input.cfg`` ([laser] omega = omega0).
"""
from __future__ import annotations

import argparse
import configparser
import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SWEEP = Path("outputs/sweeps/orenstein_phix")
OUT = Path("outputs/hhg_waterfall")


def _phix_of(name: str):
    m = re.search(r"phix=([-\d.]+)deg", name)
    return float(m.group(1)) if m else None


def _omega0(run: Path) -> float:
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cfg.optionxform = str
    cfg.read(run / "input.cfg")
    return float(cfg["laser"]["omega"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(SWEEP))
    ap.add_argument("--xmax", type=float, default=3.5, help="max harmonic order")
    ap.add_argument("--floor", type=float, default=5.0, help="decades below peak to show")
    ap.add_argument("--cmap", default="Blues")
    ap.add_argument("--ncurves", type=int, default=12, help="how many phi_x spectra to stack")
    ap.add_argument("--alpha", type=float, default=0.6, help="fill transparency")
    ap.add_argument("--elev", type=float, default=26.0)
    ap.add_argument("--azim", type=float, default=-66.0)
    args = ap.parse_args()

    sweep = Path(args.sweep)
    runs = []
    for d in sorted(sweep.iterdir()):
        if not d.is_dir():
            continue
        px = _phix_of(d.name)
        npz = d / "cmd" / "data" / "current_spectrum.npz"
        if px is None or not npz.exists():
            continue
        runs.append((px, npz, d))
    runs.sort(key=lambda r: r[0])
    if not runs:
        print(f"no completed runs in {sweep}")
        return 1
    if args.ncurves > 0 and len(runs) > args.ncurves:
        idx = np.unique(np.linspace(0, len(runs) - 1, args.ncurves).round().astype(int))
        runs = [runs[i] for i in idx]
    omega0 = _omega0(runs[0][2])
    print(f"[waterfall] {len(runs)} spectra, omega0={omega0:.5f} a.u., "
          f"phi_x {runs[0][0]:.1f}..{runs[-1][0]:.1f} deg")

    curves = []          # (phix, x_order, logJ)
    gmax = -np.inf
    for px, npz, _d in runs:
        d = np.load(npz)
        w = np.asarray(d["omega_axis"], dtype=float)
        J = np.asarray(d["current_total_magnitude"], dtype=float)
        order = w / omega0
        m = (order >= 0.0) & (order <= args.xmax) & np.isfinite(J)
        xo, Jm = order[m], J[m]
        idx = np.argsort(xo); xo, Jm = xo[idx], Jm[idx]
        if xo.size > 700:                              # downsample for light polygons
            s = xo.size // 700
            xo, Jm = xo[::s], Jm[::s]
        logJ = np.log10(np.clip(Jm, 1e-300, None))
        gmax = max(gmax, float(logJ.max()))
        curves.append((px, xo, logJ))

    zfloor = gmax - args.floor
    phis = np.array([c[0] for c in curves])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib import cm, colors as mcolors
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    norm = mcolors.Normalize(vmin=phis.min(), vmax=phis.max())
    cmap = plt.get_cmap(args.cmap)

    verts, facecolors, edgecolors, zs = [], [], [], []
    for px, xo, logJ in curves:
        z = np.clip(logJ, zfloor, None)
        poly = [(xo[0], zfloor)] + list(zip(xo, z)) + [(xo[-1], zfloor)]
        verts.append(poly)
        rgba = list(cmap(norm(px)))
        edgecolors.append((rgba[0] * 0.55, rgba[1] * 0.55, rgba[2] * 0.55, 0.95))
        rgba[3] = args.alpha
        facecolors.append(tuple(rgba))
        zs.append(px)

    fig = plt.figure(figsize=(10.5, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    pc = PolyCollection(verts, facecolors=facecolors, edgecolors=edgecolors,
                        linewidths=0.9)
    ax.add_collection3d(pc, zs=zs, zdir="y")

    ax.set_xlim(0, args.xmax)
    ax.set_ylim(phis.min(), phis.max())
    ax.set_zlim(zfloor, gmax + 0.15 * args.floor)
    ax.set_xlabel(r"$\mathrm{harmonic\ order}\ \ \omega/\omega_0$", labelpad=12)
    ax.set_ylabel(r"$\phi_x\ (\mathrm{deg})$", labelpad=14)
    ax.set_zlabel(r"$\log_{10}|J(\omega)|\ (\mathrm{arb.})$", labelpad=8)
    ax.set_title(r"$\mathrm{HHG\ spectra\ waterfall\ vs\ laser\ ellipse\ angle}\ \phi_x$"
                 "\n" r"$\mathrm{WSM\ (Orenstein),\ elliptical\ drive\ in\ the}\ k_xk_y\ \mathrm{plane}$",
                 fontsize=13, pad=2)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_box_aspect((1.0, 1.25, 0.6))
    try:   # clean, light "paper" grid: transparent panes + soft grid lines
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0.0))
            axis.pane.set_edgecolor((0.8, 0.8, 0.8, 1.0))
            axis.pane.set_linewidth(0.6)
            axis._axinfo["grid"].update(color=(0.82, 0.82, 0.86), linewidth=0.5, linestyle="-")
    except Exception:
        pass

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap); mappable.set_array([])
    cb = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.11)
    cb.set_label(r"$\phi_x\ (\mathrm{deg})$")

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "hhg_waterfall_phix.png"
    fig.savefig(f, dpi=230, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
