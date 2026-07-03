#!/usr/bin/env python3
r"""Paper-quality susceptibility / conductivity MODULUS maps for a QXTI model.

For a given config it computes, over a photon-energy axis, the analytic
    sigma^(1)_{ij}(omega)        -- linear conductivity
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


def _compute(cfg, dim, omega):
    lin = compute_linear_response_spectrum(cfg, omega, progress=False)
    sigma1 = np.asarray(lin["sigma"])                       # (nw, dim, dim)
    sim = QXTISimulation(config=cfg); ham = sim.build_hamiltonian(); kg = sim.build_kgrid(ham)
    wts = build_k_integration_weights(cfg, hamiltonian=ham, kgrid=kg)
    sigma2 = order2_full_tensor_spectrum(ham, kg, omega, wts, cfg.susceptibility_solver,
                                         progress=True)     # (nw, dim, dim, dim)
    return sigma1, sigma2


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
    args = ap.parse_args()

    cfg0 = QXTIConfig.from_file(args.config)
    dim = int(QXTISimulation(config=cfg0).build_hamiltonian().dimension)
    cfg = _grid_override(cfg0, args.grid, dim)
    pretty = args.pretty or args.label

    omega = np.linspace(args.emin / _EV, args.emax / _EV, args.nw)
    e_ev = omega * _EV
    out = Path("outputs/susceptibility_maps") / args.label
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"{args.label}_data.npz"

    tag = np.array([dim, args.grid, args.nw], dtype=float)
    erange = np.array([args.emin, args.emax], dtype=float)
    sigma1 = sigma2 = None
    if cache.exists() and not args.recompute:
        d = np.load(cache)
        if (d["tag"].shape == tag.shape and np.allclose(d["tag"], tag)
                and np.allclose(d["erange"], erange)):
            sigma1, sigma2 = d["sigma1"], d["sigma2"]
            print(f"[{args.label}] loaded cache {cache.name}")
    if sigma1 is None:
        print(f"[{args.label}] dim={dim}, grid={cfg.kgrid.k_points}, "
              f"{args.nw} energies {args.emin}-{args.emax} eV -> computing")
        sigma1, sigma2 = _compute(cfg, dim, omega)
        np.savez_compressed(cache, sigma1=sigma1, sigma2=sigma2, e_ev=e_ev,
                            tag=tag, erange=erange)
        print(f"[{args.label}] cached -> {cache}")

    chi2 = sigma2 / (2j * omega[:, None, None, None])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    # ================= sigma^(1): 2x2 in-plane, modulus =================
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
