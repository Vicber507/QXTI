#!/usr/bin/env python3
r"""Publication-quality order-2 response of the WSM with Weyl-node masks.

Two analyses, both for three cases (all nodes / without chi=+1 / without chi=-1),
using the CONVERGED grid-based analytic tensor (order2_full_tensor_spectrum):

  1. ALL nonzero components of chi^(2)_{ijk}(omega) vs photon energy (Fig-4d style).
  2. SHG circular dichroism CD(theta2) on the (112) surface (Fig-2c style):
     CD = 6 Im(chi_xzx chi_zzz*)/|chi_zzz+4 chi_xzx+2 chi_zxx|^2 * sin(2 theta2).

Node removal is a multiplicative mask on the BZ weights (same logic as
tools/plot_wsm_helicity_sigma_node_masks.py). Output: paper-ready PNGs + data npz.
"""
from __future__ import annotations

import argparse
import importlib.util
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

from qxti.analytics.theory_response import build_k_integration_weights
from qxti.core.config import QXTIConfig
from qxti.core.simulation import QXTISimulation
from plot_wsm_helicity_sigma_node_masks import _build_local_node_mask, _min_nonzero_pair_distance
from _shg_tensor_gridbased import order2_full_tensor_spectrum

_AU_TO_EV = 27.211386245988
GRID = 72
NUM_FREQ = 200
OMEGA_MIN, OMEGA_MAX = 0.006, 0.11          # a.u. (~0.16 - 3.0 eV incident)
OMEGA_800NM = (1239.84159 / 800.0) / _AU_TO_EV
XYZ = ("x", "y", "z")

CASES = (
    ("full", "All nodes", None, "#111827"),
    ("no_pos", r"Without $\chi=+1$ nodes", +1, "#0072B2"),
    ("no_neg", r"Without $\chi=-1$ nodes", -1, "#D55E00"),
)
OUT = Path("outputs/wsm_orenstein_order2_node_masks")


def _load_nodes(config):
    src = Path(config.hamiltonian.source_file)
    path = src if src.is_absolute() else (PROJECT_ROOT / "models" / src.name)
    spec = importlib.util.spec_from_file_location(f"qxti_o2_{path.stem}", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return list(module.weyl_nodes_with_chirality(dict(config.hamiltonian.params)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute publication-quality order-2 WSM tensor with Weyl-node masks. "
            "Use --num-freq for smoother paper plots."
        )
    )
    parser.add_argument("--grid", type=int, default=GRID, help="k-grid points per direction.")
    parser.add_argument("--num-freq", type=int, default=NUM_FREQ, help="number of frequency samples.")
    parser.add_argument("--omega-min", type=float, default=OMEGA_MIN, help="minimum omega in a.u.")
    parser.add_argument("--omega-max", type=float, default=OMEGA_MAX, help="maximum omega in a.u.")
    parser.add_argument("--omega-chunk", type=int, default=6, help="frequency chunk size used internally.")
    parser.add_argument("--progress", action="store_true", help="print progress for frequency chunks.")
    parser.add_argument("--out", default=str(OUT), help="output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grid = int(args.grid)
    num_freq = int(args.num_freq)
    omega_min = float(args.omega_min)
    omega_max = float(args.omega_max)
    omega_chunk = int(args.omega_chunk)
    out = Path(args.out)
    if grid <= 0:
        raise ValueError("--grid must be positive.")
    if num_freq <= 1:
        raise ValueError("--num-freq must be greater than 1.")
    if omega_chunk <= 0:
        raise ValueError("--omega-chunk must be positive.")
    if omega_min <= 0.0 or omega_max <= omega_min:
        raise ValueError("Require 0 < --omega-min < --omega-max.")

    cfg0 = QXTIConfig.from_file("inputs/inputParams.wsm_orenstein.cfg")
    config = replace(cfg0, kgrid=replace(cfg0.kgrid, k_points=[grid, grid, grid]))
    nodes = _load_nodes(config)
    sim = QXTISimulation(config=config); ham = sim.build_hamiltonian(); kgrid = sim.build_kgrid(ham)
    k_points = np.asarray(kgrid.points(), dtype=np.float64)
    radius = 0.35 * _min_nonzero_pair_distance(
        np.asarray([np.asarray(n["k"], dtype=np.float64) for n in nodes]))
    omega = np.linspace(omega_min, omega_max, num_freq)
    E_eV = omega * _AU_TO_EV
    print(f"[order2] grid {grid}^3 -> {kgrid.total_points} pts, {num_freq} freqs "
          f"({E_eV[0]:.2f}-{E_eV[-1]:.2f} eV), radio mascara={radius:.4f}")

    sigmas = {}
    for name, label, chir, _c in CASES:
        mask, _cc, _r = _build_local_node_mask(k_points, tagged_nodes=nodes,
                                               chirality_to_remove=chir, radius=radius)
        wts = build_k_integration_weights(config, hamiltonian=ham, kgrid=kgrid, extra_k_weight_mask=mask)
        print(f"[order2] caso '{name}': excluidos {int(np.count_nonzero(mask==0.0))} pts ... computando tensor")
        sig = order2_full_tensor_spectrum(ham, kgrid, omega, wts, config.susceptibility_solver,
                                          omega_chunk=omega_chunk, progress=bool(args.progress))
        sigmas[name] = sig                                   # (nw, 3,3,3) sigma
    # chi = sigma/(2 i omega)
    chis = {n: sigmas[n] / (2j * omega[:, None, None, None]) for n in sigmas}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass
    out.mkdir(parents=True, exist_ok=True)

    # ---- (1) all nonzero independent components (j<=k) ----
    chi_full = chis["full"]
    scale = float(np.max(np.abs(chi_full)))
    comps = []
    for i in range(3):
        for j in range(3):
            for k in range(j, 3):                            # symmetric in (j,k)
                if float(np.max(np.abs(chi_full[:, i, j, k]))) > 1e-3 * scale:
                    comps.append((i, j, k))
    ncomp = len(comps)
    ncols = min(4, ncomp); nrows = int(np.ceil(ncomp / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.7 * nrows),
                             squeeze=False, sharex=True)
    for idx, (i, j, k) in enumerate(comps):
        ax = axes.flat[idx]
        for name, label, _chir, color in CASES:
            ax.plot(E_eV, np.abs(chis[name][:, i, j, k]) / 1e3, color=color, lw=1.8, label=label)
        ax.set_title(rf"$|\chi_{{{XYZ[i]}{XYZ[j]}{XYZ[k]}}}|$", fontsize=11)
        ax.set_xlim(E_eV[0], E_eV[-1]); ax.set_ylim(bottom=0.0)
        if idx % ncols == 0:
            ax.set_ylabel(r"$|\chi^{(2)}|\;(10^3\,\mathrm{arb.})$")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("photon energy (eV)")
    for ax in axes.flat[ncomp:]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(r"WSM $\chi^{(2)}_{ijk}(\omega)$ - full material vs Weyl-node removed",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    f1 = out / "chi2_all_components_node_masks.png"
    fig.savefig(f1, dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print(f"[order2] wrote {f1}  ({ncomp} componentes no nulas)")

    # ---- (2) SHG circular dichroism vs analyzer angle (at 800 nm) ----
    iw = int(np.argmin(np.abs(omega - OMEGA_800NM)))
    theta = np.linspace(0.0, 2.0 * np.pi, 361); s2 = np.sin(2.0 * theta)
    amps = {}
    for name, *_ in CASES:
        s = sigmas[name][iw]
        s33, s15, s31 = s[2, 2, 2], s[0, 2, 0], s[2, 0, 0]
        amps[name] = float(6.0 * np.imag(s15 * np.conj(s33)) / np.abs(s33 + 4 * s15 + 2 * s31) ** 2)
    norm = abs(amps["full"]) or 1.0
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axhline(0.0, color="0.6", lw=0.9, zorder=0)
    for name, label, _chir, color in CASES:
        ax.plot(np.degrees(theta), (amps[name] / norm) * s2, color=color, lw=2.2, label=label)
    ax.set_xlabel(r"analyzer angle  $\theta_2$ (deg)  [from $[1,1,\bar1]$]")
    ax.set_ylabel("CD (normalized)")
    ax.set_title(rf"SHG circular dichroism, TaAs (112) - {E_eV[iw]:.2f} eV")
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 45)); ax.set_ylim(-1.15, 1.15)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    f2 = out / "shg_cd_node_masks.png"
    fig.savefig(f2, dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print(f"[order2] wrote {f2}  (A_full={amps['full']:+.4f}, A_no+={amps['no_pos']:+.4f}, A_no-={amps['no_neg']:+.4f})")

    np.savez_compressed(out / "order2_tensor_node_masks.npz",
                        omega=omega, energy_ev=E_eV, components=np.array(comps),
                        **{f"chi_{n}": chis[n] for n in chis},
                        cd_amp_full=amps["full"], cd_amp_no_pos=amps["no_pos"], cd_amp_no_neg=amps["no_neg"])
    print(f"[order2] datos en {out/'order2_tensor_node_masks.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
