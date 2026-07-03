#!/usr/bin/env python3
r"""SHG-wave phase difference (vs total), RCP & LCP filled, for NODE PAIRS.

Instead of grouping the removed nodes by chirality, this groups the 4 Weyl nodes
into two (+,-) PAIRS by ky:
    pair A : ky < 0   (one chi=+1 and one chi=-1 node)
    pair B : ky > 0   (one chi=+1 and one chi=-1 node)
Two stacked panels: (without pair A) - total  and  (without pair B) - total.
Each panel overlays RCP (solid) and LCP (dotted) in the same colour with the area
between them shaded. Axes in multiples of pi.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qxti.analytics.theory_response import build_k_integration_weights, order2_tensor_at_omega
from qxti.core.config import QXTIConfig
from qxti.core.simulation import QXTISimulation
from qxti.physics.laser import Laser
from plot_wsm_helicity_sigma_node_masks import _min_nonzero_pair_distance

NPZ = Path("outputs/wsm_orenstein_order2_node_masks/order2_tensor_node_masks.npz")
OUT = Path("outputs/wsm_orenstein_rcp_analyzer")
GRID = 40
THETAZ_DEG = 35.2644
ENERGY_EV = 1.54
COLOR_A = "#009E73"   # pair A (ky<0)
COLOR_B = "#CC79A7"   # pair B (ky>0)


def _pi_label(v, _pos):
    if abs(v) < 1e-9:
        return "0"
    fr = Fraction(v / np.pi).limit_denominator(256)
    n, d = fr.numerator, fr.denominator
    s = "-" if n < 0 else ""
    n = abs(n)
    if d == 1:
        return rf"${s}{'' if n == 1 else n}\pi$"
    if n == 1:
        return rf"${s}\pi/{d}$"
    return rf"${s}{n}\pi/{d}$"


def _load_nodes(config):
    src = Path(config.hamiltonian.source_file)
    path = src if src.is_absolute() else (PROJECT_ROOT / "models" / src.name)
    spec = importlib.util.spec_from_file_location(f"qxti_pair_{path.stem}", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return list(m.weyl_nodes_with_chirality(dict(config.hamiltonian.params)))


def _mask_removing(k_points, centers, radius):
    mask = np.ones(k_points.shape[0], dtype=np.float64)
    if len(centers) == 0:
        return mask
    c = np.asarray(centers, dtype=np.float64)
    dist = np.linalg.norm(k_points[:, None, :] - c[None, :, :], axis=2)
    mask[np.any(dist <= radius, axis=1)] = 0.0
    return mask


def main() -> int:
    d = np.load(NPZ)
    iw = int(np.argmin(np.abs(d["energy_ev"] - ENERGY_EV)))
    omega = float(d["omega"][iw])
    chi_full = d["chi_full"][iw]

    cfg0 = QXTIConfig.from_file("inputs/inputParams.wsm_orenstein.cfg")
    config = replace(cfg0, kgrid=replace(cfg0.kgrid, k_points=[GRID, GRID, GRID]))
    nodes = _load_nodes(config)
    sim = QXTISimulation(config=config); ham = sim.build_hamiltonian(); kg = sim.build_kgrid(ham)
    kpts = np.asarray(kg.points(), dtype=np.float64)
    positions = np.asarray([np.asarray(n["k"], dtype=np.float64) for n in nodes])
    radius = 0.35 * _min_nonzero_pair_distance(positions)

    pairA = [np.asarray(n["k"]) for n in nodes if float(n["k"][1]) < 0]   # ky<0
    pairB = [np.asarray(n["k"]) for n in nodes if float(n["k"][1]) > 0]   # ky>0
    print(f"[pairs] pair A (ky<0): {len(pairA)} nodos, pair B (ky>0): {len(pairB)} nodos, radio={radius:.4f}")

    def chi_without(centers):
        mask = _mask_removing(kpts, centers, radius)
        wts = build_k_integration_weights(config, hamiltonian=ham, kgrid=kg, extra_k_weight_mask=mask)
        return order2_tensor_at_omega(ham, kg, omega, wts, config.susceptibility_solver) / (2j * omega)

    chi_A = chi_without(pairA)   # material WITHOUT pair A
    chi_B = chi_without(pairB)   # material WITHOUT pair B

    tz = math.radians(THETAZ_DEG)
    phiz = np.linspace(0.0, 2.0 * np.pi, 361)

    def J(chi, pz, hel):
        L = Laser(omega=0.057, E0=3e-4, ellip=1.0, ncycles=8, phix=0.0, thetaz=tz, phiz=pz)
        E = (L.xdir + hel * 1j * L.ydir) / np.sqrt(2.0)
        return np.einsum("ijk,j,k->i", chi, E, E, optimize=True)

    def dphi(chi_masked, hel):
        return np.array([float(np.angle(np.vdot(J(chi_full, p, hel), J(chi_masked, p, hel)))) for p in phiz])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    panels = [(chi_A, r"Without pair A ($k_y<0$: one $+$, one $-$)", COLOR_A),
              (chi_B, r"Without pair B ($k_y>0$: one $+$, one $-$)", COLOR_B)]
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.6), sharex=True)
    for ax, (chi_m, title, color) in zip(axes, panels):
        rcp, lcp = dphi(chi_m, +1.0), dphi(chi_m, -1.0)
        ax.axhline(0.0, color="0.8", lw=0.9, zorder=0)
        ax.fill_between(phiz, rcp, lcp, color=color, alpha=0.22, lw=0)
        ax.plot(phiz, rcp, color=color, lw=2.2, ls="-", label="RCP")
        ax.plot(phiz, lcp, color=color, lw=2.2, ls=":", label="LCP")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(r"$\Delta\varphi$ (rad)")
        ax.yaxis.set_major_locator(MultipleLocator(np.pi / 64))
        ax.yaxis.set_major_formatter(FuncFormatter(_pi_label))
        ax.legend(frameon=False, fontsize=10, loc="upper right")
        ax.margins(x=0)
    axes[-1].set_xlabel(r"$\phi_z$")
    axes[-1].xaxis.set_major_locator(MultipleLocator(np.pi / 2))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(_pi_label))
    axes[-1].set_xlim(0, 2 * np.pi)
    fig.suptitle(rf"SHG-wave phase difference vs $\phi_z$, node PAIRS "
                 rf"($\theta_z=35.3^\circ$, {ENERGY_EV:.2f} eV)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "shg_phase_diff_pairs_fill.png"
    fig.savefig(f, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
