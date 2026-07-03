#!/usr/bin/env python3
r"""SHG-wave phase difference (vs total) with RCP & LCP overlaid and filled.

Two stacked panels:
  top    -> (without chi=+1) minus total
  bottom -> (without chi=-1) minus total
Each panel shows RCP (solid) and LCP (dotted) in the SAME colour, with the area
between them shaded (transparent). Both axes are labelled in multiples of pi:
x = phi_z in [0, 2 pi], y = Delta_phi in radians.
"""
from __future__ import annotations

import math
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.physics.laser import Laser

NPZ = Path("outputs/wsm_orenstein_order2_node_masks/order2_tensor_node_masks.npz")
OUT = Path("outputs/wsm_orenstein_rcp_analyzer")
THETAZ_DEG = 35.2644
ENERGY_EV = 1.54


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


def main() -> int:
    d = np.load(NPZ)
    iw = int(np.argmin(np.abs(d["energy_ev"] - ENERGY_EV)))
    chi = {"full": d["chi_full"][iw], "no_pos": d["chi_no_pos"][iw], "no_neg": d["chi_no_neg"][iw]}
    tz = math.radians(THETAZ_DEG)
    phiz = np.linspace(0.0, 2.0 * np.pi, 361)      # radians

    def J(key, pz, hel):
        L = Laser(omega=0.057, E0=3e-4, ellip=1.0, ncycles=8, phix=0.0, thetaz=tz, phiz=pz)
        E = (L.xdir + hel * 1j * L.ydir) / np.sqrt(2.0)
        return np.einsum("ijk,j,k->i", chi[key], E, E, optimize=True)

    def dphi(mask, hel):  # radians, (mask - total)
        return np.array([float(np.angle(np.vdot(J("full", p, hel), J(mask, p, hel)))) for p in phiz])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    panels = [("no_pos", r"Without $\chi=+1$ nodes  (no$+$ $-$ total)", "#0072B2"),
              ("no_neg", r"Without $\chi=-1$ nodes  (no$-$ $-$ total)", "#D55E00")]
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.6), sharex=True)
    for ax, (mask, title, color) in zip(axes, panels):
        rcp = dphi(mask, +1.0)
        lcp = dphi(mask, -1.0)
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
    fig.suptitle(rf"SHG-wave phase difference vs $\phi_z$ ($\theta_z=35.3^\circ$, {ENERGY_EV:.2f} eV)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "shg_phase_diff_rcp_lcp_fill.png"
    fig.savefig(f, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
