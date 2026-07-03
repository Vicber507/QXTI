#!/usr/bin/env python3
r"""Single paper-style plot: SHG (RCP) intensity vs analyzer angle on TaAs (112).

Fixed incidence geometry (the (112) surface, theta_z/phi_z from the config), RCP
drive E = (xhat + i yhat)/sqrt(2). The emitted second-harmonic current
J(2w) = chi^(2) : E E is passed through a linear analyzer rotating in the surface
plane, e(theta2) = cos(theta2) xhat + sin(theta2) yhat, giving

    I(theta2) = | e(theta2) . J |^2 .

Three curves: full material, without chi=+1 nodes, without chi=-1 nodes, using the
CONVERGED, chirality-CORRECTED tensors in order2_tensor_node_masks.npz. A polar
inset shows the same three lobed patterns. (Rebuilt from the corrected npz; the
previous inline version used the old, mislabelled chirality.)
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.core.config import QXTIConfig
from qxti.physics.laser import Laser

NPZ = Path("outputs/wsm_orenstein_order2_node_masks/order2_tensor_node_masks.npz")
OUT = Path("outputs/wsm_orenstein_rcp_analyzer")
ENERGY_EV = 1.54
NTH = 721

CASES = (
    ("chi_full", r"$\mathrm{all\ nodes}$", "#111827"),
    ("chi_no_pos", r"$\mathrm{without}\ \chi=+1$", "#0072B2"),
    ("chi_no_neg", r"$\mathrm{without}\ \chi=-1$", "#D55E00"),
)


def main() -> int:
    d = np.load(NPZ)
    iw = int(np.argmin(np.abs(d["energy_ev"] - ENERGY_EV)))
    e_ev = float(d["energy_ev"][iw])

    cfg = QXTIConfig.from_file("inputs/inputParams.wsm_orenstein.cfg")
    tz = float(cfg.laser.thetaz); pz = float(cfg.laser.phiz)
    L = Laser(omega=0.057, E0=3e-4, ellip=1.0, ncycles=8, phix=0.0, thetaz=tz, phiz=pz)
    xd, yd = L.xdir, L.ydir                       # in-plane analyzer basis on (112)
    E = (xd + 1j * yd) / np.sqrt(2.0)             # RCP drive

    theta = np.linspace(0.0, 2.0 * np.pi, NTH)
    curves = {}
    gmax = 0.0
    for key, _lab, _c in CASES:
        chi = d[key][iw]
        J = np.einsum("ijk,j,k->i", chi, E, E, optimize=True)
        Jx, Jy = xd @ J, yd @ J
        I = np.abs(np.cos(theta) * Jx + np.sin(theta) * Jy) ** 2
        curves[key] = I
        gmax = max(gmax, float(I.max()))
    gmax = gmax or 1.0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for key, lab, col in CASES:
        ax.plot(np.degrees(theta), curves[key] / gmax, color=col, lw=2.4, label=lab)
    ax.set_xlabel(r"$\mathrm{analyzer\ angle}\ \ \theta_2\ \ (\mathrm{deg})$")
    ax.set_ylabel(r"$\mathrm{SHG\ intensity\ (RCP,\ norm.)}$")
    ax.set_title(rf"$\mathrm{{SHG\ (RCP)\ vs\ analyzer\ angle:\ TaAs\ (112)\ at\ "
                 rf"{e_ev:.2f}\ eV}}$", fontsize=13)
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 45))
    ax.set_ylim(0, 1.06)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")

    # polar inset (same three patterns)
    axin = fig.add_axes((0.60, 0.52, 0.30, 0.40), projection="polar")
    for key, _lab, col in CASES:
        axin.plot(theta, curves[key] / gmax, color=col, lw=1.8)
    axin.set_xticklabels([]); axin.set_yticklabels([])
    axin.grid(alpha=0.3); axin.set_facecolor("white")

    fig.tight_layout()
    f = OUT / "shg_rcp_112_single.png"
    fig.savefig(f, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f}  (theta_z={math.degrees(tz):.2f} deg, phi_z={math.degrees(pz):.2f} deg, {e_ev:.3f} eV)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
