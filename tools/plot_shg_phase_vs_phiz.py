#!/usr/bin/env python3
r"""Phase difference (desfase) BETWEEN the two node-masked SHG waves vs phi_z.

Each Weyl-node case emits its own second-harmonic wave under RCP incidence:
    J_nopos(2w) = chi(no chi=+1) : E_RCP E_RCP
    J_noneg(2w) = chi(no chi=-1) : E_RCP E_RCP
This plots the RELATIVE PHASE between those two waves as a function of phi_z
(theta_z fixed), via the Hermitian overlap:

    Delta_phi(phi_z) = arg( sum_i  J_nopos,i * conj(J_noneg,i) ).

Delta_phi = 0 -> the two waves are in phase; +/-180 -> in antiphase.
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

from qxti.physics.laser import Laser

NPZ = Path("outputs/wsm_orenstein_order2_node_masks/order2_tensor_node_masks.npz")
OUT = Path("outputs/wsm_orenstein_rcp_analyzer")
THETAZ_DEG = 35.2644     # fixed tilt (112-type)
NUM_PHIZ = 361           # many points
ENERGY_EV = 1.54


def _J(chi, thetaz, phiz):
    L = Laser(omega=0.057, E0=3e-4, ellip=1.0, ncycles=8, phix=0.0, thetaz=thetaz, phiz=phiz)
    E = (L.xdir + 1j * L.ydir) / np.sqrt(2.0)        # RCP
    return np.einsum("ijk,j,k->i", chi, E, E, optimize=True)


def main() -> int:
    d = np.load(NPZ)
    iw = int(np.argmin(np.abs(d["energy_ev"] - ENERGY_EV)))
    chi_pos, chi_neg = d["chi_no_pos"][iw], d["chi_no_neg"][iw]
    thetaz = math.radians(THETAZ_DEG)
    phiz = np.linspace(0.0, 360.0, NUM_PHIZ)

    dphi = np.empty(NUM_PHIZ)
    for n, p in enumerate(phiz):
        Jp = _J(chi_pos, thetaz, math.radians(p))
        Jm = _J(chi_neg, thetaz, math.radians(p))
        dphi[n] = float(np.degrees(np.angle(np.vdot(Jm, Jp))))   # arg(sum Jp * conj(Jm))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for y in (-90, 0, 90):
        ax.axhline(y, color="0.85", lw=0.8, zorder=0)
    ax.plot(phiz, dphi, color="#6A0DAD", lw=2.2)
    ax.set_xlabel(r"$\phi_z$ (deg)")
    ax.set_ylabel(r"phase difference  $\Delta\varphi_{(+)-(-)}$  (deg)")
    ax.set_title(rf"Phase difference between the (no $\chi{{=}}{{+}}1$) and (no $\chi{{=}}{{-}}1$) "
                 rf"SHG waves vs $\phi_z$" "\n"
                 rf"(RCP, $\theta_z={THETAZ_DEG:.1f}^\circ$, {ENERGY_EV:.2f} eV)", fontsize=11)
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 45))
    ax.set_ylim(-190, 190); ax.set_yticks(range(-180, 181, 90))
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "shg_phase_diff_nodes_vs_phiz.png"
    fig.savefig(f, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f}  ({NUM_PHIZ} phiz points, thetaz={THETAZ_DEG} deg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
