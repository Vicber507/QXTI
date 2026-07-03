#!/usr/bin/env python3
r"""SHG intensity vs analyzer angle under RCP light, node-masked, for different
surface orientations (theta_z and phi_z).

Uses the CONVERGED chi^(2) tensors already computed (full / no chi=+1 / no chi=-1)
from outputs/wsm_orenstein_order2_node_masks/order2_tensor_node_masks.npz. The
tensor is a material (crystal-frame) property, so only the incident RCP field and
the analyzer direction change with orientation.

  I_RCP(theta2) = |a(theta2) . J(2w)|^2 ,   J = chi : E_RCP E_RCP ,
  E_RCP = (xdir + i ydir)/sqrt2 ,   a = cos(theta2) xdir + sin(theta2) ydir ,

with (xdir, ydir) the in-plane laser axes for (theta_z, phi_z) (phix=0). Two
figures: one varying theta_z, one varying phi_z. Panels x node-mask curves.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.physics.laser import Laser

NPZ = Path("outputs/wsm_orenstein_order2_node_masks/order2_tensor_node_masks.npz")
OUT = Path("outputs/wsm_orenstein_rcp_analyzer")
CASES = (("chi_full", "All nodes", "#111827"),
         ("chi_no_pos", r"Without $\chi=+1$ nodes", "#0072B2"),
         ("chi_no_neg", r"Without $\chi=-1$ nodes", "#D55E00"))
THETA = np.linspace(0.0, 2.0 * np.pi, 361)


def _basis(thetaz, phiz):
    L = Laser(omega=0.057, E0=3e-4, ellip=1.0, ncycles=8, phix=0.0, thetaz=thetaz, phiz=phiz)
    return L.xdir, L.ydir


def _I_rcp(chi, xdir, ydir):
    E = (xdir + 1j * ydir) / np.sqrt(2.0)
    J = np.einsum("ijk,j,k->i", chi, E, E, optimize=True)
    p1, p2 = xdir @ J, ydir @ J
    return np.abs(np.cos(THETA) * p1 + np.sin(THETA) * p2) ** 2


def _figure(tensors, iw, varname, angles_deg, fixed_deg, fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    sym = r"\theta_z" if varname == "thetaz" else r"\phi_z"
    data, gmax = {}, 0.0
    for ad in angles_deg:
        if varname == "thetaz":
            xd, yd = _basis(math.radians(ad), math.radians(fixed_deg))
        else:
            xd, yd = _basis(math.radians(fixed_deg), math.radians(ad))
        for key, _lab, _c in CASES:
            v = _I_rcp(tensors[key][iw], xd, yd)
            data[(ad, key)] = v
            gmax = max(gmax, float(v.max()))
    gmax = gmax or 1.0

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True, sharey=True)
    for ax, ad in zip(axes.flat, angles_deg):
        for key, label, color in CASES:
            ax.plot(np.degrees(THETA), data[(ad, key)] / gmax, color=color, lw=2.0, label=label)
        tag = ""
        if varname == "thetaz":
            if abs(ad - 35.2644) < 0.5:
                tag = " = (112)"
            elif ad == 0:
                tag = " = (001)"
        ax.set_title(rf"${sym}={ad:g}^\circ${tag}", fontsize=12)
        ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90)); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"analyzer angle  $\theta_2$ (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("SHG intensity (RCP, norm.)")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / fname}")


def main() -> int:
    d = np.load(NPZ)
    iw = int(np.argmin(np.abs(d["energy_ev"] - 1.54)))
    tensors = {k: d[k] for k, _l, _c in CASES}
    _figure(tensors, iw, "thetaz", [0, 17.6, 35.2644, 52.9], 45.0,
            "shg_rcp_vs_thetaz.png",
            r"SHG (RCP) vs analyzer angle - different $\theta_z$  ($\phi_z=45^\circ$)")
    _figure(tensors, iw, "phiz", [0, 30, 45, 90], 35.2644,
            "shg_rcp_vs_phiz.png",
            r"SHG (RCP) vs analyzer angle - different $\phi_z$  ($\theta_z=35.3^\circ$)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
