#!/usr/bin/env python3
r"""SHG-wave phase difference vs phi_z, laid out BY HELICITY.

Same physics as plot_shg_phase_rcp_lcp_fill / plot_shg_phase_pairs_fill, but the
panels are split by helicity instead of by node case:

    top panel    -> RCP incidence
    bottom panel -> LCP incidence

and each panel overlays the TWO node-removal cases of that figure, with the area
between them shaded. Two figures are produced (each a different "combination"):

  (1) SAME-chirality figure : (without chi=+1)  vs  (without chi=-1)
  (2) DIFFERENT-chirality figure : (without ky<0 pair)  vs  (without ky>0 pair)
      (each ky pair holds one chi=+1 and one chi=-1 node)

Delta_phi(case, hel) = arg( <J_full(hel) , J_case(hel)> ),  the phase of the
node-removed SHG wave relative to the full-material wave. Axes in multiples of pi.
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
NPHI = 361


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
    spec = importlib.util.spec_from_file_location(f"qxti_bh_{path.stem}", path)
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


def _dphi_curve(chi_full, chi_case, tz, phiz, hel):
    def J(chi, pz):
        L = Laser(omega=0.057, E0=3e-4, ellip=1.0, ncycles=8, phix=0.0, thetaz=tz, phiz=pz)
        E = (L.xdir + hel * 1j * L.ydir) / np.sqrt(2.0)
        return np.einsum("ijk,j,k->i", chi, E, E, optimize=True)
    return np.array([float(np.angle(np.vdot(J(chi_full, p), J(chi_case, p)))) for p in phiz])


COL_RCP = "#C1121F"   # RCP -> red family
COL_LCP = "#124E78"   # LCP -> blue family


def _make_figure(plt, MultipleLocator, FuncFormatter, chi_full, cases, phiz, tz,
                 suptitle, fname):
    """Single panel. Colour = helicity (RCP/LCP); line style = removed nodes.

    cases: list of exactly two (chi_case, label, linestyle).
    The area between the two same-helicity curves is shaded in that helicity's
    colour.
    """
    from matplotlib.lines import Line2D

    (chiA, labA, lsA), (chiB, labB, lsB) = cases
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    ax.axhline(0.0, color="0.8", lw=0.9, zorder=0)
    for hel, col in ((+1.0, COL_RCP), (-1.0, COL_LCP)):
        yA = _dphi_curve(chi_full, chiA, tz, phiz, hel)
        yB = _dphi_curve(chi_full, chiB, tz, phiz, hel)
        ax.fill_between(phiz, yA, yB, color=col, alpha=0.16, lw=0)
        ax.plot(phiz, yA, color=col, lw=2.4, ls=lsA)
        ax.plot(phiz, yB, color=col, lw=2.4, ls=lsB)

    ax.set_xlabel(r"$\phi_z$")
    ax.set_ylabel(r"$\Delta\varphi\ \ (\mathrm{rad})$")
    ax.yaxis.set_major_locator(MultipleLocator(np.pi / 64))
    ax.yaxis.set_major_formatter(FuncFormatter(_pi_label))
    ax.xaxis.set_major_locator(MultipleLocator(np.pi / 2))
    ax.xaxis.set_major_formatter(FuncFormatter(_pi_label))
    ax.set_xlim(0, 2 * np.pi)
    ax.margins(x=0)

    # two legends: colour = helicity, line style = removed nodes
    hel_handles = [Line2D([], [], color=COL_RCP, lw=2.4, label="RCP"),
                   Line2D([], [], color=COL_LCP, lw=2.4, label="LCP")]
    sty_handles = [Line2D([], [], color="0.35", lw=2.4, ls=lsA, label=labA),
                   Line2D([], [], color="0.35", lw=2.4, ls=lsB, label=labB)]
    leg1 = ax.legend(handles=hel_handles, frameon=True, facecolor="white",
                     framealpha=0.9, edgecolor="0.8", fontsize=10.5,
                     loc="upper left", title=r"$\mathrm{helicity}$", title_fontsize=10.5)
    ax.add_artist(leg1)
    ax.legend(handles=sty_handles, frameon=True, facecolor="white",
              framealpha=0.9, edgecolor="0.8", fontsize=10.5,
              loc="upper right", title=r"$\mathrm{removed\ nodes}$", title_fontsize=10.5)

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / fname, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / fname}")


def main() -> int:
    d = np.load(NPZ)
    iw = int(np.argmin(np.abs(d["energy_ev"] - ENERGY_EV)))
    omega = float(d["omega"][iw])
    chi_full = d["chi_full"][iw]
    chi_no_pos = d["chi_no_pos"][iw]
    chi_no_neg = d["chi_no_neg"][iw]

    # pairs (ky<0 / ky>0), computed fresh on the same grid
    cfg0 = QXTIConfig.from_file("inputs/inputParams.wsm_orenstein.cfg")
    config = replace(cfg0, kgrid=replace(cfg0.kgrid, k_points=[GRID, GRID, GRID]))
    nodes = _load_nodes(config)
    sim = QXTISimulation(config=config); ham = sim.build_hamiltonian(); kg = sim.build_kgrid(ham)
    kpts = np.asarray(kg.points(), dtype=np.float64)
    positions = np.asarray([np.asarray(n["k"], dtype=np.float64) for n in nodes])
    radius = 0.35 * _min_nonzero_pair_distance(positions)
    pairA = [np.asarray(n["k"]) for n in nodes if float(n["k"][1]) < 0]
    pairB = [np.asarray(n["k"]) for n in nodes if float(n["k"][1]) > 0]

    def chi_without(centers):
        mask = _mask_removing(kpts, centers, radius)
        wts = build_k_integration_weights(config, hamiltonian=ham, kgrid=kg, extra_k_weight_mask=mask)
        return order2_tensor_at_omega(ham, kg, omega, wts, config.susceptibility_solver) / (2j * omega)

    chi_A, chi_B = chi_without(pairA), chi_without(pairB)

    tz = math.radians(THETAZ_DEG)
    phiz = np.linspace(0.0, 2.0 * np.pi, NPHI)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    _make_figure(
        plt, MultipleLocator, FuncFormatter, chi_full,
        cases=[(chi_no_pos, r"without $\chi=+1$", "-"),
               (chi_no_neg, r"without $\chi=-1$", ":")],
        phiz=phiz, tz=tz,
        suptitle=r"SHG-wave phase difference vs $\phi_z$: SAME-chirality removal"
                 "\n"
                 rf"$\theta_z = 35.3^\circ,\ {ENERGY_EV:.2f}\ \mathrm{{eV}}$"
                 r" (colour: helicity, line: removed nodes)",
        fname="shg_phase_byhelicity_same_chirality.png")

    _make_figure(
        plt, MultipleLocator, FuncFormatter, chi_full,
        cases=[(chi_A, r"without $k_y<0$ pair", "-"),
               (chi_B, r"without $k_y>0$ pair", ":")],
        phiz=phiz, tz=tz,
        suptitle=r"SHG-wave phase difference vs $\phi_z$: DIFFERENT-chirality removal"
                 "\n"
                 rf"$\theta_z = 35.3^\circ,\ {ENERGY_EV:.2f}\ \mathrm{{eV}}$"
                 r" (colour: helicity, line: removed nodes)",
        fname="shg_phase_byhelicity_diff_chirality.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
