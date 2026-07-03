#!/usr/bin/env python3
r"""Add a 'without ALL 4 Weyl nodes' case and redo:
  (1) SHG (RCP) intensity vs analyzer angle, per phi_z, node-masked (+ no-all).
  (2) Phase difference of the emitted SHG wave vs phi_z, of each masked case
      relative to the FULL material: (no+ , full), (no- , full), (no-all , full),
      plus the (no+ , no-) reference.

Reuses the converged chi^(2) tensors already computed (full/no+/no-) at 1.54 eV
and computes the missing no-all tensor once (grid-based, convergent).
"""
from __future__ import annotations

import importlib.util
import math
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

from qxti.analytics.theory_response import build_k_integration_weights, order2_tensor_at_omega
from qxti.core.config import QXTIConfig
from qxti.core.simulation import QXTISimulation
from qxti.physics.laser import Laser
from plot_wsm_helicity_sigma_node_masks import _build_local_node_mask, _min_nonzero_pair_distance

NPZ = Path("outputs/wsm_orenstein_order2_node_masks/order2_tensor_node_masks.npz")
OUT = Path("outputs/wsm_orenstein_rcp_analyzer")
GRID = 40
ENERGY_EV = 1.54
THETA = np.linspace(0.0, 2.0 * np.pi, 361)


def _load_nodes(config):
    src = Path(config.hamiltonian.source_file)
    path = src if src.is_absolute() else (PROJECT_ROOT / "models" / src.name)
    spec = importlib.util.spec_from_file_location(f"qxti_all_{path.stem}", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return list(m.weyl_nodes_with_chirality(dict(config.hamiltonian.params)))


def _compute_no_all(omega):
    cfg0 = QXTIConfig.from_file("inputs/inputParams.wsm_orenstein.cfg")
    config = replace(cfg0, kgrid=replace(cfg0.kgrid, k_points=[GRID, GRID, GRID]))
    nodes = _load_nodes(config)
    sim = QXTISimulation(config=config); ham = sim.build_hamiltonian(); kg = sim.build_kgrid(ham)
    kpts = np.asarray(kg.points(), dtype=np.float64)
    radius = 0.35 * _min_nonzero_pair_distance(np.asarray([np.asarray(n["k"]) for n in nodes]))
    mp, _c, _r = _build_local_node_mask(kpts, tagged_nodes=nodes, chirality_to_remove=+1, radius=radius)
    mn, _c, _r = _build_local_node_mask(kpts, tagged_nodes=nodes, chirality_to_remove=-1, radius=radius)
    mask_all = mp * mn                      # remove BOTH chiralities (all 4 nodes)
    wts = build_k_integration_weights(config, hamiltonian=ham, kgrid=kg, extra_k_weight_mask=mask_all)
    print(f"[all] excluidos {int(np.count_nonzero(mask_all==0))}/{kpts.shape[0]} pts (los 4 nodos)")
    sigma = order2_tensor_at_omega(ham, kg, omega, wts, config.susceptibility_solver)
    return sigma / (2j * omega)             # -> chi convention (matches npz)


def _J(chi, thetaz, phiz):
    L = Laser(omega=0.057, E0=3e-4, ellip=1.0, ncycles=8, phix=0.0, thetaz=thetaz, phiz=phiz)
    E = (L.xdir + 1j * L.ydir) / np.sqrt(2.0)
    return L.xdir, L.ydir, np.einsum("ijk,j,k->i", chi, E, E, optimize=True)


def main() -> int:
    d = np.load(NPZ)
    iw = int(np.argmin(np.abs(d["energy_ev"] - ENERGY_EV)))
    omega = float(d["omega"][iw])
    chi = {"full": d["chi_full"][iw], "no_pos": d["chi_no_pos"][iw], "no_neg": d["chi_no_neg"][iw]}
    chi["no_all"] = _compute_no_all(omega)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- (1) SHG (RCP) vs analyzer angle, per phi_z, with no-all ----
    rcp_cases = (("full", "All nodes", "#111827"),
                 ("no_pos", r"Without $\chi=+1$", "#0072B2"),
                 ("no_neg", r"Without $\chi=-1$", "#D55E00"),
                 ("no_all", "Without all 4 nodes", "#009E73"))
    phiz_panels = [0, 30, 45, 90]
    thetaz = math.radians(35.2644)
    data, gmax = {}, 0.0
    for pz in phiz_panels:
        xd, yd, _ = _J(chi["full"], thetaz, math.radians(pz))
        for key, _l, _c in rcp_cases:
            _x, _y, J = _J(chi[key], thetaz, math.radians(pz))
            I = np.abs(np.cos(THETA) * (xd @ J) + np.sin(THETA) * (yd @ J)) ** 2
            data[(pz, key)] = I; gmax = max(gmax, float(I.max()))
    gmax = gmax or 1.0
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True, sharey=True)
    for ax, pz in zip(axes.flat, phiz_panels):
        for key, label, color in rcp_cases:
            ax.plot(np.degrees(THETA), data[(pz, key)] / gmax, color=color, lw=2.0, label=label)
        ax.set_title(rf"$\phi_z={pz}^\circ$", fontsize=12)
        ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90)); ax.set_ylim(0, 1.05); ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"analyzer angle  $\theta_2$ (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("SHG intensity (RCP, norm.)")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(r"SHG (RCP) vs analyzer angle - different $\phi_z$ ($\theta_z=35.3^\circ$)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "shg_rcp_vs_phiz.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'shg_rcp_vs_phiz.png'}")

    # ---- (2) phase difference vs phi_z: each masked case relative to FULL ----
    phiz = np.linspace(0.0, 360.0, 361)
    pairs = (("no_pos", "full", r"(no $+$) $-$ total", "#0072B2"),
             ("no_neg", "full", r"(no $-$) $-$ total", "#D55E00"),
             ("no_all", "full", r"(no all) $-$ total", "#009E73"),
             ("no_pos", "no_neg", r"(no $+$) $-$ (no $-$)", "#6A0DAD"))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.axhline(0, color="0.8", lw=0.9, zorder=0)
    for a, b, label, color in pairs:
        dphi = np.empty(phiz.size)
        for n, pz in enumerate(phiz):
            _x, _y, Ja = _J(chi[a], thetaz, math.radians(pz))
            _x, _y, Jb = _J(chi[b], thetaz, math.radians(pz))
            dphi[n] = float(np.degrees(np.angle(np.vdot(Jb, Ja))))
        ax.plot(phiz, dphi, color=color, lw=2.0, label=label)
    ax.set_xlabel(r"$\phi_z$ (deg)")
    ax.set_ylabel(r"desfase  $\Delta\varphi$  (deg)")
    ax.set_title(rf"SHG-wave phase difference vs $\phi_z$ (RCP, $\theta_z=35.3^\circ$, {ENERGY_EV:.2f} eV)",
                 fontsize=11)
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 45))
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "shg_phase_diff_nodes_vs_phiz.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'shg_phase_diff_nodes_vs_phiz.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
