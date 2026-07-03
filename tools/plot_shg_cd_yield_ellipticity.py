#!/usr/bin/env python3
r"""SHG circular dichroism from the THEORY HHG engine + harmonic yield.

Physical / time-domain route (what an experiment does): for each ellipticity and
each helicity (ellip = +e / -e), IRRADIATE with the theory HHG engine
(compute_hhg_spectrum), grab the SECOND HARMONIC, take its band-integrated yield
through a rotating analyzer, and form

    CD_e(theta2) = Y_+(theta2) - Y_-(theta2),
    Y_pm(theta2) = \int_{2w +/- dw} |a(theta2) . J_pm(w)|^2 dw ,   a = cos e1 + sin e2

for three Weyl-node cases (all / without chi=+1 / without chi=-1). One panel per
ellipticity. Each (e, helicity, case) is an INDIVIDUAL theory run.
"""
from __future__ import annotations

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

from qxti.analytics.theory_response import compute_hhg_spectrum
from qxti.core.config import QXTIConfig
from qxti.core.simulation import QXTISimulation
from plot_wsm_helicity_sigma_node_masks import _build_local_node_mask, _min_nonzero_pair_distance

_AU_TO_EV = 27.211386245988
OMEGA_800NM = (1239.84159 / 800.0) / _AU_TO_EV
BAND_HW = 0.1
ELLIPTICITIES = (0.25, 0.5, 0.75, 1.0)
E1 = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
E2 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
CASES = (("full", "All nodes", None, "#111827"),
         ("no_pos", r"Without $\chi=+1$ nodes", +1, "#0072B2"),
         ("no_neg", r"Without $\chi=-1$ nodes", -1, "#D55E00"))
OUT = Path("outputs/wsm_orenstein_cd_yield_ellipticity")


def _load_nodes(config):
    src = Path(config.hamiltonian.source_file)
    path = src if src.is_absolute() else (PROJECT_ROOT / "models" / src.name)
    spec = importlib.util.spec_from_file_location(f"qxti_cdy_{path.stem}", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return list(m.weyl_nodes_with_chirality(dict(config.hamiltonian.params)))


def _spectrum(config, ellip, mask):
    cfg = replace(config,
                  laser=replace(config.laser, ellip=float(ellip), omega=OMEGA_800NM),
                  cmd=replace(config.cmd, max_order=2))
    r = compute_hhg_spectrum(cfg, max_order=2, progress=False, extra_k_weight_mask=mask)
    freq = np.asarray(r["omega_axis"], dtype=np.float64)
    J = np.asarray(r["J_total"], dtype=np.complex128)
    Jv = np.zeros((freq.size, 3), dtype=np.complex128); Jv[:, :J.shape[1]] = J
    return freq, Jv, float(r["omega0"])


def _cd_yield(freq, Jp, Jm, w0, theta):
    integ = getattr(np, "trapezoid", np.trapz)
    band = (freq >= (2 - BAND_HW) * w0) & (freq <= (2 + BAND_HW) * w0)
    fb = freq[band]
    p1p, p2p = Jp[band] @ E1, Jp[band] @ E2
    p1m, p2m = Jm[band] @ E1, Jm[band] @ E2
    ct, st = np.cos(theta)[:, None], np.sin(theta)[:, None]
    Yp = integ(np.abs(ct * p1p[None] + st * p2p[None]) ** 2, fb, axis=1)
    Ym = integ(np.abs(ct * p1m[None] + st * p2m[None]) ** 2, fb, axis=1)
    return Yp - Ym


def main() -> int:
    cfg0 = QXTIConfig.from_file("inputs/inputParams.wsm_orenstein.cfg")
    config = cfg0
    nodes = _load_nodes(config)
    sim = QXTISimulation(config=config); ham = sim.build_hamiltonian(); kgrid = sim.build_kgrid(ham)
    k_points = np.asarray(kgrid.points(), dtype=np.float64)
    radius = 0.35 * _min_nonzero_pair_distance(
        np.asarray([np.asarray(n["k"], dtype=np.float64) for n in nodes]))
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    print(f"[cd-yield] 800 nm, grid {kgrid.total_points} pts, {len(ELLIPTICITIES)} ellipticities "
          f"x 2 helicities x {len(CASES)} casos = {len(ELLIPTICITIES)*2*len(CASES)} corridas theory")

    masks = {}
    for name, _l, chir, _c in CASES:
        masks[name], _cc, _r = _build_local_node_mask(
            k_points, tagged_nodes=nodes, chirality_to_remove=chir, radius=radius)

    curves = {}   # (ellip, case) -> CD(theta)
    for e in ELLIPTICITIES:
        for name, _l, _chir, _c in CASES:
            freq, Jp, w0 = _spectrum(config, +e, masks[name])
            _f, Jm, _w = _spectrum(config, -e, masks[name])
            curves[(e, name)] = _cd_yield(freq, Jp, Jm, w0, theta)
            print(f"[cd-yield]   e={e}, {name}: max|CD|={np.max(np.abs(curves[(e,name)])):.3e}")

    gmax = max(np.max(np.abs(v)) for v in curves.values()) or 1.0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True, sharey=True)
    for ax, e in zip(axes.flat, ELLIPTICITIES):
        ax.axhline(0.0, color="0.6", lw=0.9, zorder=0)
        for name, label, _chir, color in CASES:
            ax.plot(np.degrees(theta), curves[(e, name)] / gmax, color=color, lw=2.0, label=label)
        ax.set_title(rf"$\epsilon={e}$" + ("  (circular)" if e == 1.0 else ""), fontsize=12)
        ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 90)); ax.set_ylim(-1.1, 1.1)
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"analyzer angle  $\theta_2$ (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("CD (normalized)")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(r"SHG circular dichroism (theory HHG, 2nd-harmonic yield) vs ellipticity, TaAs (112)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fpng = OUT / "shg_cd_yield_vs_ellipticity.png"
    fig.savefig(fpng, dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)
    np.savez_compressed(OUT / "shg_cd_yield_vs_ellipticity.npz", theta2_deg=np.degrees(theta),
                        gmax=gmax, **{f"cd_e{e}_{n}": curves[(e, n)] for e in ELLIPTICITIES for n, *_ in CASES})
    print(f"[cd-yield] wrote {fpng}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
