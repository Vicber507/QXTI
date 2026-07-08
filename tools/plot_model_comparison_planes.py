#!/usr/bin/env python3
r"""Compare Orenstein and McCormick WSM models at two crystallographic planes.

Computes sigma^(1) and chi^(2) tensors for both models at:
  - plane_112:  (112) surface, thetaz=0.6155 rad, phiz=pi/4
  - plane_common: kx-polarized (thetaz=0, phiz=0)

Projects onto the respective laser frames to extract 4 observables:
  sigma_ll, sigma_tl, chi_lll, chi_ltt

Produces 3 publication-quality figures saved to outputs/comparison_planes/.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qxti.core.config import QXTIConfig
from qxti.core.simulation import QXTISimulation
from qxti.analytics.theory_response import (
    build_k_integration_weights,
    compute_linear_response_spectrum,
)
from _shg_tensor_gridbased import order2_full_tensor_spectrum

_EV = 27.211386245988  # Hartree to eV
XYZ = ("l", "t")

# Plane definitions: (thetaz_rad, phiz_rad, label)
PLANE_112    = (0.6154797086703873, 0.7853981633974483, "112")
PLANE_COMMON = (0.0, 0.0, "common")

# Model configs
CFG_ORENSTEIN  = "inputs/inputParams.wsm_orenstein.cfg"
CFG_MCCORMICK  = "inputs/inputParams.wsm.cfg"

OUT_DIR = Path("outputs/comparison_planes")

# Plot styling
COLOR_OR = "#1F77B4"
COLOR_MC = "#D62728"
FILL_ALPHA = 0.18


def laser_frame_vectors(thetaz: float, phiz: float, phix: float = 0.0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Return (xdir, ydir) unit vectors for a given laser orientation.

    Replicates qxti/physics/laser.py _build_field_basis() lines 346-367.
    """
    trans_x0 = np.array([
        math.cos(thetaz) * math.cos(phiz),
        math.cos(thetaz) * math.sin(phiz),
        -math.sin(thetaz),
    ])
    trans_y0 = np.array([
        -math.sin(phiz),
        math.cos(phiz),
        0.0,
    ])
    xdir = math.cos(phix) * trans_x0 + math.sin(phix) * trans_y0
    ydir = -math.sin(phix) * trans_x0 + math.cos(phix) * trans_y0
    xdir /= np.linalg.norm(xdir)
    ydir /= np.linalg.norm(ydir)
    return xdir, ydir


def build_config(cfg_path: str, *,
                 k_points: tuple[int, ...] | None = None) -> QXTIConfig:
    """Load config from file and optionally override k-grid."""
    cfg = QXTIConfig.from_file(cfg_path)
    if k_points is not None:
        cfg = replace(cfg, kgrid=replace(cfg.kgrid, k_points=k_points))
    return cfg


def compute_and_cache(cfg: QXTIConfig, cache_path: Path,
                      omega: np.ndarray, *,
                      label: str, recompute: bool = False
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return (sigma1[nw,3,3], sigma2[nw,3,3,3]), loading from cache if valid."""
    dim = 3
    n_kpts = cfg.kgrid.k_points
    tag = np.array([*n_kpts, omega.size, omega[0] * _EV, omega[-1] * _EV])

    sigma1 = sigma2 = None
    if cache_path.exists() and not recompute:
        try:
            d = np.load(cache_path)
            if d["tag"].shape == tag.shape and np.allclose(d["tag"], tag):
                sigma1, sigma2 = d["sigma1"], d["sigma2"]
                print(f"[{label}] loaded cache {cache_path.name}")
        except Exception:
            pass

    if sigma1 is None:
        print(f"[{label}] computing tensors (kgrid={n_kpts}) ...")
        lin = compute_linear_response_spectrum(cfg, omega, progress=True)
        sigma1 = np.asarray(lin["sigma"], dtype=np.complex128)  # (nw, 3, 3)
        sim = QXTISimulation(config=cfg)
        ham = sim.build_hamiltonian()
        kg = sim.build_kgrid(ham)
        wts = build_k_integration_weights(cfg, hamiltonian=ham, kgrid=kg)
        sigma2 = order2_full_tensor_spectrum(
            ham, kg, omega, wts, cfg.susceptibility_solver, progress=True)  # (nw,3,3,3)
        np.savez_compressed(cache_path, sigma1=sigma1, sigma2=sigma2,
                            e_ev=omega * _EV, tag=tag)
        print(f"[{label}] cached -> {cache_path}")

    return sigma1, sigma2


def project_observables(sigma1: np.ndarray, sigma2: np.ndarray,
                        omega: np.ndarray,
                        xdir: np.ndarray, ydir: np.ndarray
                        ) -> dict[str, np.ndarray]:
    """Project full tensors onto laser frame; return dict of 4 observables.

    Keys: 'sigma_ll', 'sigma_tl', 'chi_lll', 'chi_ltt'  (all shape [nw], complex)
    """
    chi2 = sigma2 / (2j * omega[:, None, None, None])

    sigma_ll = np.einsum("i,wij,j->w", xdir, sigma1, xdir)
    sigma_tl = np.einsum("i,wij,j->w", ydir, sigma1, xdir)
    chi_lll = np.einsum("i,j,k,wijk->w", xdir, xdir, xdir, chi2)
    chi_ltt = np.einsum("i,j,k,wijk->w", xdir, ydir, ydir, chi2)

    return {
        "sigma_ll": sigma_ll,
        "sigma_tl": sigma_tl,
        "chi_lll": chi_lll,
        "chi_ltt": chi_ltt,
    }


def panel(ax, e_ev: np.ndarray, curves: list[tuple[np.ndarray, str, str]],
          title: str, ylabel: str) -> None:
    """Draw one comparison panel: modulus of each curve in a different color."""
    for z, color, label in curves:
        mod = np.abs(z)
        ax.fill_between(e_ev, mod, alpha=FILL_ALPHA, color=color, lw=0, zorder=2)
        ax.plot(e_ev, mod, color=color, lw=1.8, label=label, zorder=3)
    ax.set_title(title, fontsize=10, pad=3)
    ax.set_xlim(e_ev[0], e_ev[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$\hbar\omega\ (\mathrm{eV})$", fontsize=8, labelpad=1.5)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=1.5)
    ax.tick_params(labelsize=7.5)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    ax.yaxis.get_offset_text().set_fontsize(7)
    ax.legend(fontsize=7, framealpha=0.8)


def make_figure1(e_ev, obs_or, obs_mc, out_dir: Path) -> Path:
    """Figure 1: both models at (112) plane, 2x2 grid of 4 observables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.6))
    keys = [
        ("sigma_ll", r"$|\sigma_{ll}(\omega)|$", r"$|\sigma^{(1)}|\ \mathrm{(arb.)}$"),
        ("sigma_tl", r"$|\sigma_{tl}(\omega)|$", r"$|\sigma^{(1)}|\ \mathrm{(arb.)}$"),
        ("chi_lll",  r"$|\chi_{lll}(2\omega)|$", r"$|\chi^{(2)}|\ \mathrm{(arb.)}$"),
        ("chi_ltt",  r"$|\chi_{ltt}(2\omega)|$", r"$|\chi^{(2)}|\ \mathrm{(arb.)}$"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, keys):
        panel(ax, e_ev,
              [(obs_or[key], COLOR_OR, "Orenstein"),
               (obs_mc[key], COLOR_MC, "McCormick")],
              title, ylabel)
    fig.suptitle(r"TaAs — both models at $(112)$ plane", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6, w_pad=1.4)
    path = out_dir / "fig1_comparison_112.png"
    fig.savefig(path, dpi=230, facecolor="white")
    plt.close(fig)
    return path


def make_figure2(e_ev, obs_or, obs_mc, out_dir: Path) -> Path:
    """Figure 2: both models at common plane (kx-polarized), 2x2 grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.6))
    keys = [
        ("sigma_ll", r"$|\sigma_{ll}(\omega)|$", r"$|\sigma^{(1)}|\ \mathrm{(arb.)}$"),
        ("sigma_tl", r"$|\sigma_{tl}(\omega)|$", r"$|\sigma^{(1)}|\ \mathrm{(arb.)}$"),
        ("chi_lll",  r"$|\chi_{lll}(2\omega)|$", r"$|\chi^{(2)}|\ \mathrm{(arb.)}$"),
        ("chi_ltt",  r"$|\chi_{ltt}(2\omega)|$", r"$|\chi^{(2)}|\ \mathrm{(arb.)}$"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, keys):
        panel(ax, e_ev,
              [(obs_or[key], COLOR_OR, "Orenstein"),
               (obs_mc[key], COLOR_MC, "McCormick")],
              title, ylabel)
    fig.suptitle(r"TaAs — both models at common plane ($k_x$-polarized)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6, w_pad=1.4)
    path = out_dir / "fig2_comparison_common.png"
    fig.savefig(path, dpi=230, facecolor="white")
    plt.close(fig)
    return path


def make_figure3(e_ev, obs_or_112, obs_or_common, obs_mc_112, obs_mc_common,
                 out_dir: Path) -> Path:
    """Figure 3: sigma_ll at (112) vs common for each model; 1x2 panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))

    # Left: Orenstein
    panel(axes[0], e_ev,
          [(obs_or_112["sigma_ll"],    COLOR_OR, r"Orenstein (112)"),
           (obs_or_common["sigma_ll"], "#7FB3D3", r"Orenstein (common)")],
          r"Orenstein: $|\sigma_{ll}|$ vs plane", r"$|\sigma^{(1)}|$")

    # Right: McCormick
    panel(axes[1], e_ev,
          [(obs_mc_112["sigma_ll"],    COLOR_MC, r"McCormick (112)"),
           (obs_mc_common["sigma_ll"], "#F0908A", r"McCormick (common)")],
          r"McCormick: $|\sigma_{ll}|$ vs plane", r"$|\sigma^{(1)}|$")

    fig.suptitle(r"Effect of crystallographic plane on $|\sigma_{ll}|$", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.6, w_pad=1.6)
    path = out_dir / "fig3_sigma_ll_plane_effect.png"
    fig.savefig(path, dpi=230, facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare Orenstein and McCormick WSM models at two planes.")
    ap.add_argument("--emin", type=float, default=0.1,
                    help="Minimum photon energy in eV (default: 0.1)")
    ap.add_argument("--emax", type=float, default=3.0,
                    help="Maximum photon energy in eV (default: 3.0)")
    ap.add_argument("--nw", type=int, default=200,
                    help="Number of frequency points (default: 200)")
    ap.add_argument("--grid-or", type=int, default=0,
                    help="Override Orenstein k-grid (0=use config default)")
    ap.add_argument("--grid-mc", type=int, default=30,
                    help="Override McCormick k-grid (default: 30)")
    ap.add_argument("--recompute", action="store_true",
                    help="Ignore cached .npz files and recompute")
    ap.add_argument("--no-fig3", action="store_true",
                    help="Skip optional Figure 3")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    omega = np.linspace(args.emin / _EV, args.emax / _EV, args.nw)
    e_ev = omega * _EV

    # --- Load and override configs ---
    cfg_or = build_config(CFG_ORENSTEIN,
                          k_points=tuple([args.grid_or] * 3) if args.grid_or > 0 else None)
    cfg_mc = build_config(CFG_MCCORMICK,
                          k_points=tuple([args.grid_mc] * 3))

    # --- Compute tensors (model-level, laser-angle-independent) ---
    s1_or, s2_or = compute_and_cache(cfg_or, OUT_DIR / "orenstein_tensor.npz",
                                      omega, label="orenstein",
                                      recompute=args.recompute)
    s1_mc, s2_mc = compute_and_cache(cfg_mc, OUT_DIR / "mccormick_tensor.npz",
                                      omega, label="mccormick",
                                      recompute=args.recompute)

    # --- Laser frames ---
    xd_112,    yd_112    = laser_frame_vectors(*PLANE_112[:2])
    xd_common, yd_common = laser_frame_vectors(*PLANE_COMMON[:2])

    # --- Project onto each plane ---
    obs_or_112    = project_observables(s1_or, s2_or, omega, xd_112, yd_112)
    obs_or_common = project_observables(s1_or, s2_or, omega, xd_common, yd_common)
    obs_mc_112    = project_observables(s1_mc, s2_mc, omega, xd_112, yd_112)
    obs_mc_common = project_observables(s1_mc, s2_mc, omega, xd_common, yd_common)

    # --- Matplotlib setup ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    # --- Generate figures ---
    f1 = make_figure1(e_ev, obs_or_112, obs_mc_112, OUT_DIR)
    f2 = make_figure2(e_ev, obs_or_common, obs_mc_common, OUT_DIR)
    f3 = None
    if not args.no_fig3:
        f3 = make_figure3(e_ev, obs_or_112, obs_or_common,
                          obs_mc_112, obs_mc_common, OUT_DIR)

    print(f"[comparison] Figure 1 -> {f1}")
    print(f"[comparison] Figure 2 -> {f2}")
    if f3:
        print(f"[comparison] Figure 3 -> {f3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
