#!/usr/bin/env python3
r"""SHG circular dichroism of TaAs (112) vs analyzer angle, with Weyl-node masks.

Reproduces the paper's Fig. 2c from the model, the RIGHT way: the CD is built from
the CONVERGED grid-based analytic chi^(2) tensor (the same BZ-integral method the
paper uses for Fig. 4d), NOT the per-k HHG recursion (which does not converge).

From the 4mm formula (Wu/Orenstein SI Eq. 7), the SHG circular dichroism on the
(112) surface is pure sin(2 theta2):

    CD(theta2) = 6 * Im(sigma_xzx sigma_zzz*) / |sigma_zzz + 4 sigma_xzx + 2 sigma_zxx|^2 * sin(2 theta2)

(the sigma->chi and d=chi/2 factors cancel in this normalized ratio). Laser is
800 nm (the paper). Three cases via a multiplicative node mask on the BZ weights:
all nodes, without chi=+1 nodes, without chi=-1 nodes.
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

from qxti.analytics.theory_response import build_k_integration_weights
from qxti.core.config import QXTIConfig
from qxti.core.simulation import QXTISimulation
from plot_wsm_helicity_sigma_node_masks import _build_local_node_mask, _min_nonzero_pair_distance
from _shg_tensor_gridbased import order2_full_tensor

_AU_TO_EV = 27.211386245988
OMEGA_800NM = (1239.84159 / 800.0) / _AU_TO_EV   # paper's fundamental (~0.05696 a.u.)
GRID = 60                                         # converged grid for the tensor
DEFAULT_CONFIG = Path("inputs/inputParams.wsm_orenstein.cfg")

CASES = (
    ("full", "All nodes", None, "#111827"),
    ("no_pos", r"Without $\chi=+1$ nodes", +1, "#0072B2"),
    ("no_neg", r"Without $\chi=-1$ nodes", -1, "#D55E00"),
)


def _load_nodes(config: QXTIConfig):
    src = Path(config.hamiltonian.source_file)
    path = src if src.is_absolute() else (PROJECT_ROOT / "models" / src.name)
    spec = importlib.util.spec_from_file_location(f"qxti_cdmask_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.weyl_nodes_with_chirality(dict(config.hamiltonian.params)))


def _cd_amplitude(config, ham, kgrid, mask):
    """A = 6 Im(s_xzx s_zzz*)/|s_zzz+4 s_xzx+2 s_zxx|^2  (the sin(2theta2) amplitude)."""
    wts = build_k_integration_weights(config, hamiltonian=ham, kgrid=kgrid, extra_k_weight_mask=mask)
    sig = order2_full_tensor(ham, kgrid, OMEGA_800NM, wts, config.susceptibility_solver)
    s33, s15, s31 = sig[2, 2, 2], sig[0, 2, 0], sig[2, 0, 0]      # zzz, xzx, zxx
    A = 6.0 * np.imag(s15 * np.conj(s33)) / np.abs(s33 + 4.0 * s15 + 2.0 * s31) ** 2
    return float(A), (s33, s15, s31)


def main() -> int:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = replace(QXTIConfig.from_file(cfg_path),
                     kgrid=replace(QXTIConfig.from_file(cfg_path).kgrid, k_points=[GRID, GRID, GRID]))

    nodes = _load_nodes(config)
    sim = QXTISimulation(config=config)
    ham = sim.build_hamiltonian()
    kgrid = sim.build_kgrid(ham)
    k_points = np.asarray(kgrid.points(), dtype=np.float64)
    positions = np.asarray([np.asarray(n["k"], dtype=np.float64) for n in nodes])
    radius = 0.35 * _min_nonzero_pair_distance(positions)
    print(f"[cd-mask] 800 nm (omega={OMEGA_800NM:.5f} a.u.={OMEGA_800NM*_AU_TO_EV:.3f} eV), "
          f"grid {GRID}^3 -> {kgrid.total_points} pts, radio mascara={radius:.4f}")
    for i, n in enumerate(nodes, 1):
        print(f"[cd-mask]   nodo {i}: chi={int(n['chirality']):+d}  k={np.round(n['k'],4)}")

    amps = {}
    for name, label, chir, _color in CASES:
        mask, _c, _r = _build_local_node_mask(
            k_points, tagged_nodes=nodes, chirality_to_remove=chir, radius=radius)
        removed = int(np.count_nonzero(mask == 0.0))
        A, (s33, s15, s31) = _cd_amplitude(config, ham, kgrid, mask)
        amps[name] = A
        print(f"[cd-mask] '{name}': excluidos {removed}/{k_points.shape[0]}  "
              f"A_CD={A:+.4f}  |s33|={abs(s33):.2e} |s15|={abs(s15):.2e} |s31|={abs(s31):.2e}")

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    s2 = np.sin(2.0 * theta)
    norm = abs(amps["full"]) or 1.0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhline(0.0, color="0.6", lw=0.9, zorder=0)
    deg = np.degrees(theta)
    for name, label, _chir, color in CASES:
        ax.plot(deg, (amps[name] / norm) * s2, color=color, lw=2.2, label=label)
    ax.set_xlabel(r"analyzer angle  $\theta_2$ (deg)  [from $[1,1,\bar1]$]")
    ax.set_ylabel("CD (normalized)")
    ax.set_title(r"SHG circular dichroism, TaAs (112) - 800 nm")
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 45)); ax.set_ylim(-1.15, 1.15)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    out_dir = Path("outputs/wsm_orenstein_cd_node_masks")
    out_dir.mkdir(parents=True, exist_ok=True)
    fpng = out_dir / "shg_cd_node_masks.png"
    fig.savefig(fpng, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(out_dir / "shg_cd_node_masks.npz", theta2_deg=deg,
                        amp_full=amps["full"], amp_no_pos=amps["no_pos"], amp_no_neg=amps["no_neg"])
    print(f"[cd-mask] wrote {fpng}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
