#!/usr/bin/env python3
r"""Reproduce Fig. 2c of Wu/Orenstein et al. (Nat. Phys. 2017): the SHG circular
dichroism vs analyzer angle on the TaAs (112) surface.

Setup (same as the paper):
  * Incident light is circular: sigma+ (ellip=+1) and sigma- (ellip=-1), normally
    incident on the (112) surface (polarization plane = (112) plane).
  * An analyzer selects the emitted 2w field along a direction rotating in the
    (112) plane, at angle theta2 measured from the [1,1,-1] crystal axis:
        a(theta2) = cos(theta2) e1 + sin(theta2) e2 ,
        e1 = [1,1,-1]/sqrt3  (theta2 = 0),   e2 = [1,-1,0]/sqrt2  (theta2 = 90).
  * Detected SHG intensity through the analyzer:  I(theta2) = |a . J(2w)|^2
    (only the in-plane part of J(2w) radiates into the reflected beam; the
    component along the surface normal [1,1,2] is not analyzed).
  * Circular dichroism:  CD(theta2) = I_sigma+(theta2) - I_sigma-(theta2).

The 4mm point group predicts CD(theta2) ~ (2/9) Im{d15 d33*} sin(2 theta2)
(SI Eq. 7), i.e. a pure sin(2 theta2). The plot is normalized (as in the paper)
by the peak of the parallel-generator scan Fig. 1b, if a linear-scan directory is
supplied via --norm-sweep; otherwise it is normalized to its own peak.

Usage
-----
    python tools/plot_shg_cd_analyzer.py \
        --sigma-plus  outputs/sweeps/orenstein_112_cd/001_ellip=1 \
        --sigma-minus outputs/sweeps/orenstein_112_cd/000_ellip=-1 \
        --norm-sweep  outputs/sweeps/orenstein_112_linear \
        --out outputs/sweeps/orenstein_112_cd
"""
from __future__ import annotations

import argparse
import configparser
from pathlib import Path

import numpy as np

# High-symmetry in-plane axes of the (112) surface (paper's convention).
E1 = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)   # [1,1,-1], theta2 = 0
E2 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)   # [1,-1,0], theta2 = 90
NORMAL = np.array([1.0, 1.0, 2.0]) / np.sqrt(6.0)  # surface normal [1,1,2]


def _omega0(run_dir: Path) -> float:
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cfg.optionxform = str
    cfg.read(run_dir / "input.cfg")
    return float(cfg["laser"]["omega"])


def _J_at(run_dir: Path, harmonic: int) -> np.ndarray:
    """Complex Cartesian current vector at the given harmonic peak."""
    d = np.load(run_dir / "cmd" / "data" / "current_spectrum.npz", allow_pickle=True)
    omega = np.asarray(d["omega_axis"], dtype=float)
    sp = np.asarray(d["current_spectrum"], dtype=np.complex128)
    idx = int(np.argmin(np.abs(omega - harmonic * _omega0(run_dir))))
    return sp[idx]


def _parallel_peak(norm_sweep: Path) -> float | None:
    """Peak of the parallel-generator SHG scan (Fig. 1b): I_para(theta1) =
    |g . J(2w)|^2 with generator = analyzer = in-plane linear direction."""
    runs = sorted(p for p in norm_sweep.iterdir()
                  if p.is_dir() and (p / "cmd" / "data" / "current_spectrum.npz").exists())
    if not runs:
        return None
    from qxti.physics.laser import Laser
    best = 0.0
    for rd in runs:
        cfg = configparser.ConfigParser(inline_comment_prefixes=("#",))
        cfg.optionxform = str
        cfg.read(rd / "input.cfg")
        lz = cfg["laser"]
        L = Laser(omega=float(lz["omega"]), E0=float(lz["E0"]), ellip=float(lz["ellip"]),
                  ncycles=float(lz["ncycles"]), phix=float(lz["phix"]),
                  thetaz=float(lz["thetaz"]), phiz=float(lz["phiz"]))
        g = L.xdir  # generator (= analyzer) linear direction, in-plane
        J2 = _J_at(rd, 2)
        best = max(best, float(np.abs(np.dot(g, J2)) ** 2))
    return best if best > 0 else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sigma-plus", required=True, help="Run dir, ellip=+1 (RCP).")
    p.add_argument("--sigma-minus", required=True, help="Run dir, ellip=-1 (LCP).")
    p.add_argument("--norm-sweep", default=None,
                   help="Linear generator scan dir for Fig-1b normalization (optional).")
    p.add_argument("--out", default=None, help="Output dir (default: parent of --sigma-plus).")
    p.add_argument("--num-theta", type=int, default=361, help="Analyzer-angle samples.")
    p.add_argument("--remove-dc", action="store_true",
                   help="Subtract the theta2-averaged CD (TRS forbids it -> pure artifact), "
                        "so the physical CD oscillates around zero (crosses sign).")
    args = p.parse_args()

    sp_dir, sm_dir = Path(args.sigma_plus), Path(args.sigma_minus)
    out_dir = Path(args.out) if args.out else sp_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    Jp = _J_at(sp_dir, 2)   # sigma+
    Jm = _J_at(sm_dir, 2)   # sigma-

    theta = np.linspace(0.0, 2.0 * np.pi, args.num_theta)
    ct, st = np.cos(theta), np.sin(theta)
    # a(theta).J = cos*(e1.J) + sin*(e2.J)
    ap = ct * np.dot(E1, Jp) + st * np.dot(E2, Jp)
    am = ct * np.dot(E1, Jm) + st * np.dot(E2, Jm)
    Ip = np.abs(ap) ** 2
    Im = np.abs(am) ** 2
    cd = Ip - Im

    dc = float(np.mean(cd))   # theta2-averaged CD == analyzer-independent part (TRS: 0)
    if args.remove_dc:
        cd = cd - dc

    norm = None
    if args.norm_sweep:
        try:
            from qxti.core import QXTIConfig  # noqa: F401  (ensure import path ok)
        except Exception:
            pass
        norm = _parallel_peak(Path(args.norm_sweep))
    if norm is None or norm <= 0:
        norm = float(np.max(np.abs(cd))) or 1.0
        norm_label = "own peak"
    else:
        norm_label = "Fig-1b parallel peak"
    cd_n = cd / norm

    # sin(2 theta2) fit (amplitude by projection).
    s2 = np.sin(2.0 * theta)
    amp = float(np.trapz(cd_n * s2, theta) / np.trapz(s2 * s2, theta))
    deg = np.degrees(theta)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(0.0, color="0.6", lw=0.9, zorder=0)
    ax.plot(deg, cd_n, color="#6A0DAD", lw=2.2, label=r"$I_{\sigma^+}-I_{\sigma^-}$  (model)")
    ax.plot(deg, amp * s2, color="#D55E00", lw=1.3, ls="--",
            label=rf"fit $\propto\sin 2\theta_2$  (A={amp:+.3f})")
    ax.set_xlabel(r"analyzer angle  $\theta_2$ (deg)  [from $[1,1,\bar1]$]")
    ax.set_ylabel(r"circular dichroism  $I_{\sigma^+}-I_{\sigma^-}$")
    dc_note = " (TRS-forbidden DC removed)" if args.remove_dc else ""
    ax.set_title(f"SHG circular dichroism, TaAs (112){dc_note}")
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 45))
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fpng = out_dir / "shg_circular_dichroism_analyzer.png"
    fig.savefig(fpng, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(out_dir / "shg_circular_dichroism_analyzer.npz",
                        theta2_deg=deg, cd=cd, cd_normalized=cd_n,
                        I_sigma_plus=Ip, I_sigma_minus=Im, norm=norm, sin2_amplitude=amp)

    print(f"[CD-analyzer] J(2w) sigma+: in-plane |e1.J|={abs(np.dot(E1,Jp)):.3e} "
          f"|e2.J|={abs(np.dot(E2,Jp)):.3e}")
    print(f"[CD-analyzer] normalization = {norm:.4e} ({norm_label})")
    print(f"[CD-analyzer] sin(2 theta2) amplitude = {amp:+.4f}  (paper Fig 2c ~ +/-0.06)")
    print(f"[CD-analyzer] wrote {fpng}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
