#!/usr/bin/env python3
"""Second-harmonic circular dichroism (CD) from an RCP/LCP run pair.

Circular dichroism = differential nonlinear response to right- vs left-circularly
polarized drive. Given two CMD/HHG runs -- one with ellip=+1 (RCP) and one with
ellip=-1 (LCP), same geometry otherwise -- this computes, per harmonic n, the
band-integrated yield

    Y^{+/-}_n = \\int_{(n-hw) w0}^{(n+hw) w0} |J(w)|^2 dw

for each helicity and the normalized circular dichroism

    CD_n = (Y^+_n - Y^-_n) / (Y^+_n + Y^-_n) .

It also plots the full |J(w)|^2 spectra of both helicities and the spectral CD(w)
around the fundamental and the second harmonic.

Usage
-----
    python tools/plot_circular_dichroism.py \
        --rcp outputs/sweeps/orenstein_112_cd/001_ellip=1 \
        --lcp outputs/sweeps/orenstein_112_cd/000_ellip=-1 \
        --out outputs/sweeps/orenstein_112_cd
"""
from __future__ import annotations

import argparse
import configparser
from pathlib import Path

import numpy as np

_AU_TO_EV = 27.211386245988


def _omega0(run_dir: Path) -> float:
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cfg.optionxform = str
    cfg.read(run_dir / "input.cfg")
    return float(cfg["laser"]["omega"])


def _load(run_dir: Path):
    d = np.load(run_dir / "cmd" / "data" / "current_spectrum.npz", allow_pickle=True)
    omega = np.asarray(d["omega_axis"], dtype=float)
    sp = np.asarray(d["current_spectrum"], dtype=np.complex128)
    inten = np.abs(sp[:, 0]) ** 2 + np.abs(sp[:, 1]) ** 2 + np.abs(sp[:, 2]) ** 2  # |J|^2
    return omega, inten


def _band_yield(omega, inten, lo, hi):
    m = (omega >= lo) & (omega <= hi)
    if m.sum() < 2:
        return 0.0
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(inten[m], omega[m]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rcp", required=True, help="Run dir with ellip=+1 (right circular).")
    p.add_argument("--lcp", required=True, help="Run dir with ellip=-1 (left circular).")
    p.add_argument("--out", default=None, help="Output dir (default: parent of --rcp).")
    p.add_argument("--half-window", type=float, default=0.1,
                   help="Half window in harmonic-order units (default 0.1).")
    p.add_argument("--max-harmonic", type=int, default=3, help="Report CD up to this harmonic.")
    args = p.parse_args()

    rcp_dir, lcp_dir = Path(args.rcp), Path(args.lcp)
    out_dir = Path(args.out) if args.out else rcp_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    w0 = _omega0(rcp_dir)
    wr, ir = _load(rcp_dir)
    wl, il = _load(lcp_dir)
    # Put both on the RCP grid (they share it, but interpolate to be safe).
    if wl.shape != wr.shape or not np.allclose(wl, wr):
        il = np.interp(wr, wl, il)
    omega = wr

    hw = args.half_window
    print(f"[CD] w0 = {w0:g} a.u. = {w0 * _AU_TO_EV:.4f} eV,  window = n*w0 +/- {hw}*w0")
    rows = []
    for n in range(1, args.max_harmonic + 1):
        lo, hi = (n - hw) * w0, (n + hw) * w0
        yp = _band_yield(omega, ir, lo, hi)   # RCP
        ym = _band_yield(omega, il, lo, hi)   # LCP
        cd = (yp - ym) / (yp + ym) if (yp + ym) > 0 else 0.0
        rows.append((n, yp, ym, cd))
        print(f"[CD] H{n} (E={n * w0 * _AU_TO_EV:.3f} eV):  Y_RCP={yp:.4e}  Y_LCP={ym:.4e}  "
              f"CD=(R-L)/(R+L)={cd:+.4f}")

    # ---- plot: spectra + spectral CD ------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    order = omega / w0  # harmonic-order axis
    sel = (order > 0.3) & (order < args.max_harmonic + 0.6)
    denom = ir + il
    cd_spec = np.where(denom > 0, (ir - il) / denom, 0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.semilogy(order[sel], ir[sel], color="#0072B2", lw=1.8, label=r"RCP ($\epsilon=+1$)  $|J|^2$")
    ax1.semilogy(order[sel], il[sel], color="#CC79A7", lw=1.8, ls="--", label=r"LCP ($\epsilon=-1$)  $|J|^2$")
    for n in range(1, args.max_harmonic + 1):
        ax1.axvspan(n - hw, n + hw, color="0.85", zorder=0)
    ax1.set_ylabel(r"$|J(\omega)|^2$  (a.u.)")
    ax1.legend(frameon=False)
    ax1.set_title(f"Second-harmonic circular dichroism - {out_dir.name}")

    ax2.axhline(0.0, color="0.6", lw=0.9)
    ax2.plot(order[sel], cd_spec[sel], color="#111827", lw=1.6)
    for n in range(1, args.max_harmonic + 1):
        ax2.axvspan(n - hw, n + hw, color="0.85", zorder=0)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_xlabel(r"harmonic order  $\omega/\omega_0$")
    ax2.set_ylabel(r"CD$(\omega)=\dfrac{|J_R|^2-|J_L|^2}{|J_R|^2+|J_L|^2}$")
    # Annotate the H2 CD value.
    cd2 = next(cd for (nn, _a, _b, cd) in rows if nn == 2)
    ax2.annotate(rf"$\mathrm{{CD}}_{{2\omega}}={cd2:+.3f}$", xy=(2.0, cd2),
                 xytext=(2.15, 0.6), fontsize=11,
                 arrowprops=dict(arrowstyle="->", color="#D55E00"), color="#D55E00")
    fig.tight_layout()

    fpng = out_dir / "circular_dichroism_shg.png"
    fig.savefig(fpng, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(out_dir / "circular_dichroism_shg.npz",
                        harmonic=np.array([r[0] for r in rows]),
                        yield_rcp=np.array([r[1] for r in rows]),
                        yield_lcp=np.array([r[2] for r in rows]),
                        cd=np.array([r[3] for r in rows]),
                        order_axis=order, cd_spectrum=cd_spec, half_window=hw)
    print(f"[CD] wrote {fpng}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
