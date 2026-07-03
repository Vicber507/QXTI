#!/usr/bin/env python3
"""Harmonic-yield-vs-parameter plot for a QXTI parameter sweep.

For a sweep produced by ``tools/run_parameter_sweep.py`` (one run per swept
value, each with ``cmd/data/current_spectrum.npz``), this builds a single plot of
the **harmonic yield** of selected harmonics as a function of the swept parameter
(default ``phix`` in degrees).

Harmonic yield of harmonic ``n`` = integrated spectral intensity around the peak,

    Y_n = \\int_{(n-hw) w0}^{(n+hw) w0} |J(w)|^2 dw ,

with the half-window ``hw`` given in units of the harmonic order (default 0.1, so
the window is n*w0 +/- 0.1*w0). Curves plotted:

  * H1 total  = |J(w)| = sqrt(|Jx|^2+|Jy|^2+|Jz|^2)
  * H2 total
  * H1 J_R    = (Jx - i Jy)/sqrt(2)   (right circular, in-plane)
  * H1 J_L    = (Jx + i Jy)/sqrt(2)   (left  circular, in-plane)

Usage
-----
    python tools/plot_sweep_harmonic_yield.py \
        --sweep outputs/sweeps/orenstein_phix

Options: --half-window (harmonic-order half width), --deg/--native (x axis),
--out (output png). The x value is parsed from each run folder name
(``NNN_phix=<value>deg``).
"""
from __future__ import annotations

import argparse
import configparser
import re
from pathlib import Path

import numpy as np

_AU_TO_EV = 27.211386245988


def _run_omega0(run_dir: Path) -> float:
    """Read the fundamental omega (a.u.) from the per-run input.cfg."""
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cfg.optionxform = str
    cfg.read(run_dir / "input.cfg")
    return float(cfg["laser"]["omega"])


def _parse_x(label: str) -> tuple[float, bool]:
    """Return (value, is_deg) parsed from a folder label like '007_phix=50.4deg'."""
    m = re.search(r"=([-+0-9.eE]+)(deg)?", label)
    if not m:
        return float("nan"), False
    return float(m.group(1)), bool(m.group(2))


def _band_yield(omega: np.ndarray, weight: np.ndarray, lo: float, hi: float) -> float:
    """Integral of ``weight`` (already |.|^2) over omega in [lo, hi]."""
    mask = (omega >= lo) & (omega <= hi)
    if mask.sum() < 2:
        return 0.0
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(weight[mask], omega[mask]))


def collect(sweep_dir: Path, half_window: float):
    runs = sorted(p for p in sweep_dir.iterdir()
                  if p.is_dir() and (p / "cmd" / "data" / "current_spectrum.npz").exists())
    xs, y1_tot, y2_tot, y1_R, y1_L, y2_R, y2_L = [], [], [], [], [], [], []
    is_deg_any = False
    for rd in runs:
        x, is_deg = _parse_x(rd.name)
        is_deg_any = is_deg_any or is_deg
        w0 = _run_omega0(rd)
        d = np.load(rd / "cmd" / "data" / "current_spectrum.npz", allow_pickle=True)
        omega = np.asarray(d["omega_axis"], dtype=float)
        sp = np.asarray(d["current_spectrum"], dtype=np.complex128)  # (Nw, 3)
        jx, jy, jz = sp[:, 0], sp[:, 1], sp[:, 2]

        total2 = np.abs(jx) ** 2 + np.abs(jy) ** 2 + np.abs(jz) ** 2   # |J|^2
        R2 = np.abs((jx - 1j * jy) / np.sqrt(2.0)) ** 2
        L2 = np.abs((jx + 1j * jy) / np.sqrt(2.0)) ** 2

        def win(n: int) -> tuple[float, float]:
            return ((n - half_window) * w0, (n + half_window) * w0)

        lo1, hi1 = win(1)
        lo2, hi2 = win(2)
        xs.append(x)
        y1_tot.append(_band_yield(omega, total2, lo1, hi1))
        y2_tot.append(_band_yield(omega, total2, lo2, hi2))
        y1_R.append(_band_yield(omega, R2, lo1, hi1))
        y1_L.append(_band_yield(omega, L2, lo1, hi1))
        y2_R.append(_band_yield(omega, R2, lo2, hi2))
        y2_L.append(_band_yield(omega, L2, lo2, hi2))

    order = np.argsort(xs)
    arr = lambda v: np.asarray(v)[order]
    return (arr(xs), arr(y1_tot), arr(y2_tot), arr(y1_R), arr(y1_L),
            arr(y2_R), arr(y2_L), is_deg_any)


def _xlabel(is_deg: bool) -> str:
    return r"$\varphi_x$ (deg)" if is_deg else r"$\varphi_x$ (a.u.)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default="outputs/sweeps/orenstein_phix",
                   help="Sweep directory (contains the per-run folders).")
    p.add_argument("--half-window", type=float, default=0.1,
                   help="Half window in harmonic-order units (default 0.1).")
    p.add_argument("--out", default=None, help="Combined-plot PNG (default <sweep>/harmonic_yield_vs_phix.png).")
    p.add_argument("--log", action="store_true", help="Log scale on the (raw) combined yield axis.")
    args = p.parse_args()

    sweep_dir = Path(args.sweep)
    xs, y1, y2, yR, yL, y2R, y2L, is_deg = collect(sweep_dir, args.half_window)
    if xs.size == 0:
        print(f"No runs with current_spectrum.npz under {sweep_dir}")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    xl = _xlabel(is_deg)
    # (label, data, color, marker) for the four curves.
    curves = [
        ("H1 total", r"H1 total $\int|J|^2$", y1, "#111827", "o", "h1_total"),
        ("H2 total", r"H2 total $\int|J|^2$", y2, "#D55E00", "s", "h2_total"),
        ("H1 J_R", r"H1 $J_\mathrm{R}$", yR, "#0072B2", "^", "h1_R"),
        ("H1 J_L", r"H1 $J_\mathrm{L}$", yL, "#CC79A7", "v", "h1_L"),
    ]

    def _norm(v: np.ndarray) -> np.ndarray:
        m = float(np.nanmax(v))
        return v / m if m > 0 else v

    # ---- 1) combined RAW yield -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for _name, lab, data, color, mk, _stem in curves:
        ax.plot(xs, data, "-" + mk, color=color, lw=1.9, ms=4, label=lab)
    ax.set_xlabel(xl)
    ax.set_ylabel(r"harmonic yield  $\int_{\mathrm{band}}|J(\omega)|^2\,d\omega$  (a.u.)")
    ax.set_title(f"Harmonic yield vs phix - {sweep_dir.name}")
    if args.log:
        ax.set_yscale("log")
    ax.legend(frameon=False, ncol=2)
    ax.margins(x=0.02)
    fig.tight_layout()
    out = Path(args.out) if args.out else sweep_dir / "harmonic_yield_vs_phix.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[yield] wrote {out}")

    # ---- 2) combined NORMALIZED yield (each curve / its own max) ----------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for _name, lab, data, color, mk, _stem in curves:
        ax.plot(xs, _norm(data), "-" + mk, color=color, lw=1.9, ms=4, label=lab)
    ax.set_xlabel(xl)
    ax.set_ylabel(r"normalized harmonic yield  $Y/Y_\mathrm{max}$")
    ax.set_title(f"Normalized harmonic yield vs phix - {sweep_dir.name}")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, ncol=2)
    ax.margins(x=0.02)
    fig.tight_layout()
    out_norm = sweep_dir / "harmonic_yield_vs_phix_normalized.png"
    fig.savefig(out_norm, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[yield] wrote {out_norm}")

    # ---- 3) individual plots (one per curve), normalized 0..1 -------------------
    indiv_dir = sweep_dir / "harmonic_yield_individual"
    indiv_dir.mkdir(exist_ok=True)
    for name, lab, data, color, mk, stem in curves:
        ymax = float(np.nanmax(data))
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.plot(xs, _norm(data), "-" + mk, color=color, lw=2.0, ms=4.5)
        ax.fill_between(xs, _norm(data), color=color, alpha=0.12)
        ax.set_xlabel(xl)
        ax.set_ylabel(r"normalized yield  $Y/Y_\mathrm{max}$")
        ax.set_title(f"{name} - normalized ($Y_\\mathrm{{max}}={ymax:.3g}$ a.u.)")
        ax.set_ylim(-0.02, 1.05)
        ax.margins(x=0.02)
        fig.tight_layout()
        fpng = indiv_dir / f"{stem}.png"
        fig.savefig(fpng, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"[yield]   individual: {fpng}  (Ymax={ymax:.3g})")

    # ---- 4) mirror-difference about 180 deg, per curve -------------------------
    # D(delta) = Y_norm(center + delta) - Y_norm(center - delta), on the
    # normalized curve so the asymmetry is read as a fraction of the max.
    center = 180.0
    reach = min(center - float(xs.min()), float(xs.max()) - center)
    deltas = np.asarray([d for d in np.unique(np.round(xs - center, 6)) if 0.0 < d <= reach + 1e-9])
    mdiff_dir = sweep_dir / "harmonic_yield_mirrordiff"
    mdiff_dir.mkdir(exist_ok=True)
    for name, _lab, data, color, mk, stem in curves:
        yn = _norm(data)
        plus = np.interp(center + deltas, xs, yn)
        minus = np.interp(center - deltas, xs, yn)
        diff = plus - minus
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.axhline(0.0, color="0.6", lw=0.9, zorder=0)
        ax.plot(deltas, diff, "-" + mk, color=color, lw=2.0, ms=4.5)
        ax.set_xlabel(r"offset from $180^\circ$:  $\delta$ (deg)")
        ax.set_ylabel(r"$Y(180^\circ{+}\delta)-Y(180^\circ{-}\delta)$  (norm.)")
        ax.set_title(f"{name} - mirror difference about $180^\\circ$")
        ax.margins(x=0.02)
        fig.tight_layout()
        fpng = mdiff_dir / f"{stem}_mirrordiff.png"
        fig.savefig(fpng, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"[yield]   mirror-diff: {fpng}  (max|D|={np.nanmax(np.abs(diff)):.3g})")

    # ---- 5) circular dichroism of the EMITTED harmonic vs phix -----------------
    # CD_n = (|J_R|^2 - |J_L|^2) / (|J_R|^2 + |J_L|^2)  per harmonic, band-integrated.
    def _cd(r: np.ndarray, l: np.ndarray) -> np.ndarray:
        s = r + l
        return np.where(s > 0, (r - l) / s, 0.0)

    cd1 = _cd(yR, yL)
    cd2 = _cd(y2R, y2L)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axhline(0.0, color="0.6", lw=0.9, zorder=0)
    ax.plot(xs, cd1, "-o", color="#0072B2", lw=2.0, ms=4.5, label="H1  (fundamental)")
    ax.plot(xs, cd2, "-s", color="#D55E00", lw=2.0, ms=4.5, label="H2  (SHG)")
    ax.set_xlabel(xl)
    ax.set_ylabel(r"circular dichroism  $\dfrac{|J_R|^2-|J_L|^2}{|J_R|^2+|J_L|^2}$")
    ax.set_title(f"Emitted circular dichroism vs phix - {sweep_dir.name}")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(frameon=False)
    ax.margins(x=0.02)
    fig.tight_layout()
    out_cd = sweep_dir / "circular_dichroism_vs_phix.png"
    fig.savefig(out_cd, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[yield] wrote {out_cd}  (H1 max|CD|={np.nanmax(np.abs(cd1)):.3g}, "
          f"H2 max|CD|={np.nanmax(np.abs(cd2)):.3g})")

    np.savez_compressed(sweep_dir / "harmonic_yield_vs_phix.npz",
                        phix=xs, h1_total=y1, h2_total=y2, h1_R=yR, h1_L=yL,
                        h2_R=y2R, h2_L=y2L, cd_h1=cd1, cd_h2=cd2,
                        half_window=args.half_window)
    print(f"[yield] {xs.size} runs, phix range [{xs.min():g}, {xs.max():g}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
