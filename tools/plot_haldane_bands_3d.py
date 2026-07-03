#!/usr/bin/env python3
r"""Paper-quality 3D band structure of the Haldane model: E(kx, ky).

Both bands of the honeycomb Haldane Hamiltonian plotted as smooth surfaces over
the Brillouin zone, coloured by energy, with a translucent Fermi plane at E=0 and
a shadow contour projected on the floor. Axes: kx, ky (AA^-1), E (eV).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT = Path("outputs/haldane_bands_3d")
NGRID = 161
_EV = 27.211386245988
_AA = 1.8897259886   # a.u.^-1 -> AA^-1


def _load():
    spec = importlib.util.spec_from_file_location("hald", PROJECT_ROOT / "models" / "haldane.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main() -> int:
    m = _load()
    p = m.default_params()
    a0 = float(p["a0"])
    # window covering the six Dirac points K = 4 pi / (3 sqrt3 a0)
    kD = 4.0 * np.pi / (3.0 * np.sqrt(3.0) * a0)
    kmax = 1.55 * kD
    kx = np.linspace(-kmax, kmax, NGRID)
    ky = np.linspace(-kmax, kmax, NGRID)
    KX, KY = np.meshgrid(kx, ky)

    Elo = np.empty_like(KX); Ehi = np.empty_like(KX)
    for j in range(NGRID):
        for i in range(NGRID):
            ev = np.linalg.eigvalsh(m.H(KX[j, i], KY[j, i], 0.0, p))
            Elo[j, i], Ehi[j, i] = ev[0] * _EV, ev[1] * _EV
    KXa, KYa = KX * _AA, KY * _AA

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.0, 7.4))
    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(vmin=min(Elo.min(), Ehi.min()), vmax=max(Elo.max(), Ehi.max()))
    cmap = plt.get_cmap("Spectral_r")
    for Z in (Elo, Ehi):
        ax.plot_surface(KXa, KYa, Z, facecolors=cmap(norm(Z)), rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False, alpha=0.97)

    # shadow contour of both bands on the floor
    zfloor = norm.vmin - 0.12 * (norm.vmax - norm.vmin)
    ax.contourf(KXa, KYa, Ehi, zdir="z", offset=zfloor, cmap="Spectral_r", alpha=0.55, levels=30)
    ax.contour(KXa, KYa, Ehi, zdir="z", offset=zfloor, colors="0.3", linewidths=0.4, levels=14)

    # translucent Fermi plane E = 0
    xxp = np.array([[KXa.min(), KXa.max()], [KXa.min(), KXa.max()]])
    yyp = np.array([[KYa.min(), KYa.min()], [KYa.max(), KYa.max()]])
    ax.plot_surface(xxp, yyp, np.zeros_like(xxp), color="0.55", alpha=0.12,
                    linewidth=0, shade=False, zorder=0)

    ax.set_xlabel(r"$k_x\ (\mathrm{\AA}^{-1})$", labelpad=12)
    ax.set_ylabel(r"$k_y\ (\mathrm{\AA}^{-1})$", labelpad=12)
    ax.set_zlabel(r"$E\ (\mathrm{eV})$", labelpad=10)
    ax.set_zlim(zfloor, norm.vmax * 1.02)
    ax.set_title(r"$\mathrm{Haldane\ model:\ band\ structure}\ E(k_x,k_y)$", fontsize=15, pad=6)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1, 1, 0.62))
    try:
        ax.set_facecolor("white"); ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0); ax.zaxis.pane.set_alpha(0.0)
    except Exception:
        pass

    mapp = cm.ScalarMappable(norm=norm, cmap=cmap); mapp.set_array([])
    cb = fig.colorbar(mapp, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(r"$E\ (\mathrm{eV})$")

    f = OUT / "haldane_bands_3d.png"
    fig.savefig(f, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
