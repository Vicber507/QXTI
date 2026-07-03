#!/usr/bin/env python3
r"""Paper-quality 3D band structure of the Wu/Orenstein WSM at kz=0: E(kx, ky).

The two central bands of the 4-band Hamiltonian (the ones that touch at the four
Weyl nodes) plotted as smooth surfaces over the kz=0 plane, coloured by energy,
with a translucent E=0 plane and the four Weyl nodes marked (red +, blue -).
Axes: kx, ky (AA^-1), E (eV).
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

OUT = Path("outputs/wsm_orenstein_bands_3d")
NGRID = 181
_EV = 27.211386245988
_AA = 1.8897259886   # a.u.^-1 -> AA^-1


def _load():
    spec = importlib.util.spec_from_file_location("wsm", PROJECT_ROOT / "models" / "wsm_orenstein.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main() -> int:
    m = _load()
    p = m.default_params()
    a = float(p["a"])
    # zoom around the nodes (kx ~ +-1.68, ky ~ +-0.46); show a bit beyond
    kxmax = 0.80 * np.pi / a
    kymax = 0.42 * np.pi / a
    kx = np.linspace(-kxmax, kxmax, NGRID)
    ky = np.linspace(-kymax, kymax, NGRID)
    KX, KY = np.meshgrid(kx, ky)

    E1 = np.empty_like(KX); E2 = np.empty_like(KX)   # two central bands (index 1,2)
    for j in range(NGRID):
        for i in range(NGRID):
            ev = np.linalg.eigvalsh(m.H(KX[j, i], KY[j, i], 0.0, p))
            E1[j, i], E2[j, i] = ev[1] * _EV, ev[2] * _EV
    KXa, KYa = KX * _AA, KY * _AA

    nodes = m.weyl_nodes_with_chirality(p)

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
    fig = plt.figure(figsize=(9.4, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    vlim = max(abs(E1.min()), abs(E2.max()))
    norm = plt.Normalize(vmin=-vlim, vmax=vlim)
    cmap = plt.get_cmap("Spectral_r")
    for Z in (E1, E2):
        ax.plot_surface(KXa, KYa, Z, facecolors=cmap(norm(Z)), rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False, alpha=0.96)

    # translucent E = 0 plane through the Weyl points
    xxp = np.array([[KXa.min(), KXa.max()], [KXa.min(), KXa.max()]])
    yyp = np.array([[KYa.min(), KYa.min()], [KYa.max(), KYa.max()]])
    ax.plot_surface(xxp, yyp, np.zeros_like(xxp), color="0.55", alpha=0.13,
                    linewidth=0, shade=False)

    col = {+1: "#E8000B", -1: "#1E64C8"}
    seen = set()
    for nd in nodes:
        k = nd["k"]; c = int(nd["chirality"])
        lab = rf"$\chi={c:+d}$" if c not in seen else None
        seen.add(c)
        ax.scatter([k[0] * _AA], [k[1] * _AA], [0.0], s=150, color=col[c],
                   edgecolors="white", linewidths=1.8, depthshade=False, zorder=20, label=lab)

    ax.set_xlabel(r"$k_x\ (\mathrm{\AA}^{-1})$", labelpad=12)
    ax.set_ylabel(r"$k_y\ (\mathrm{\AA}^{-1})$", labelpad=10)
    ax.set_zlabel(r"$E\ (\mathrm{eV})$", labelpad=8)
    ax.set_title(r"$\mathrm{Wu/Orenstein\ Weyl\ semimetal\ (}k_z{=}0\mathrm{):}\ E(k_x,k_y)$",
                 fontsize=15, pad=6)
    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect((1.0, 0.62, 0.55))
    try:
        ax.set_facecolor("white"); ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0); ax.zaxis.pane.set_alpha(0.0)
    except Exception:
        pass
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=11)

    mapp = cm.ScalarMappable(norm=norm, cmap=cmap); mapp.set_array([])
    cb = fig.colorbar(mapp, ax=ax, shrink=0.55, pad=0.11)
    cb.set_label(r"$E\ (\mathrm{eV})$")

    f = OUT / "wsm_orenstein_bands_3d.png"
    fig.savefig(f, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
