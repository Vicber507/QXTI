#!/usr/bin/env python3
r"""Bulk spectral map A(kx,ky,E=0) at kz=0 with the four Weyl nodes marked.

This is the BULK spectral function (no surface) -> it shows the Weyl points
themselves as the high-weight loci, NOT Fermi arcs (those need the z-slab, see
tools/plot_wsm_orenstein_fermi_arcs.py). Nodes carry their Berry-flux-verified
chirality: chi=+1 in red, chi=-1 in blue.
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

OUT = Path("outputs/wsm_orenstein_ldos")
NGRID = 320
ETA = 0.03      # broadening (fraction not used; absolute a.u.-ish on band scale)


def _load_model():
    spec = importlib.util.spec_from_file_location("wsm_o", PROJECT_ROOT / "models" / "wsm_orenstein.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    m = _load_model()
    p = m.default_params()
    a = float(p["a"])
    eta = ETA * float(p["t"])

    kx = np.linspace(-np.pi / a, np.pi / a, NGRID)
    ky = np.linspace(-np.pi / a, np.pi / a, NGRID)
    A = np.zeros((NGRID, NGRID))
    for j, kyj in enumerate(ky):
        for i, kxi in enumerate(kx):
            ev = np.linalg.eigvalsh(m.H(kxi, kyj, 0.0, p))
            A[j, i] = np.sum((eta / np.pi) / (ev ** 2 + eta ** 2))
        if j % 60 == 0:
            print(f"  row {j+1}/{NGRID}")

    nodes = m.weyl_nodes_with_chirality(p)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    from matplotlib.colors import LinearSegmentedColormap
    # same "cotton-candy sunset" palette as the Fermi-arc map.
    cute = LinearSegmentedColormap.from_list("cotton_candy", [
        "#1b1033", "#5c2a72", "#b83a91", "#f47fa4", "#ffe5b4"])

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.pcolormesh(kx, ky, np.log10(A + A.max() * 1e-4), cmap=cute, shading="auto")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10} A(k_x,k_y,E{=}0,\,k_z{=}0)$")

    col = {+1: "#E8000B", -1: "#1E64C8"}   # + red, - blue
    mrk = {+1: "$+$", -1: "$-$"}
    seen = set()
    for nd in nodes:
        k = nd["k"]; c = int(nd["chirality"])
        lab = rf"$\chi={c:+d}$" if c not in seen else None
        seen.add(c)
        ax.scatter([k[0]], [k[1]], s=240, facecolors=col[c], edgecolors="white",
                   linewidths=2.2, zorder=5, label=lab)
        ax.text(k[0], k[1], mrk[c], color="white", ha="center", va="center",
                fontsize=13, fontweight="bold", zorder=6)

    ax.set_xlabel(r"$k_x$"); ax.set_ylabel(r"$k_y$")
    ax.set_title(r"TaAs Weyl model - bulk spectral map ($k_z=0$)"
                 "\n" r"Weyl points with Berry-flux chirality", fontsize=11)
    ax.set_aspect("equal")
    ax.legend(frameon=True, facecolor="white", framealpha=0.85, fontsize=10, loc="upper right")
    fig.tight_layout()
    f = OUT / "spectral_map_weyl_nodes.png"
    fig.savefig(f, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
