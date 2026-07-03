#!/usr/bin/env python3
r"""Surface Fermi arcs of the Wu/Orenstein TaAs Weyl model (z-normal slab).

A bulk kx-ky cut at kz=0 can only show the bulk Weyl points, never the arcs:
Fermi arcs are SURFACE states and require an open boundary. Here we build a slab
that is finite (open) along z and keeps (kx, ky) as good quantum numbers, so all
four nodes project into the kx-ky surface BZ. We plot the surface-projected
spectral weight at E=0,

    A_surf(kx,ky) = sum_a  w_a  eta/pi / (E_a^2 + eta^2),

where w_a is the weight of eigenstate a on the outermost layers (top surface).
The arcs connect opposite-chirality node projections; nodes are overlaid with
their (corrected, Berry-flux-verified) chirality.

Slab construction (Fourier transform only kz):
    onsite H0(kx,ky) = t[ (cos kxa + my(1-cos kya) + mz) sx@I
                          + sin kya sy@I + Delta cos kya sy@sx ]
    inter-layer  T   = t[ -mz/2 sx@I + (-i/2) sz@sx ]   (coeff of e^{i kz a})
so H(kz) = H0 + T e^{ikza} + T^dag e^{-ikza} reproduces the bulk model exactly.
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
NZ = 30            # slab layers along z
NGRID = 181        # kx, ky samples
N_SURF = 4         # outermost layers defining the "surface"
ETA_FRAC = 0.02    # broadening as fraction of t

_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)
SX_I = np.kron(_sx, _I2)
SY_I = np.kron(_sy, _I2)
SY_SX = np.kron(_sy, _sx)
SZ_SX = np.kron(_sz, _sx)


def _load_model():
    spec = importlib.util.spec_from_file_location("wsm_o", PROJECT_ROOT / "models" / "wsm_orenstein.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    m = _load_model()
    p = m.default_params()
    t = float(p["t"]); my = float(p["my"]); mz = float(p["mz"])
    delta = float(p["Delta"]); a = float(p["a"])
    eta = ETA_FRAC * t

    # inter-layer hopping (coeff of e^{i kz a}); Hermitian conjugate goes down.
    T = t * (-0.5 * mz * SX_I + (-0.5j) * SZ_SX)
    Tdag = T.conj().T

    # surface projection weights: outermost N_SURF layers, both faces summed.
    surf = np.zeros(4 * NZ)
    for n in list(range(N_SURF)) + list(range(NZ - N_SURF, NZ)):
        surf[4 * n:4 * n + 4] = 1.0

    kx = np.linspace(-np.pi / a, np.pi / a, NGRID)
    ky = np.linspace(-np.pi / a, np.pi / a, NGRID)
    A = np.zeros((NGRID, NGRID))

    off_upper = np.zeros((4 * NZ, 4 * NZ), dtype=complex)
    off_lower = np.zeros((4 * NZ, 4 * NZ), dtype=complex)
    for n in range(NZ - 1):
        off_upper[4 * n:4 * n + 4, 4 * (n + 1):4 * (n + 1) + 4] = T
        off_lower[4 * (n + 1):4 * (n + 1) + 4, 4 * n:4 * n + 4] = Tdag
    off = off_upper + off_lower

    for j, kyj in enumerate(ky):
        for i, kxi in enumerate(kx):
            kxa, kya = a * kxi, a * kyj
            f0 = np.cos(kxa) + my * (1.0 - np.cos(kya)) + mz  # onsite sx@I coeff
            H0 = t * (f0 * SX_I + np.sin(kya) * SY_I + delta * np.cos(kya) * SY_SX)
            Hs = off.copy()
            for n in range(NZ):
                Hs[4 * n:4 * n + 4, 4 * n:4 * n + 4] = H0
            evals, evecs = np.linalg.eigh(Hs)
            w = surf @ (np.abs(evecs) ** 2)              # surface weight per state
            A[j, i] = np.sum(w * (eta / np.pi) / (evals ** 2 + eta ** 2))
        if j % 30 == 0:
            print(f"  row {j+1}/{NGRID}")

    nodes = m.weyl_nodes_with_chirality(p)

    # Convert wavevectors from atomic units (a.u.^-1 = per Bohr) to inverse
    # angstrom for display: 1 Bohr^-1 = AU_PER_ANGSTROM AA^-1.
    to_inv_ang = float(getattr(m, "AU_PER_ANGSTROM", 1.8897259886))
    kx_A, ky_A = kx * to_inv_ang, ky * to_inv_ang

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
        apply_paper_style()
    except Exception:
        pass

    from matplotlib.colors import LinearSegmentedColormap
    # "cotton-candy sunset": deep plum -> purple -> magenta -> pink -> cream.
    cute = LinearSegmentedColormap.from_list("cotton_candy", [
        "#1b1033", "#5c2a72", "#b83a91", "#f47fa4", "#ffe5b4"])

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.pcolormesh(kx_A, ky_A, np.log10(A + A.max() * 1e-4), cmap=cute, shading="auto")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}\, A_{\mathrm{surf}}\!\left(k_x,k_y,\,E{=}0\right)$"
                 "\n" r"$(\mathrm{arb.\ units})$", fontsize=13)

    col = {+1: "#E8000B", -1: "#1E64C8"}   # + red, - blue
    mrk = {+1: "$+$", -1: "$-$"}
    seen = set()
    for nd in nodes:
        k = nd["k"]; c = int(nd["chirality"])
        lab = None
        if c not in seen:
            lab = rf"$\chi = {c:+d}$"; seen.add(c)
        ax.scatter([k[0] * to_inv_ang], [k[1] * to_inv_ang], s=250, facecolors=col[c],
                   edgecolors="white", linewidths=2.2, zorder=5, label=lab)
        ax.text(k[0] * to_inv_ang, k[1] * to_inv_ang, mrk[c], color="white",
                ha="center", va="center", fontsize=13, fontweight="bold", zorder=6)

    ax.set_xlabel(r"$k_x \;\; (\mathrm{\AA}^{-1})$")
    ax.set_ylabel(r"$k_y \;\; (\mathrm{\AA}^{-1})$")
    ax.set_title(r"$\mathrm{TaAs\ Weyl\ model:\ surface\ Fermi\ arcs}$"
                 "\n"
                 rf"$z\mathrm{{-slab}},\ N_z = {NZ},\ E = 0\ \mathrm{{surface\ spectral\ weight}}$",
                 fontsize=13)
    ax.set_aspect("equal")
    ax.legend(title=r"$\mathrm{Weyl\ node}$", frameon=True, facecolor="white",
              framealpha=0.9, fontsize=11, title_fontsize=11, loc="upper right")
    fig.tight_layout()
    f = OUT / "fermi_arcs_surface_weyl_nodes.png"
    fig.savefig(f, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
