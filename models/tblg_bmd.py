"""Twisted bilayer graphene (TBLG) — Bistritzer–MacDonald continuum model.

"Graphene with a twist/rotation parameter": two graphene layers rotated by an
angle ``theta``. This is the canonical low-energy continuum model (Bistritzer &
MacDonald, PNAS 108, 12233 (2011)) — the *seed* on which the Carr–Fang–Zhu–
Kaxiras 2019 "exact continuum model" (Phys. Rev. Research 1, 013001) is built
(their reference [13]). It reproduces the moiré flat bands and the magic angle
(~1.05°) where the Fermi velocity collapses.

=============================================================================
                    LIMITATIONS — READ BEFORE USING
=============================================================================
This is the *minimal* BMD model, NOT the full relaxed Carr et al. (2019) model.
What is and is NOT here, and where it fails:

1. NOT the relaxed ab-initio model. Carr et al. add (a) lattice relaxation
   (in-plane strain A(r) + out-of-plane corrugation), (b) higher k·p shells,
   and (c) k-dependent interlayer terms fitted to DFT. Those need their fitted
   parameter tables (github.com/stcarr/kp_tblg) and are NOT reproduced here.
   Consequence: the magic angle and particle-hole asymmetry are only
   qualitatively right; with the simple constants below the magic angle lands
   near ~1.1–1.2°, not exactly 1.05°. Relaxation (w_AA < w_AB) is included as a
   crude knob but not the full r-dependent fields.

2. Large, truncation-dependent basis. The continuum H(k) is an (infinite)
   plane-wave matrix on the moiré reciprocal lattice, truncated to ``_N_RINGS``
   shells -> BASIS_SIZE = 4*(1+3N(N+1)) bands (N=2 -> 76 bands). Increasing
   ``_N_RINGS`` improves convergence but CHANGES BASIS_SIZE and makes everything
   slower (QXTI diagonalises per k-point and loops over band PAIRS ~ nb^2).
   For optical response this is heavy.

3. Continuum periodicity is only up to a gauge. H(k + G_moire) equals H(k) only
   after a unitary RELABELLING of the plane-wave basis, not literally. QXTI's
   covariant-gradient / Wilson-link transport and the FFT propagation assume
   literal periodicity H(k+G)=H(k). They are therefore only approximately valid
   at the moiré-BZ boundary. Keep k-grids in the BZ interior; expect small
   boundary artifacts in 2nd+ order response.

4. The moiré BZ scale depends on theta — AND IS HANDLED AUTOMATICALLY. The
   ``default_lattice(params)`` hook below recomputes the reciprocal box from the
   actual ``theta_deg`` of your input (k_theta ∝ sin(theta/2)), so the k-grid
   always covers exactly one moiré Brillouin zone. Just set ``theta_deg`` in
   ``[hamiltonian]`` and the BZ follows. (No manual rescaling needed.)

5. Gapless Dirac cones. The model has Dirac touchings (gap = 0) at the moiré K,
   K'. Like graphene, a k-grid landing on them gives ambiguous eigenvectors;
   rely on ``[kgrid] shifted = true`` + the automatic degeneracy guard. Near the
   magic angle the flat bands are also nearly degenerate (meV), so the band
   gauge is delicate.

6. Energy/optical scale is meV / THz, not eV. The flat-band physics lives at
   ~1–50 meV, so the interesting optical response is in the THz range
   (hbar*omega ~ meV). The eV-scale laser parameters used for the other models
   are NOT appropriate here; use omega ~ 1e-4–1e-3 a.u. and long pulses.

USABILITY FOR OUR PURPOSE (QXTI optical response):
  - Band structure / DOS / magic-angle studies: YES, works well (tested below).
  - Linear response sigma^(1)(omega) via the THEORY engine on a modest moiré
    k-grid: feasible but heavy (76 bands) and only quantitatively trustworthy in
    the BZ interior (see #3).
  - Time-domain CMD / HHG: NOT recommended — large basis x long THz pulses is
    very expensive, and the gauge/periodicity caveats (#3) bite hardest at
    high order. Treat any 2nd+ order result as exploratory.
  - For quantitative TBLG nonlinear optics, the relaxed Carr model (#1) and a
    proper moiré-periodic gauge implementation would be required.

All quantities are in atomic units (Hartree, Bohr).
=============================================================================
"""
from __future__ import annotations

import numpy as np

MODEL_NAME = "tblg-bmd-continuum"
DIMENSION = 2
BASIS_TYPE = "moire_planewave_layer_sublattice"
IS_PERIODIC = True  # only up to a basis relabelling — see LIMITATIONS #3

# ── Pauli matrices ──────────────────────────────────────────────────────────
_S0 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_OMEGA = np.exp(2j * np.pi / 3.0)  # e^{i 2pi/3}

# ── physical constants (atomic units) ───────────────────────────────────────
# a = 2.46 Angstrom (graphene lattice constant) in Bohr.
_A_BOHR = 2.46 * 1.8897259886
# hbar*v_F ≈ 5.253 eV·Angstrom (Koshino 2018) in Hartree·Bohr.
_HBAR_VF = 5.253 * 0.0367493 * 1.8897259886
# Interlayer tunnelling (Hartree). Relaxed values w_AA < w_AB.
_W_AA = 0.0817 * 0.0367493   # ≈ 0.0030 Ha
_W_AB = 0.1100 * 0.0367493   # ≈ 0.0040 Ha

# Moiré-lattice truncation. BASIS_SIZE = 4 * (1 + 3 N (N+1)).
#   N=1 -> 7 pts -> 28 bands (coarse), N=2 -> 19 pts -> 76 bands (default),
#   N=3 -> 37 pts -> 148 bands (more converged, slow).
_N_RINGS = 2

DEFAULT_PARAMS: dict[str, float] = {
    "theta_deg": 1.10,     # twist angle (degrees)
    "a": _A_BOHR,          # graphene lattice constant (Bohr)
    "hbar_vF": _HBAR_VF,   # Dirac velocity (Hartree·Bohr)
    "w_aa": _W_AA,         # interlayer AA coupling (Hartree)
    "w_ab": _W_AB,         # interlayer AB coupling (Hartree)
}


def _hex_points(n_rings: int) -> list[tuple[int, int]]:
    """Integer moiré-lattice indices (i, j) within ``n_rings`` hexagonal shells."""
    pts = []
    for i in range(-n_rings, n_rings + 1):
        for j in range(-n_rings, n_rings + 1):
            if max(abs(i), abs(j), abs(i + j)) <= n_rings:
                pts.append((i, j))
    return pts


# Fixed integer point set (theta-independent) -> fixes BASIS_SIZE at import time.
_POINTS = _hex_points(_N_RINGS)
_INDEX = {ij: idx for idx, ij in enumerate(_POINTS)}
BASIS_SIZE = 4 * len(_POINTS)


def _geometry(theta_deg: float, a: float):
    """Return moiré reciprocal vectors g1, g2 and the interlayer momentum q1."""
    theta = np.radians(float(theta_deg))
    K = 4.0 * np.pi / (3.0 * a)              # |K| of monolayer graphene
    k_theta = 2.0 * K * np.sin(theta / 2.0)  # moiré BZ scale
    s3 = np.sqrt(3.0)
    q1 = k_theta * np.array([0.0, -1.0])
    q2 = k_theta * np.array([s3 / 2.0, 0.5])
    q3 = k_theta * np.array([-s3 / 2.0, 0.5])
    g1 = q2 - q1                              # moiré reciprocal lattice vectors
    g2 = q3 - q1
    return g1, g2, q1, theta, k_theta


def _interlayer_T(w0: float = _W_AA, w1: float = _W_AB) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three interlayer tunnelling matrices (sublattice space).

    ``w0`` (AA) and ``w1`` (AB) are taken from the resolved model parameters so
    the input file actually controls the interlayer coupling.
    """
    # T_{j} = w0 * s0 + w1 * (cos(j phi) sx + sin(j phi) sy), phi = 2pi/3.
    T1 = w0 * _S0 + w1 * _SX
    T2 = w0 * _S0 + w1 * np.array([[0, _OMEGA.conjugate()], [_OMEGA, 0]], dtype=complex)
    T3 = w0 * _S0 + w1 * np.array([[0, _OMEGA], [_OMEGA.conjugate(), 0]], dtype=complex)
    return T1, T2, T3


def _dirac(px: float, py: float, phi: float, hbar_vF: float) -> np.ndarray:
    """Rotated monolayer Dirac block h = hbar vF (R(phi) p) · (sx, sy)."""
    c, s = np.cos(phi), np.sin(phi)
    rx = c * px - s * py
    ry = s * px + c * py
    return hbar_vF * (rx * _SX + ry * _SY)


def default_params() -> dict[str, float]:
    return dict(DEFAULT_PARAMS)


def H(kx: float, ky: float, kz: float, params: dict[str, object] | None = None) -> np.ndarray:
    """BMD continuum Hamiltonian H(k) for momentum (kx, ky) in the moiré BZ.

    kz is ignored (2D). Returns a Hermitian (BASIS_SIZE, BASIS_SIZE) matrix in
    atomic units, in the basis [moiré point, layer(bottom/top), sublattice(A/B)].
    """
    p = default_params()
    if params:
        p.update({k: v for k, v in params.items() if k in p})
    a = float(p["a"])
    hbar_vF = float(p["hbar_vF"])
    g1, g2, q1, theta, _ = _geometry(p["theta_deg"], a)
    T1, T2, T3 = _interlayer_T(float(p["w_aa"]), float(p["w_ab"]))

    nb = BASIS_SIZE
    Hm = np.zeros((nb, nb), dtype=complex)
    k = np.array([float(kx), float(ky)])

    # Diagonal: per moiré point, a bottom (−θ/2) and top (+θ/2) Dirac block.
    for idx, (i, j) in enumerate(_POINTS):
        base = 4 * idx
        Gm = i * g1 + j * g2
        pb = k + Gm                    # bottom layer momentum
        pt = k + Gm + q1               # top layer momentum (shifted by q1)
        Hm[base:base + 2, base:base + 2] = _dirac(pb[0], pb[1], -theta / 2.0, hbar_vF)
        Hm[base + 2:base + 4, base + 2:base + 4] = _dirac(pt[0], pt[1], +theta / 2.0, hbar_vF)

    # Off-diagonal: bottom(i,j) couples to top(i,j) [T1], top(i+1,j) [T2],
    # top(i,j+1) [T3]. Hermitian partner added automatically.
    couplings = (((0, 0), T1), ((1, 0), T2), ((0, 1), T3))
    for idx, (i, j) in enumerate(_POINTS):
        b = 4 * idx                    # bottom block rows [b, b+2)
        for (di, dj), T in couplings:
            nbr = _INDEX.get((i + di, j + dj))
            if nbr is None:
                continue               # truncated neighbour: dropped (see #2)
            t = 4 * nbr + 2            # top block of the neighbour
            Hm[b:b + 2, t:t + 2] += T
            Hm[t:t + 2, b:b + 2] += T.conj().T

    return Hm


def H_batch(kpts, params=None):
    """Vectorised version of :func:`H` over many k-points.

    ``kpts`` is an ``(nk, 3)`` array of momenta; returns an
    ``(nk, BASIS_SIZE, BASIS_SIZE)`` complex128 stack. The interlayer
    tunnelling blocks and the moiré geometry are k-independent, so only the
    diagonal Dirac blocks vary with k — these are filled as ``(nk,)`` arrays.
    Bit-exact replica of :func:`H` (kz is ignored, 2D).
    """
    p = default_params()
    if params:
        p.update({k: v for k, v in params.items() if k in p})
    a = float(p["a"])
    hbar_vF = float(p["hbar_vF"])
    g1, g2, q1, theta, _ = _geometry(p["theta_deg"], a)
    T1, T2, T3 = _interlayer_T(float(p["w_aa"]), float(p["w_ab"]))

    kpts = np.asarray(kpts, dtype=np.float64)
    kx, ky = kpts[:, 0], kpts[:, 1]
    nk = kx.shape[0]
    nb = BASIS_SIZE
    Hm = np.zeros((nk, nb, nb), dtype=complex)

    # Rotation matrices for the bottom (−θ/2) and top (+θ/2) Dirac blocks.
    cb, sb = np.cos(-theta / 2.0), np.sin(-theta / 2.0)
    ct, st = np.cos(+theta / 2.0), np.sin(+theta / 2.0)

    # Diagonal: per moiré point, a bottom (−θ/2) and top (+θ/2) Dirac block.
    # _dirac(px, py, phi) = hbar_vF * ((c*px - s*py) * _SX + (s*px + c*py) * _SY).
    for idx, (i, j) in enumerate(_POINTS):
        base = 4 * idx
        Gm = i * g1 + j * g2
        pbx = kx + Gm[0]               # bottom layer momentum
        pby = ky + Gm[1]
        ptx = kx + Gm[0] + q1[0]       # top layer momentum (shifted by q1)
        pty = ky + Gm[1] + q1[1]

        rbx = cb * pbx - sb * pby
        rby = sb * pbx + cb * pby
        rtx = ct * ptx - st * pty
        rty = st * ptx + ct * pty

        # bottom Dirac block (rows/cols base..base+2)
        Hm[:, base + 0, base + 1] = hbar_vF * (rbx - 1j * rby)  # sx - i sy off-diag
        Hm[:, base + 1, base + 0] = hbar_vF * (rbx + 1j * rby)
        # top Dirac block (rows/cols base+2..base+4)
        Hm[:, base + 2, base + 3] = hbar_vF * (rtx - 1j * rty)
        Hm[:, base + 3, base + 2] = hbar_vF * (rtx + 1j * rty)

    # Off-diagonal (k-independent): bottom(i,j) -> top(i,j) [T1], top(i+1,j)
    # [T2], top(i,j+1) [T3]. Hermitian partner added automatically.
    couplings = (((0, 0), T1), ((1, 0), T2), ((0, 1), T3))
    for idx, (i, j) in enumerate(_POINTS):
        b = 4 * idx                    # bottom block rows [b, b+2)
        for (di, dj), T in couplings:
            nbr = _INDEX.get((i + di, j + dj))
            if nbr is None:
                continue               # truncated neighbour: dropped (see #2)
            t = 4 * nbr + 2            # top block of the neighbour
            Hm[:, b:b + 2, t:t + 2] += T
            Hm[:, t:t + 2, b:b + 2] += T.conj().T

    return Hm


def _moire_reciprocal(theta_deg: float = None, a: float = None):
    """Moiré reciprocal lattice vectors g1, g2 (for BZ bounds / k-paths)."""
    g1, g2, _, _, _ = _geometry(
        DEFAULT_PARAMS["theta_deg"] if theta_deg is None else theta_deg,
        DEFAULT_PARAMS["a"] if a is None else a,
    )
    return g1, g2


def default_lattice(params: dict[str, object] | None = None) -> dict:
    """Lattice metadata whose moiré BZ box AUTO-SCALES with the twist angle.

    QXTI calls this with the resolved model parameters, so the reciprocal box
    tracks ``theta_deg`` automatically (k_theta ∝ sin(theta/2)) — no manual
    rescaling needed when you change the angle in the input file.

    The box is the bounding rectangle of the moiré reciprocal primitive cell
    {alpha g1 + beta g2 : |alpha|,|beta| <= 1/2}, which covers exactly one moiré
    Brillouin zone:  kx in [-(sqrt3/2) k_theta, +], ky in [-(3/2) k_theta, +].
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if k in p})
    k_theta = _geometry(p["theta_deg"], float(p["a"]))[4]
    s3 = np.sqrt(3.0)
    return {
        "lattice_type": "moire superlattice (continuum)",
        "BZorigin": [0.0, 0.0, 0.0],
        # Full widths along x and y of the moiré reciprocal primitive cell.
        "BZaxis": [
            [s3 * k_theta, 0.0, 0.0],   # full width in kx = sqrt(3) k_theta
            [0.0, 3.0 * k_theta, 0.0],  # full width in ky = 3 k_theta
            [0.0, 0.0, 0.0],
        ],
    }
