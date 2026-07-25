"""
taas_tb.py
==========
Symmetry-faithful, spinful tight-binding model for TaAs on the *real* crystal
structure: body-centered tetragonal lattice, nonsymmorphic space group
I4_1md (No. 109, point group C4v = 4mm), with 2 Ta + 2 As per primitive cell
and spin-orbit coupling.  8 bands per primitive cell.

QXTI interface added at the bottom (H, DEFAULT_PARAMS, DEFAULT_LATTICE, ...).
The model itself works in eV / Angstrom; the QXTI wrapper H(kx,ky,kz,params)
converts k from atomic units (1/Bohr) to 1/Angstrom and energies eV -> Hartree.
"""

import numpy as np
from numpy.linalg import det, inv, norm, eigh, eigvalsh
import itertools

# ----------------------------------------------------------------------
# 1. Crystal structure of TaAs (conventional cell, Angstrom)
# ----------------------------------------------------------------------
a = 3.4348          # tetragonal in-plane lattice constant
c = 11.641          # tetragonal c axis
uAs = 0.4176        # As z-parameter (units of c); Ta sits at z = 0

# Primitive vectors of the body-centered tetragonal lattice (rows)
A = np.array([[-a/2,  a/2,  c/2],
              [ a/2, -a/2,  c/2],
              [ a/2,  a/2, -c/2]])
Ainv = inv(A)
Bm = 2*np.pi*Ainv.T          # rows = reciprocal primitive vectors b1,b2,b3

species = ["Ta", "Ta", "As", "As"]
pos = np.array([
    [0.0, 0.0, 0.0        ],   # Ta_A
    [0.0, a/2, c/4        ],   # Ta_B
    [0.0, 0.0, uAs*c      ],   # As_A
    [0.0, a/2, (uAs+.25)*c],   # As_B
])
nsite, nspin = 4, 2
nb = nsite*nspin             # 8 bands

s0 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = np.array([sx, sy, sz])


def frac(r):
    return r @ Ainv


def cart(f):
    return np.asarray(f, float) @ A


def identify_site(r, tol=1e-6):
    for j in range(nsite):
        f = frac(r - pos[j])
        n = np.rint(f)
        if np.max(np.abs(f - n)) < tol:
            return j, n.astype(int)
    raise ValueError(f"position {r} is not an atomic site of the structure")


# ----------------------------------------------------------------------
# 2. Space group I4_1md: generate the 8 coset representatives {R|t}
# ----------------------------------------------------------------------
R4 = np.array([[0., -1., 0.],
               [1.,  0., 0.],
               [0.,  0., 1.]])                       # C4z
MX = np.diag([-1., 1., 1.])                          # mirror x -> -x
GEN = [(R4, np.array([0.0, a/2, c/4])),              # 4_1 screw {C4z | (0,a/2,c/4)}
       (MX, np.zeros(3))]                            # {m_x | 0}


def _canon(op):
    Rop, t = op
    ft = frac(t)
    ft = ft - np.floor(ft + 1e-8)
    key = tuple(np.rint(Rop.flatten()*1e6)/1e6) + tuple(np.rint(ft*1e6)/1e6)
    return key


def space_group_ops():
    ops = {_canon((np.eye(3), np.zeros(3))): (np.eye(3), np.zeros(3))}
    frontier = list(ops.values())
    while frontier:
        new = []
        for (R1, t1) in frontier:
            for (R2, t2) in GEN:
                R3, t3 = R2 @ R1, R2 @ t1 + t2
                k = _canon((R3, t3))
                if k not in ops:
                    ops[k] = (R3, t3)
                    new.append((R3, t3))
        frontier = new
    ops = list(ops.values())
    for (R, t) in ops:
        for j in range(nsite):
            j2, _ = identify_site(R @ pos[j] + t)
            assert species[j2] == species[j], "operation does not preserve the crystal"
    return ops


def spin_rep(R):
    Rp = R if det(R) > 0 else -R
    ang = np.arccos(np.clip((np.trace(Rp) - 1.0)/2.0, -1.0, 1.0))
    if ang < 1e-9:
        return np.eye(2, dtype=complex)
    if abs(ang - np.pi) < 1e-9:
        M = (Rp + np.eye(3))/2.0
        i0 = int(np.argmax(np.diag(M)))
        n = M[:, i0]/np.sqrt(M[i0, i0])
    else:
        n = np.array([Rp[2, 1]-Rp[1, 2], Rp[0, 2]-Rp[2, 0], Rp[1, 0]-Rp[0, 1]])
        n = n/(2.0*np.sin(ang))
    n = n/norm(n)
    nsig = n[0]*sx + n[1]*sy + n[2]*sz
    return np.cos(ang/2)*s0 - 1j*np.sin(ang/2)*nsig


# ----------------------------------------------------------------------
# 3. Tight-binding container with group-projection symmetrization
# ----------------------------------------------------------------------
class Model:
    def __init__(self, ops):
        self.ops = ops
        self.hop = {}
        self.terms = None

    def _add_raw(self, i, j, nvec, M):
        key = (int(i), int(j), tuple(int(x) for x in nvec))
        self.hop[key] = self.hop.get(key, np.zeros((2, 2), complex)) + M

    def add_sym(self, i, j, nvec, M):
        images = []
        rj_full = pos[j] + cart(nvec)
        for (R, t) in self.ops:
            D = spin_rep(R)
            i2, ni = identify_site(R @ pos[i] + t)
            j2, nj = identify_site(R @ rj_full + t)
            Mg = D @ M @ D.conj().T
            images.append((i2, j2, tuple(nj - ni), Mg))
            images.append((i2, j2, tuple(nj - ni), sy @ Mg.conj() @ sy))
        w = 1.0/len(images)
        for (ii, jj, nn, MM) in images:
            self._add_raw(ii, jj, nn, MM*w)

    def finalize(self):
        newh = {}
        for (i, j, nv), M in self.hop.items():
            rev = (j, i, tuple(-x for x in nv))
            Mrev = self.hop.get(rev, np.zeros((2, 2), complex))
            newh[(i, j, nv)] = 0.5*(M + Mrev.conj().T)
        full = {}
        for (i, j, nv), M in newh.items():
            full[(i, j, nv)] = M
            full[(j, i, tuple(-x for x in nv))] = M.conj().T
        self.hop = {k: v for k, v in full.items() if np.max(np.abs(v)) > 1e-12}
        self.terms = []
        for (i, j, nv), M in self.hop.items():
            d = pos[j] + cart(nv) - pos[i]
            self.terms.append((i, j, d, M))
        return self

    def Hk(self, k):
        H = np.zeros((nb, nb), complex)
        for i, j, d, M in self.terms:
            H[2*j:2*j+2, 2*i:2*i+2] += np.exp(1j*np.dot(k, d))*M
        return H

    def Hk_batch(self, ks):
        ks = np.atleast_2d(ks)
        H = np.zeros((len(ks), nb, nb), complex)
        for i, j, d, M in self.terms:
            ph = np.exp(1j*(ks @ d))
            H[:, 2*j:2*j+2, 2*i:2*i+2] += ph[:, None, None]*M
        return H

    def dHk(self, k):
        dH = np.zeros((3, nb, nb), complex)
        for i, j, d, M in self.terms:
            ph = 1j*d*np.exp(1j*np.dot(k, d))
            for mu in range(3):
                dH[mu, 2*j:2*j+2, 2*i:2*i+2] += ph[mu]*M
        return dH


# ----------------------------------------------------------------------
# 4. The TaAs model: bond classes and parameters
# ----------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    eTa=+1.073, eAs=-0.544,
    t1=0.931, t1p=1.015, t2=0.620, t3=0.615, t4=0.812, t5=0.927,
    l1=0.28, l1p=0.22, l2=0.18, l3=0.16, l4=0.10, l5=0.16,
    soc_scale=1.0,
)

_SEED_N = {
    'l1':  np.array([0.7, 0.5, 0.4]),
    'l1p': np.array([0.3, 0.8, 0.5]),
    'l2':  np.array([0.6, 0.4, 0.7]),
    'l3':  np.array([0.2, 0.9, 0.4]),
    'l4':  np.array([0.5, 0.3, 0.8]),
    'l5':  np.array([0.4, 0.7, 0.6]),
}


def _soc(lam, key):
    n = _SEED_N[key]/norm(_SEED_N[key])
    return 1j*lam*(n[0]*sx + n[1]*sy + n[2]*sz)


def build_taas_model(params=None):
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if k in DEFAULT_PARAMS})
    s = p['soc_scale']
    ops = space_group_ops()
    m = Model(ops)
    m.add_sym(0, 0, (0, 0, 0), p['eTa']*s0)
    m.add_sym(2, 2, (0, 0, 0), p['eAs']*s0)
    m.add_sym(0, 2, (0, 0, 1), p['t1']*s0 + s*_soc(p['l1'], 'l1'))
    m.add_sym(0, 3, (-1, 0, 0), p['t1p']*s0 + s*_soc(p['l1p'], 'l1p'))
    m.add_sym(0, 1, (0, 0, 0), p['t2']*s0 + s*_soc(p['l2'], 'l2'))
    m.add_sym(0, 0, (0, 1, 1), p['t3']*s0 + s*_soc(p['l3'], 'l3'))
    m.add_sym(2, 2, (0, 1, 1), p['t4']*s0 + s*_soc(p['l4'], 'l4'))
    m.add_sym(2, 3, (0, 0, 0), p['t5']*s0 + s*_soc(p['l5'], 'l5'))
    return m.finalize()


# ----------------------------------------------------------------------
# 5. Verification utility
# ----------------------------------------------------------------------
def verify_symmetry(model, ntest=10, seed=0, tol=1e-9):
    rng = np.random.default_rng(seed)
    report = {"space_group": True, "trs": True}
    for _ in range(ntest):
        k = rng.uniform(-1, 1, 3) @ Bm
        e0 = eigvalsh(model.Hk(k))
        for (R, t) in model.ops:
            if not np.allclose(e0, eigvalsh(model.Hk(R @ k)), atol=tol):
                report["space_group"] = False
        if not np.allclose(e0, eigvalsh(model.Hk(-k)), atol=tol):
            report["trs"] = False
    eG = eigvalsh(model.Hk(np.zeros(3)))
    report["kramers_at_Gamma"] = bool(np.allclose(eG[0::2], eG[1::2], atol=1e-9))
    k = np.array([0.31, 0.17, 0.23]) @ Bm
    ev = eigvalsh(model.Hk(k))
    report["min_spin_splitting_generic_k"] = float(np.min(np.diff(ev)))
    report["inversion_broken"] = bool(report["min_spin_splitting_generic_k"] > 1e-4)
    report["n_ops"] = len(model.ops)
    report["n_hopping_matrices"] = len(model.hop)
    return report


# ======================================================================
# QXTI interface
# ======================================================================
MODEL_NAME = "taas-tb-8band"
BASIS_SIZE = 8
DIMENSION = 3
BASIS_TYPE = "orbital+spin"
IS_PERIODIC = True

_AU_PER_ANGSTROM = 1.8897259886          # 1 Angstrom in Bohr (a.u. of length)
_EV_TO_HARTREE = 1.0 / 27.211386245988   # 1 eV in Hartree
_a_bohr = a * _AU_PER_ANGSTROM
_c_bohr = c * _AU_PER_ANGSTROM

# NOTE on the BZ box: the model lives on a body-centered tetragonal lattice, so
# the exact primitive Brillouin zone is not an axis-aligned box.  For an
# illustrative optical-response run we sample the CONVENTIONAL tetragonal box
# in Cartesian coordinates (so that z = the polar C4 axis, the relevant one for
# chi^(2)_zzz).  Frequencies/resonances are physical (energies in Hartree); the
# absolute magnitude / exact BZ average is approximate.  For quantitative work
# use the primitive BZ or a Wannier90 model.
DEFAULT_LATTICE = {
    "lattice_type": "body-centered tetragonal (TaAs, I4_1md, C4v)",
    "lattice_constants": {"a": a, "c": c, "z_As": uAs},
    "BZorigin": [0.0, 0.0, 0.0],
    "BZaxis": [
        [2.0 * np.pi / _a_bohr, 0.0, 0.0],
        [0.0, 2.0 * np.pi / _a_bohr, 0.0],
        [0.0, 0.0, 2.0 * np.pi / _c_bohr],
    ],
    "notes": "Conventional tetragonal Cartesian sampling box (approx BCT BZ).",
}

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = build_taas_model()
    return _MODEL


def default_params():
    return dict(DEFAULT_PARAMS)


def H(kx, ky, kz, params=None):
    """QXTI Hamiltonian: k in atomic units (1/Bohr), returns 8x8 in Hartree."""
    kv = np.array([kx, ky, kz], dtype=float) * _AU_PER_ANGSTROM   # 1/Bohr -> 1/Angstrom
    return _model().Hk(kv) * _EV_TO_HARTREE                        # eV -> Hartree


def H_batch(kpts, params=None):
    """Vectorized QXTI Hamiltonian over many k-points: kpts (nk,3) -> (nk,8,8).

    Exact vectorized replica of ``H`` (same k-scaling 1/Bohr -> 1/Angstrom and
    the same eV -> Hartree factor), evaluated for all k at once via the model's
    ``Hk_batch`` (the batched twin of ``Hk`` with identical phase convention).
    Like ``H``, ``params`` is accepted for signature compatibility but the cached
    default model is used.  QXTI uses this as ``h_batch`` to avoid the per-k loop.
    """
    ks = np.atleast_2d(np.asarray(kpts, dtype=float)) * _AU_PER_ANGSTROM
    return _model().Hk_batch(ks) * _EV_TO_HARTREE
