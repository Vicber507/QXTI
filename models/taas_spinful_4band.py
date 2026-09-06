"""Modelo de 4 bandas CON espin para TaAs (2 sitios: Ta, As), red TETRAGONAL simple.

(copia para revision - ver docstring original)
"""
from __future__ import annotations

import numpy as np

MODEL_NAME = "taas-spinful-2site-4band"
BASIS_SIZE = 4
DIMENSION = 3
BASIS_TYPE = "spinorbital"   # (Ta,As) x (up,dn)
IS_PERIODIC = True

# --- unidades -----------------------------------------------------------------------
AU_PER_ANGSTROM = 1.8897259886          # 1 Ang = ... Bohr
ANGSTROM_PER_AU = 1.0 / AU_PER_ANGSTROM
EV_PER_HARTREE = 27.211386245988
HARTREE_PER_EV = 1.0 / EV_PER_HARTREE

# --- geometria (Angstrom) -----------------------------------------------------------
A_ANG = 3.4348
C_TAAS = 11.641
U_AS = 0.4176
CP_ANG = C_TAAS / 4.0                    # 2.910250
Z0_ANG = (U_AS - 0.25) * C_TAAS          # 1.951008

# --- matrices de Pauli --------------------------------------------------------------
_s0 = np.eye(2, dtype=complex)
_sx = np.array([[0, 1], [1, 0]], complex)
_sy = np.array([[0, -1j], [1j, 0]], complex)
_sz = np.array([[1, 0], [0, -1]], complex)
_SIG = np.array([_sx, _sy, _sz])

DEFAULT_PARAMS = {
    "t1": 0.551219, "t2": -0.775686, "tT": -1.913595, "tA": 1.844665,
    "tTz": -1.877263, "tAz": 0.012071, "t2T": -0.095731, "t2A": -0.363665,
    "l1": -0.998485, "l2": -1.190587, "l3": 0.052546, "Delta": 2.531344,
}
W1_FRAC_DEFAULT = (0.0072, 0.450)

PRESET_CONO_EXACTO = {
    "t1": 1.956314, "t2": -2.372135, "tT": -2.499720, "tA": 2.499683,
    "tTz": -0.324683, "tAz": -0.017062, "t2T": 0.007608, "t2A": 0.031763,
    "l1": -0.885541, "l2": -1.088883, "l3": 0.035097, "Delta": 0.283832,
}
PRESET_ENERGETICO = {
    "t1": 1.1981, "t2": -1.6208, "tT": -2.2014, "tA": 2.2808, "tTz": -1.2388, "tAz": 0.3805,
    "t2T": -0.0927, "t2A": -0.1570, "l1": -1.1692, "l2": -1.4165, "l3": 0.0379, "Delta": 1.7561,
}
PRESETS = {"hhg": DEFAULT_PARAMS, "cono_exacto": PRESET_CONO_EXACTO, "energetico": PRESET_ENERGETICO}
W1_FRAC_PRESET = {"hhg": (0.0072, 0.450), "cono_exacto": (0.0072, 0.4827), "energetico": (0.0078, 0.4831)}
_PARAM_ORDER = ["t1", "t2", "tT", "tA", "tTz", "tAz", "t2T", "t2A", "l1", "l2", "l3"]


def _lattice(a=A_ANG, cp=CP_ANG, z0=Z0_ANG):
    A = np.diag([a, a, cp])
    pos = np.array([[0.0, 0.0, 0.0], [a / 2, a / 2, z0]])
    return A, pos


def _point_ops():
    """8 operaciones de C4v y su representacion SU(2) de espin."""
    C4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    Mx = np.diag([-1.0, 1.0, 1.0])
    Rs = []
    for n in range(4):
        R = np.linalg.matrix_power(C4, n)
        Rs.append(R); Rs.append(R @ Mx)

    def spin_rep(R):
        det = np.linalg.det(R)
        Rp = R * det
        ang = np.arccos(np.clip((np.trace(Rp) - 1) / 2, -1, 1))
        if abs(ang) < 1e-9:
            return _s0.copy()
        if abs(ang - np.pi) < 1e-9:
            w, v = np.linalg.eigh(Rp); n = v[:, np.argmin(np.abs(w - 1))]
            return -1j * np.tensordot(n, _SIG, 1)
        n = np.array([Rp[2, 1] - Rp[1, 2], Rp[0, 2] - Rp[2, 0], Rp[1, 0] - Rp[0, 1]]) / (2 * np.sin(ang))
        return np.cos(ang / 2) * _s0 - 1j * np.sin(ang / 2) * np.tensordot(n, _SIG, 1)

    return [(R, spin_rep(R)) for R in Rs]


def _site_of(A, pos, r, tol=1e-6):
    for j in range(len(pos)):
        f = np.linalg.solve(A.T, r - pos[j])
        if np.allclose(f, np.round(f), atol=tol):
            return j
    raise ValueError("no es sitio de la red")


def _unique_vecs(vs, tol=1e-6):
    out = []
    for v in vs:
        if all(np.linalg.norm(v - w) > tol for w in out):
            out.append(v)
    return out


class _Builder:
    def __init__(self, a=A_ANG, cp=CP_ANG, z0=Z0_ANG):
        self.A, self.pos = _lattice(a, cp, z0)
        self.a, self.cp, self.z0 = a, cp, z0
        self.ops = _point_ops()
        self.perms = [[_site_of(self.A, self.pos, R @ self.pos[j]) for j in range(2)]
                      for (R, U) in self.ops]
        self.B = 2 * np.pi * np.linalg.inv(self.A).T
        self.terms = {}

    def _symmetrize(self, seed):
        acc = {}

        def add(i, j, d, M):
            key = (i, j) + tuple(np.round(d, 5) + 0.0)
            if key in acc:
                acc[key][3] = acc[key][3] + M
            else:
                acc[key] = [i, j, d.copy(), M.copy()]

        for (i, j, d, M) in seed:
            for gi, (R, U) in enumerate(self.ops):
                ii, jj, dd, MM = self.perms[gi][i], self.perms[gi][j], R @ d, U @ M @ U.conj().T
                N = MM
                add(ii, jj, dd, N)
                add(jj, ii, -dd, N.conj().T)
                add(ii, jj, dd, _sy @ N.conj() @ _sy)                 # TRS
                add(jj, ii, -dd, (_sy @ N.conj() @ _sy).conj().T)
        out = [tuple(v) for v in acc.values()]
        nmax = max(np.linalg.norm(M) for (_, _, _, M) in out)
        n0 = max(np.linalg.norm(M) for (_, _, _, M) in seed)
        return [(i, j, d, M * (n0 / nmax)) for (i, j, d, M) in out]

    def add(self, name, seed):
        self.terms[name] = self._symmetrize(seed)

    def build(self):
        a, z0, cp = self.a, self.z0, self.cp
        d_dn = np.array([a / 2, a / 2, z0 - cp])   # Ta -> As corto (2.611 Ang)
        d_up = np.array([a / 2, a / 2, z0])        # Ta -> As largo (3.109 Ang)
        self.add("t1", [(0, 1, d_dn, _s0)])
        self.add("t2", [(0, 1, d_up, _s0)])
        self.add("tT", [(0, 0, np.array([a, 0, 0]), _s0)])
        self.add("tA", [(1, 1, np.array([a, 0, 0]), _s0)])
        self.add("tTz", [(0, 0, np.array([0, 0, cp]), _s0)])
        self.add("tAz", [(1, 1, np.array([0, 0, cp]), _s0)])
        self.add("t2T", [(0, 0, np.array([a, a, 0.0]), _s0)])
        self.add("t2A", [(1, 1, np.array([a, a, 0.0]), _s0)])
        for tag, dv in (("l1", d_dn), ("l2", d_up)):
            seed = []
            for i in (0, 1):
                dijs = _unique_vecs([R @ (dv if i == 0 else -dv) for (R, U) in self.ops])
                for dij in dijs:
                    for djk in [-x for x in dijs]:
                        dik = dij + djk
                        n = np.cross(dij, djk)
                        if np.linalg.norm(dik) < 1e-6 or np.linalg.norm(n) < 1e-9:
                            continue
                        n = n / np.linalg.norm(n)
                        seed.append((i, i, dik, 1j * np.tensordot(n, _SIG, 1)))
            self.add(tag, seed)
        n = np.cross([0, 0, 1.0], d_dn); n = n / np.linalg.norm(n)
        self.add("l3", [(0, 1, d_dn, 1j * np.tensordot(n, _SIG, 1))])
        return self

    def tables(self):
        """Devuelve (D (nb,3), orb (nb,), S (nb,16)) para H vectorizado."""
        D = []; orb = []; rows = []
        for oi, name in enumerate(_PARAM_ORDER):
            for (i, j, d, M) in self.terms[name]:
                v = np.zeros(16, complex)
                for s in range(2):
                    for sp in range(2):
                        v[(2 * j + sp) * 4 + (2 * i + s)] = M[sp, s]
                D.append(d); orb.append(oi); rows.append(v)
        return np.array(D), np.array(orb), np.array(rows)


_BUILD = _Builder().build()
_D, _ORB, _S = _BUILD.tables()
_A = _BUILD.A
_B = _BUILD.B
_POS = _BUILD.pos
_OPS = _BUILD.ops
_PERMS = _BUILD.perms


def default_params() -> dict:
    return dict(DEFAULT_PARAMS)


def _resolved(params):
    r = default_params()
    if params:
        r.update(params)
    return r


def _pvec(params):
    r = _resolved(params)
    return np.array([r[k] for k in _PARAM_ORDER], float), float(r["Delta"])


def _H_from_k(k, pvec, delta):
    """k: (...,3) en 1/Ang.  Devuelve (...,4,4) en eV."""
    k = np.asarray(k, float)
    sh = k.shape[:-1]
    ph = np.exp(1j * np.tensordot(k, _D, axes=([-1], [1]))) * pvec[_ORB]
    H = (ph @ _S).reshape(sh + (4, 4))
    H[..., [0, 1, 2, 3], [0, 1, 2, 3]] += np.array([delta, delta, -delta, -delta])
    return H


# ====================================================================================
#  API PUBLICA EN UNIDADES ATOMICAS (lo que QXTI espera)
#  ----------------------------------------------------------------------------------
#  QXTI es atomico de punta a punta: el k-grid sale de default_lattice()["BZaxis"] y el
#  laser ([laser] omega, E0), T2, temperature y fermi_level del cfg estan en Hartree/Bohr.
#  Ademas qxti.analytics.theory_response._model_h_batch busca literalmente el nombre
#  'H_batch' -> por eso H/H_batch DEBEN ser atomicas.  Las versiones en eV/Angstrom
#  siguen disponibles como H_ev / H_batch_ev para analisis y comparacion con DFT.
# ====================================================================================
def H_batch(kpts, params=None):
    """Interfaz QXTI (VECTORIZADA): kpts (nk,3) en 1/Bohr -> (nk,4,4) en Hartree."""
    kpts_ang = np.asarray(kpts, float) * AU_PER_ANGSTROM        # 1/Bohr -> 1/Ang
    return _H_from_k(kpts_ang, *_pvec(params)) * HARTREE_PER_EV


def H(kx, ky, kz, params=None):
    """Interfaz QXTI: kx,ky,kz en 1/Bohr -> (4,4) en Hartree."""
    return H_batch(np.array([[float(kx), float(ky), float(kz)]]), params)[0]


hamiltonian = H

# alias historico (mismo objeto que H_batch): k en 1/Bohr -> Hartree
h_batch_au = H_batch


def H_batch_ev(kpts, params=None):
    """Version en unidades de estructura de bandas: kpts (nk,3) en 1/Ang -> (nk,4,4) en eV."""
    pvec, delta = _pvec(params)
    return _H_from_k(np.asarray(kpts, float), pvec, delta)


def H_ev(kx, ky, kz, params=None):
    """Hamiltoniano 4x4 en eV, con kx,ky,kz en 1/Ang (para comparar con DFT)."""
    pvec, delta = _pvec(params)
    return _H_from_k(np.array([float(kx), float(ky), float(kz)]), pvec, delta)


def band_energies(kx, ky, kz, params=None):
    """Cuatro energias ordenadas [E0<=E1<=E2<=E3] en HARTREE, k en 1/Bohr (interfaz QXTI)."""
    return np.linalg.eigvalsh(H(kx, ky, kz, params))


def band_energies_ev(kx, ky, kz, params=None):
    """Cuatro energias ordenadas en eV, k en 1/Ang."""
    return np.linalg.eigvalsh(H_ev(kx, ky, kz, params))


def current_matrices(kx, ky, kz, params=None):
    """j_i = -dH/dk_i en unidades ATOMICAS (Hartree*Bohr), k en 1/Bohr.

    Referencia analitica para validar la velocidad por diferencias finitas de QXTI.
    """
    pvec, delta = _pvec(params)
    k_ang = np.array([float(kx), float(ky), float(kz)]) * AU_PER_ANGSTROM
    ph = np.exp(1j * (k_ang @ _D.T)) * pvec[_ORB]
    # d/dk_bohr = (dk_ang/dk_bohr) d/dk_ang = AU_PER_ANGSTROM * d/dk_ang ; luego eV->Ha
    scale = AU_PER_ANGSTROM * HARTREE_PER_EV
    out = {}
    for a, name in zip(range(3), ("jx", "jy", "jz")):
        M = (((1j * _D[:, a]) * ph) @ _S).reshape(4, 4)
        out[name] = -M * scale
    return out


def w1_nodes_frac(params=None, kxky=None):
    kx, ky = kxky if kxky is not None else W1_FRAC_DEFAULT
    return np.array([[kx, ky, 0.0], [-kx, -ky, 0.0], [-kx, ky, 0.0], [kx, -ky, 0.0],
                     [-ky, kx, 0.0], [ky, -kx, 0.0], [-ky, -kx, 0.0], [ky, kx, 0.0]])


def find_w1_node(params=None, guess=None):
    from scipy.optimize import minimize
    pvec, delta = _pvec(params)
    fa = 2 * np.pi / A_ANG
    g0 = np.array(guess if guess is not None else W1_FRAC_DEFAULT, float)
    f = lambda q: (lambda e: e[2] - e[1])(np.linalg.eigvalsh(_H_from_k(np.array([q[0] * fa, q[1] * fa, 0.0]), pvec, delta)))
    r = minimize(f, g0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=4000))
    e = np.linalg.eigvalsh(_H_from_k(np.array([r.x[0] * fa, r.x[1] * fa, 0.0]), pvec, delta))
    return np.array([r.x[0], r.x[1], 0.0]), float(r.fun), float(0.5 * (e[1] + e[2]))


def weyl_node_momenta(params=None):
    """Las 8 posiciones de los nodos de Weyl W1 en 1/BOHR (interfaz QXTI)."""
    return weyl_node_momenta_ang(params) * ANGSTROM_PER_AU      # 1/Ang -> 1/Bohr


def weyl_node_momenta_ang(params=None):
    """Las 8 posiciones de los nodos de Weyl W1 en 1/Ang."""
    kf, gap, _ = find_w1_node(params)
    fa = 2 * np.pi / A_ANG
    return w1_nodes_frac(params, kxky=(kf[0], kf[1])) * np.array([fa, fa, 2 * np.pi / CP_ANG])


def chemical_potential(params=None, nk=20, filling=2):
    pvec, delta = _pvec(params)
    f = [np.linspace(-0.5, 0.5, nk, endpoint=False)] * 3
    K = np.stack(np.meshgrid(*f, indexing="ij"), -1).reshape(-1, 3) @ _B
    E = np.sort(np.linalg.eigvalsh(_H_from_k(K, pvec, delta)).ravel())
    n = filling * len(K)
    return float(0.5 * (E[n - 1] + E[n]))


def default_lattice(params=None):
    """Metadatos de red en UNIDADES ATOMICAS (Bohr / 1-Bohr), como espera QXTI."""
    a = A_ANG * AU_PER_ANGSTROM          # 6.4909 Bohr
    cp = CP_ANG * AU_PER_ANGSTROM        # 5.4993 Bohr
    z0 = Z0_ANG * AU_PER_ANGSTROM        # 3.6866 Bohr
    return {
        "lattice_type": "3D simple tetragonal P4mm (TaAs 'desenroscado', 2 sitios con espin)",
        "lattice_constants": {"a": a, "a1_length": a, "a2_length": cp,
                              "c_prime": cp, "gamma_deg": 90.0, "u_As": U_AS, "z0": z0},
        "sites": {"Ta": [0.0, 0.0, 0.0], "As": [a / 2, a / 2, z0]},
        "real_space_vectors": {"a1": [a, 0.0, 0.0], "a2": [0.0, a, 0.0], "a3": [0.0, 0.0, cp]},
        "BZorigin": [0.0, 0.0, 0.0],
        "BZaxis": [[2 * np.pi / a, 0.0, 0.0], [0.0, 2 * np.pi / a, 0.0], [0.0, 0.0, 2 * np.pi / cp]],
        "point_group": "C4v (4mm), polar axis z; no inversion; TRS T=i sy K (T^2=-1)",
    }


def brillouin_zone_bounds(params=None):
    """Caja reciproca en 1/Bohr (interfaz QXTI)."""
    a = A_ANG * AU_PER_ANGSTROM
    cp = CP_ANG * AU_PER_ANGSTROM
    return {"kx": (-np.pi / a, np.pi / a),
            "ky": (-np.pi / a, np.pi / a),
            "kz": (-np.pi / cp, np.pi / cp)}


def stitch_phases_au(G_bohr):
    G = np.asarray(G_bohr, float) * AU_PER_ANGSTROM   # 1/Bohr -> 1/Ang
    ph = np.array([np.exp(1j * (G @ _POS[j])) for j in range(2)])
    return np.array([ph[0], ph[0], ph[1], ph[1]], complex)


def symmetry_report(params=None, nk=6, seed=0):
    pvec, delta = _pvec(params)
    ks = np.random.default_rng(seed).uniform(-2, 2, (nk, 3))
    errs = []
    for gi, (R, U) in enumerate(_OPS):
        P = np.zeros((2, 2)); P[_PERMS[gi][0], 0] = 1; P[_PERMS[gi][1], 1] = 1
        Ug = np.kron(P, U)
        errs.append(max(np.abs(_H_from_k(R @ k, pvec, delta)
                               - Ug @ _H_from_k(k, pvec, delta) @ Ug.conj().T).max() for k in ks))
    SY = np.kron(np.eye(2), _sy)
    trs = max(np.abs(_H_from_k(-k, pvec, delta)
                     - SY @ _H_from_k(k, pvec, delta).conj() @ SY).max() for k in ks)
    herm = max(np.abs(_H_from_k(k, pvec, delta) - _H_from_k(k, pvec, delta).conj().T).max() for k in ks)
    return {"C4v_max_err": float(np.max(errs)), "TRS_err": float(trs), "hermiticity_err": float(herm)}
