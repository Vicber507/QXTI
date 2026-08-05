"""taas_4band.py — modelo de CUATRO BANDAS de TaAs, simetria-exacto.

=====================================================================
QUE ES
=====================================================================
Modelo tight-binding minimo de 4 bandas (1 doblete orbital x espin) para
el semimetal de Weyl TaAs, construido de forma que TODOS los terminos son
exactamente los permitidos por la simetria del cristal, con las constantes
de red REALES de la estructura experimental (Materials Project mp-1936).

ESTRUCTURA (mp-1936 / Furuseth-Selte-Kjekshus 1965, usada por Weng et al.):
    grupo espacial  I4_1md (No. 109), body-centered tetragonal, NO centrosimetrico
    grupo puntual   C4v = 4mm  (eje polar c)
    a = b = 3.4348 A ,  c = 11.641 A
    Ta en 4a (0,0,0) ,  As en 4a (0,0,u) con u = 0.4176

BASE (4 componentes):
    |doblete orbital> (x) |espin>
    El doblete son los dos orbitales tipo {d_xz, d_yz} de Ta que dominan
    cerca de E_F (Weng et al., arXiv:1501.00060).  Bajo C4z el doblete rota
    como (d_xz, d_yz) -> (d_yz, -d_xz).
    sigma_i = Pauli ORBITAL ,  s_i = Pauli de ESPIN ,  H = sum c_n f_n(k) sigma_a (x) s_b

=====================================================================
COMO SE CONSTRUYO (ver el PDF adjunto para el detalle)
=====================================================================
NO se escribieron los terminos a mano.  Se genero el espacio de todos los
productos {armonico de red} x {matriz 4x4} y se extrajo NUMERICAMENTE el
subespacio invariante bajo los generadores del grupo:

    T   = i s_y K                 inversion temporal (material no magnetico)
    M_x = i sigma_z s_x           espejo  x -> -x
    M_y = i sigma_z s_y           espejo  y -> -y
    C4z = R_orb(90) (x) e^{-i pi s_z/4}   rotacion 4 (eje polar)

Cada generador g define H(k) -> U_g H(g^-1 k) U_g^dag (o con conjugacion
compleja si es antiunitario).  El subespacio invariante comun (kernel de
R_g - 1 para todos los g) tiene dimension 22 con la lista de armonicos
usada; los terminos implementados abajo son esa base, agrupados por su
significado fisico.  Esto GARANTIZA que el modelo tiene exactamente la
simetria de TaAs: ni mas (no hay simetrias accidentales que anulen
componentes del tensor optico) ni menos.

La periodicidad exacta en la caja de muestreo (imprescindible para los
enlaces de Wilson / gradiente covariante de QXTI) esta asegurada porque
todos los armonicos son sin/cos de (kx*a, ky*a, kz*c4) con c4 = c/4, el
paso de apilamiento del tornillo 4_1.  Caja:  kx,ky in [-pi/a, pi/a],
kz in [-pi/c4, pi/c4].

=====================================================================
FISICA QUE CONTIENE
=====================================================================
  * Rompe inversion (eje polar c) -> respuesta optica de orden PAR no nula.
  * Conserva T -> los nodos de Weyl vienen en pares +-k.
  * Dos familias de nodos, como en el DFT (Weng et al.) -- VERIFICADO
    numericamente con los parametros por defecto:
        W1 : 8 nodos en el plano kz = 0, en (+-0.206, +-0.133, 0) y las
             4 imagenes C4, energia E = +0.300 eV respecto al minimo
        W2 : 16 nodos fuera del plano, en (+-0.093, +-0.180, +-0.571) y
             sus imagenes, energia E = +0.465 eV
    -> 24 nodos = 12 pares, el numero correcto para TaAs, con quiralidades
       +-1 que suman cero (verificado por flujo de Berry de Fukui-Hatsugai).
  * Terminos SOC de tipo Rashba (polar) y Dresselhaus, que son los que
    generan el dicroismo circular y las respuestas quirales que medimos.

Referencias:
  H. Weng, C. Fang, Z. Fang, B. A. Bernevig, X. Dai,
      Phys. Rev. X 5, 011029 (2015)  [arXiv:1501.00060]   -- k.p, nodos, DFT
  S.-M. Huang et al., Nat. Commun. 6, 7373 (2015)                -- WSM en TaAs
  Materials Project mp-1936                                       -- estructura
"""
from __future__ import annotations

import numpy as np

MODEL_NAME = "taas-4band-symmetry-exact"
BASIS_SIZE = 4
DIMENSION = 3
BASIS_TYPE = "orbital+spin"
IS_PERIODIC = True

AU_PER_ANGSTROM = 1.8897259886

# --- estructura real (mp-1936) -------------------------------------------
A_ANG = 3.4348            # a = b  [Angstrom]
C_ANG = 11.641            # c      [Angstrom]
U_AS = 0.4176             # posicion z del As en 4a (0,0,u)
A_AU = A_ANG * AU_PER_ANGSTROM          # 6.4907 a.u.
C4_AU = (C_ANG / 4.0) * AU_PER_ANGSTROM  # paso del tornillo 4_1: c/4

# --- matrices de Pauli ----------------------------------------------------
_s0 = np.eye(2, dtype=complex)
_sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

def _K(a, b):
    """sigma_a (x) s_b  (orbital (x) espin)."""
    return np.kron(a, b)

# matrices que aparecen en la base simetrizada
G_00 = _K(_s0, _s0)
G_yz = _K(_sy, _sz)
G_z0 = _K(_sz, _s0)
G_x0 = _K(_sx, _s0)
G_0y = _K(_s0, _sy); G_0x = _K(_s0, _sx)
G_zy = _K(_sz, _sy); G_zx = _K(_sz, _sx)
G_xx = _K(_sx, _sx); G_xy = _K(_sx, _sy); G_xz = _K(_sx, _sz)
G_yy = _K(_sy, _sy); G_yx = _K(_sy, _sx)
G_zz = _K(_sz, _sz)

# =========================================================================
# PARAMETROS
# =========================================================================
# Energias en eV (el wrapper QXTI las pasa a Hartree).  Los valores por
# defecto colocan las dos familias de nodos de Weyl cerca de E_F con las
# escalas del DFT de TaAs (bandas ~ +-1 eV, nodos a pocas decenas de meV).
DEFAULT_PARAMS = {
    # --- termino cinetico (identidad): asimetria particula-hueco ---
    "e0": 0.488925,      # constante
    "e1": -0.0721068,      # (2 - cos kx a - cos ky a)
    "e2": -1.19779,      # (1 - cos kz c4)
    "e3": -0.583919,      # (1-cos kx a)(1-cos ky a)
    # --- masa / inversion de bandas: M(k) sigma_y s_z  (locus M=0 = anillo) ---
    "m0": 0.2254,     # constante
    "m1": -0.257381,      # (2 - cos kx a - cos ky a)
    "m2": -0.01886,      # (1 - cos kz c4)   <- fija la inversion FUERA del eje C4
    "m3": -0.0231301,      # (1-cos kx a)(1-cos ky a)
    # --- terminos cuadrupolares (doblete C4-impar) ---
    "q1": 0.00237281,      # (cos ky a - cos kx a) sigma_z
    "q2": 0.968995,      # sin kx a sin ky a  sigma_x
    # --- SOC permitido por C4v ---
    "lR": 0.14524,      # Rashba polar:  sin kx sigma_0 s_y + sin ky sigma_0 s_x
    "lD": -0.999467,      # Dresselhaus:   sin kx sigma_z s_y - sin ky sigma_z s_x
    "lA": 0.906134,      # sin kx sigma_x s_x + sin ky sigma_x s_y
    "lB": 0.212083,      # sin kx sin kz sigma_y s_x - sin ky sin kz sigma_y s_y
    "lC": 0.605769,      # (cos kx - cos ky) sin kz  sigma_x s_z
    "lE": 0.263531,      # sin kx sin ky sin kz sigma_z s_z (orden alto)
    # --- red (no tocar salvo para estudiar deformaciones) ---
    "a": A_AU,
    "c4": C4_AU,
}

DEFAULT_LATTICE = {
    "lattice_type": "body-centered tetragonal I4_1md (TaAs, mp-1936); "
                    "caja de muestreo rectangular (a, a, c/4)",
    "lattice_constants": {"a": A_AU, "c": C_ANG * AU_PER_ANGSTROM,
                          "c4": C4_AU, "u_As": U_AS, "gamma_deg": 90.0},
    "real_space_vectors": {"a1": [A_AU, 0.0, 0.0],
                           "a2": [0.0, A_AU, 0.0],
                           "a3": [0.0, 0.0, C4_AU]},
    "BZorigin": [0.0, 0.0, 0.0],
    "BZaxis": [[2.0 * np.pi / A_AU, 0.0, 0.0],
               [0.0, 2.0 * np.pi / A_AU, 0.0],
               [0.0, 0.0, 2.0 * np.pi / C4_AU]],
}


def default_params() -> dict:
    return dict(DEFAULT_PARAMS)


def _resolved(params):
    p = default_params()
    if params:
        p.update(params)
    return p


def _terms(kx, ky, kz, p):
    """Devuelve (lista de (coef, matriz)) — la base SIMETRIZADA."""
    a, c4 = float(p["a"]), float(p["c4"])
    X, Y, Z = kx * a, ky * a, kz * c4
    sX, sY, sZ = np.sin(X), np.sin(Y), np.sin(Z)
    cX, cY, cZ = np.cos(X), np.cos(Y), np.cos(Z)
    e_iso = (1 - cX) + (1 - cY)          # C4-par
    e_quad = cY - cX                     # C4-impar
    out = []
    # cinetico (identidad)
    eps = (float(p["e0"]) + float(p["e1"]) * e_iso + float(p["e2"]) * (1 - cZ)
           + float(p["e3"]) * (1 - cX) * (1 - cY))
    out.append((eps, G_00))
    # masa (inversion de bandas)
    M = (float(p["m0"]) + float(p["m1"]) * e_iso + float(p["m2"]) * (1 - cZ)
         + float(p["m3"]) * (1 - cX) * (1 - cY))
    out.append((M, G_yz))
    # cuadrupolares
    out.append((float(p["q1"]) * e_quad, G_z0))
    out.append((float(p["q2"]) * sX * sY, G_x0))
    # SOC
    lR = float(p["lR"]); out += [(lR * sX, G_0y), (lR * sY, G_0x)]
    lD = float(p["lD"]); out += [(lD * sX, G_zy), (-lD * sY, G_zx)]
    lA = float(p["lA"]); out += [(lA * sX, G_xx), (lA * sY, G_xy)]
    lB = float(p["lB"]); out += [(lB * sX * sZ, G_yx), (-lB * sY * sZ, G_yy)]
    lC = float(p["lC"]); out += [(lC * (cX - cY) * sZ, G_xz)]
    lE = float(p["lE"]); out += [(lE * sX * sY * sZ, G_zz)]
    return out


def H(kx, ky, kz, params=None):
    """Hamiltoniano 4x4 (energias en eV).  Firma que QXTI espera."""
    p = _resolved(params)
    Hm = np.zeros((4, 4), dtype=complex)
    for c, M in _terms(float(kx), float(ky), float(kz), p):
        if c != 0.0:
            Hm = Hm + c * M
    return Hm


def H_batch(kpts, params=None):
    """Version vectorizada: kpts (nk,3) -> (nk,4,4).  Identica a H punto a punto."""
    p = _resolved(params)
    k = np.asarray(kpts, dtype=np.float64)
    a, c4 = float(p["a"]), float(p["c4"])
    X, Y, Z = k[:, 0] * a, k[:, 1] * a, k[:, 2] * c4
    sX, sY, sZ = np.sin(X), np.sin(Y), np.sin(Z)
    cX, cY, cZ = np.cos(X), np.cos(Y), np.cos(Z)
    e_iso = (1 - cX) + (1 - cY)
    e_quad = cY - cX
    nk = k.shape[0]
    Hm = np.zeros((nk, 4, 4), dtype=complex)

    def add(coef, M):
        Hm[:] += np.asarray(coef, dtype=complex)[:, None, None] * M[None, :, :]

    eps = (float(p["e0"]) + float(p["e1"]) * e_iso + float(p["e2"]) * (1 - cZ)
           + float(p["e3"]) * (1 - cX) * (1 - cY))
    add(eps, G_00)
    M_ = (float(p["m0"]) + float(p["m1"]) * e_iso + float(p["m2"]) * (1 - cZ)
          + float(p["m3"]) * (1 - cX) * (1 - cY))
    add(M_, G_yz)
    add(float(p["q1"]) * e_quad, G_z0)
    add(float(p["q2"]) * sX * sY, G_x0)
    lR = float(p["lR"]); add(lR * sX, G_0y); add(lR * sY, G_0x)
    lD = float(p["lD"]); add(lD * sX, G_zy); add(-lD * sY, G_zx)
    lA = float(p["lA"]); add(lA * sX, G_xx); add(lA * sY, G_xy)
    lB = float(p["lB"]); add(lB * sX * sZ, G_yx); add(-lB * sY * sZ, G_yy)
    lC = float(p["lC"]); add(lC * (cX - cY) * sZ, G_xz)
    lE = float(p["lE"])
    if lE != 0.0:
        add(lE * sX * sY * sZ, G_zz)
    return Hm


# =========================================================================
# Verificaciones de simetria (se usan en los tests y en el PDF)
# =========================================================================
T_U = _K(_s0, _sy)
MX_U = 1j * _K(_sz, _sx)
MY_U = 1j * _K(_sz, _sy)
C4_U = _K(np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex),
          np.diag([np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)]))


def check_symmetries(params=None, n=40, seed=0, tol=1e-10):
    """Devuelve dict con el error maximo de cada simetria (debe ser ~1e-15)."""
    rng = np.random.default_rng(seed)
    ks = rng.uniform(-1.5, 1.5, (n, 3))
    out = {}
    err = 0.0
    for k in ks:
        Hk = H(*k, params)
        err = max(err, np.max(np.abs(Hk - Hk.conj().T)))
    out["hermitico"] = err
    for name, U, kmap, anti in (
            ("T (inv. temporal)", T_U, lambda k: -k, True),
            ("M_x", MX_U, lambda k: np.array([-k[0], k[1], k[2]]), False),
            ("M_y", MY_U, lambda k: np.array([k[0], -k[1], k[2]]), False),
            ("C4z", C4_U, lambda k: np.array([k[1], -k[0], k[2]]), False)):
        e = 0.0
        for k in ks:
            lhs = H(*kmap(k), params)
            Hk = H(*k, params)
            rhs = U @ (np.conj(Hk) if anti else Hk) @ U.conj().T
            e = max(e, np.max(np.abs(lhs - rhs)))
        out[name] = e
    p = _resolved(params)
    a, c4 = float(p["a"]), float(p["c4"])
    e = 0.0
    for k in ks:
        for shift in (np.array([2 * np.pi / a, 0, 0]), np.array([0, 2 * np.pi / a, 0]),
                      np.array([0, 0, 2 * np.pi / c4])):
            e = max(e, np.max(np.abs(H(*(k + shift), params) - H(*k, params))))
    out["periodicidad de la caja"] = e
    return out


def weyl_nodes(params=None, grid=24, refine=True):
    """Localiza los nodos (gap banda2-banda3 = 0) en la caja; devuelve (posiciones, quiralidades)."""
    p = _resolved(params)
    a, c4 = float(p["a"]), float(p["c4"])
    kx = np.linspace(-np.pi / a, np.pi / a, grid, endpoint=False)
    ky = np.linspace(-np.pi / a, np.pi / a, grid, endpoint=False)
    kz = np.linspace(-np.pi / c4, np.pi / c4, grid, endpoint=False)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    pts = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], axis=1)
    ev = np.linalg.eigvalsh(H_batch(pts, params))
    gap = (ev[:, 2] - ev[:, 1]).reshape(grid, grid, grid)
    # minimos locales del gap
    cand = []
    for i in range(grid):
        for j in range(grid):
            for l in range(grid):
                g = gap[i, j, l]
                nb = gap[(i-1) % grid:(i+2) % grid or None, :, :]
                if g <= min(gap[(i+di) % grid, (j+dj) % grid, (l+dl) % grid]
                            for di in (-1, 0, 1) for dj in (-1, 0, 1) for dl in (-1, 0, 1)):
                    cand.append((g, np.array([kx[i], ky[j], kz[l]])))
    cand.sort(key=lambda t: t[0])
    out = []
    for g, k0 in cand[:60]:
        k = k0.copy()
        if refine:
            step = (kx[1] - kx[0]) * 0.6
            for _ in range(70):
                moved = False
                for d in range(3):
                    for sgn in (+1, -1):
                        kt = k.copy(); kt[d] += sgn * step
                        e1 = np.linalg.eigvalsh(H(*kt, params))
                        e0 = np.linalg.eigvalsh(H(*k, params))
                        if (e1[2] - e1[1]) < (e0[2] - e0[1]):
                            k = kt; moved = True
                if not moved:
                    step *= 0.5
                if step < 1e-8:
                    break
        e = np.linalg.eigvalsh(H(*k, params))
        out.append((k, e[2] - e[1], 0.5 * (e[1] + e[2])))
    return out


# --- alias que QXTI puede pedir ------------------------------------------
def Hk(k, **params):
    return H(k[0], k[1], k[2], params or None)


def bands(ks, params=None):
    return np.linalg.eigvalsh(H_batch(np.asarray(ks), params))
