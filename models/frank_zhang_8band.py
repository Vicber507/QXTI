"""frank_zhang_8band.py — replica exacta en Python del modelo de 8 bandas de
Frank & Zhang, "Tight-binding models and Landau bands for Weyl semimetals in
transition metal monopnictide" (Cornell, 2017), transcrito verbatim del
programa Fortran TaAs2_weyldisp.f suministrado.

Estructura: 4 subredes (Ta, As, Ta, As apiladas con fase kz/4 por capa,
red BCT idealizada con a = c = 1) x 2 espines. Base (indices 0..7):
    (s1 up, s1 dn, s2 up, s2 dn, s3 up, s3 dn, s4 up, s4 dn)
con potencial alternante -delta en s1, s3 (Ta) y +delta en s2, s4 (As).

Parametros del archivo Fortran (por defecto aqui):
    tp   = 1.0    hopping del "2-bond" (el del "4-bond", t, esta fijado a 1)
    lam  = 0.1    SOC en el plano (lambda)
    lamp = 0.1    SOC inter-plano (lambda')
    delta= 0.75   potencial alternante (rompe inversion)
    ani  = 1.2    anisotropia de los terminos espin-conservantes de lambda'

El Hamiltoniano preserva TRS (T^2 = -1); delta != 0 rompe inversion y el
modelo realiza una fase de Weyl (el paper reporta un nodo en k=(0.7,2.56,0)
para el modelo minimo t=1, delta=0.7, lambda=0.1).

--------------------------------------------------------------------------
Interfaz QXTI (al final del archivo)
--------------------------------------------------------------------------
El modelo es adimensional (a = c = 1): kx, ky, kz se pasan directamente
(unidades reciprocas de la red idealizada) y las energias quedan en unidades
del hopping t = 1.  El Hamiltoniano se escribe en el "convenio de fase con las
posiciones de subred" (factores e^{i kz/4}, cos(kx/2), ...), por lo que la
MATRIZ H(k) es exactamente periodica sobre la caja
    kx, ky in [-2pi, 2pi]  (periodo 4pi),   kz in [-4pi, 4pi]  (periodo 8pi)
Esa es la caja que se muestrea (BZaxis diag(4pi, 4pi, 8pi)) para que el
gradiente covariante / enlaces de Wilson de QXTI (np.roll) sean exactos en el
borde.  Contiene 8 zonas de Brillouin primitivas BCT (muestreo redundante pero
fisicamente correcto: promedia la misma fisica 8 veces).  Para la ZB primitiva
exacta (no rectangular) usa un muestreo tipo Wannier.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Metadata QXTI
# ---------------------------------------------------------------------------
MODEL_NAME = "frank-zhang-taas-8band"
BASIS_SIZE = 8
DIMENSION = 3
BASIS_TYPE = "orbital+spin"
IS_PERIODIC = True


DEFAULTS = dict(tp=1.0, lam=0.1, lamp=0.1, delta=0.75, ani=1.2, M0=0.0, M0z=0.0)
DEFAULT_PARAMS = dict(DEFAULTS)

# reciprocal primitive vectors of the model's BCT lattice (a = c = 1):
#   A1=(-1/2,1/2,1/2), A2=(1/2,-1/2,1/2), A3=(1/2,1/2,-1/2)
B1 = 2 * np.pi * np.array([0., 1., 1.])
B2 = 2 * np.pi * np.array([1., 0., 1.])
B3 = 2 * np.pi * np.array([1., 1., 0.])


# Caja rectangular en la que la MATRIZ H(k) es exactamente periodica:
#   kx, ky : periodo 4pi (por cos(kx/2), sin(kx/2))
#   kz     : periodo 8pi (por e^{i kz/4}, sin(kz/4))
DEFAULT_LATTICE = {
    "lattice_type": "body-centered tetragonal (Frank-Zhang TaAs, a=c=1, adimensional)",
    "lattice_constants": {"a": 1.0, "c": 1.0},
    "BZorigin": [0.0, 0.0, 0.0],
    "BZaxis": [
        [4.0 * np.pi, 0.0, 0.0],
        [0.0, 4.0 * np.pi, 0.0],
        [0.0, 0.0, 8.0 * np.pi],
    ],
    "notes": (
        "Caja de periodicidad EXACTA de la matriz (4pi,4pi,8pi) = 8 ZB primitivas "
        "BCT (redundante pero periodica-exacta para el gradiente covariante). "
        "Reciprocal primitives: B1=2pi(0,1,1), B2=2pi(1,0,1), B3=2pi(1,1,0)."
    ),
}


def default_params() -> dict[str, float]:
    return dict(DEFAULT_PARAMS)


def _resolved_params(params: dict | None) -> dict:
    resolved = dict(DEFAULT_PARAMS)
    if params:
        resolved.update({k: v for k, v in params.items() if k in DEFAULT_PARAMS})
    return resolved


def H8(kx, ky, kz, tp=1.0, lam=0.1, lamp=0.1, delta=0.75, ani=1.2, M0=0.0, M0z=0.0):
    """Verbatim transcription of the Fortran Hamiltonian (0-based indices).

    ``M0`` (AÑADIDO, no esta en el Fortran original) es una MASA DE WILSON
    dependiente de k, el analogo directo del ``M0`` del modelo de 2 bandas:
    alli entra sumada al hopping en el MISMO canal inter-orbital,

        B1 = -[ M0 (1 - cos(a2 kz) - cos(a1 ky)) + 2 tx cos(a0 kx) ]

    Aqui se suma al "4-bond" (el hopping inter-subred sin fase interplano):

        hop4 = 4 cos(kx/2) cos(ky/2)  ->  hop4 + M0 (1 - cos(ky) - cos(kz))

    Compite con el hopping en el mismo canal, abre/cierra el gap y desplaza
    los nodos de Weyl.  M0 = 0 -> modelo original EXACTO.  El termino es real
    y simetrico (conserva hermiticidad) y sus periodos (2pi en ky, kz) son
    submultiplos de la caja (4pi, 8pi), asi que la periodicidad exacta de H(k)
    -- y con ella los enlaces de Wilson -- se conserva para cualquier M0.
    """
    H = np.zeros((8, 8), complex)
    c4, s4 = np.cos(kz / 4), np.sin(kz / 4)
    ez = c4 + 1j * s4                                 # e^{+i kz/4}
    cx2, cy2 = np.cos(kx / 2), np.cos(ky / 2)

    # -- staggered potential ------------------------------------------------
    for i, sgn in enumerate((-1, -1, +1, +1, -1, -1, +1, +1)):
        H[i, i] = sgn * delta

    # -- normal hopping (spin up: odd Fortran indices -> python 0,2,4,6) ----
    # masa de Wilson: M0z en el canal 2-bond (INTER-PLANO, lleva la fase e^{ikz/4});
    #                 M0  en el canal 4-bond (en el plano).
    wilson = (1.0 - np.cos(ky) - np.cos(kz))
    hop2y = 2 * tp * cy2 + M0z * wilson
    hop2x = 2 * tp * cx2 + M0z * wilson
    hop4 = 4 * cx2 * cy2 + M0 * wilson
    for s in (0, 1):                                  # spin up / spin down
        H[2 + s, 0 + s] = hop2y * ez
        H[4 + s, 2 + s] = hop4
        H[6 + s, 4 + s] = hop2x * ez
        H[0 + s, 6 + s] = hop4
        H[0 + s, 2 + s] = hop2y * np.conj(ez)
        H[2 + s, 4 + s] = hop4
        H[4 + s, 6 + s] = hop2x * np.conj(ez)
        H[6 + s, 0 + s] = hop4

    # -- in-plane SOI (lambda) ---------------------------------------------
    H[1, 0] = 4 * lam * np.sin(kx) * 1j
    H[0, 1] = 4 * lam * np.sin(kx) * (-1j)
    H[5, 4] = -4 * lam * np.sin(ky)
    H[4, 5] = -4 * lam * np.sin(ky)
    H[3, 2] = 4 * lam * np.sin(kx) * (-1j)
    H[2, 3] = 4 * lam * np.sin(kx) * 1j
    H[7, 6] = 4 * lam * np.sin(ky)
    H[6, 7] = 4 * lam * np.sin(ky)

    # -- inter-plane SOI (lambdap), spin-flip blocks ------------------------
    P = np.sin(kx / 2) * (1 + np.cos(ky))
    Q = np.sin(ky / 2) * (1 + np.cos(kx))
    H[5, 0] = 4 * lamp * ((P * s4 + Q * c4) + 1j * (-P * c4 - Q * s4))     # (6,1)
    H[4, 1] = 4 * lamp * ((-P * s4 + Q * c4) + 1j * (P * c4 - Q * s4))     # (5,2)
    H[0, 5] = 4 * lamp * ((P * s4 + Q * c4) + 1j * (P * c4 + Q * s4))      # (1,6)
    H[1, 4] = 4 * lamp * ((-P * s4 + Q * c4) + 1j * (-P * c4 + Q * s4))    # (2,5)
    H[3, 6] = -4 * lamp * ((P * s4 + Q * c4) + 1j * (-P * c4 - Q * s4))    # (4,7)
    H[2, 7] = -4 * lamp * ((-P * s4 + Q * c4) + 1j * (P * c4 - Q * s4))    # (3,8)
    H[6, 3] = -4 * lamp * ((P * s4 + Q * c4) + 1j * (P * c4 + Q * s4))     # (7,4)
    H[7, 2] = -4 * lamp * ((-P * s4 + Q * c4) + 1j * (-P * c4 + Q * s4))   # (8,3)

    R = np.sin(ky) * np.cos(kx / 2)
    S = np.sin(kx) * np.cos(ky / 2)
    H[5, 0] += 2 * lamp * ((R * c4 - S * s4) + 1j * (R * s4 - S * c4))     # (6,1) +=
    H[4, 1] += 2 * lamp * ((R * c4 + S * s4) + 1j * (R * s4 + S * c4))     # (5,2) +=
    H[0, 5] += 2 * lamp * ((R * c4 - S * s4) + 1j * (-R * s4 + S * c4))    # (1,6) +=
    H[1, 4] += 2 * lamp * ((R * c4 + S * s4) + 1j * (-R * s4 - S * c4))    # (2,5) +=
    H[3, 6] -= 2 * lamp * ((R * c4 - S * s4) + 1j * (R * s4 - S * c4))     # (4,7) -=
    H[2, 7] -= 2 * lamp * ((R * c4 + S * s4) + 1j * (R * s4 + S * c4))     # (3,8) -=
    H[6, 3] -= 2 * lamp * ((R * c4 - S * s4) + 1j * (-R * s4 + S * c4))    # (7,4) -=
    H[7, 2] -= 2 * lamp * ((R * c4 + S * s4) + 1j * (-R * s4 - S * c4))    # (8,3) -=

    # -- anisotropic spin-conserving lambdap terms (ani) --------------------
    U = np.sin(ky) * np.sin(kx / 2)
    V = np.sin(ky / 2) * np.sin(kx)
    Zp = U * (-s4 + 1j * c4) + V * (s4 + 1j * c4)                          # dcmplx pairs
    Zm = U * (-s4 - 1j * c4) + V * (s4 - 1j * c4)
    H[4, 0] += 4 * ani * lamp * Zp                                        # (5,1)
    H[5, 1] += -4 * ani * lamp * Zp                                       # (6,2)
    H[0, 4] += 4 * ani * lamp * Zm                                        # (1,5)
    H[1, 5] += -4 * ani * lamp * Zm                                       # (2,6)
    H[2, 6] += -4 * ani * lamp * Zp                                       # (3,7)
    H[3, 7] += 4 * ani * lamp * Zp                                        # (4,8)
    H[6, 2] += -4 * ani * lamp * Zm                                       # (7,3)
    H[7, 3] += 4 * ani * lamp * Zm                                        # (8,4)
    return H


def H_batch(kpts, params=None):
    """Version VECTORIZADA de H8 sobre muchos k: kpts (nk,3) -> (nk,8,8).

    Replica EXACTA de H8 (cada ``H[i,j] = expr`` -> ``H[:,i,j] = expr`` con
    kx,ky,kz como arrays).  Evita el loop Python de _build_H_mesh: imprescindible
    para grids grandes (120^3 = 1.7M puntos).  QXTI la usa como ``h_batch``.
    Un test compara H_batch vs H8 a precision de maquina.
    """
    p = _resolved_params(params)
    tp = float(p["tp"]); lam = float(p["lam"]); lamp = float(p["lamp"])
    delta = float(p["delta"]); ani = float(p["ani"]); M0 = float(p["M0"]); M0z = float(p["M0z"])
    kpts = np.asarray(kpts, dtype=np.float64)
    kx, ky, kz = kpts[:, 0], kpts[:, 1], kpts[:, 2]
    nk = kx.shape[0]
    H = np.zeros((nk, 8, 8), dtype=np.complex128)
    c4, s4 = np.cos(kz / 4), np.sin(kz / 4)
    ez = c4 + 1j * s4
    ezc = c4 - 1j * s4
    cx2, cy2 = np.cos(kx / 2), np.cos(ky / 2)

    for i, sgn in enumerate((-1, -1, +1, +1, -1, -1, +1, +1)):
        H[:, i, i] = sgn * delta

    wilson = (1.0 - np.cos(ky) - np.cos(kz))
    hop2y = 2 * tp * cy2 + M0z * wilson
    hop2x = 2 * tp * cx2 + M0z * wilson
    hop4 = 4 * cx2 * cy2 + M0 * wilson
    for s in (0, 1):
        H[:, 2 + s, 0 + s] = hop2y * ez
        H[:, 4 + s, 2 + s] = hop4
        H[:, 6 + s, 4 + s] = hop2x * ez
        H[:, 0 + s, 6 + s] = hop4
        H[:, 0 + s, 2 + s] = hop2y * ezc
        H[:, 2 + s, 4 + s] = hop4
        H[:, 4 + s, 6 + s] = hop2x * ezc
        H[:, 6 + s, 0 + s] = hop4

    skx, sky = np.sin(kx), np.sin(ky)
    H[:, 1, 0] = 4 * lam * skx * 1j
    H[:, 0, 1] = 4 * lam * skx * (-1j)
    H[:, 5, 4] = -4 * lam * sky
    H[:, 4, 5] = -4 * lam * sky
    H[:, 3, 2] = 4 * lam * skx * (-1j)
    H[:, 2, 3] = 4 * lam * skx * 1j
    H[:, 7, 6] = 4 * lam * sky
    H[:, 6, 7] = 4 * lam * sky

    P = np.sin(kx / 2) * (1 + np.cos(ky))
    Q = np.sin(ky / 2) * (1 + np.cos(kx))
    H[:, 5, 0] = 4 * lamp * ((P * s4 + Q * c4) + 1j * (-P * c4 - Q * s4))
    H[:, 4, 1] = 4 * lamp * ((-P * s4 + Q * c4) + 1j * (P * c4 - Q * s4))
    H[:, 0, 5] = 4 * lamp * ((P * s4 + Q * c4) + 1j * (P * c4 + Q * s4))
    H[:, 1, 4] = 4 * lamp * ((-P * s4 + Q * c4) + 1j * (-P * c4 + Q * s4))
    H[:, 3, 6] = -4 * lamp * ((P * s4 + Q * c4) + 1j * (-P * c4 - Q * s4))
    H[:, 2, 7] = -4 * lamp * ((-P * s4 + Q * c4) + 1j * (P * c4 - Q * s4))
    H[:, 6, 3] = -4 * lamp * ((P * s4 + Q * c4) + 1j * (P * c4 + Q * s4))
    H[:, 7, 2] = -4 * lamp * ((-P * s4 + Q * c4) + 1j * (-P * c4 + Q * s4))

    R = sky * np.cos(kx / 2)
    S = skx * np.cos(ky / 2)
    H[:, 5, 0] += 2 * lamp * ((R * c4 - S * s4) + 1j * (R * s4 - S * c4))
    H[:, 4, 1] += 2 * lamp * ((R * c4 + S * s4) + 1j * (R * s4 + S * c4))
    H[:, 0, 5] += 2 * lamp * ((R * c4 - S * s4) + 1j * (-R * s4 + S * c4))
    H[:, 1, 4] += 2 * lamp * ((R * c4 + S * s4) + 1j * (-R * s4 - S * c4))
    H[:, 3, 6] -= 2 * lamp * ((R * c4 - S * s4) + 1j * (R * s4 - S * c4))
    H[:, 2, 7] -= 2 * lamp * ((R * c4 + S * s4) + 1j * (R * s4 + S * c4))
    H[:, 6, 3] -= 2 * lamp * ((R * c4 - S * s4) + 1j * (-R * s4 + S * c4))
    H[:, 7, 2] -= 2 * lamp * ((R * c4 + S * s4) + 1j * (-R * s4 - S * c4))

    Uu = sky * np.sin(kx / 2)
    Vv = np.sin(ky / 2) * skx
    Zp = Uu * (-s4 + 1j * c4) + Vv * (s4 + 1j * c4)
    Zm = Uu * (-s4 - 1j * c4) + Vv * (s4 - 1j * c4)
    H[:, 4, 0] += 4 * ani * lamp * Zp
    H[:, 5, 1] += -4 * ani * lamp * Zp
    H[:, 0, 4] += 4 * ani * lamp * Zm
    H[:, 1, 5] += -4 * ani * lamp * Zm
    H[:, 2, 6] += -4 * ani * lamp * Zp
    H[:, 3, 7] += 4 * ani * lamp * Zp
    H[:, 6, 2] += -4 * ani * lamp * Zm
    H[:, 7, 3] += 4 * ani * lamp * Zm
    return H


def H(kx, ky, kz, params=None):
    """Hamiltoniano 8x8 con la firma que QXTI espera: H(kx, ky, kz, params).

    k adimensional (a = c = 1); energias en unidades del hopping t = 1.
    """
    p = _resolved_params(params)
    return H8(float(kx), float(ky), float(kz),
              tp=float(p["tp"]), lam=float(p["lam"]), lamp=float(p["lamp"]),
              delta=float(p["delta"]), ani=float(p["ani"]),
              M0=float(p.get("M0", 0.0)), M0z=float(p.get("M0z", 0.0)))


def Hk(k, **params):
    p = dict(DEFAULTS)
    p.update(params)
    return H8(k[0], k[1], k[2], **p)


def bands(ks, **params):
    return np.array([np.linalg.eigvalsh(Hk(k, **params)) for k in ks])


# ----------------------------------------------------------------------
# High-symmetry path: same fractional BCT2 path used in the TaAs
# band-structure literature (Gamma-Sigma-N-Sigma1-Z-Gamma-X), evaluated in
# this model's reciprocal basis; eta from the real TaAs c/a.
# ----------------------------------------------------------------------
ETA = (1 + (3.437 / 11.646) ** 2) / 4.0          # = 0.2718

FRAC = {
    "G":  np.array([0.0, 0.0, 0.0]),
    "S":  np.array([-ETA, ETA, ETA]),       # Sigma
    "N":  np.array([0.0, 0.5, 0.0]),
    "S1": np.array([ETA, 1 - ETA, -ETA]),   # Sigma_1
    "Z":  np.array([0.5, 0.5, -0.5]),
    "X":  np.array([0.0, 0.0, 0.5]),
}


def kcart(label):
    f = FRAC[label]
    return f[0] * B1 + f[1] * B2 + f[2] * B3


def kpath(labels, nseg=80):
    pts = [kcart(l) for l in labels]
    ks, xs, ticks, x0 = [], [], [0.0], 0.0
    for p0, p1 in zip(pts[:-1], pts[1:]):
        seg = np.linspace(0, 1, nseg, endpoint=False)[:, None]
        ks.append(p0 + seg * (p1 - p0))
        xs.append(x0 + seg[:, 0] * np.linalg.norm(p1 - p0))
        x0 += np.linalg.norm(p1 - p0)
        ticks.append(x0)
    ks.append(pts[-1][None, :])
    xs.append(np.array([x0]))
    return np.vstack(ks), np.concatenate(xs), ticks


# Alias para compatibilidad: QXTI busca la funcion 'H' por defecto.
hamiltonian = H
