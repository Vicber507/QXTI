"""Modelo minimo de semimetal de Weyl de Wu/Orenstein et al., Nature Physics 2016.

Implementacion del modelo de cuatro nodos de Weyl que rompe inversion (ecuacion 2
de Wu et al., "Giant anisotropic nonlinear optical response in transition metal
monopnictide Weyl semimetals", Nat. Phys. 13, 350 (2017); DOI 10.1038/NPHYS3969),
usado para explicar la respuesta SHG (chi^(2)) gigante de TaAs.

Hamiltoniano (ec. 2), con sigma_i (orbital) y s_i (spin) matrices de Pauli:

    H = t { [cos(kx a) + my(1 - cos(ky a)) + mz(1 - cos(kz a))] sigma_x
            + [sin(ky a) + Delta cos(ky a) s_x] sigma_y
            + sin(kz a) s_x sigma_z }

En el espacio orbital (x) spin (4x4):

    H = t [ f_x (sigma_x o I) + sin(ky a)(sigma_y o I)
            + Delta cos(ky a)(sigma_y o s_x) + sin(kz a)(sigma_z o s_x) ]

    f_x = cos(kx a) + my(1 - cos(ky a)) + mz(1 - cos(kz a))

Fisica:
-------
* t          : escala de energia (bandwidth). Paper: t = 0.8 eV.
* my, mz     : anisotropia. Paper: my = 1, mz = 5.
* Delta      : ruptura de INVERSION. Es esencial: chi^(2) -> 0 cuando Delta -> 0.
               Paper: Delta = 0.5.
* a          : constante de red (modelo esquematico cubico). k.a es adimensional.

Como s_x conmuta con H (no aparecen s_y, s_z), el Hamiltoniano es bloque-diagonal
en s_x = +/-1; cada bloque 2x2 aloja un par de nodos de Weyl.

Nodos de Weyl:
    Delta = 0 : pares de quiralidad opuesta se SUPERPONEN en (+/- pi/2a, 0, 0)
                (caso centrosimetrico -> sin SHG).
    Delta != 0: los nodos se separan a lo largo de ky, Delta_ky ~ Delta/a
                (rompe inversion -> SHG no nulo).

El Hamiltoniano preserva la rotacion C2 alrededor de z y los espejos Mx, My
(subconjunto del grupo puntual 4mm de TaAs), y es invariante bajo TRS.

Nota sobre el k-grid:
--------------------
Ademas de los nodos de Weyl, el modelo tiene una degeneracion doble (interna a
la banda de valencia y a la de conduccion) en la linea ky = 0, kz = 0. El gap
optico (valencia-conduccion) ahi es no nulo, pero conviene evitar esa linea para
que el suavizado de la conexion de Berry sea limpio. Con shifted = true, usa un
numero PAR de puntos por eje (p.ej. [8,8,8]) para que el grid no incluya ky=0.

Referencia:
    Wu, Patankar, Morimoto, ... Orenstein, Nat. Phys. 13, 350 (2017).
    Modelo construido siguiendo Morimoto & Nagaosa, Sci. Adv. 2, e1501524 (2016).
"""
from __future__ import annotations

import numpy as np


MODEL_NAME = "wsm-orenstein-taas-shg"
BASIS_SIZE = 4
DIMENSION = 3
BASIS_TYPE = "orbital_spin"
IS_PERIODIC = True


AU_PER_ANGSTROM = 1.8897259886
EV_PER_HARTREE = 27.211386245988

# Pauli matrices (shared by orbital sigma and spin s sectors).
_sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)

# 4x4 building blocks: orbital (x) spin, with Kronecker order (orbital, spin).
_SIGX_I = np.kron(_sx, _I2)
_SIGY_I = np.kron(_sy, _I2)
_SIGY_SX = np.kron(_sy, _sx)
_SIGZ_SX = np.kron(_sz, _sx)


# Paper parameters (Fig. 4d): t = 0.8 eV, Delta = 0.5, mz = 5, my = 1.
_T_AU = 0.8 / EV_PER_HARTREE  # 0.8 eV in atomic units

DEFAULT_PARAMS = {
    "t": _T_AU,      # bandwidth / energy scale [a.u.]  (0.8 eV)
    "my": 1.0,       # anisotropy along ky
    "mz": 5.0,       # anisotropy along kz
    "Delta": 0.5,    # inversion-breaking parameter (chi^(2)=0 if Delta=0)
    "a": 1.0,        # lattice constant [a.u.] (schematic cubic model; k*a dimensionless)
}


def _lattice_for(a: float) -> dict:
    return {
        "lattice_type": "3D simple cubic (schematic Weyl model)",
        "lattice_constants": {"a0": a, "a": a, "a1_length": a, "a2_length": a, "gamma_deg": 90.0},
        "real_space_vectors": {
            "a1": [a, 0.0, 0.0],
            "a2": [0.0, a, 0.0],
            "a3": [0.0, 0.0, a],
        },
        "BZorigin": [0.0, 0.0, 0.0],
        "BZaxis": [
            [2.0 * np.pi / a, 0.0, 0.0],
            [0.0, 2.0 * np.pi / a, 0.0],
            [0.0, 0.0, 2.0 * np.pi / a],
        ],
    }


DEFAULT_LATTICE = _lattice_for(float(DEFAULT_PARAMS["a"]))


def default_params() -> dict[str, object]:
    """Return a copy of the default parameters."""
    return dict(DEFAULT_PARAMS)


def _resolved_params(params: dict[str, object] | None) -> dict[str, object]:
    resolved = default_params()
    if params:
        resolved.update(params)
    return resolved


def _validate_params(params: dict[str, object]) -> None:
    if float(params["a"]) <= 0.0:
        raise ValueError("a debe ser estrictamente positivo.")
    if float(params["t"]) == 0.0:
        raise ValueError("t (bandwidth) no puede ser cero.")


def H(kx: float, ky: float, kz: float, params: dict[str, object] | None = None) -> np.ndarray:
    """4x4 Weyl Hamiltonian of Wu/Orenstein et al. (eq. 2) with the QXTI signature.

    Parameters
    ----------
    kx, ky, kz : float
        Wave-vector components in a.u.^{-1}.
    params : dict, optional
        Overrides any subset of DEFAULT_PARAMS.

    Returns
    -------
    np.ndarray, shape (4, 4), dtype=complex
        Hermitian H(k).
    """
    p = _resolved_params(params)
    _validate_params(p)
    t = float(p["t"])
    my = float(p["my"])
    mz = float(p["mz"])
    delta = float(p["Delta"])
    a = float(p["a"])

    kxa, kya, kza = a * float(kx), a * float(ky), a * float(kz)
    f_x = np.cos(kxa) + my * (1.0 - np.cos(kya)) + mz * (1.0 - np.cos(kza))

    matrix = t * (
        f_x * _SIGX_I
        + np.sin(kya) * _SIGY_I
        + delta * np.cos(kya) * _SIGY_SX
        + np.sin(kza) * _SIGZ_SX
    )
    return np.asarray(matrix, dtype=complex)


def H_batch(kpts, params: dict[str, object] | None = None) -> np.ndarray:
    """Vectorized version of ``H`` over many k-points: kpts (nk,3) -> (nk,4,4).

    Exact replica of ``H`` (each scalar coefficient ``coeff(kx,ky,kz)`` becomes an
    array of shape (nk,) multiplying the corresponding constant 4x4 building
    block). Avoids the per-k Python loop over ``H`` for large grids. QXTI uses
    this as ``h_batch``; a test compares it against ``H`` at machine precision.
    """
    p = _resolved_params(params)
    _validate_params(p)
    t = float(p["t"])
    my = float(p["my"])
    mz = float(p["mz"])
    delta = float(p["Delta"])
    a = float(p["a"])

    kpts = np.asarray(kpts, dtype=np.float64)
    kx, ky, kz = kpts[:, 0], kpts[:, 1], kpts[:, 2]
    nk = kx.shape[0]

    kxa, kya, kza = a * kx, a * ky, a * kz
    f_x = np.cos(kxa) + my * (1.0 - np.cos(kya)) + mz * (1.0 - np.cos(kza))

    # coefficients: (nk, 1, 1) broadcast against constant (4, 4) blocks.
    c_sigx_i = (t * f_x)[:, None, None]
    c_sigy_i = (t * np.sin(kya))[:, None, None]
    c_sigy_sx = (t * delta * np.cos(kya))[:, None, None]
    c_sigz_sx = (t * np.sin(kza))[:, None, None]

    H = (
        c_sigx_i * _SIGX_I
        + c_sigy_i * _SIGY_I
        + c_sigy_sx * _SIGY_SX
        + c_sigz_sx * _SIGZ_SX
    )
    return np.asarray(H, dtype=np.complex128)


def dH_dk(kx: float, ky: float, kz: float, direction: str,
          params: dict[str, object] | None = None) -> np.ndarray:
    """ANALYTIC velocity operator dH/dk along 'x'|'y'|'z' (exact, no finite diff).

    Differentiating eq. (2) term by term (chain rule d(a k)/dk = a):
        dH/dkx = t (-a sin kxa) sigma_x(x)I
        dH/dky = t [ my a sin kya  sigma_x(x)I + a cos kya  sigma_y(x)I
                     - Delta a sin kya  sigma_y(x)s_x ]
        dH/dkz = t [ mz a sin kza  sigma_x(x)I + a cos kza  sigma_z(x)s_x ]

    Detected automatically by the QXTI loader (CustomHamiltonian.dH_dk), so the
    band velocities / Berry pieces use exact derivatives instead of finite
    differences.
    """
    p = _resolved_params(params)
    _validate_params(p)
    t = float(p["t"]); my = float(p["my"]); mz = float(p["mz"])
    delta = float(p["Delta"]); a = float(p["a"])
    kxa, kya, kza = a * float(kx), a * float(ky), a * float(kz)

    d = str(direction).lower()
    if d in ("x", "kx", "0"):
        matrix = t * (-a * np.sin(kxa)) * _SIGX_I
    elif d in ("y", "ky", "1"):
        matrix = t * (
            my * a * np.sin(kya) * _SIGX_I
            + a * np.cos(kya) * _SIGY_I
            - delta * a * np.sin(kya) * _SIGY_SX
        )
    elif d in ("z", "kz", "2"):
        matrix = t * (
            mz * a * np.sin(kza) * _SIGX_I
            + a * np.cos(kza) * _SIGZ_SX
        )
    else:
        raise ValueError(f"direction invalido: {direction!r} (usa 'x'|'y'|'z').")
    return np.asarray(matrix, dtype=complex)


def band_energies(kx: float, ky: float, kz: float, params: dict[str, object] | None = None) -> np.ndarray:
    """Return the four band energies (sorted ascending)."""
    return np.linalg.eigvalsh(H(kx, ky, kz, params))


def _weyl_nodes_with_block(params: dict[str, object] | None = None) -> list[tuple[np.ndarray, int]]:
    """Return the four Weyl nodes as (position, s_x-block eigenvalue) pairs.

    The Hamiltonian is block-diagonal in ``s_x = +/-1``; each block hosts a pair
    of opposite-position nodes:

    * s_x = +1 block: sin(ky a) + Delta cos(ky a) = 0  ->  ky a = -arctan(Delta)
    * s_x = -1 block: sin(ky a) - Delta cos(ky a) = 0  ->  ky a = +arctan(Delta)

    and kz = 0. Because the sigma_x coefficient f_x depends on ky through
    ``my (1 - cos(ky a))``, the node's kx is shifted away from pi/2a and is
    obtained from f_x = 0:  cos(kx a) = -my (1 - cos(ky a)).
    """
    p = _resolved_params(params)
    a = float(p["a"])
    delta = float(p["Delta"])
    my = float(p["my"])

    def _kx_for(kya: float) -> float:
        # cos(kx a) = -my (1 - cos(ky a)); take the root near pi/2.
        rhs = -my * (1.0 - np.cos(kya))
        rhs = float(np.clip(rhs, -1.0, 1.0))
        return np.arccos(rhs) / a

    kya_plus = -np.arctan(delta)   # s_x = +1
    kya_minus = +np.arctan(delta)  # s_x = -1
    kx_p = _kx_for(kya_plus)
    kx_m = _kx_for(kya_minus)
    return [
        (np.array([+kx_p, kya_plus / a, 0.0], dtype=float), +1),
        (np.array([-kx_p, kya_plus / a, 0.0], dtype=float), +1),
        (np.array([+kx_m, kya_minus / a, 0.0], dtype=float), -1),
        (np.array([-kx_m, kya_minus / a, 0.0], dtype=float), -1),
    ]


def weyl_nodes(params: dict[str, object] | None = None) -> np.ndarray:
    """Return the four Weyl-node positions in the kz = 0 plane (shape (4, 3))."""
    return np.array([pos for pos, _s in _weyl_nodes_with_block(params)], dtype=float)


def weyl_nodes_with_chirality(params: dict[str, object] | None = None) -> list[dict[str, object]]:
    """Return Weyl-node positions together with their chirality.

    The chirality is the sign of the Jacobian ``det(d d_i / d k_j)`` of the 2x2
    Weyl block at the node. In the ``s_x = s`` block the d-vector is

        d_x = t f_x
        d_y = t [sin(ky a) + s Delta cos(ky a)]
        d_z = t s sin(kz a)

    so at kz = 0 the Jacobian determinant is proportional to

        -s sin(kx a) [cos(ky a) - s Delta sin(ky a)].

    The ``s`` factor in ``d_z`` (the sign flip between the two s_x blocks) is what
    makes the chirality follow the DIAGONAL pattern chi = sign(kx * ky), i.e. two
    +1 and two -1 nodes as required by Nielsen-Ninomiya. (An earlier version keyed
    chirality on sign(kx) alone and mislabelled the two ky>0 nodes; verified here
    against the numerical Berry-flux Chern number.)
    """
    p = _resolved_params(params)
    a = float(p["a"])
    delta = float(p["Delta"])
    tagged: list[dict[str, object]] = []
    for pos, s in _weyl_nodes_with_block(params):
        kxa = a * float(pos[0])
        kya = a * float(pos[1])
        det_j = -s * np.sin(kxa) * (np.cos(kya) - s * delta * np.sin(kya))
        chirality = 1 if det_j > 0.0 else -1
        tagged.append(
            {
                "k": np.asarray(pos, dtype=float),
                "chirality": int(chirality),
            }
        )
    return tagged


# QXTI looks for the function 'H' by default.
hamiltonian = H
