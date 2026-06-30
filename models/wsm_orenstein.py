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


def band_energies(kx: float, ky: float, kz: float, params: dict[str, object] | None = None) -> np.ndarray:
    """Return the four band energies (sorted ascending)."""
    return np.linalg.eigvalsh(H(kx, ky, kz, params))


def weyl_nodes(params: dict[str, object] | None = None) -> np.ndarray:
    """Return the four Weyl-node positions in the kz = 0 plane.

    For Delta = 0 the opposite-chirality nodes coincide at (+-pi/2a, 0, 0).
    For Delta != 0 they split along ky:

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
    return np.array(
        [
            [+kx_p, kya_plus / a, 0.0],
            [-kx_p, kya_plus / a, 0.0],
            [+kx_m, kya_minus / a, 0.0],
            [-kx_m, kya_minus / a, 0.0],
        ],
        dtype=float,
    )


def weyl_nodes_with_chirality(params: dict[str, object] | None = None) -> list[dict[str, object]]:
    """Return Weyl-node positions together with their chirality.

    For the 2x2 Weyl blocks of this model the chirality is the sign of the
    Jacobian ``det(∂d_i / ∂k_j)`` at the node. In this Hamiltonian the ``ky``
    and ``kz`` derivatives contribute with the same positive sign at every node,
    so the chirality is fixed by the sign of ``-sin(kx a)``:

    * ``kx < 0``  -> chirality ``+1``
    * ``kx > 0``  -> chirality ``-1``

    This yields two positive-chirality nodes and two negative-chirality nodes,
    as required by Nielsen-Ninomiya.
    """
    nodes = np.asarray(weyl_nodes(params), dtype=float)
    tagged: list[dict[str, object]] = []
    for node in nodes:
        chirality = +1 if float(node[0]) < 0.0 else -1
        tagged.append(
            {
                "k": np.asarray(node, dtype=float),
                "chirality": int(chirality),
            }
        )
    return tagged


# QXTI looks for the function 'H' by default.
hamiltonian = H
