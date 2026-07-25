"""Hamiltoniano tight-binding de grafeno bilayer para QXTI.

Modelo de 4 bandas con una orbital pz por sitio en la base:

    (A1, B1, A2, B2)

donde la capa 1 es la inferior y la capa 2 la superior.

El parametro ``stacking`` selecciona la geometria:

- ``AB``: Bernal estandar, con sitios dimer ``B1`` y ``A2``
- ``BA``: Bernal invertido, con sitios dimer ``A1`` y ``B2``
- ``AA``: apilamiento eclipsado; aqui se usa una rama minimal de 4 bandas
  con acoplamiento vertical ``gamma1`` entre ``A1 <-> A2`` y ``B1 <-> B2``

Las ramas ``AB`` y ``BA`` siguen el modelo tight-binding estandar discutido en:

    McCann & Koshino, Rep. Prog. Phys. 76, 056503 (2013)

incluyendo:

- ``gamma0``: hopping intralayer de primeros vecinos
- ``gamma1``: acoplamiento vertical entre sitios alineados
- ``gamma3``: hopping skew entre sitios no-dimer en Bernal
- ``gamma4``: hopping skew entre sitios dimer y no-dimer en Bernal
- ``delta_prime``: corrimiento energetico de sitios dimer en Bernal
- ``u``: asimetria entre capas (abre gap al romper inversion)

La rama ``AA`` es una extension minimal consistente con la misma base y la
misma convencion reciproca del repo. En esa rama solo entran ``gamma0``,
``gamma1`` y ``u``.
"""

from __future__ import annotations

import numpy as np


MODEL_NAME = "graphene-bilayer"
BASIS_SIZE = 4
DIMENSION = 2
BASIS_TYPE = "layer-sublattice"
IS_PERIODIC = True


EV_TO_HARTREE = 1.0 / 27.2114


# Estos defaults siguen los ordenes de magnitud usados en la literatura para
# bilayer AB y se expresan en atomic units. El parametro a0 se mantiene con la
# misma convencion usada en models/graphene.py para que ambos modelos vivan en
# el mismo dominio reciproco dentro de este repo.
DEFAULT_PARAMS = {
    "a0": 4.6479,
    "gamma0": 3.16 * EV_TO_HARTREE,
    "gamma1": 0.381 * EV_TO_HARTREE,
    "gamma3": 0.38 * EV_TO_HARTREE,
    "gamma4": 0.14 * EV_TO_HARTREE,
    "delta_prime": 0.022 * EV_TO_HARTREE,
    "u": 0.0,
    "stacking": "AB",
}

def default_params() -> dict[str, object]:
    return dict(DEFAULT_PARAMS)


def _resolved_params(params: dict[str, object] | None) -> dict[str, object]:
    resolved = default_params()
    if params:
        resolved.update(params)
    return resolved


def _validate_graphene_bilayer_params(params: dict[str, object]) -> None:
    if float(params["a0"]) <= 0.0:
        raise ValueError("a0 debe ser estrictamente positivo.")
    if np.isclose(float(params["gamma0"]), 0.0):
        raise ValueError("gamma0 no puede ser cero.")
    _normalize_stacking(params.get("stacking", "AB"))


def _normalize_stacking(stacking: object) -> str:
    value = str(stacking).strip().upper()
    if value in {"AB", "AB/BERNAL", "BERNAL", "BERNAL_AB"}:
        return "AB"
    if value in {"BA", "BA/BERNAL", "BERNAL_BA"}:
        return "BA"
    if value == "AA":
        return "AA"
    raise ValueError("stacking debe ser 'AB', 'BA' o 'AA'.")


def default_lattice(params: dict[str, object] | None = None) -> dict[str, object]:
    resolved = _resolved_params(params)
    a0 = float(resolved["a0"])
    stacking = _normalize_stacking(resolved.get("stacking", "AB"))
    stacking_label = {
        "AB": "AB/Bernal",
        "BA": "BA/Bernal invertido",
        "AA": "AA",
    }[stacking]
    return {
        "lattice_type": f"2D honeycomb bilayer graphene ({stacking_label})",
        "lattice_constants": {
            "a0": a0,
            "a": a0,
            "gamma_deg": 120.0,
        },
        "real_space_vectors": {
            "a1": [np.sqrt(3.0) * a0, 0.0],
            "a2": [np.sqrt(3.0) * 0.5 * a0, 1.5 * a0],
        },
        "BZorigin": [0.0, 0.0, 0.0],
        "BZaxis": [
            [2.0 * np.pi / (np.sqrt(3.0) * a0), 0.0, 0.0],
            [0.0, 2.0 * np.pi / (1.5 * a0), 0.0],
            [0.0, 0.0, 0.0],
        ],
        "stacking": stacking_label,
        "basis_order": ["A1", "B1", "A2", "B2"],
    }


def _nn_vectors(a0: float) -> np.ndarray:
    """Vectores A -> B de primeros vecinos, consistentes con graphene.py."""

    return np.array(
        [
            [0.0, a0],
            [-np.sqrt(3.0) * 0.5 * a0, -0.5 * a0],
            [np.sqrt(3.0) * 0.5 * a0, -0.5 * a0],
        ],
        dtype=float,
    )


def _structure_factor(kx: float, ky: float, a0: float) -> np.complex128:
    deltas = _nn_vectors(a0)
    k = np.array([float(kx), float(ky)], dtype=float)
    phases = deltas @ k
    return np.asarray(np.exp(1.0j * phases).sum(), dtype=np.complex128)


def _hamiltonian_ab(
    f: np.complex128,
    gamma0: float,
    gamma1: float,
    gamma3: float,
    gamma4: float,
    delta_prime: float,
    u: float,
) -> np.ndarray:
    f_conj = np.conjugate(f)

    # Sitios no-dimer: A1, B2
    # Sitios dimer:    B1, A2
    e_a1 = 0.5 * u
    e_b1 = 0.5 * u + delta_prime
    e_a2 = -0.5 * u + delta_prime
    e_b2 = -0.5 * u

    return np.array(
        [
            [e_a1, gamma0 * f, gamma4 * f_conj, gamma3 * f],
            [gamma0 * f_conj, e_b1, gamma1, gamma4 * f_conj],
            [gamma4 * f, gamma1, e_a2, gamma0 * f],
            [gamma3 * f_conj, gamma4 * f, gamma0 * f_conj, e_b2],
        ],
        dtype=complex,
    )


def _hamiltonian_ba(
    f: np.complex128,
    gamma0: float,
    gamma1: float,
    gamma3: float,
    gamma4: float,
    delta_prime: float,
    u: float,
) -> np.ndarray:
    f_conj = np.conjugate(f)

    # Sitios no-dimer: B1, A2
    # Sitios dimer:    A1, B2
    e_a1 = 0.5 * u + delta_prime
    e_b1 = 0.5 * u
    e_a2 = -0.5 * u
    e_b2 = -0.5 * u + delta_prime

    return np.array(
        [
            [e_a1, gamma0 * f, gamma4 * f, gamma1],
            [gamma0 * f_conj, e_b1, gamma3 * f_conj, gamma4 * f],
            [gamma4 * f_conj, gamma3 * f, e_a2, gamma0 * f],
            [gamma1, gamma4 * f_conj, gamma0 * f_conj, e_b2],
        ],
        dtype=complex,
    )


def _hamiltonian_aa(
    f: np.complex128,
    gamma0: float,
    gamma1: float,
    u: float,
) -> np.ndarray:
    f_conj = np.conjugate(f)
    e_layer1 = 0.5 * u
    e_layer2 = -0.5 * u

    return np.array(
        [
            [e_layer1, gamma0 * f, gamma1, 0.0],
            [gamma0 * f_conj, e_layer1, 0.0, gamma1],
            [gamma1, 0.0, e_layer2, gamma0 * f],
            [0.0, gamma1, gamma0 * f_conj, e_layer2],
        ],
        dtype=complex,
    )


def H(kx: float, ky: float, kz: float, params: dict[str, object] | None) -> np.ndarray:
    """Retorna H(k) del grafeno bilayer con firma compatible con QXTI."""

    del kz
    resolved = _resolved_params(params)
    _validate_graphene_bilayer_params(resolved)

    a0 = float(resolved["a0"])
    gamma0 = float(resolved["gamma0"])
    gamma1 = float(resolved["gamma1"])
    gamma3 = float(resolved["gamma3"])
    gamma4 = float(resolved["gamma4"])
    delta_prime = float(resolved["delta_prime"])
    u = float(resolved["u"])
    stacking = _normalize_stacking(resolved.get("stacking", "AB"))

    f = _structure_factor(float(kx), float(ky), a0)
    if stacking == "AB":
        return _hamiltonian_ab(f, gamma0, gamma1, gamma3, gamma4, delta_prime, u)
    if stacking == "BA":
        return _hamiltonian_ba(f, gamma0, gamma1, gamma3, gamma4, delta_prime, u)
    return _hamiltonian_aa(f, gamma0, gamma1, u)


def H_batch(kpts, params=None):
    """Version VECTORIZADA de H sobre muchos k: kpts (nk,3) -> (nk,4,4).

    Replica EXACTA de H (cada ``H[i,j] = expr`` -> ``H[:,i,j] = expr`` con
    kx,ky como arrays). El stacking es un parametro escalar por llamada, asi
    que la rama AB/BA/AA se elige una sola vez; el factor de estructura ``f``
    y todos los elementos de matriz se construyen vectorizados sobre k.
    """
    resolved = _resolved_params(params)
    _validate_graphene_bilayer_params(resolved)

    a0 = float(resolved["a0"])
    gamma0 = float(resolved["gamma0"])
    gamma1 = float(resolved["gamma1"])
    gamma3 = float(resolved["gamma3"])
    gamma4 = float(resolved["gamma4"])
    delta_prime = float(resolved["delta_prime"])
    u = float(resolved["u"])
    stacking = _normalize_stacking(resolved.get("stacking", "AB"))

    kpts = np.asarray(kpts, dtype=np.float64)
    kx, ky = kpts[:, 0], kpts[:, 1]
    nk = kx.shape[0]

    # Factor de estructura vectorizado (equivalente a _structure_factor).
    deltas = _nn_vectors(a0)
    phases = kx[:, None] * deltas[:, 0][None, :] + ky[:, None] * deltas[:, 1][None, :]
    f = np.exp(1.0j * phases).sum(axis=1).astype(np.complex128)
    f_conj = np.conjugate(f)

    H = np.zeros((nk, 4, 4), dtype=np.complex128)

    if stacking == "AB":
        # Sitios no-dimer: A1, B2 ; sitios dimer: B1, A2
        e_a1 = 0.5 * u
        e_b1 = 0.5 * u + delta_prime
        e_a2 = -0.5 * u + delta_prime
        e_b2 = -0.5 * u

        H[:, 0, 0] = e_a1
        H[:, 0, 1] = gamma0 * f
        H[:, 0, 2] = gamma4 * f_conj
        H[:, 0, 3] = gamma3 * f
        H[:, 1, 0] = gamma0 * f_conj
        H[:, 1, 1] = e_b1
        H[:, 1, 2] = gamma1
        H[:, 1, 3] = gamma4 * f_conj
        H[:, 2, 0] = gamma4 * f
        H[:, 2, 1] = gamma1
        H[:, 2, 2] = e_a2
        H[:, 2, 3] = gamma0 * f
        H[:, 3, 0] = gamma3 * f_conj
        H[:, 3, 1] = gamma4 * f
        H[:, 3, 2] = gamma0 * f_conj
        H[:, 3, 3] = e_b2
        return H

    if stacking == "BA":
        # Sitios no-dimer: B1, A2 ; sitios dimer: A1, B2
        e_a1 = 0.5 * u + delta_prime
        e_b1 = 0.5 * u
        e_a2 = -0.5 * u
        e_b2 = -0.5 * u + delta_prime

        H[:, 0, 0] = e_a1
        H[:, 0, 1] = gamma0 * f
        H[:, 0, 2] = gamma4 * f
        H[:, 0, 3] = gamma1
        H[:, 1, 0] = gamma0 * f_conj
        H[:, 1, 1] = e_b1
        H[:, 1, 2] = gamma3 * f_conj
        H[:, 1, 3] = gamma4 * f
        H[:, 2, 0] = gamma4 * f_conj
        H[:, 2, 1] = gamma3 * f
        H[:, 2, 2] = e_a2
        H[:, 2, 3] = gamma0 * f
        H[:, 3, 0] = gamma1
        H[:, 3, 1] = gamma4 * f_conj
        H[:, 3, 2] = gamma0 * f_conj
        H[:, 3, 3] = e_b2
        return H

    # stacking == "AA"
    e_layer1 = 0.5 * u
    e_layer2 = -0.5 * u

    H[:, 0, 0] = e_layer1
    H[:, 0, 1] = gamma0 * f
    H[:, 0, 2] = gamma1
    H[:, 0, 3] = 0.0
    H[:, 1, 0] = gamma0 * f_conj
    H[:, 1, 1] = e_layer1
    H[:, 1, 2] = 0.0
    H[:, 1, 3] = gamma1
    H[:, 2, 0] = gamma1
    H[:, 2, 1] = 0.0
    H[:, 2, 2] = e_layer2
    H[:, 2, 3] = gamma0 * f
    H[:, 3, 0] = 0.0
    H[:, 3, 1] = gamma1
    H[:, 3, 2] = gamma0 * f_conj
    H[:, 3, 3] = e_layer2
    return H
