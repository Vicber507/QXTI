"""Hamiltoniano tight-binding de grafeno bilayer AB (Bernal) para QXTI.

Modelo de 4 bandas con una orbital pz por sitio en la base:

    (A1, B1, A2, B2)

donde la capa 1 es la inferior y la capa 2 la superior. Los sitios dimer son
``B1`` y ``A2`` porque estan alineados verticalmente en el apilamiento AB.

La implementacion sigue el modelo tight-binding estandar discutido en:

    McCann & Koshino, Rep. Prog. Phys. 76, 056503 (2013)

incluyendo:

- ``gamma0``: hopping intralayer de primeros vecinos
- ``gamma1``: acoplamiento vertical entre sitios dimer B1 <-> A2
- ``gamma3``: hopping skew entre sitios no-dimer A1 <-> B2
- ``gamma4``: hopping skew entre sitios dimer y no-dimer
- ``delta_prime``: corrimiento energetico de sitios dimer respecto a no-dimer
- ``u``: asimetria entre capas (abre gap al romper inversion)

Para mantener compatibilidad con el ``models/graphene.py`` existente del repo,
este archivo reutiliza exactamente la misma convencion de red y el mismo factor
de estructura ``f(k)``.
"""

from __future__ import annotations

import numpy as np


MODEL_NAME = "graphene-bilayer-ab"
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
}


DEFAULT_LATTICE = {
    "lattice_type": "2D honeycomb bilayer AB (Bernal)",
    "lattice_constants": {
        "a0": 4.6479,
        "a": 4.6479,
        "gamma_deg": 120.0,
    },
    "real_space_vectors": {
        "a1": [8.0508, 0.0],
        "a2": [4.0254, 6.9719],
    },
    "BZorigin": [0.0, 0.0, 0.0],
    "BZaxis": [
        [2.0 * np.pi / (np.sqrt(3.0) * 4.6479), 0.0, 0.0],
        [0.0, 2.0 * np.pi / (1.5 * 4.6479), 0.0],
        [0.0, 0.0, 0.0],
    ],
    "stacking": "AB/Bernal",
    "basis_order": ["A1", "B1", "A2", "B2"],
}


def default_params() -> dict[str, float]:
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


def H(kx: float, ky: float, kz: float, params: dict[str, object] | None) -> np.ndarray:
    """Retorna H(k) del grafeno bilayer AB con firma compatible con QXTI."""

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

    f = _structure_factor(float(kx), float(ky), a0)
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
