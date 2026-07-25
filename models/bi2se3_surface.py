"""Hamiltoniano de superficie de Bi2Se3 en un modulo autocontenido."""

from __future__ import annotations

import numpy as np


MODEL_NAME = "bi2se3-surface"
BASIS_SIZE = 2
DIMENSION = 2
BASIS_TYPE = "spin"
IS_PERIODIC = True


sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_0 = np.eye(2, dtype=complex)


DEFAULT_PARAMS = {
    "a0": 7.8234655927,
    "A0": -0.000937,
    "B0": 0.00060,
    "A11": 0.00711836,
    "A12": 0.00823187,
    "A14": 0.00202489,
    "B11": 0.00442095,
    "B14": 0.0,
}

DEFAULT_LATTICE = {
    "lattice_type": "2D hexagonal surface",
    "lattice_constants": {
        "a0": 7.8234655927,
        "a1_length": 7.8234655927,
        "a2_length": 7.8234655927,
        "gamma_deg": 120.0,
    },
    "real_space_vectors": {
        "a1": [7.8234655927, 0.0],
        "a2": [-3.91173279635, 6.775091094748115],
    },
    "BZorigin": [0.0, 0.0, 0.0],
    "BZaxis": [
        [2.0 * np.pi / 7.8234655927, 0.0, 0.0],
        [0.0, 4.0 * np.pi / (np.sqrt(3.0) * 7.8234655927), 0.0],
        [0.0, 0.0, 0.0],
    ],
}


Omega = np.array([0.0, -2.0 * np.pi / 3.0, -4.0 * np.pi / 3.0], dtype=float)


def default_params() -> dict[str, float]:
    return dict(DEFAULT_PARAMS)


def _resolved_params(params: dict[str, object] | None) -> dict[str, object]:
    resolved = default_params()
    if params:
        resolved.update(params)
    return resolved


def _validate_surface_params(params: dict[str, object]) -> None:
    if float(params["a0"]) <= 0.0:
        raise ValueError("a0 debe ser estrictamente positivo.")
    if np.isclose(float(params["B11"]), 0.0):
        raise ValueError("B11 no puede ser cero.")
    if abs(float(params["B0"]) / float(params["B11"])) > 1.0:
        raise ValueError("|B0 / B11| debe ser <= 1 para que gamma sea real.")


def _surface_vectors(a0: float) -> tuple[np.ndarray, np.ndarray]:
    sqrt3 = np.sqrt(3.0)
    surface_avec = np.array(
        [
            [a0, 0.0],
            [-a0 / 2.0, (a0 * sqrt3) / 2.0],
            [-a0 / 2.0, -(a0 * sqrt3) / 2.0],
        ],
        dtype=float,
    )
    surface_bvec = np.array(
        [
            [0.0, (a0 * sqrt3) / 3.0],
            [-a0 / 2.0, -(a0 * sqrt3) / 6.0],
            [a0 / 2.0, -(a0 * sqrt3) / 6.0],
        ],
        dtype=float,
    )
    return surface_avec, surface_bvec


def _effective_coefficients(params: dict[str, object]) -> tuple[float, float, float, float, float]:
    B0 = float(params["B0"])
    B11 = float(params["B11"])
    A11 = float(params["A11"])
    A12 = float(params["A12"])
    A14 = float(params["A14"])
    B14 = float(params["B14"])
    A0 = float(params["A0"])

    gamma = np.sqrt(1.0 - (B0 / B11) ** 2)
    eps = 6.0 * B0 * (1.0 + (A11 / B11))
    t0 = A0 - B0 * (A11 / B11)
    lamb_a = A14 * gamma
    lamb_b = B14 * gamma
    lamb_z = A12 * gamma
    return eps, t0, lamb_a, lamb_b, lamb_z


def H(kx: float, ky: float, kz: float, params: dict[str, object] | None) -> np.ndarray:
    """Hamiltoniano natural de superficie de Bi2Se3 con firma QXTI."""

    del kz
    resolved = _resolved_params(params)
    _validate_surface_params(resolved)

    a0 = float(resolved["a0"])
    surface_avec, surface_bvec = _surface_vectors(a0)
    eps, t0, lamb_a, lamb_b, lamb_z = _effective_coefficients(resolved)

    k = np.array([float(kx), float(ky)], dtype=float)
    h0 = 0.0
    h1 = 0.0
    h2 = 0.0
    h3 = 0.0

    for i in range(3):
        pa = np.dot(k, surface_avec[i])
        pb = np.dot(k, surface_bvec[i])

        h0 += np.cos(pa)
        h1 += -2.0 * lamb_a * np.sin(Omega[i]) * np.sin(pa)
        h1 += 2.0 * lamb_b * np.cos(Omega[i]) * np.sin(pb)
        h2 += -2.0 * lamb_a * np.cos(Omega[i]) * np.sin(pa)
        h2 += -2.0 * lamb_b * np.sin(Omega[i]) * np.sin(pb)
        h3 += 2.0 * lamb_z * np.sin(pa)

    h0 = eps + 2.0 * t0 * h0
    return h0 * sigma_0 + h1 * sigma_x + h2 * sigma_y + h3 * sigma_z


def H_batch(kpts, params=None):
    """Version VECTORIZADA de H sobre muchos k: kpts (nk,3) -> (nk,2,2).

    Replica EXACTA de H (cada ``H[i,j] = expr`` -> ``H[:,i,j] = expr`` con
    kx,ky como arrays).  Evita el loop Python por k: imprescindible para grids
    grandes.  QXTI la usa como ``h_batch``; un test la compara contra H a
    precision de maquina.
    """
    resolved = _resolved_params(params)
    _validate_surface_params(resolved)

    a0 = float(resolved["a0"])
    surface_avec, surface_bvec = _surface_vectors(a0)
    eps, t0, lamb_a, lamb_b, lamb_z = _effective_coefficients(resolved)

    kpts = np.asarray(kpts, dtype=np.float64)
    kx, ky = kpts[:, 0], kpts[:, 1]
    nk = kx.shape[0]

    h0 = np.zeros(nk, dtype=np.float64)
    h1 = np.zeros(nk, dtype=np.float64)
    h2 = np.zeros(nk, dtype=np.float64)
    h3 = np.zeros(nk, dtype=np.float64)

    for i in range(3):
        pa = kx * surface_avec[i, 0] + ky * surface_avec[i, 1]
        pb = kx * surface_bvec[i, 0] + ky * surface_bvec[i, 1]

        h0 += np.cos(pa)
        h1 += -2.0 * lamb_a * np.sin(Omega[i]) * np.sin(pa)
        h1 += 2.0 * lamb_b * np.cos(Omega[i]) * np.sin(pb)
        h2 += -2.0 * lamb_a * np.cos(Omega[i]) * np.sin(pa)
        h2 += -2.0 * lamb_b * np.sin(Omega[i]) * np.sin(pb)
        h3 += 2.0 * lamb_z * np.sin(pa)

    h0 = eps + 2.0 * t0 * h0

    H = np.zeros((nk, 2, 2), dtype=np.complex128)
    H += h0[:, None, None] * sigma_0
    H += h1[:, None, None] * sigma_x
    H += h2[:, None, None] * sigma_y
    H += h3[:, None, None] * sigma_z
    return H
