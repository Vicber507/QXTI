"""Plantilla minima y comentada para programar un Hamiltoniano en QXTI."""

from __future__ import annotations

import numpy as np


# Recomendado: metadata del modelo.
MODEL_NAME = "example-two-band-dirac"
BASIS_SIZE = 2
DIMENSION = 2

# Opcional: ayuda a documentar la base y periodicidad.
BASIS_TYPE = "spin"
IS_PERIODIC = True


# Opcional pero muy util: parametros fisicos por defecto.
DEFAULT_PARAMS = {
    "mass": 0.35,
    "velocity": 2.10,
    "curvature": 1.00,
    "energy_shift": 0.0,
}


# Opcional: informacion de red para lectura/documentacion.
DEFAULT_LATTICE = {
    "lattice_type": "2D square lattice",
    "lattice_constants": {
        "a": 1.0,
        "b": 1.0,
        "gamma_deg": 90.0,
    },
    "real_space_vectors": {
        "a1": [1.0, 0.0],
        "a2": [0.0, 1.0],
    },
    "notes": "Puedes guardar aqui informacion extra del modelo.",
}


sigma_0 = np.eye(2, dtype=complex)
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def default_params() -> dict[str, float]:
    """Opcional: si existe, QXTI tambien puede leer defaults desde aqui."""

    return dict(DEFAULT_PARAMS)


def _resolved_params(params: dict[str, object] | None) -> dict[str, object]:
    resolved = default_params()
    if params:
        resolved.update(params)
    return resolved


def H(kx: float, ky: float, kz: float, params: dict[str, object] | None) -> np.ndarray:
    """Funcion obligatoria.

    Debe aceptar exactamente: ``H(kx, ky, kz, params)``.
    Debe devolver una matriz Hermitica compleja de tamano ``BASIS_SIZE x BASIS_SIZE``.
    """

    del kz
    resolved = _resolved_params(params)

    mass = float(resolved["mass"])
    velocity = float(resolved["velocity"])
    curvature = float(resolved["curvature"])
    energy_shift = float(resolved["energy_shift"])

    k2 = float(kx) ** 2 + float(ky) ** 2

    return (
        energy_shift * sigma_0
        + velocity * float(kx) * sigma_x
        + velocity * float(ky) * sigma_y
        + (mass - curvature * k2) * sigma_z
    )
