"""Modelo de 2 bandas en red cuadrada (Qi-Wu-Zhang / Chern aislante 2D).

    H(k) = sin(kx)*sigma_x + sin(ky)*sigma_y + (m - cos(kx) - cos(ky))*sigma_z

La zona de Brillouin es EXACTAMENTE [-pi, pi] x [-pi, pi], que coincide con la
caja rectangular que QXTI muestrea. Esto lo hace ideal para validar la teoria
analitica (formulas de Hipolito+2018) contra la simulacion numerica CMD: la
integral de BZ es exacta, sin artefactos de geometria.

Con m=2.5 el sistema tiene gap en TODA la BZ (no hay degeneraciones), asi que
el gradiente covariante y la conexion de Berry son suaves en todo el grid.
"""
from __future__ import annotations
import numpy as np

MODEL_NAME = "square-2band-qwz"
BASIS_SIZE = 2
DIMENSION = 2
BASIS_TYPE = "orbital"
IS_PERIODIC = True

_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)

DEFAULT_PARAMS = {
    "m": 2.5,   # parametro de masa; m=2.5 -> gap en toda la BZ (~0.5 minimo)
    "t": 1.0,   # escala de hopping
}

DEFAULT_LATTICE = {
    "lattice_type": "2D square",
    "lattice_constants": {"a": 1.0, "b": 1.0, "gamma_deg": 90.0},
    "BZorigin": [0.0, 0.0, 0.0],
    # BZaxis da el ANCHO COMPLETO de la caja (half-width = 0.5*valor).
    # H es periodico con periodo 2*pi -> caja = [-pi, pi] -> ancho = 2*pi.
    "BZaxis": [
        [2.0 * np.pi, 0.0, 0.0],
        [0.0, 2.0 * np.pi, 0.0],
        [0.0, 0.0, 0.0],
    ],
}


def default_params() -> dict[str, float]:
    return dict(DEFAULT_PARAMS)


def H(kx: float, ky: float, kz: float, params: dict[str, object] | None) -> np.ndarray:
    """Hamiltoniano QWZ de 2 bandas con firma QXTI.

    H(k) = t*[sin(kx) sx + sin(ky) sy + (m - cos kx - cos ky) sz]
    """
    p = default_params()
    if params:
        p.update(params)
    m = float(p["m"])
    t = float(p["t"])
    dx = np.sin(kx)
    dy = np.sin(ky)
    dz = m - np.cos(kx) - np.cos(ky)
    return t * (dx * _sx + dy * _sy + dz * _sz)
