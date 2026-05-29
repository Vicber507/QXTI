"""Hamiltoniano del modelo de Haldane para QXTI.

El modelo de Haldane es un modelo de tight-binding en red hexagonal (honeycomb)
con hopping de primeros vecinos (t1) y segundos vecinos con fase compleja (t2 * e^{i*phi0}).
Describe un aislante topologico de Chern sin campo magnetico neto.

Referencias:
    Haldane, F. D. M. (1988). Model for a Quantum Hall Effect without Landau Levels.
    Physical Review Letters, 61(18), 2015-2018.
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Metadata del modelo
# ---------------------------------------------------------------------------
MODEL_NAME  = "haldane-honeycomb"
BASIS_SIZE  = 2        # dos sitios por celda unitaria: sublattice A y B
DIMENSION   = 2        # sistema puramente 2D

BASIS_TYPE  = "sublattice"   # base: {|A,k>, |B,k>}
IS_PERIODIC = True

# ---------------------------------------------------------------------------
# Parametros fisicos por defecto
# (equivalentes a los leidos por libconfig en el codigo C++)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "t1"  : 0.075,                 # hopping primeros vecinos A->B  [a.u.]
    "t2"  : 0.025,                 # hopping segundos vecinos A->A / B->B  [a.u.]
    "M0"  : 2.54 * 0.025,          # M0 = Mt2 * t2 = 0.0635  [a.u.]
    "phi0": 1.16,                  # fase del hopping NNN  [rad]
    "a0"  : 1.0 / 0.529177,        # 1.0 Angstrom -> a.u. (1 Bohr = 0.529177 Å)
}

# ---------------------------------------------------------------------------
# Informacion de red
# ---------------------------------------------------------------------------
# La red honeycomb tiene dos subredes triangulares A y B.
# Vectores NN  (de A hacia B):
#   a0 = [0,      a0]
#   a1 = [-√3/2·a, -a/2]
#   a2 = [ √3/2·a, -a/2]
#
# Vectores NNN (de A hacia A, o de B hacia B):
#   b0 = [ √3·a,   0   ]
#   b1 = [-√3/2·a,  3/2·a]
#   b2 = [-√3/2·a, -3/2·a]
#
# Zona de Brillouin: hexagonal con
#   kx_max = pi / (√3 · a0)
#   ky_max = 2*pi / (3 · a0)

_a0_au = 1.0 / 0.529177   # 1.0 Angstrom en unidades atomicas ~ 1.8897 a.u.
_kx_max = np.pi / (np.sqrt(3.0) * _a0_au)
_ky_max = 2.0 * np.pi / (3.0 * _a0_au)

DEFAULT_LATTICE = {
    "lattice_type": "2D honeycomb (hexagonal)",
    "lattice_constants": {
        "a0"       : _a0_au,   # constante de red en a.u. (~1.8897); clave leida por QXTI como fallback
        "a"        : _a0_au,   # alias por si QXTI busca la clave "a"
        "gamma_deg": 120.0,
    },
    "real_space_vectors": {
        # Vectores primitivos de la red triangular de Bravais (en a.u.)
        # a1 = sqrt(3)*a0 * x_hat
        # a2 = sqrt(3)/2*a0 * x_hat + 3/2*a0 * y_hat
        "a1": [ np.sqrt(3.0) * _a0_au,             0.0          ],
        "a2": [ np.sqrt(3.0) / 2.0 * _a0_au,  1.5 * _a0_au     ],
    },
    "BZorigin": [0.0, 0.0, 0.0],
    "BZaxis": [
        [2.0 * _kx_max, 0.0, 0.0],
        [0.0, 2.0 * _ky_max, 0.0],
        [0.0, 0.0, 0.0],
    ],
    "notes": (
        "a0 = 1.0 Ang = 1.8897 a.u. | "
        "t1 = 0.075 a.u. | t2 = 0.025 a.u. | "
        "M0 = Mt2*t2 = 2.54*0.025 = 0.0635 a.u. | "
        "phi0 = 1.16 rad. "
        "El numero de Chern es C=+/-1 segun el signo de M0 y phi0."
    ),
}

# ---------------------------------------------------------------------------
# Matrices de Pauli (base de la algebra del Hamiltoniano 2x2)
# ---------------------------------------------------------------------------
sigma_0 = np.eye(2, dtype=complex)
sigma_x = np.array([[0.0,  1.0 ], [ 1.0, 0.0 ]], dtype=complex)
sigma_y = np.array([[0.0, -1.0j], [ 1.0j, 0.0]], dtype=complex)
sigma_z = np.array([[1.0,  0.0 ], [ 0.0, -1.0]], dtype=complex)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_params() -> dict[str, float]:
    """Devuelve una copia de los parametros por defecto."""
    return dict(DEFAULT_PARAMS)


def _resolved_params(params: dict[str, object] | None) -> dict[str, object]:
    resolved = default_params()
    if params:
        resolved.update(params)
    return resolved


def _nn_vectors(a0: float):
    """Vectores de primeros vecinos (A -> B) para constante de red a0."""
    return np.array([
        [ 0.0,              a0      ],
        [-np.sqrt(3.0)/2*a0, -0.5*a0],
        [ np.sqrt(3.0)/2*a0, -0.5*a0],
    ])  # shape (3, 2)


def _nnn_vectors(a0: float):
    """Vectores de segundos vecinos (A -> A  o  B -> B) para constante de red a0."""
    return np.array([
        [ np.sqrt(3.0)*a0,       0.0      ],
        [-np.sqrt(3.0)/2*a0,  1.5*a0],
        [-np.sqrt(3.0)/2*a0, -1.5*a0],
    ])  # shape (3, 2)


def _bcomp(kx: float, ky: float, t1: float, t2: float,
           M0: float, phi0: float, a0: float):
    """
    Calcula los cuatro componentes del vector 'B' del Hamiltoniano.

    El Hamiltoniano se escribe como:
        H(k) = B0*sigma_0 + B1*sigma_x + B2*sigma_y + B3*sigma_z

    con:
        B0 = 2*t2*cos(phi0) * sum_i cos(k · b_i)
        B1 = t1              * sum_i cos(k · a_i)
        B2 = t1              * sum_i sin(k · a_i)
        B3 = M0 - 2*t2*sin(phi0) * sum_i sin(k · b_i)

    Retorna (B0, B1, B2, B3).
    """
    vec_a = _nn_vectors(a0)
    vec_b = _nnn_vectors(a0)

    k = np.array([kx, ky])
    angles_a = vec_a @ k   # shape (3,)
    angles_b = vec_b @ k   # shape (3,)

    B0 = 2.0 * t2 * np.cos(phi0)  * np.cos(angles_b).sum()
    B1 = t1                        * np.cos(angles_a).sum()
    B2 = t1                        * np.sin(angles_a).sum()
    B3 = M0 - 2.0 * t2 * np.sin(phi0) * np.sin(angles_b).sum()

    return B0, B1, B2, B3


# ---------------------------------------------------------------------------
# Funcion obligatoria: H(kx, ky, kz, params)
# ---------------------------------------------------------------------------

def H(kx: float, ky: float, kz: float,
      params: dict[str, object] | None = None) -> np.ndarray:
    """Hamiltoniano del modelo de Haldane en el espacio k.

    Parametros
    ----------
    kx, ky : float
        Componentes del vector de onda en la primera zona de Brillouin.
    kz : float
        Ignorado (sistema 2D).
    params : dict, opcional
        Sobreescribe cualquier subconjunto de DEFAULT_PARAMS.

    Retorna
    -------
    np.ndarray, shape (2, 2), dtype=complex
        Matriz Hermitiana H(k).

    Notas
    -----
    La matriz se construye como:

        H = B0*I + B1*σx + B2*σy + B3*σz

          = | B0 + B3       B1 - i*B2 |
            | B1 + i*B2     B0 - B3   |

    que coincide exactamente con `_hstore` en `GenHamiltonian` del codigo C++.
    """
    del kz  # no utilizado

    p = _resolved_params(params)
    t1   = float(p["t1"])
    t2   = float(p["t2"])
    M0   = float(p["M0"])
    phi0 = float(p["phi0"])
    a0   = float(p["a0"])

    B0, B1, B2, B3 = _bcomp(float(kx), float(ky), t1, t2, M0, phi0, a0)

    return (
        B0 * sigma_0
        + B1 * sigma_x
        + B2 * sigma_y
        + B3 * sigma_z
    )
