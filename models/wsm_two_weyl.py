"""Semimetal de Weyl de dos bandas (modelo de cuatro nodos) para QXTI.

Modelo tight-binding 3D de dos bandas que reproduce cuatro nodos de Weyl en el
plano ky = 0, inspirado en el modelo de dos bandas de TaAs.

El Hamiltoniano se escribe como

    H(k) = B0(k) sigma_0 + B1(k) sigma_x + B2(k) sigma_y + B3(k) sigma_z

         = | B0 + B3        B1 - i B2 |
           | B1 + i B2      B0 - B3   |

con autovalores E_pm(k) = B0(k) +/- sqrt(B1^2 + B2^2 + B3^2).

Coeficientes (kw es la posicion del nodo de Weyl):
    B0 = gamma [cos(2 a0 kx) - cos(a0 kw)] [cos(a2 kz) - cos(a0 kw)]
    B1 = -{ M0 [1 - cos(a2 kz) - cos(a1 ky)] + 2 tx [cos(a0 kx) - cos(a0 kw)] }
    B2 = -2 ty sin(a1 ky)
    B3 = -2 tz cos(a2 kz)

Nodos de Weyl (donde B1 = B2 = B3 = 0 con gamma = 0) en

    (+/- kw, 0, +/- kw),   kw = pi / (2 a0)  por defecto.

Reglas de seleccion / fisica:
--------------------------------
1. Con gamma = 0 el espectro es simetrico en energia (E_- = -E_+); el termino
   gamma (tilt) rompe esa simetria e inclina los conos de Weyl.
2. Los cuatro nodos de Weyl son puntos de toque de banda (gap = 0). Igual que en
   el grafeno, un punto del k-grid que caiga EXACTAMENTE sobre un nodo de Weyl
   hace singular la conexion de Berry y genera artefactos. USA shifted = true en
   [kgrid] para evitarlo (ver [[dirac-degeneracy-grid]]).
3. La corriente J = -dH/dk es la cantidad relevante para la respuesta optica no
   lineal; QXTI la calcula por diferencias finitas. Aqui se incluye tambien la
   forma analitica en current_matrices() como referencia/validacion.

Parametros por defecto (en atomic units), modelo tipo TaAs TETRAGONAL, ajustados
para reproducir la figura experimental del 4to armonico (tools/replica_paper_4th.py):
    tx = 0.0041,  ty = tz = 0.0082,  M0 = 0.014,  gamma = 0.0
    a0 = a1 = 3.4 Ang = 6.4250683612 a.u.,  a2 = 12.0 a.u. (distorsion tetragonal)
Valores TaAs cubicos originales: M0 = 0.0164, a2 = a0.

Referencias:
    McCormick, Kimchi, Trivedi, Phys. Rev. B 95, 075133 (2017).
"""
from __future__ import annotations

import numpy as np


MODEL_NAME = "wsm-two-weyl-nodes"
BASIS_SIZE = 2
DIMENSION = 3
BASIS_TYPE = "orbital"
IS_PERIODIC = True


AU_PER_ANGSTROM = 1.8897259886

sigma_0 = np.eye(2, dtype=complex)
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


# Constante de red: 3.4 Ang -> a.u.
_A_LATTICE_AU = 3.4 * AU_PER_ANGSTROM  # 6.4250683612 a.u.


# Parametros del modelo en atomic units.
# kw es la posicion del nodo de Weyl; None -> pi / (2 a0).
DEFAULT_PARAMS = {
    "gamma": 0.0,        # tilt / inclinacion de los conos de Weyl
    "tx": 0.0041,        # hopping a lo largo de kx
    "ty": 0.0082,        # hopping a lo largo de ky
    "tz": 0.0082,        # hopping a lo largo de kz
    "M0": 0.014,         # termino de masa   (TaAs original: 0.0164)
    "a0": _A_LATTICE_AU,  # constante de red eje x [a.u.]
    "a1": _A_LATTICE_AU,  # constante de red eje y [a.u.]
    "a2": 12.0,          # eje z TETRAGONAL [a.u.]  (cubico original: _A_LATTICE_AU)
    "kw": None,          # posicion del nodo de Weyl; None -> pi/(2 a0)
}
# NOTA: M0=0.014, a2=12.0 son los parametros del barrido fino que reproducen la
# figura del paper (replica 4to armonico TaAs: S conectada con verticales rectas
# en eps~+-0.5, doble pico en +-90 grados).  Ver tools/replica_paper_4th.py.
# Los valores TaAs originales (McCormick et al.): M0=0.0164, a2=_A_LATTICE_AU.


DEFAULT_LATTICE = {
    "lattice_type": "3D simple orthorhombic (two-Weyl-node model)",
    "lattice_constants": {
        "a0": _A_LATTICE_AU,
        "a": _A_LATTICE_AU,
        "a1_length": _A_LATTICE_AU,
        "a2_length": _A_LATTICE_AU,
        "gamma_deg": 90.0,
    },
    "real_space_vectors": {
        "a1": [_A_LATTICE_AU, 0.0, 0.0],
        "a2": [0.0, _A_LATTICE_AU, 0.0],
        "a3": [0.0, 0.0, _A_LATTICE_AU],
    },
    "BZorigin": [0.0, 0.0, 0.0],
    "BZaxis": [
        [2.0 * np.pi / _A_LATTICE_AU, 0.0, 0.0],
        [0.0, 2.0 * np.pi / _A_LATTICE_AU, 0.0],
        [0.0, 0.0, 2.0 * np.pi / _A_LATTICE_AU],
    ],
}


def default_params() -> dict[str, object]:
    """Devuelve una copia de los parametros por defecto."""
    return dict(DEFAULT_PARAMS)


def _resolved_params(params: dict[str, object] | None) -> dict[str, object]:
    resolved = default_params()
    if params:
        resolved.update(params)
    return resolved


def _validate_params(params: dict[str, object]) -> None:
    for key in ("a0", "a1", "a2"):
        if float(params[key]) <= 0.0:
            raise ValueError(f"{key} debe ser estrictamente positivo.")


def _kw_internal(params: dict[str, object]) -> float:
    """Posicion del nodo de Weyl: usa params['kw'] o pi/(2 a0) si es None/ausente."""
    kw = params.get("kw", None)
    if kw is None:
        return float(np.pi / (2.0 * float(params["a0"])))
    return float(kw)


def _b_components(kx: float, ky: float, kz: float, params: dict[str, object]) -> np.ndarray:
    """Devuelve los cuatro coeficientes escalares [B0, B1, B2, B3]."""
    gamma = float(params["gamma"])
    tx = float(params["tx"])
    ty = float(params["ty"])
    tz = float(params["tz"])
    M0 = float(params["M0"])
    a0 = float(params["a0"])
    a1 = float(params["a1"])
    a2 = float(params["a2"])
    kw = _kw_internal(params)

    cos_k0 = np.cos(a0 * kw)
    cos_shift_x = np.cos(a0 * kx) - cos_k0
    cos_shift_2x = np.cos(2.0 * a0 * kx) - cos_k0
    cos_shift_z = np.cos(a2 * kz) - cos_k0

    b0 = gamma * cos_shift_2x * cos_shift_z
    b1 = -(M0 * (1.0 - np.cos(a2 * kz) - np.cos(a1 * ky)) + 2.0 * tx * cos_shift_x)
    b2 = -2.0 * ty * np.sin(a1 * ky)
    b3 = -2.0 * tz * np.cos(a2 * kz)
    return np.array([b0, b1, b2, b3], dtype=float)


def H(kx: float, ky: float, kz: float, params: dict[str, object] | None = None) -> np.ndarray:
    """Hamiltoniano 2x2 del semimetal de Weyl con la firma que QXTI espera.

    Parametros
    ----------
    kx, ky, kz : float
        Componentes del vector de onda en a.u.^{-1}.
    params : dict, opcional
        Sobreescribe cualquier subconjunto de DEFAULT_PARAMS.

    Retorna
    -------
    np.ndarray, shape (2, 2), dtype=complex
        Matriz Hermitiana H(k).
    """
    resolved = _resolved_params(params)
    _validate_params(resolved)
    b0, b1, b2, b3 = _b_components(float(kx), float(ky), float(kz), resolved)
    return np.array(
        [
            [b0 + b3, b1 - 1j * b2],
            [b1 + 1j * b2, b0 - b3],
        ],
        dtype=complex,
    )


def H_batch(kpts, params=None):
    """Version VECTORIZADA de H sobre muchos k: kpts (nk,3) -> (nk,2,2).

    Replica EXACTA de H (cada ``H[i,j] = expr`` -> ``H[:,i,j] = expr`` con
    kx,ky,kz como arrays).  Evita el loop Python punto a punto: imprescindible
    para grids grandes.  QXTI la usa como ``h_batch``.  Un test compara
    H_batch vs H a precision de maquina.
    """
    resolved = _resolved_params(params)
    _validate_params(resolved)
    gamma = float(resolved["gamma"])
    tx = float(resolved["tx"])
    ty = float(resolved["ty"])
    tz = float(resolved["tz"])
    M0 = float(resolved["M0"])
    a0 = float(resolved["a0"])
    a1 = float(resolved["a1"])
    a2 = float(resolved["a2"])
    kw = _kw_internal(resolved)

    kpts = np.asarray(kpts, dtype=np.float64)
    kx, ky, kz = kpts[:, 0], kpts[:, 1], kpts[:, 2]
    nk = kx.shape[0]

    cos_k0 = np.cos(a0 * kw)
    cos_shift_x = np.cos(a0 * kx) - cos_k0
    cos_shift_2x = np.cos(2.0 * a0 * kx) - cos_k0
    cos_shift_z = np.cos(a2 * kz) - cos_k0

    b0 = gamma * cos_shift_2x * cos_shift_z
    b1 = -(M0 * (1.0 - np.cos(a2 * kz) - np.cos(a1 * ky)) + 2.0 * tx * cos_shift_x)
    b2 = -2.0 * ty * np.sin(a1 * ky)
    b3 = -2.0 * tz * np.cos(a2 * kz)

    H = np.zeros((nk, 2, 2), dtype=np.complex128)
    H[:, 0, 0] = b0 + b3
    H[:, 0, 1] = b1 - 1j * b2
    H[:, 1, 0] = b1 + 1j * b2
    H[:, 1, 1] = b0 - b3
    return H


def band_energies(kx: float, ky: float, kz: float, params: dict[str, object] | None = None) -> np.ndarray:
    """Devuelve las dos energias de banda ordenadas como [E_menos, E_mas]."""
    resolved = _resolved_params(params)
    b0, b1, b2, b3 = _b_components(float(kx), float(ky), float(kz), resolved)
    gap = np.sqrt(b1 * b1 + b2 * b2 + b3 * b3)
    return np.array([b0 - gap, b0 + gap], dtype=float)


def current_matrices(kx: float, ky: float, kz: float, params: dict[str, object] | None = None) -> dict[str, np.ndarray]:
    """Matrices de corriente analiticas j_i = -dH/dk_i (referencia/validacion).

    QXTI calcula la velocidad por diferencias finitas; estas formas analiticas
    sirven para verificar la implementacion numerica.
    """
    resolved = _resolved_params(params)
    gamma = float(resolved["gamma"])
    tx = float(resolved["tx"])
    ty = float(resolved["ty"])
    tz = float(resolved["tz"])
    M0 = float(resolved["M0"])
    a0 = float(resolved["a0"])
    a1 = float(resolved["a1"])
    a2 = float(resolved["a2"])
    kw = _kw_internal(resolved)

    cos_k0 = np.cos(a0 * kw)
    cos_shift_2x = np.cos(2.0 * a0 * float(kx)) - cos_k0
    cos_shift_z = np.cos(a2 * float(kz)) - cos_k0

    xder = np.zeros(4, dtype=float)
    yder = np.zeros(4, dtype=float)
    zder = np.zeros(4, dtype=float)

    xder[0] = -2.0 * a0 * gamma * np.sin(2.0 * a0 * float(kx)) * cos_shift_z
    xder[1] = 2.0 * a0 * tx * np.sin(a0 * float(kx))

    yder[1] = -a1 * M0 * np.sin(a1 * float(ky))
    yder[2] = -2.0 * a1 * ty * np.cos(a1 * float(ky))

    zder[0] = -a2 * gamma * cos_shift_2x * np.sin(a2 * float(kz))
    zder[1] = -a2 * M0 * np.sin(a2 * float(kz))
    zder[3] = 2.0 * a2 * tz * np.sin(a2 * float(kz))

    def _assemble(d: np.ndarray) -> np.ndarray:
        return -np.array(
            [
                [d[0] + d[3], d[1] - 1j * d[2]],
                [d[1] + 1j * d[2], d[0] - d[3]],
            ],
            dtype=complex,
        )

    return {"jx": _assemble(xder), "jy": _assemble(yder), "jz": _assemble(zder)}


def weyl_node_momenta(params: dict[str, object] | None = None) -> np.ndarray:
    """Devuelve las cuatro posiciones de los nodos de Weyl (+/-kw, 0, +/-kw)."""
    resolved = _resolved_params(params)
    k0 = _kw_internal(resolved)
    return np.array(
        [
            [-k0, 0.0, -k0],
            [-k0, 0.0, +k0],
            [+k0, 0.0, -k0],
            [+k0, 0.0, +k0],
        ],
        dtype=float,
    )


def brillouin_zone_bounds(params: dict[str, object] | None = None) -> dict[str, tuple[float, float]]:
    """Limites de la zona de Brillouin en unidades reciprocas atomicas."""
    resolved = _resolved_params(params)
    a0 = float(resolved["a0"])
    a1 = float(resolved["a1"])
    a2 = float(resolved["a2"])
    return {
        "kx": (-np.pi / a0, np.pi / a0),
        "ky": (-np.pi / a1, np.pi / a1),
        "kz": (-np.pi / a2, np.pi / a2),
    }


# Alias para compatibilidad: QXTI busca la funcion 'H' por defecto.
hamiltonian = H
