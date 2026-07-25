"""Hamiltoniano tight-binding de primeros vecinos del grafeno para QXTI.

Red honeycomb con dos sublattices A y B, hopping t entre primeros vecinos.
Sin gap: punto de Dirac en K y K'. Con gap (M != 0): aislante topologico trivial.

Reglas de seleccion:
--------------------
1. Simetria: grupo puntual C6v de la red honeycomb.
2. Sin gap (M=0, simetria de inversion intacta):
   - Transiciones interbanda permitidas para luz lineal en x e y en toda la BZ.
   - No hay distincion de helicidad entre K y K': son degenerados por inversion.
   - Armónicos pares suprimidos exactamente por simetria C2 (k -> -k impares la corriente).
3. Con gap (M != 0, simetria de inversion rota):
   - Regla de seleccion de valle contrastada:
       Valle K  -> acopla a luz circularmente polarizada sigma+ (m = +1)
       Valle K' -> acopla a luz circularmente polarizada sigma- (m = -1)
   - Armonicos pares reaparecen porque C2 se rompe con M != 0.
4. La corriente J = dH/dk (relacion de Peierls) es la cantidad relevante
   para calcular la respuesta optica no lineal en QXTI.

Parametros fisicos (en a.u.):
------------------------------
    t   : hopping NN  ~ 2.7 eV  = 0.09923 a.u.
    M   : gap de masa ~ 0       = 0.0     a.u.  (grafeno puro sin gap)
    a0  : constante de red ~ 2.46 Ang = 4.6479 a.u.

Referencias:
    Castro Neto et al., Rev. Mod. Phys. 81, 109 (2009).
    Yao et al., Phys. Rev. B 77, 235406 (2008). [reglas de seleccion de valle]
"""
from __future__ import annotations
import numpy as np

MODEL_NAME = "graphene-nearest-neighbor"
BASIS_SIZE = 2
DIMENSION = 2
BASIS_TYPE = "sublattice"
IS_PERIODIC = True

sigma_x = np.array([[0, 1],  [1,  0]],  dtype=complex)
sigma_y = np.array([[0, -1j],[1j, 0]],  dtype=complex)
sigma_z = np.array([[1, 0],  [0,  -1]], dtype=complex)
sigma_0 = np.eye(2, dtype=complex)

# a0 = 2.46 Ang / 0.529177 Ang/bohr = 4.6479 a.u.
# t  = 2.7 eV  / 27.2114 eV/a.u.   = 0.09923 a.u.
DEFAULT_PARAMS = {
    "a0": 4.6479,     # constante de red [a.u.]
    "t" : 0.09923,    # hopping primeros vecinos A->B [a.u.]
    "M" : 0.0,        # gap de masa [a.u.]; 0 = grafeno puro
}

DEFAULT_LATTICE = {
    "lattice_type": "2D honeycomb (hexagonal)",
    "lattice_constants": {
        "a0"       : 4.6479,   # constante de red en a.u.; clave leida por QXTI
        "a"        : 4.6479,   # alias por si QXTI busca la clave "a"
        "gamma_deg": 120.0,
    },
    "real_space_vectors": {
        # Vectores primitivos de Bravais en a.u.
        # a1 = sqrt(3)*a0 * x_hat
        # a2 = sqrt(3)/2*a0 * x_hat + 3/2*a0 * y_hat
        "a1": [ 8.0508, 0.0    ],
        "a2": [ 4.0254, 6.9719 ],
    },
    "BZorigin": [0.0, 0.0, 0.0],
    "BZaxis": [
        [2.0 * np.pi / (np.sqrt(3.0) * 4.6479), 0.0, 0.0],
        [0.0, 2.0 * np.pi / (1.5 * 4.6479),     0.0],
        [0.0, 0.0,                                0.0],
    ],
}


def default_params() -> dict[str, float]:
    """Devuelve una copia de los parametros por defecto."""
    return dict(DEFAULT_PARAMS)


def _resolved_params(params: dict[str, object] | None) -> dict[str, object]:
    resolved = default_params()
    if params:
        resolved.update(params)
    return resolved


def _validate_graphene_params(params: dict[str, object]) -> None:
    if float(params["a0"]) <= 0.0:
        raise ValueError("a0 debe ser estrictamente positivo.")
    if float(params["t"]) == 0.0:
        raise ValueError("t (hopping NN) no puede ser cero.")


def _nn_vectors(a0: float) -> np.ndarray:
    """Tres vectores de primeros vecinos delta_i (A -> B), shape (3, 2), en a.u."""
    return np.array([
        [ 0.0,                a0      ],
        [-np.sqrt(3.0)/2*a0, -0.5*a0 ],
        [ np.sqrt(3.0)/2*a0, -0.5*a0 ],
    ], dtype=float)


def _fk_components(kx: float, ky: float, a0: float) -> tuple[float, float]:
    """
    Parte real e imaginaria del factor de estructura:

        f(k) = sum_{i=0}^{2} exp(i * k . delta_i)

    Retorna (Re f, Im f).
    """
    deltas = _nn_vectors(a0)
    k = np.array([kx, ky], dtype=float)
    phases = deltas @ k
    return float(np.cos(phases).sum()), float(np.sin(phases).sum())


def H(kx: float, ky: float, kz: float, params: dict[str, object] | None) -> np.ndarray:
    """Hamiltoniano tight-binding de primeros vecinos del grafeno con firma QXTI.

    Parametros
    ----------
    kx, ky : float
        Componentes del vector de onda en a.u.^{-1}.
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
    Escribiendo f(k) = fre + i*fim:

        H(k) = t*fre*sigma_x - t*fim*sigma_y + M*sigma_z

             = |   M        t*f*(k) |
               | t*f(k)       -M    |

    Autovalores: E±(k) = ± sqrt(M^2 + t^2 |f(k)|^2)

    En k = K o K': f(K) = 0  =>  E± = ±M  (gap de masa 2M).
    Con M = 0: puntos de Dirac, semimetal.

    Regla de seleccion emergente (corriente J = dH/dk):
        dH/dkx y dH/dky solo mezclan sigma_x y sigma_y.
        sigma_z (gap de masa) no contribuye a la corriente:
        la brecha abre el espectro pero no acopla directamente al campo.
    """
    del kz
    resolved = _resolved_params(params)
    _validate_graphene_params(resolved)

    a0  = float(resolved["a0"])
    t   = float(resolved["t"])
    M   = float(resolved["M"])

    fre, fim = _fk_components(float(kx), float(ky), a0)

    return (
          t * fre * sigma_x
        - t * fim * sigma_y
        +     M   * sigma_z
    )


def H_batch(kpts, params=None):
    """Version VECTORIZADA de H sobre muchos k: kpts (nk,3) -> (nk,2,2).

    Replica EXACTA de H (cada ``H[i,j] = expr`` -> ``H[:,i,j] = expr`` con
    kx,ky como arrays).  Evita el loop Python punto a punto: imprescindible
    para grids grandes.  QXTI la usa como ``h_batch``.

    Con f(k) = fre + i*fim:

        H(k) = t*fre*sigma_x - t*fim*sigma_y + M*sigma_z

             = |   M              t*(fre + i*fim) |
               | t*(fre - i*fim)       -M         |
    """
    resolved = _resolved_params(params)
    _validate_graphene_params(resolved)

    a0 = float(resolved["a0"])
    t  = float(resolved["t"])
    M  = float(resolved["M"])

    kpts = np.asarray(kpts, dtype=np.float64)
    kx, ky = kpts[:, 0], kpts[:, 1]  # kz ignorado (sistema 2D)
    nk = kx.shape[0]

    deltas = _nn_vectors(a0)                     # (3, 2)
    phases = kx[:, None] * deltas[:, 0] + ky[:, None] * deltas[:, 1]  # (nk, 3)
    fre = np.cos(phases).sum(axis=1)             # (nk,)
    fim = np.sin(phases).sum(axis=1)             # (nk,)

    H = np.zeros((nk, 2, 2), dtype=np.complex128)
    H[:, 0, 0] = M
    H[:, 1, 1] = -M
    H[:, 0, 1] = t * fre + 1j * t * fim
    H[:, 1, 0] = t * fre - 1j * t * fim
    return H