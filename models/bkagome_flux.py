"""Modelo kagome breathing con flujo, portado de ``BKagomeFlux.h``.

La base orbital es ``(A, B, C)``.  Los hoppings ``ta`` y ``tb`` recorren los
dos triangulos inequivalentes de la red kagome y adquieren las fases de Peierls
``phi_a / 3`` y ``phi_b / 3``, respectivamente.

Todas las cantidades que entran y salen del modulo estan en unidades atomicas:
``a0`` en Bohr, ``k`` en Bohr^-1, y ``ta``/``tb`` en Hartree.  Las fases se
expresan en radianes.
"""

from __future__ import annotations

import numpy as np


MODEL_NAME = "bkagome-flux"
BASIS_SIZE = 3
DIMENSION = 2
BASIS_TYPE = "sublattice"
IS_PERIODIC = True


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# Los valores reproducen el caso base usado por los inputs de Antelope.  En el
# C++ a0=7.0 se interpreta en Angstrom y se convierte a unidades atomicas en el
# constructor; aqui almacenamos directamente el valor convertido.
DEFAULT_PARAMS = {
    "a0": 7.0 * ANGSTROM_TO_BOHR,
    "ta": 0.0123,
    "tb": 0.0065,
    "phi_a": 0.0,
    "phi_b": 0.0,
}


def default_params() -> dict[str, float]:
    """Devuelve una copia de los parametros por defecto."""

    return dict(DEFAULT_PARAMS)


def _resolved_params(params: dict[str, object] | None) -> dict[str, object]:
    resolved: dict[str, object] = default_params()
    if params:
        resolved.update(params)
    return resolved


def _validate_params(params: dict[str, object]) -> None:
    a0 = float(params["a0"])
    values = np.array(
        [
            a0,
            float(params["ta"]),
            float(params["tb"]),
            float(params["phi_a"]),
            float(params["phi_b"]),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Los parametros de BKagomeFlux deben ser finitos.")
    if a0 <= 0.0:
        raise ValueError("a0 debe ser estrictamente positivo.")


def _bond_vectors(a0: float) -> np.ndarray:
    """Vectores ``(a1, a2, a3)`` de ``BKagomeFlux::SetBasis``."""

    return np.array(
        [
            [0.25 * a0, np.sqrt(3.0) * 0.25 * a0],
            [-0.25 * a0, np.sqrt(3.0) * 0.25 * a0],
            [-0.5 * a0, 0.0],
        ],
        dtype=float,
    )


def default_lattice(params: dict[str, object] | None = None) -> dict[str, object]:
    """Informacion de red y caja de integracion usada por el modelo C++."""

    resolved = _resolved_params(params)
    _validate_params(resolved)
    a0 = float(resolved["a0"])
    sqrt3 = np.sqrt(3.0)
    return {
        "lattice_type": "2D breathing kagome with flux",
        "lattice_constants": {
            "a0": a0,
            "a": a0,
            "gamma_deg": 60.0,
        },
        "real_space_vectors": {
            # Vectores primitivos de Bravais; los enlaces del Hamiltoniano son
            # la mitad de estos vectores.
            "a1": [0.5 * a0, 0.5 * sqrt3 * a0],
            "a2": [-0.5 * a0, 0.5 * sqrt3 * a0],
        },
        "bond_vectors": {
            "a1": [0.25 * a0, 0.25 * sqrt3 * a0],
            "a2": [-0.25 * a0, 0.25 * sqrt3 * a0],
            "a3": [-0.5 * a0, 0.0],
        },
        "BZorigin": [0.0, 0.0, 0.0],
        "BZaxis": [
            [2.0 * np.pi / a0, 0.0, 0.0],
            [0.0, 4.0 * np.pi / (sqrt3 * a0), 0.0],
            [0.0, 0.0, 0.0],
        ],
        "basis_order": ["A", "B", "C"],
        "notes": "Caja rectangular de igual area que la BZ usada por BKagomeFlux.h.",
    }


DEFAULT_LATTICE = default_lattice()


def _hopping(
    dot_k: float | np.ndarray,
    phase_sign: float,
    ta: float,
    tb: float,
    phi_a: float,
    phi_b: float,
) -> np.complex128 | np.ndarray:
    return (
        ta * np.exp(1.0j * phase_sign * phi_a / 3.0) * np.exp(-1.0j * dot_k)
        + tb * np.exp(1.0j * phase_sign * phi_b / 3.0) * np.exp(1.0j * dot_k)
    )


def H(
    kx: float,
    ky: float,
    kz: float,
    params: dict[str, object] | None = None,
) -> np.ndarray:
    """Hamiltoniano hermitico 3x3 con la firma requerida por QXTI."""

    point = np.array([[float(kx), float(ky), float(kz)]], dtype=np.float64)
    return H_batch(point, params)[0]


def H_batch(
    kpts: np.ndarray,
    params: dict[str, object] | None = None,
) -> np.ndarray:
    """Version vectorizada: ``kpts (nk, >=2) -> H (nk, 3, 3)``."""

    resolved = _resolved_params(params)
    _validate_params(resolved)
    points = np.asarray(kpts, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("kpts debe tener forma (nk, >=2).")

    a0 = float(resolved["a0"])
    ta = float(resolved["ta"])
    tb = float(resolved["tb"])
    phi_a = float(resolved["phi_a"])
    phi_b = float(resolved["phi_b"])
    vectors = _bond_vectors(a0)
    # H escalar reutiliza esta ruta: ambos resultados son bit-exactos. La forma
    # explicita tambien evita redondeos dependientes de una llamada a BLAS.
    dots = (
        points[:, 0, None] * vectors[None, :, 0]
        + points[:, 1, None] * vectors[None, :, 1]
    )

    h12 = _hopping(dots[:, 2], -1.0, ta, tb, phi_a, phi_b)
    h13 = _hopping(dots[:, 1], 1.0, ta, tb, phi_a, phi_b)
    h23 = _hopping(dots[:, 0], -1.0, ta, tb, phi_a, phi_b)

    matrices = np.zeros((points.shape[0], 3, 3), dtype=np.complex128)
    matrices[:, 0, 1] = h12
    matrices[:, 0, 2] = h13
    matrices[:, 1, 2] = h23
    matrices[:, 1, 0] = np.conjugate(h12)
    matrices[:, 2, 0] = np.conjugate(h13)
    matrices[:, 2, 1] = np.conjugate(h23)
    return matrices


def _hopping_derivative(
    dot_k: float,
    vector_component: float,
    phase_sign: float,
    ta: float,
    tb: float,
    phi_a: float,
    phi_b: float,
) -> np.complex128:
    return np.complex128(
        1.0j
        * vector_component
        * (
            -ta * np.exp(1.0j * phase_sign * phi_a / 3.0) * np.exp(-1.0j * dot_k)
            + tb * np.exp(1.0j * phase_sign * phi_b / 3.0) * np.exp(1.0j * dot_k)
        )
    )


def dH_dk(
    kx: float,
    ky: float,
    kz: float,
    direction: str,
    params: dict[str, object] | None = None,
) -> np.ndarray:
    """Derivada analitica ``dH/dk_direction`` usada como operador velocidad."""

    del kz
    normalized_direction = str(direction).strip().lower()
    resolved = _resolved_params(params)
    _validate_params(resolved)
    if normalized_direction == "z":
        return np.zeros((3, 3), dtype=np.complex128)
    if normalized_direction not in {"x", "y"}:
        raise ValueError("direction debe ser 'x', 'y' o 'z'.")

    vectors = _bond_vectors(float(resolved["a0"]))
    dots = vectors[:, 0] * float(kx) + vectors[:, 1] * float(ky)
    component = 0 if normalized_direction == "x" else 1
    args = (
        float(resolved["ta"]),
        float(resolved["tb"]),
        float(resolved["phi_a"]),
        float(resolved["phi_b"]),
    )

    dh12 = _hopping_derivative(dots[2], vectors[2, component], -1.0, *args)
    dh13 = _hopping_derivative(dots[1], vectors[1, component], 1.0, *args)
    dh23 = _hopping_derivative(dots[0], vectors[0, component], -1.0, *args)

    derivative = np.zeros((3, 3), dtype=np.complex128)
    derivative[0, 1] = dh12
    derivative[0, 2] = dh13
    derivative[1, 2] = dh23
    derivative[1, 0] = np.conjugate(dh12)
    derivative[2, 0] = np.conjugate(dh13)
    derivative[2, 1] = np.conjugate(dh23)
    return derivative
