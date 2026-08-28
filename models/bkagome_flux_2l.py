"""Modelo kagome breathing de dos capas, portado de ``BKagomeFlux2L.h``.

La base es ``(A1, B1, C1, A2, B2, C2)``. La segunda capa es una copia de la
primera rotada un angulo ``rot`` alrededor de z. El header original no contiene
acoplamiento entre capas, por lo que el Hamiltoniano es exactamente bloque
diagonal. El parametro ``FB`` tampoco cambia ``H``: en Antelope solo selecciona
el llenado (2 bandas para ``FB=1`` y 4 para ``FB=-1``).

Unidades: ``a0`` en Bohr, ``k`` en Bohr^-1, ``ta``/``tb`` en Hartree y
``phi_a``, ``phi_b`` y ``rot`` en radianes.
"""

from __future__ import annotations

import numpy as np


MODEL_NAME = "bkagome-flux-2l"
BASIS_SIZE = 6
DIMENSION = 2
BASIS_TYPE = "layer-sublattice"
IS_PERIODIC = True


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

DEFAULT_PARAMS = {
    "a0": 7.0 * ANGSTROM_TO_BOHR,
    "ta": 0.0123,
    "tb": 0.0065,
    "phi_a": 0.0,
    "phi_b": 0.0,
    "rot": 0.0,
    "FB": 1,
}


def default_params() -> dict[str, float | int]:
    """Devuelve una copia de los parametros por defecto."""

    return dict(DEFAULT_PARAMS)


def _resolved_params(params: dict[str, object] | None) -> dict[str, object]:
    resolved: dict[str, object] = default_params()
    if params:
        resolved.update(params)
    return resolved


def _fb_value(value: object) -> int:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric not in {-1.0, 1.0}:
        raise ValueError(
            "FB debe ser 1 (dos bandas ocupadas) o -1 (cuatro bandas ocupadas)."
        )
    return int(numeric)


def _validate_params(params: dict[str, object]) -> None:
    a0 = float(params["a0"])
    values = np.array(
        [
            a0,
            float(params["ta"]),
            float(params["tb"]),
            float(params["phi_a"]),
            float(params["phi_b"]),
            float(params["rot"]),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Los parametros de BKagomeFlux2L deben ser finitos.")
    if a0 <= 0.0:
        raise ValueError("a0 debe ser estrictamente positivo.")
    _fb_value(params["FB"])


def occupied_bands_from_fb(params: dict[str, object] | None = None) -> int:
    """Traduce la bandera de Antelope a su numero de bandas ocupadas.

    QXTI controla las ocupaciones desde ``[cmd]``/``[xtp]`` y no desde el
    Hamiltoniano. Este helper conserva de forma explicita la semantica del
    parametro original para clientes que construyan su propia distribucion.
    """

    resolved = _resolved_params(params)
    fb = _fb_value(resolved["FB"])
    return 2 if fb == 1 else 4


def _bond_vectors(a0: float) -> np.ndarray:
    return np.array(
        [
            [0.25 * a0, np.sqrt(3.0) * 0.25 * a0],
            [-0.25 * a0, np.sqrt(3.0) * 0.25 * a0],
            [-0.5 * a0, 0.0],
        ],
        dtype=float,
    )


def _rotate(vectors: np.ndarray, angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=float)
    return vectors @ rotation.T


def default_lattice(params: dict[str, object] | None = None) -> dict[str, object]:
    """Metadatos de ambas capas y la caja reciproca del header original."""

    resolved = _resolved_params(params)
    _validate_params(resolved)
    a0 = float(resolved["a0"])
    rot = float(resolved["rot"])
    sqrt3 = np.sqrt(3.0)
    primitive_a = np.array(
        [[0.5 * a0, 0.5 * sqrt3 * a0], [-0.5 * a0, 0.5 * sqrt3 * a0]],
        dtype=float,
    )
    primitive_b = _rotate(primitive_a, rot)
    bonds_a = _bond_vectors(a0)
    bonds_b = _rotate(bonds_a, rot)
    return {
        "lattice_type": "2D twisted breathing-kagome bilayer without interlayer hopping",
        "lattice_constants": {
            "a0": a0,
            "a": a0,
            "gamma_deg": 60.0,
            "rotation_rad": rot,
        },
        # QXTI usa a1/a2 como red de referencia; corresponden a la capa 1.
        "real_space_vectors": {
            "a1": primitive_a[0].tolist(),
            "a2": primitive_a[1].tolist(),
        },
        "layer_real_space_vectors": {
            "layer1": primitive_a.tolist(),
            "layer2": primitive_b.tolist(),
        },
        "layer_bond_vectors": {
            "layer1": bonds_a.tolist(),
            "layer2": bonds_b.tolist(),
        },
        "BZorigin": [0.0, 0.0, 0.0],
        "BZaxis": [
            [2.0 * np.pi / a0, 0.0, 0.0],
            [0.0, 4.0 * np.pi / (sqrt3 * a0), 0.0],
            [0.0, 0.0, 0.0],
        ],
        "basis_order": ["A1", "B1", "C1", "A2", "B2", "C2"],
        "interlayer_coupling": 0.0,
        "occupied_bands_from_FB": occupied_bands_from_fb(resolved),
        "notes": (
            "La caja de integracion permanece fija a la capa 1, como en "
            "BKagomeFlux2L.h; un rot generico puede ser incommensurable."
        ),
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
    """Hamiltoniano 6x6 de las dos capas kagome desacopladas."""

    point = np.array([[float(kx), float(ky), float(kz)]], dtype=np.float64)
    return H_batch(point, params)[0]


def _layer_batch(
    points: np.ndarray,
    vectors: np.ndarray,
    ta: float,
    tb: float,
    phi_a: float,
    phi_b: float,
) -> np.ndarray:
    # H escalar reutiliza esta misma ruta, de modo que ambos son bit-exactos.
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


def H_batch(
    kpts: np.ndarray,
    params: dict[str, object] | None = None,
) -> np.ndarray:
    """Version vectorizada: ``kpts (nk, >=2) -> H (nk, 6, 6)``."""

    resolved = _resolved_params(params)
    _validate_params(resolved)
    points = np.asarray(kpts, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("kpts debe tener forma (nk, >=2).")

    vectors_a = _bond_vectors(float(resolved["a0"]))
    vectors_b = _rotate(vectors_a, float(resolved["rot"]))
    args = (
        float(resolved["ta"]),
        float(resolved["tb"]),
        float(resolved["phi_a"]),
        float(resolved["phi_b"]),
    )
    matrices = np.zeros((points.shape[0], 6, 6), dtype=np.complex128)
    matrices[:, :3, :3] = _layer_batch(points, vectors_a, *args)
    matrices[:, 3:, 3:] = _layer_batch(points, vectors_b, *args)
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


def _layer_derivative(
    kx: float,
    ky: float,
    vectors: np.ndarray,
    component: int,
    ta: float,
    tb: float,
    phi_a: float,
    phi_b: float,
) -> np.ndarray:
    dots = vectors[:, 0] * float(kx) + vectors[:, 1] * float(ky)
    dh12 = _hopping_derivative(
        dots[2], vectors[2, component], -1.0, ta, tb, phi_a, phi_b
    )
    dh13 = _hopping_derivative(
        dots[1], vectors[1, component], 1.0, ta, tb, phi_a, phi_b
    )
    dh23 = _hopping_derivative(
        dots[0], vectors[0, component], -1.0, ta, tb, phi_a, phi_b
    )
    derivative = np.zeros((3, 3), dtype=np.complex128)
    derivative[0, 1] = dh12
    derivative[0, 2] = dh13
    derivative[1, 2] = dh23
    derivative[1, 0] = np.conjugate(dh12)
    derivative[2, 0] = np.conjugate(dh13)
    derivative[2, 1] = np.conjugate(dh23)
    return derivative


def dH_dk(
    kx: float,
    ky: float,
    kz: float,
    direction: str,
    params: dict[str, object] | None = None,
) -> np.ndarray:
    """Derivada analitica de ambos bloques con respecto a kx, ky o kz."""

    del kz
    normalized_direction = str(direction).strip().lower()
    resolved = _resolved_params(params)
    _validate_params(resolved)
    if normalized_direction == "z":
        return np.zeros((6, 6), dtype=np.complex128)
    if normalized_direction not in {"x", "y"}:
        raise ValueError("direction debe ser 'x', 'y' o 'z'.")

    vectors_a = _bond_vectors(float(resolved["a0"]))
    vectors_b = _rotate(vectors_a, float(resolved["rot"]))
    component = 0 if normalized_direction == "x" else 1
    args = (
        float(resolved["ta"]),
        float(resolved["tb"]),
        float(resolved["phi_a"]),
        float(resolved["phi_b"]),
    )
    derivative = np.zeros((6, 6), dtype=np.complex128)
    derivative[:3, :3] = _layer_derivative(
        float(kx), float(ky), vectors_a, component, *args
    )
    derivative[3:, 3:] = _layer_derivative(
        float(kx), float(ky), vectors_b, component, *args
    )
    return derivative
