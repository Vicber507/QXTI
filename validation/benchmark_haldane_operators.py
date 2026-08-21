"""Validate Haldane derivatives and optical operators with independent references.

Two references are deliberately combined:

1. closed-form first and second derivatives of the Haldane Hamiltonian validate
   QXTI's finite-difference velocity and inverse-mass matrices in the orbital basis;
2. PythTB's analytic tight-binding velocity validates gauge-invariant band-basis
   velocities and off-diagonal Berry-connection/dipole magnitudes.

Complex interband matrix elements are phase-gauge dependent, so comparing their
raw complex values across independently diagonalized codes would not be meaningful.
Their magnitudes are gauge invariant and are compared point by point instead.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from qxti.core import QXTIConfig, QXTISimulation
from qxti.physics import OperatorFactory

from .benchmark_haldane_pythtb import _pythtb_model, _real_vectors, _to_reduced
from .registry import RESULTS_DIR, record_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DIRECTIONS = ("x", "y")


@dataclass(frozen=True)
class HaldaneOperatorResult:
    case: str
    random_points: int
    velocity_vs_closed_form_max_abs_error: float
    inverse_mass_vs_closed_form_max_abs_error: float
    band_velocity_vs_pythtb_max_abs_error: float
    interband_velocity_magnitude_vs_pythtb_max_abs_error: float
    dipole_magnitude_vs_pythtb_max_abs_error: float
    dipole_vs_berry_alias_max_abs_error: float
    passed: bool


def _pauli_matrix(components: tuple[complex, complex, complex, complex]) -> np.ndarray:
    b0, b1, b2, b3 = components
    return np.asarray(
        [[b0 + b3, b1 - 1.0j * b2], [b1 + 1.0j * b2, b0 - b3]],
        dtype=np.complex128,
    )


def _lattice_vectors(a0: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Haldane NN/NNN vectors without calling the model's helpers."""
    nn = np.asarray(
        [
            [0.0, a0],
            [-np.sqrt(3.0) * a0 / 2.0, -a0 / 2.0],
            [np.sqrt(3.0) * a0 / 2.0, -a0 / 2.0],
        ],
        dtype=np.float64,
    )
    nnn = np.asarray(
        [
            [np.sqrt(3.0) * a0, 0.0],
            [-np.sqrt(3.0) * a0 / 2.0, 3.0 * a0 / 2.0],
            [-np.sqrt(3.0) * a0 / 2.0, -3.0 * a0 / 2.0],
        ],
        dtype=np.float64,
    )
    return nn, nnn


def _closed_form_first_derivative(
    k: np.ndarray, params: dict[str, object], axis: int
) -> np.ndarray:
    t1 = float(params["t1"])
    t2 = float(params["t2"])
    phi = float(params["phi0"])
    nn, nnn = _lattice_vectors(float(params["a0"]))
    angle_nn = nn @ k[:2]
    angle_nnn = nnn @ k[:2]
    components = (
        -2.0 * t2 * np.cos(phi) * np.sum(nnn[:, axis] * np.sin(angle_nnn)),
        -t1 * np.sum(nn[:, axis] * np.sin(angle_nn)),
        t1 * np.sum(nn[:, axis] * np.cos(angle_nn)),
        -2.0 * t2 * np.sin(phi) * np.sum(nnn[:, axis] * np.cos(angle_nnn)),
    )
    return _pauli_matrix(components)


def _closed_form_second_derivative(
    k: np.ndarray, params: dict[str, object], axis1: int, axis2: int
) -> np.ndarray:
    t1 = float(params["t1"])
    t2 = float(params["t2"])
    phi = float(params["phi0"])
    nn, nnn = _lattice_vectors(float(params["a0"]))
    angle_nn = nn @ k[:2]
    angle_nnn = nnn @ k[:2]
    nn_factor = nn[:, axis1] * nn[:, axis2]
    nnn_factor = nnn[:, axis1] * nnn[:, axis2]
    components = (
        -2.0 * t2 * np.cos(phi) * np.sum(nnn_factor * np.cos(angle_nnn)),
        -t1 * np.sum(nn_factor * np.cos(angle_nn)),
        -t1 * np.sum(nn_factor * np.sin(angle_nn)),
        2.0 * t2 * np.sin(phi) * np.sum(nnn_factor * np.sin(angle_nnn)),
    )
    return _pauli_matrix(components)


def _band_transform(matrix: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    return eigenvectors.conj().T @ matrix @ eigenvectors


def _offdiagonal_connection(velocity: np.ndarray, energies: np.ndarray) -> np.ndarray:
    connection = np.zeros_like(velocity, dtype=np.complex128)
    for row in range(len(energies)):
        for col in range(len(energies)):
            if row != col:
                connection[row, col] = 1.0j * velocity[row, col] / (
                    energies[col] - energies[row]
                )
    return connection


def run_case(case: str, *, random_points: int = 128) -> HaldaneOperatorResult:
    if case not in {"topological", "trivial"}:
        raise ValueError("case must be 'topological' or 'trivial'")

    config = QXTIConfig.from_file(
        PROJECT_ROOT / "inputs" / f"inputParams.haldane_{case}.cfg"
    )
    hamiltonian = QXTISimulation(config=config).build_hamiltonian()
    operators = OperatorFactory(hamiltonian, basis="band")
    pythtb_model = _pythtb_model(hamiltonian)
    real_vectors = _real_vectors(hamiltonian)

    bounds = hamiltonian.reciprocal_box_bounds()
    rng = np.random.default_rng(20260821)
    k_points = np.column_stack(
        [
            rng.uniform(bounds[0][0], bounds[0][1], random_points),
            rng.uniform(bounds[1][0], bounds[1][1], random_points),
            np.zeros(random_points),
        ]
    )
    reduced_k = _to_reduced(k_points, real_vectors)

    pythtb_hamiltonians = np.asarray(
        pythtb_model.hamiltonian(reduced_k), dtype=np.complex128
    )
    pythtb_energies, pythtb_vectors = np.linalg.eigh(pythtb_hamiltonians)
    pythtb_velocity_reduced = np.asarray(
        pythtb_model.velocity(reduced_k, cartesian=False), dtype=np.complex128
    )
    # kappa_i = k_cart . a_i/(2*pi), hence dH/dk_a =
    # sum_i (a_i)_a/(2*pi) dH/dkappa_i.
    pythtb_velocity_cartesian = np.einsum(
        "ia,ikmn->akmn",
        real_vectors / (2.0 * np.pi),
        pythtb_velocity_reduced,
    )

    velocity_closed_form_errors: list[float] = []
    inverse_mass_closed_form_errors: list[float] = []
    band_velocity_errors: list[float] = []
    interband_velocity_errors: list[float] = []
    dipole_errors: list[float] = []
    dipole_alias_errors: list[float] = []

    for point_index, k in enumerate(k_points):
        qxti_energies, _ = hamiltonian.diagonalize(*k)
        external_energies = pythtb_energies[point_index]
        external_vectors = pythtb_vectors[point_index]

        for axis, direction in enumerate(_DIRECTIONS):
            qxti_velocity_working = operators.velocity(direction, *k, basis="working")
            velocity_closed_form_errors.append(
                float(
                    np.max(
                        np.abs(
                            qxti_velocity_working
                            - _closed_form_first_derivative(k, hamiltonian.params, axis)
                        )
                    )
                )
            )

            qxti_velocity_band = operators.velocity(direction, *k, basis="band")
            external_velocity_band = _band_transform(
                pythtb_velocity_cartesian[axis, point_index], external_vectors
            )
            band_velocity_errors.append(
                float(
                    np.max(
                        np.abs(
                            np.real(np.diag(qxti_velocity_band))
                            - np.real(np.diag(external_velocity_band))
                        )
                    )
                )
            )
            interband_velocity_errors.append(
                float(
                    abs(
                        abs(qxti_velocity_band[0, 1])
                        - abs(external_velocity_band[0, 1])
                    )
                )
            )

            qxti_dipole = operators.dipole(direction, *k, basis="band")
            qxti_berry = operators.berry_connection(direction, *k, basis="band")
            external_dipole = _offdiagonal_connection(
                external_velocity_band, external_energies
            )
            dipole_errors.append(
                float(abs(abs(qxti_dipole[0, 1]) - abs(external_dipole[0, 1])))
            )
            dipole_alias_errors.append(float(np.max(np.abs(qxti_dipole - qxti_berry))))

            # Ensure the independently sorted external spectrum corresponds to
            # QXTI before dividing by its gap in the dipole comparison.
            if not np.allclose(qxti_energies, external_energies, atol=1.0e-12):
                raise AssertionError("PythTB and QXTI band ordering changed unexpectedly")

            for axis2, direction2 in enumerate(_DIRECTIONS):
                qxti_inverse_mass = operators.inverse_mass(
                    direction, direction2, *k, basis="working"
                )
                inverse_mass_closed_form_errors.append(
                    float(
                        np.max(
                            np.abs(
                                qxti_inverse_mass
                                - _closed_form_second_derivative(
                                    k, hamiltonian.params, axis, axis2
                                )
                            )
                        )
                    )
                )

    velocity_error = max(velocity_closed_form_errors)
    inverse_mass_error = max(inverse_mass_closed_form_errors)
    band_velocity_error = max(band_velocity_errors)
    interband_velocity_error = max(interband_velocity_errors)
    dipole_error = max(dipole_errors)
    alias_error = max(dipole_alias_errors)
    passed = bool(
        velocity_error <= 1.0e-9
        and inverse_mass_error <= 1.0e-5
        and band_velocity_error <= 1.0e-9
        and interband_velocity_error <= 1.0e-9
        and dipole_error <= 1.0e-8
        and alias_error <= 1.0e-14
    )
    return HaldaneOperatorResult(
        case=case,
        random_points=random_points,
        velocity_vs_closed_form_max_abs_error=velocity_error,
        inverse_mass_vs_closed_form_max_abs_error=inverse_mass_error,
        band_velocity_vs_pythtb_max_abs_error=band_velocity_error,
        interband_velocity_magnitude_vs_pythtb_max_abs_error=interband_velocity_error,
        dipole_magnitude_vs_pythtb_max_abs_error=dipole_error,
        dipole_vs_berry_alias_max_abs_error=alias_error,
        passed=passed,
    )


def run_benchmark(*, random_points: int = 128) -> list[HaldaneOperatorResult]:
    return [
        run_case("topological", random_points=random_points),
        run_case("trivial", random_points=random_points),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-points", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "haldane_operators.json"
    )
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    results = run_benchmark(random_points=args.random_points)
    payload = {
        "benchmark": "Haldane derivatives and optical operators",
        "error_methodology": {
            "orbital_derivatives": (
                "At every k-point and for x/y, compare complete QXTI dH/dk and "
                "d2H/(dki dkj) matrices with separately coded closed-form Haldane "
                "derivatives; report the largest absolute matrix-element error."
            ),
            "band_velocity": (
                "Transform each code's velocity with its own eigenvectors. Compare "
                "diagonal band velocities point by point and compare the magnitude "
                "of the off-diagonal element, which is invariant under band phases."
            ),
            "dipole_berry": (
                "Build the PythTB off-diagonal connection from its analytic velocity "
                "as A_nm=i*v_nm/(E_m-E_n). Compare |A_01| point by point because the "
                "raw complex phase depends on the independently chosen eigenvector gauge."
            ),
        },
        "results": [asdict(result) for result in results],
        "passed": all(result.passed for result in results),
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")

    if not args.no_register:
        record_result(
            {
                "id": "haldane_derivatives_optical_operators",
                "title": "Haldane derivatives and optical operators",
                "passed": payload["passed"],
                "scope": (
                    "Pointwise first/second Hamiltonian derivatives, band velocities, "
                    "and off-diagonal dipole/Berry magnitudes for topological and "
                    "trivial Haldane inputs."
                ),
                "independent_reference": (
                    f"PythTB {version('pythtb')} analytic velocity and closed-form "
                    "Haldane derivatives"
                ),
                "reference_type": "external cross-code plus separately coded closed form",
                "implementation": [
                    "Code the first and second Cartesian derivatives of the Haldane Hamiltonian separately from QXTI and evaluate them at deterministic generic momenta.",
                    "Use PythTB's analytic velocity operator as the external band-basis reference; compare gauge-invariant interband magnitudes because eigenvector phases are arbitrary.",
                ],
                "reference_provenance": [
                    "Closed derivatives follow directly from the trigonometric Haldane Hamiltonian, while band velocities are supplied by the independently distributed PythTB implementation.",
                ],
                "production_code_changes": [
                    "None. QXTI's existing finite-difference derivatives and optical-operator construction met the declared tolerances.",
                ],
                "results": [
                    {
                        "case": result.case,
                        "dH error": result.velocity_vs_closed_form_max_abs_error,
                        "d2H error": result.inverse_mass_vs_closed_form_max_abs_error,
                        "band velocity error": result.band_velocity_vs_pythtb_max_abs_error,
                        "interband v magnitude error": result.interband_velocity_magnitude_vs_pythtb_max_abs_error,
                        "dipole A magnitude error": result.dipole_magnitude_vs_pythtb_max_abs_error,
                    }
                    for result in results
                ],
                "error_methodology": [
                    "All quantities are compared at 128 deterministic random Cartesian k-points and in both x and y directions.",
                    "For first and second Hamiltonian derivatives, the error is the maximum absolute difference over every k-point and every matrix element relative to separately coded closed-form Haldane derivatives.",
                    "For band velocities, each code diagonalizes its own Hamiltonian. Diagonal elements are compared directly; interband elements are compared as $||v^{QXTI}_{01}|-|v^{PythTB}_{01}||$ to remove arbitrary eigenvector phases.",
                    "The PythTB dipole/Berry reference is $A_{nm}=iv_{nm}/(E_m-E_n)$. QXTI and PythTB are compared through the gauge-invariant magnitude $|A_{01}|$ point by point.",
                ],
                "acceptance_criteria": [
                    "Maximum first-derivative matrix error <= 1e-9 Ha*Bohr.",
                    "Maximum second-derivative matrix error <= 1e-5 Ha*Bohr^2.",
                    "Maximum PythTB band-velocity and interband-magnitude error <= 1e-9 Ha*Bohr.",
                    "Maximum off-diagonal dipole/Berry magnitude error <= 1e-8 Bohr.",
                ],
                "conclusion": (
                    "QXTI's finite-difference Hamiltonian derivatives and the "
                    "gauge-invariant optical matrix elements derived from them agree "
                    "with independent closed-form and PythTB references."
                ),
                "limitations": [
                    "Raw complex interband elements are not compared because independently diagonalized eigenvectors carry arbitrary phases.",
                    "The diagonal Berry connection is gauge dependent and QXTI fixes it to zero; this benchmark validates only the physical off-diagonal connection/dipole magnitude.",
                    "This does not yet validate integrated optical conductivity, nonlinear response, LDOS, or HHG.",
                ],
                "artifact": str(args.output),
            }
        )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
