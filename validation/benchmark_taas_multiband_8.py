"""Verify the eight-band TaAs wrapper, exact hopping sum and symmetries."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from qxti.core import QXTIConfig, QXTISimulation
from qxti.physics import OperatorFactory

from .registry import PROJECT_ROOT, RESULTS_DIR, record_result


@dataclass(frozen=True)
class TaAsMultibandResult:
    random_points: int
    matrix_max_abs_error_ha: float
    band_max_abs_error_ha: float
    velocity_max_abs_error_ha_bohr: float
    reciprocal_periodicity_max_abs_error_ha: float
    time_reversal_max_abs_error_ha: float
    space_group_max_abs_error_ha: float
    gamma_kramers_max_splitting_ha: float
    number_of_space_group_operations: int
    number_of_hopping_matrices: int
    passed: bool


def _real_space_reference(
    k_bohr: np.ndarray, terms, au_per_angstrom: float, ev_to_hartree: float
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble H and dH/dk directly from the real-space hopping list."""
    k_angstrom = np.asarray(k_bohr, dtype=np.float64) * au_per_angstrom
    matrix = np.zeros((8, 8), dtype=np.complex128)
    derivatives = np.zeros((3, 8, 8), dtype=np.complex128)
    for source, target, displacement_angstrom, hopping_ev in terms:
        displacement = np.asarray(displacement_angstrom, dtype=np.float64)
        phase = np.exp(1.0j * np.dot(k_angstrom, displacement))
        row = slice(2 * target, 2 * target + 2)
        col = slice(2 * source, 2 * source + 2)
        matrix[row, col] += phase * hopping_ev
        for axis in range(3):
            derivatives[axis, row, col] += (
                1.0j
                * displacement[axis]
                * phase
                * hopping_ev
                * au_per_angstrom
            )
    return matrix * ev_to_hartree, derivatives * ev_to_hartree


def run_benchmark(*, random_points: int = 64) -> TaAsMultibandResult:
    config = QXTIConfig.from_file(PROJECT_ROOT / "inputs" / "inputParams.taas_tb.cfg")
    hamiltonian = QXTISimulation(config=config).build_hamiltonian()
    operators = OperatorFactory(hamiltonian, basis="working")
    module = hamiltonian._module
    native_model = module._model()
    au_per_angstrom = float(module._AU_PER_ANGSTROM)
    ev_to_hartree = float(module._EV_TO_HARTREE)

    bounds = hamiltonian.reciprocal_box_bounds()
    rng = np.random.default_rng(20260823)
    k_points = np.column_stack(
        [rng.uniform(lower, upper, random_points) for lower, upper in bounds]
    )

    matrix_error = 0.0
    band_error = 0.0
    velocity_error = 0.0
    trs_error = 0.0
    space_group_error = 0.0
    for point in k_points:
        reference_matrix, reference_velocity = _real_space_reference(
            point, native_model.terms, au_per_angstrom, ev_to_hartree
        )
        qxti_matrix = hamiltonian.H(*point)
        matrix_error = max(matrix_error, float(np.max(np.abs(qxti_matrix - reference_matrix))))
        qxti_bands = np.linalg.eigvalsh(qxti_matrix)
        reference_bands = np.linalg.eigvalsh(reference_matrix)
        band_error = max(band_error, float(np.max(np.abs(qxti_bands - reference_bands))))
        for axis, direction in enumerate(("x", "y", "z")):
            qxti_velocity = operators.velocity(direction, *point, basis="working")
            velocity_error = max(
                velocity_error,
                float(np.max(np.abs(qxti_velocity - reference_velocity[axis]))),
            )
        trs_error = max(
            trs_error,
            float(
                np.max(
                    np.abs(
                        qxti_bands
                        - np.linalg.eigvalsh(hamiltonian.H(*(-point)))
                    )
                )
            ),
        )

        k_angstrom = point * au_per_angstrom
        reference_native = np.linalg.eigvalsh(native_model.Hk(k_angstrom))
        for rotation, _translation in native_model.ops:
            rotated = rotation @ k_angstrom
            rotated_bands = np.linalg.eigvalsh(native_model.Hk(rotated))
            space_group_error = max(
                space_group_error,
                float(np.max(np.abs(reference_native - rotated_bands)) * ev_to_hartree),
            )

    reciprocal_error = 0.0
    primitive_reciprocal_bohr = np.asarray(module.Bm, dtype=np.float64) / au_per_angstrom
    for point in k_points[:16]:
        reference = hamiltonian.eigenvalues(*point)
        for vector in primitive_reciprocal_bohr:
            reciprocal_error = max(
                reciprocal_error,
                float(
                    np.max(
                        np.abs(
                            reference - hamiltonian.eigenvalues(*(point + vector))
                        )
                    )
                ),
            )

    gamma_bands = hamiltonian.eigenvalues(0.0, 0.0, 0.0)
    kramers_splitting = float(np.max(np.abs(gamma_bands[0::2] - gamma_bands[1::2])))
    passed = bool(
        matrix_error <= 1.0e-12
        and band_error <= 1.0e-12
        and velocity_error <= 1.0e-8
        and reciprocal_error <= 1.0e-12
        and trs_error <= 1.0e-12
        and space_group_error <= 1.0e-10
        and kramers_splitting <= 1.0e-12
    )
    return TaAsMultibandResult(
        random_points=random_points,
        matrix_max_abs_error_ha=matrix_error,
        band_max_abs_error_ha=band_error,
        velocity_max_abs_error_ha_bohr=velocity_error,
        reciprocal_periodicity_max_abs_error_ha=reciprocal_error,
        time_reversal_max_abs_error_ha=trs_error,
        space_group_max_abs_error_ha=space_group_error,
        gamma_kramers_max_splitting_ha=kramers_splitting,
        number_of_space_group_operations=len(native_model.ops),
        number_of_hopping_matrices=len(native_model.hop),
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-points", type=int, default=64)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "taas_multiband_8.json"
    )
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(random_points=args.random_points)
    payload = {
        "benchmark": "MULTI-8 TaAs tight-binding",
        "configuration": {"random_points": args.random_points},
        "result": asdict(result),
        "passed": result.passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.no_register:
        record_result(
            {
                "id": "taas_multiband_8",
                "title": "Eight-band TaAs hopping reconstruction and symmetries",
                "passed": result.passed,
                "scope": "Eight-band matrix assembly, unit conversion, velocities, reciprocal periodicity, time reversal, space-group spectra and Kramers degeneracy.",
                "independent_reference": "Independent real-space hopping-sum evaluator and exact symmetry identities",
                "reference_type": "in-repository reconstruction and invariants; not external software",
                "implementation": [
                    "Read the finalized real-space hopping terms and rebuild each 8x8 Bloch matrix in a separate loop implementing the Fourier sum, rather than calling the QXTI wrapper's H(k).",
                    "Differentiate the phase factors analytically for the velocity reference and evaluate reciprocal, time-reversal, space-group and Kramers spectral identities independently of QXTI's finite-difference operator.",
                ],
                "reference_provenance": [
                    "The lattice constants and I4_1md structure describe TaAs, but the hopping amplitudes in `models/taas_tb.py` are explicitly illustrative project parameters rather than a traced published Wannier parametrization.",
                    "Consequently the reference is the exact Fourier reconstruction of the shipped hopping model; it validates implementation and units, not agreement with real TaAs bands.",
                ],
                "production_code_changes": [
                    "None. The TaAs wrapper, unit conversions and declared spectral symmetries passed without a production-code modification.",
                    "Only the validation benchmark and automated regression test were added.",
                ],
                "results": [
                    {
                        "matrix error (Ha)": result.matrix_max_abs_error_ha,
                        "band error (Ha)": result.band_max_abs_error_ha,
                        "velocity error (Ha Bohr)": result.velocity_max_abs_error_ha_bohr,
                        "periodicity error (Ha)": result.reciprocal_periodicity_max_abs_error_ha,
                        "TRS error (Ha)": result.time_reversal_max_abs_error_ha,
                        "space-group error (Ha)": result.space_group_max_abs_error_ha,
                        "Gamma Kramers splitting (Ha)": result.gamma_kramers_max_splitting_ha,
                    }
                ],
                "error_methodology": [
                    "Reassemble every 8x8 matrix directly from the finalized real-space hopping list, including the independent 1/Bohr-to-1/Angstrom and eV-to-Hartree factors, then compare all matrix elements and sorted bands point by point.",
                    "Differentiate the real-space phase factors analytically and compare the full x/y/z velocity matrices with QXTI's finite differences.",
                    "Compare spectra at k and k+G, k and -k, and all space-group-rotated momenta; compare Kramers pairs at Gamma.",
                ],
                "acceptance_criteria": [
                    "Matrix and band errors <= 1e-12 Ha.",
                    "Velocity matrix error <= 1e-8 Ha Bohr.",
                    "Reciprocal-periodicity and time-reversal spectral errors <= 1e-12 Ha.",
                    "Space-group spectral residual <= 1e-10 Ha and Gamma Kramers splitting <= 1e-12 Ha.",
                ],
                "conclusion": "The QXTI wrapper preserves the eight-band real-space hopping Hamiltonian, physical units, analytic velocities and declared TaAs spectral symmetries.",
                "limitations": [
                    "This is exact code verification against an independently assembled hopping sum, not validation against ab-initio or experimental TaAs bands.",
                    "The shipped conventional tetragonal sampling box is explicitly approximate for the body-centered tetragonal primitive BZ, so absolute integrated response is not validated here.",
                ],
                "artifact": str(args.output),
            }
        )
    print(json.dumps(payload, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
