"""Verify the 76-band TBLG implementation and quantify basis truncation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from qxti.core import QXTIConfig, QXTISimulation

from .registry import PROJECT_ROOT, RESULTS_DIR, record_result


@dataclass(frozen=True)
class TBLGTruncationResult:
    random_points: int
    path_points: int
    qxti_basis_size: int
    reference_basis_sizes: dict[str, int]
    matrix_max_abs_error_ha: float
    band_max_abs_error_ha: float
    n1_to_n3_flat_band_max_error_ha: float
    n2_to_n3_flat_band_max_error_ha: float
    n1_to_n3_normalized_error: float
    n2_to_n3_normalized_error: float
    convergence_improvement_factor: float
    n2_flat_bandwidth_ha: float
    n3_flat_bandwidth_ha: float
    flat_bandwidth_relative_error: float
    passed: bool


def _hexagonal_indices(number_of_rings: int) -> list[tuple[int, int]]:
    """Generate a hexagonal reciprocal cutoff independently of the model."""
    return [
        (i, j)
        for i in range(-number_of_rings, number_of_rings + 1)
        for j in range(-number_of_rings, number_of_rings + 1)
        if max(abs(i), abs(j), abs(i + j)) <= number_of_rings
    ]


def _reference_geometry(theta_deg: float, lattice_constant: float):
    theta = np.deg2rad(theta_deg)
    monolayer_k = 4.0 * np.pi / (3.0 * lattice_constant)
    moire_k = 2.0 * monolayer_k * np.sin(theta / 2.0)
    root_three = np.sqrt(3.0)
    q1 = moire_k * np.asarray([0.0, -1.0])
    q2 = moire_k * np.asarray([root_three / 2.0, 0.5])
    q3 = moire_k * np.asarray([-root_three / 2.0, 0.5])
    return q2 - q1, q3 - q1, q1, theta, moire_k


def _reference_hamiltonian(
    momentum: np.ndarray, number_of_rings: int, params: dict[str, float]
) -> np.ndarray:
    """Separately coded Bistritzer--MacDonald plane-wave Hamiltonian."""
    points = _hexagonal_indices(number_of_rings)
    point_index = {point: index for index, point in enumerate(points)}
    g1, g2, q1, theta, _ = _reference_geometry(
        params["theta_deg"], params["a"]
    )
    velocity = params["hbar_vF"]
    w_aa = params["w_aa"]
    w_ab = params["w_ab"]
    omega = np.exp(2.0j * np.pi / 3.0)
    tunnel_matrices = (
        np.asarray([[w_aa, w_ab], [w_ab, w_aa]], dtype=np.complex128),
        np.asarray(
            [[w_aa, w_ab * omega.conjugate()], [w_ab * omega, w_aa]],
            dtype=np.complex128,
        ),
        np.asarray(
            [[w_aa, w_ab * omega], [w_ab * omega.conjugate(), w_aa]],
            dtype=np.complex128,
        ),
    )
    matrix = np.zeros((4 * len(points), 4 * len(points)), dtype=np.complex128)

    def dirac_block(vector: np.ndarray, rotation: float) -> np.ndarray:
        cosine, sine = np.cos(rotation), np.sin(rotation)
        rotated_x = cosine * vector[0] - sine * vector[1]
        rotated_y = sine * vector[0] + cosine * vector[1]
        off_diagonal = velocity * (rotated_x - 1.0j * rotated_y)
        return np.asarray(
            [[0.0, off_diagonal], [off_diagonal.conjugate(), 0.0]],
            dtype=np.complex128,
        )

    k = np.asarray(momentum, dtype=np.float64)[:2]
    neighbour_offsets = ((0, 0), (1, 0), (0, 1))
    for index, (i, j) in enumerate(points):
        start = 4 * index
        reciprocal_shift = i * g1 + j * g2
        matrix[start : start + 2, start : start + 2] = dirac_block(
            k + reciprocal_shift, -theta / 2.0
        )
        matrix[start + 2 : start + 4, start + 2 : start + 4] = dirac_block(
            k + reciprocal_shift + q1, theta / 2.0
        )
        for offset, tunnel in zip(neighbour_offsets, tunnel_matrices, strict=True):
            neighbour = point_index.get((i + offset[0], j + offset[1]))
            if neighbour is None:
                continue
            top_start = 4 * neighbour + 2
            matrix[start : start + 2, top_start : top_start + 2] += tunnel
            matrix[top_start : top_start + 2, start : start + 2] += tunnel.conj().T
    return matrix


def _high_symmetry_path(params: dict[str, float], points_per_segment: int) -> np.ndarray:
    g1, g2, _q1, _theta, _moire_k = _reference_geometry(
        params["theta_deg"], params["a"]
    )
    gamma = np.zeros(2)
    corner_k = (g1 + g2) / 3.0
    edge_m = g1 / 2.0
    vertices = (gamma, corner_k, edge_m, gamma)
    segments = []
    for index, (start, stop) in enumerate(zip(vertices[:-1], vertices[1:], strict=True)):
        segment = np.linspace(start, stop, points_per_segment, endpoint=True)
        if index:
            segment = segment[1:]
        segments.append(segment)
    return np.vstack(segments)


def _central_bands(matrix: np.ndarray) -> np.ndarray:
    energies = np.linalg.eigvalsh(matrix)
    middle = energies.size // 2
    return energies[middle - 1 : middle + 1]


def run_benchmark(
    *, random_points: int = 8, points_per_segment: int = 17
) -> TBLGTruncationResult:
    config = QXTIConfig.from_file(PROJECT_ROOT / "inputs" / "inputParams.tblg.cfg")
    hamiltonian = QXTISimulation(config=config).build_hamiltonian()
    params = {key: float(value) for key, value in hamiltonian.params.items()}

    bounds = hamiltonian.reciprocal_box_bounds()
    rng = np.random.default_rng(20260824)
    random_k = np.column_stack(
        (
            rng.uniform(*bounds[0], size=random_points),
            rng.uniform(*bounds[1], size=random_points),
            np.zeros(random_points),
        )
    )
    matrix_error = 0.0
    band_error = 0.0
    for point in random_k:
        qxti_matrix = hamiltonian.H(*point)
        reference_matrix = _reference_hamiltonian(point, 2, params)
        matrix_error = max(
            matrix_error, float(np.max(np.abs(qxti_matrix - reference_matrix)))
        )
        band_error = max(
            band_error,
            float(
                np.max(
                    np.abs(
                        np.linalg.eigvalsh(qxti_matrix)
                        - np.linalg.eigvalsh(reference_matrix)
                    )
                )
            ),
        )

    path = _high_symmetry_path(params, points_per_segment)
    central_bands: dict[int, np.ndarray] = {}
    for number_of_rings in (1, 2, 3):
        central_bands[number_of_rings] = np.asarray(
            [
                _central_bands(
                    _reference_hamiltonian(np.r_[point, 0.0], number_of_rings, params)
                )
                for point in path
            ]
        )

    scale = params["hbar_vF"] * _reference_geometry(
        params["theta_deg"], params["a"]
    )[4]
    n1_error = float(np.max(np.abs(central_bands[1] - central_bands[3])))
    n2_error = float(np.max(np.abs(central_bands[2] - central_bands[3])))
    n1_normalized = n1_error / scale
    n2_normalized = n2_error / scale
    improvement = n1_error / n2_error
    n2_bandwidth = float(np.ptp(central_bands[2]))
    n3_bandwidth = float(np.ptp(central_bands[3]))
    bandwidth_error = abs(n2_bandwidth - n3_bandwidth) / n3_bandwidth
    basis_sizes = {
        f"N={number_of_rings}": 4 * len(_hexagonal_indices(number_of_rings))
        for number_of_rings in (1, 2, 3)
    }
    passed = bool(
        hamiltonian.basis_size == basis_sizes["N=2"] == 76
        and matrix_error <= 1.0e-12
        and band_error <= 1.0e-12
        and n2_normalized <= 0.05
        and improvement >= 2.0
        and bandwidth_error <= 0.10
    )
    return TBLGTruncationResult(
        random_points=random_points,
        path_points=len(path),
        qxti_basis_size=hamiltonian.basis_size,
        reference_basis_sizes=basis_sizes,
        matrix_max_abs_error_ha=matrix_error,
        band_max_abs_error_ha=band_error,
        n1_to_n3_flat_band_max_error_ha=n1_error,
        n2_to_n3_flat_band_max_error_ha=n2_error,
        n1_to_n3_normalized_error=n1_normalized,
        n2_to_n3_normalized_error=n2_normalized,
        convergence_improvement_factor=improvement,
        n2_flat_bandwidth_ha=n2_bandwidth,
        n3_flat_bandwidth_ha=n3_bandwidth,
        flat_bandwidth_relative_error=bandwidth_error,
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-points", type=int, default=8)
    parser.add_argument("--points-per-segment", type=int, default=17)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "tblg_truncation.json"
    )
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        random_points=args.random_points, points_per_segment=args.points_per_segment
    )
    payload = {
        "benchmark": "TRUNC-MB TBLG",
        "configuration": {
            "random_points": args.random_points,
            "points_per_segment": args.points_per_segment,
            "path": "Gamma-K-M-Gamma",
            "reference_cutoffs": [1, 2, 3],
        },
        "result": asdict(result),
        "passed": result.passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.no_register:
        record_result(
            {
                "id": "tblg_truncation_many_band",
                "title": "TBLG 76-band implementation and basis truncation",
                "passed": result.passed,
                "scope": "Bistritzer--MacDonald matrix assembly and flat-band convergence of the fixed N=2 (76-band) reciprocal cutoff.",
                "independent_reference": "Separately coded BM plane-wave evaluator at N=1, N=2 and N=3",
                "reference_type": "in-repository independent implementation and convergence reference; not external software",
                "implementation": [
                    "Implement a second BM constructor that independently generates the hexagonal reciprocal cutoff, rotated layer Dirac blocks and T1/T2/T3 interlayer couplings, without calling private helpers from `models/tblg_bmd.py`.",
                    "First compare the complete N=2 matrix and all 76 eigenvalues at random momenta; then compute N=1, N=2 and N=3 central bands at 49 points including Gamma, K and M.",
                ],
                "reference_provenance": [
                    "The continuum construction follows Bistritzer and MacDonald, PNAS 108, 12233 (2011), with corrugation-inspired unequal interlayer couplings as in later continuum work such as Koshino et al., Phys. Rev. X 8, 031087 (2018).",
                    "The current hbar*vF is consistent with Koshino, but the exact pair wAA=81.7 meV and wAB=110 meV is a project-specific hybrid and is not the 79.7/97.5 meV pair reported by Koshino.",
                    "The N=1/2/3 cutoffs, use of N=3 as numerical reference and acceptance thresholds are choices of this benchmark, not external published band data.",
                ],
                "production_code_changes": [
                    "None. The shipped 76-band TBLG Hamiltonian matched the separate implementation without changing production code.",
                    "The benchmark path construction was corrected during validation to include the Gamma, K and M vertices exactly; that change affects validation code only.",
                ],
                "results": [
                    {
                        "matrix error (Ha)": result.matrix_max_abs_error_ha,
                        "band error (Ha)": result.band_max_abs_error_ha,
                        "N2-N3 normalized error": result.n2_to_n3_normalized_error,
                        "N1/N2 improvement": result.convergence_improvement_factor,
                        "bandwidth relative error": result.flat_bandwidth_relative_error,
                    }
                ],
                "error_methodology": [
                    "Rebuild the BM Hamiltonian independently at deterministic random momenta and compare every matrix element and every sorted eigenvalue point by point with QXTI's 76-band model.",
                    "Along Gamma-K-M-Gamma, compare the central valence/conduction energies from N=1 and N=2 cutoffs against N=3; normalize the maximum error by the natural continuum scale hbar*vF*k_theta.",
                    "Compute the combined two-flat-band energy range for N=2 and N=3 and report their relative difference; require the N=2 error to improve materially over N=1.",
                ],
                "acceptance_criteria": [
                    "QXTI has 76 bands and matrix/band implementation errors are <= 1e-12 Ha.",
                    "N=2 maximum central-band error relative to N=3 is <= 5% of hbar*vF*k_theta.",
                    "N=2 improves the maximum error by at least a factor of two over N=1.",
                    "N=2 versus N=3 flat-bandwidth relative error is <= 10%.",
                ],
                "conclusion": "The shipped 76-band BM Hamiltonian matches an independent assembly and its central bands are converged against the 148-band cutoff within the stated thresholds.",
                "limitations": [
                    "N=3 is a higher-cutoff numerical reference, not an exact infinite-basis solution or external ab-initio validation.",
                    "Only the central bands on Gamma-K-M-Gamma at theta=1.2 degrees are covered; nonlinear response and the approximate boundary gauge are not validated here.",
                ],
                "artifact": str(args.output),
            }
        )
    print(json.dumps(payload, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
