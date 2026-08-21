"""Verify QXTI's gapless graphene model and degeneracy-safe k sampling."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from qxti.core import QXTIConfig, QXTISimulation
from qxti.physics import OperatorFactory

from .benchmark_haldane_pythtb import _real_vectors, _to_reduced
from .registry import PROJECT_ROOT, RESULTS_DIR, record_result


@dataclass(frozen=True)
class GrapheneGaplessResult:
    random_points: int
    band_max_abs_error_ha: float
    dirac_node_max_abs_energy_ha: float
    cone_velocity_expected_ha_bohr: float
    cone_velocity_max_relative_error: float
    reciprocal_periodicity_max_abs_error_ha: float
    shifted_grid_shape: tuple[int, int, int]
    shifted_grid_min_gap_ha: float
    shifted_grid_connection_all_finite: bool
    passed: bool


def _pythtb_graphene(hamiltonian):
    try:
        from pythtb.models import graphene
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PythTB is required for GAPLESS-2D") from exc
    return graphene(
        delta=-float(hamiltonian.params["M"]),
        t=float(hamiltonian.params["t"]),
    )


def _dirac_points(a0: float) -> np.ndarray:
    kx = 4.0 * np.pi / (3.0 * np.sqrt(3.0) * a0)
    return np.asarray([[kx, 0.0, 0.0], [-kx, 0.0, 0.0]], dtype=np.float64)


def _reciprocal_vectors(real_vectors: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.linalg.inv(real_vectors).T


def run_benchmark(*, random_points: int = 128, grid_size: int = 24) -> GrapheneGaplessResult:
    base = QXTIConfig.from_file(PROJECT_ROOT / "inputs" / "inputParams.graphene.cfg")
    config = replace(
        base,
        kgrid=replace(
            base.kgrid,
            k_points=(grid_size, grid_size),
            shifted=True,
            auto_degeneracy_guard=True,
            berry_singularity_guard=True,
        ),
    )
    simulation = QXTISimulation(config=config)
    hamiltonian = simulation.build_hamiltonian()
    pythtb_model = _pythtb_graphene(hamiltonian)
    real_vectors = _real_vectors(hamiltonian)

    bounds = hamiltonian.reciprocal_box_bounds()
    rng = np.random.default_rng(20260822)
    random_k = np.column_stack(
        [
            rng.uniform(bounds[0][0], bounds[0][1], random_points),
            rng.uniform(bounds[1][0], bounds[1][1], random_points),
            np.zeros(random_points),
        ]
    )
    qxti_bands = np.linalg.eigvalsh(
        np.asarray([hamiltonian.H(*point) for point in random_k])
    )
    pythtb_bands = np.asarray(
        pythtb_model.solve_ham(_to_reduced(random_k, real_vectors)), dtype=np.float64
    )
    band_error = float(np.max(np.abs(qxti_bands - pythtb_bands)))

    a0 = float(hamiltonian.params["a0"])
    hopping = abs(float(hamiltonian.params["t"]))
    nodes = _dirac_points(a0)
    node_energies = np.asarray(
        [hamiltonian.eigenvalues(*node) for node in nodes], dtype=np.float64
    )
    node_error = float(np.max(np.abs(node_energies)))

    expected_velocity = 1.5 * hopping * a0
    radius = 1.0e-6
    directions = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    velocity_errors = []
    for node in nodes:
        for angle in directions:
            displaced = node + radius * np.asarray(
                [np.cos(angle), np.sin(angle), 0.0], dtype=np.float64
            )
            upper_energy = float(hamiltonian.eigenvalues(*displaced)[1])
            measured_velocity = upper_energy / radius
            velocity_errors.append(abs(measured_velocity - expected_velocity) / expected_velocity)
    cone_velocity_error = float(max(velocity_errors))

    reciprocal = _reciprocal_vectors(real_vectors)
    periodicity_error = 0.0
    for point in random_k[:32]:
        reference = hamiltonian.eigenvalues(*point)
        for vector in reciprocal:
            translated = point.copy()
            translated[:2] += vector
            periodicity_error = max(
                periodicity_error,
                float(np.max(np.abs(reference - hamiltonian.eigenvalues(*translated)))),
            )

    kgrid = simulation.build_kgrid(hamiltonian)
    grid_points = kgrid.points()
    grid_energies = np.linalg.eigvalsh(
        np.asarray([hamiltonian.H(*point) for point in grid_points])
    )
    grid_min_gap = float(np.min(grid_energies[:, 1] - grid_energies[:, 0]))
    operators = OperatorFactory(hamiltonian, basis="band")
    all_finite = True
    for point in grid_points:
        for direction in ("x", "y"):
            all_finite = all_finite and bool(
                np.all(np.isfinite(operators.berry_connection(direction, *point, basis="band")))
            )

    passed = bool(
        band_error <= 1.0e-12
        and node_error <= 1.0e-12
        and cone_velocity_error <= 1.0e-4
        and periodicity_error <= 1.0e-12
        and grid_min_gap > 1.0e-8
        and all_finite
    )
    return GrapheneGaplessResult(
        random_points=random_points,
        band_max_abs_error_ha=band_error,
        dirac_node_max_abs_energy_ha=node_error,
        cone_velocity_expected_ha_bohr=expected_velocity,
        cone_velocity_max_relative_error=cone_velocity_error,
        reciprocal_periodicity_max_abs_error_ha=periodicity_error,
        shifted_grid_shape=kgrid.shape,
        shifted_grid_min_gap_ha=grid_min_gap,
        shifted_grid_connection_all_finite=all_finite,
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-points", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "graphene_gapless_2d.json"
    )
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(random_points=args.random_points, grid_size=args.grid_size)
    payload = {
        "benchmark": "GAPLESS-2D graphene",
        "configuration": {
            "random_points": args.random_points,
            "requested_shifted_grid": [args.grid_size, args.grid_size],
            "cone_sampling_radius_inverse_bohr": 1.0e-6,
        },
        "result": asdict(result),
        "passed": result.passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.no_register:
        record_result(
            {
                "id": "graphene_gapless_2d",
                "title": "Graphene gapless spectrum and degeneracy-safe sampling",
                "passed": result.passed,
                "scope": "Gapless 2-D bands, Dirac cone, reciprocal periodicity and finite Berry-connection sampling for graphene.",
                "independent_reference": f"PythTB {version('pythtb')} graphene model and closed-form Dirac limit",
                "reference_type": "external cross-code plus analytic Dirac identities",
                "implementation": [
                    "Build PythTB's graphene model with the same nearest-neighbour hopping and onsite mass, then compare sorted bands after an independent Cartesian-to-reduced momentum conversion.",
                    "Evaluate both closed-form Dirac points, estimate the cone slope radially, test spectral reciprocal periodicity, and run every point of a shifted QXTI grid through the Berry-connection operator.",
                ],
                "reference_provenance": [
                    "PythTB is the external band reference; zero node energy and $v_F=3|t|a_0/2$ are analytic nearest-neighbour graphene results.",
                ],
                "production_code_changes": [
                    "Replace rounded decimal real-space vectors in `models/graphene.py` with expressions derived from the configured lattice constant. The rounded metadata corrupted Cartesian-to-reduced conversion and produced a spurious cross-code error of about 1.36e-5 Ha even though H(k) was correct.",
                    "No graphene Hamiltonian term was changed; only the lattice metadata was made numerically consistent with the Hamiltonian parameters.",
                ],
                "results": [
                    {
                        "band max error (Ha)": result.band_max_abs_error_ha,
                        "Dirac energy error (Ha)": result.dirac_node_max_abs_energy_ha,
                        "cone velocity relative error": result.cone_velocity_max_relative_error,
                        "periodicity error (Ha)": result.reciprocal_periodicity_max_abs_error_ha,
                        "minimum shifted-grid gap (Ha)": result.shifted_grid_min_gap_ha,
                    }
                ],
                "error_methodology": [
                    "Compare sorted bands point by point at 128 deterministic random Cartesian k-points after an independent Cartesian-to-reduced conversion for PythTB.",
                    "Evaluate both analytic Dirac points and compare their energies with zero; estimate the radial cone slope in 12 directions around each valley and compare with $v_F=3|t|a_0/2$.",
                    "Translate generic k-points by both primitive reciprocal vectors and compare spectra, which is gauge invariant.",
                    "Build QXTI's shifted guarded grid and require a finite nonzero sampled gap and finite off-diagonal Berry connections at every grid point.",
                ],
                "acceptance_criteria": [
                    "Maximum PythTB band error <= 1e-12 Ha.",
                    "Maximum Dirac-node energy <= 1e-12 Ha.",
                    "Maximum cone-velocity relative error <= 1e-4.",
                    "Maximum reciprocal-periodicity spectral error <= 1e-12 Ha.",
                    "The shifted grid avoids exact degeneracies and every sampled connection is finite.",
                ],
                "conclusion": "QXTI reproduces the gapless graphene spectrum and samples the Dirac model without injecting singular grid observables.",
                "limitations": [
                    "This verifies the static gapless model and grid guard, not graphene optical conductivity or HHG.",
                    "Berry connection exactly at a Dirac point is undefined and is deliberately not evaluated.",
                ],
                "artifact": str(args.output),
            }
        )
    print(json.dumps(payload, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
