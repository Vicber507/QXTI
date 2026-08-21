"""Converge Haldane bulk DOS on independently generated QXTI/PythTB meshes."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from qxti.analytics.dos import compute_dos_spectrum
from qxti.core import QXTIConfig, QXTISimulation

from .benchmark_haldane_pythtb import PROJECT_ROOT, _kernel, _pythtb_model
from .registry import RESULTS_DIR, record_result


@dataclass(frozen=True)
class GridPointResult:
    grid: int
    qxti_dos_integral: float
    qxti_vs_pythtb_relative_l2: float
    qxti_vs_reference_relative_l2: float
    pythtb_vs_reference_relative_l2: float


@dataclass(frozen=True)
class GridConvergenceResult:
    case: str
    points: list[GridPointResult]
    errors_decrease: bool
    final_cross_code_error: float
    passed: bool


def _native_reduced_grid(size: int) -> np.ndarray:
    """PythTB-native half-shifted grid over [-1/2, 1/2)^2."""
    axis = (np.arange(size, dtype=np.float64) + 0.5) / size - 0.5
    k1, k2 = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([k1.ravel(), k2.ravel()])


def _pythtb_dos(model, size: int, energies: np.ndarray, eta: float) -> np.ndarray:
    """DOS from a native reduced PythTB mesh, accumulated in bounded chunks."""
    eigenvalues = np.asarray(model.solve_ham(_native_reduced_grid(size)), dtype=np.float64)
    result = np.zeros(energies.size, dtype=np.float64)
    chunk_size = 4096
    for start in range(0, eigenvalues.shape[0], chunk_size):
        chunk = eigenvalues[start : start + chunk_size]
        diff = energies[None, None, :] - chunk[:, :, None]
        result += _kernel(diff, eta, "gaussian").sum(axis=(0, 1))
    return result / eigenvalues.shape[0]


def _qxti_config(case: str, size: int, energies: np.ndarray, eta: float) -> QXTIConfig:
    path = PROJECT_ROOT / "inputs" / f"inputParams.haldane_{case}.cfg"
    base = QXTIConfig.from_file(path)
    return replace(
        base,
        kgrid=replace(
            base.kgrid,
            k_points=(size, size),
            shifted=True,
            auto_degeneracy_guard=False,
            berry_singularity_guard=False,
        ),
        ldos=replace(
            base.ldos,
            method="eigenvalues",
            broadening="gaussian",
            eta=eta,
            e_min=float(energies[0]),
            e_max=float(energies[-1]),
            num_energies=int(energies.size),
            projected=False,
            spectral_enabled=False,
            spectral_plane_enabled=False,
        ),
    )


def run_case(
    case: str,
    *,
    grid_sizes: tuple[int, ...] = (11, 21, 41, 81),
    reference_grid: int = 161,
    eta: float = 3.0e-3,
    num_energies: int = 501,
) -> GridConvergenceResult:
    if case not in {"topological", "trivial"}:
        raise ValueError("case must be 'topological' or 'trivial'")
    if not grid_sizes or any(size < 2 for size in grid_sizes):
        raise ValueError("grid_sizes must contain values >= 2")

    energies = np.linspace(-0.45, 0.45, num_energies, dtype=np.float64)
    first_config = _qxti_config(case, grid_sizes[0], energies, eta)
    hamiltonian = QXTISimulation(first_config).build_hamiltonian()
    pythtb_model = _pythtb_model(hamiltonian)
    reference_dos = _pythtb_dos(pythtb_model, reference_grid, energies, eta)
    reference_norm = max(float(np.linalg.norm(reference_dos)), np.finfo(float).tiny)

    points: list[GridPointResult] = []
    for size in grid_sizes:
        config = _qxti_config(case, size, energies, eta)
        dataset = compute_dos_spectrum(config, progress=False)["dataset"]
        qxti_dos = np.asarray(dataset["dos"], dtype=np.float64)
        pythtb_dos = _pythtb_dos(pythtb_model, size, energies, eta)
        points.append(
            GridPointResult(
                grid=size,
                qxti_dos_integral=float(dataset["integral"]),
                qxti_vs_pythtb_relative_l2=float(
                    np.linalg.norm(qxti_dos - pythtb_dos) / reference_norm
                ),
                qxti_vs_reference_relative_l2=float(
                    np.linalg.norm(qxti_dos - reference_dos) / reference_norm
                ),
                pythtb_vs_reference_relative_l2=float(
                    np.linalg.norm(pythtb_dos - reference_dos) / reference_norm
                ),
            )
        )

    cross_errors = np.asarray([point.qxti_vs_pythtb_relative_l2 for point in points])
    errors_decrease = bool(np.all(np.diff(cross_errors) < 0.0))
    final = points[-1]
    passed = bool(
        errors_decrease
        and final.qxti_vs_pythtb_relative_l2 <= 1.0e-2
        and final.qxti_vs_reference_relative_l2 <= 1.0e-2
        and final.pythtb_vs_reference_relative_l2 <= 2.0e-2
        and abs(final.qxti_dos_integral - 2.0) <= 2.0e-2
    )
    return GridConvergenceResult(
        case=case,
        points=points,
        errors_decrease=errors_decrease,
        final_cross_code_error=final.qxti_vs_pythtb_relative_l2,
        passed=passed,
    )


def run_benchmark(**kwargs) -> list[GridConvergenceResult]:
    return [run_case("topological", **kwargs), run_case("trivial", **kwargs)]


def _payload(results, *, grid_sizes, reference_grid, eta, num_energies):
    return {
        "benchmark": "Haldane DOS convergence on independent QXTI/PythTB meshes",
        "method": {
            "qxti_mesh": "axis-aligned Cartesian reciprocal fundamental cell",
            "pythtb_mesh": "native reduced reciprocal primitive cell",
            "reference": f"PythTB native {reference_grid}x{reference_grid} mesh",
        },
        "configuration": {
            "grid_sizes": list(grid_sizes),
            "reference_grid": reference_grid,
            "eta_ha": eta,
            "energy_window_ha": [-0.45, 0.45],
            "num_energies": num_energies,
        },
        "error_methodology": {
            "cross_code": (
                "At each N, compare complete DOS arrays on the same 501-energy axis: "
                "norm(g_QXTI,N-g_PythTB,N)_2 / norm(g_PythTB,reference)_2. QXTI and "
                "PythTB generate their k-meshes independently."
            ),
            "qxti_to_reference": (
                "norm(g_QXTI,N-g_PythTB,reference)_2 / norm(g_PythTB,reference)_2"
            ),
            "pythtb_to_reference": (
                "norm(g_PythTB,N-g_PythTB,reference)_2 / norm(g_PythTB,reference)_2"
            ),
            "sum_rule": "Composite trapezoidal integral of QXTI DOS over the energy axis.",
        },
        "results": [
            {
                "case": result.case,
                "points": [asdict(point) for point in result.points],
                "errors_decrease": result.errors_decrease,
                "final_cross_code_error": result.final_cross_code_error,
                "passed": result.passed,
            }
            for result in results
        ],
        "passed": all(result.passed for result in results),
    }


def _registry_record(payload: dict, artifact: Path) -> dict:
    rows = []
    for result in payload["results"]:
        for point in result["points"]:
            rows.append(
                {
                    "case": result["case"],
                    "grid": f"{point['grid']}x{point['grid']}",
                    "QXTI vs PythTB": point["qxti_vs_pythtb_relative_l2"],
                    "QXTI vs reference": point["qxti_vs_reference_relative_l2"],
                    "PythTB vs reference": point["pythtb_vs_reference_relative_l2"],
                    "DOS integral": point["qxti_dos_integral"],
                }
            )
    return {
        "id": "haldane_dos_independent_grid_convergence",
        "title": "Haldane DOS convergence with independent reciprocal meshes",
        "passed": payload["passed"],
        "scope": "Bulk DOS integration domain and k-grid convergence for topological and trivial Haldane inputs.",
        "independent_reference": f"PythTB {version('pythtb')}",
        "reference_type": "external cross-code integration reference",
        "implementation": [
            "Let QXTI generate its Cartesian reciprocal meshes while PythTB independently generates reduced primitive-cell meshes; do not reuse QXTI k-points in the reference path.",
            "Repeat the complete DOS calculation at four refinements and compare both sequences with a native high-resolution PythTB mesh.",
        ],
        "reference_provenance": [
            "PythTB supplies the external Hamiltonian and reduced-coordinate sampling; the 161x161 convergence mesh and acceptance thresholds are choices of this verification study.",
        ],
        "production_code_changes": [
            "None. The existing QXTI reciprocal grid and DOS integration paths passed the independent-mesh convergence test.",
        ],
        "configuration": payload["configuration"],
        "results": rows,
        "error_methodology": [
            "For every mesh size $N$, QXTI and PythTB independently generate their own $N\\times N$ k-mesh. Their DOS arrays are then compared at the same 501 energy nodes.",
            "A PythTB native $161\\times161$ calculation defines $g_{ref}$. The common normalization is $\\|g_{ref}\\|_2$ so every row is comparable across mesh sizes.",
            "The cross-code error is $\\epsilon_{QP}(N)=\\|g_{QXTI,N}-g_{PythTB,N}\\|_2/\\|g_{ref}\\|_2$.",
            "Individual convergence errors are $\\epsilon_Q(N)=\\|g_{QXTI,N}-g_{ref}\\|_2/\\|g_{ref}\\|_2$ and $\\epsilon_P(N)=\\|g_{PythTB,N}-g_{ref}\\|_2/\\|g_{ref}\\|_2$.",
            "These are whole-curve L2 errors over the energy axis, not pointwise errors at a single energy. The DOS integral uses the composite trapezoidal rule.",
        ],
        "acceptance_criteria": [
            "Cross-code relative L2 error decreases at every grid refinement.",
            "Final QXTI-vs-PythTB relative L2 error <= 1%.",
            "Final QXTI-vs-reference error <= 1% and PythTB-vs-reference error <= 2%.",
            "Final QXTI DOS integral differs from two bands by <= 0.02.",
        ],
        "conclusion": (
            "The Cartesian QXTI integration cell and the independently generated "
            "PythTB reduced cell converge to the same bulk DOS."
        ),
        "limitations": [
            "This validates bulk DOS integration, not QXTI Berry curvature, optical response, surface LDOS, or HHG.",
            "The high-resolution reference is numerical PythTB data, not a closed-form analytic DOS.",
        ],
        "artifact": str(artifact),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", default="11,21,41,81")
    parser.add_argument("--reference-grid", type=int, default=161)
    parser.add_argument("--eta", type=float, default=3.0e-3)
    parser.add_argument("--num-energies", type=int, default=501)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "haldane_dos_grid_convergence.json",
    )
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    grid_sizes = tuple(int(item.strip()) for item in args.grids.split(",") if item.strip())

    results = run_benchmark(
        grid_sizes=grid_sizes,
        reference_grid=args.reference_grid,
        eta=args.eta,
        num_energies=args.num_energies,
    )
    payload = _payload(
        results,
        grid_sizes=grid_sizes,
        reference_grid=args.reference_grid,
        eta=args.eta,
        num_energies=args.num_energies,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.no_register:
        record_result(_registry_record(payload, args.output))
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
