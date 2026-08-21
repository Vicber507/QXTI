"""Verify 3-D Brillouin-zone integration with the two-band Weyl model."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

from qxti.analytics.dos import compute_dos_spectrum
from qxti.core import QXTIConfig, QXTISimulation

from .benchmark_haldane_pythtb import _kernel
from .registry import PROJECT_ROOT, RESULTS_DIR, record_result


@dataclass(frozen=True)
class WSMGridPoint:
    grid: int
    qxti_dos_integral: float
    same_grid_relative_l2: float
    qxti_vs_reference_relative_l2: float
    independent_vs_reference_relative_l2: float


@dataclass(frozen=True)
class WSMGrid3DResult:
    points: list[WSMGridPoint]
    qxti_errors_decrease: bool
    low_energy_dos_exponent: float
    final_same_grid_error: float
    final_reference_error: float
    passed: bool


def _native_reduced_grid(size: int, params: dict[str, object]) -> np.ndarray:
    axis = (np.arange(size, dtype=np.float64) + 0.5) / size - 0.5
    r1, r2, r3 = np.meshgrid(axis, axis, axis, indexing="ij")
    reduced = np.column_stack([r1.ravel(), r2.ravel(), r3.ravel()])
    widths = 2.0 * np.pi / np.asarray(
        [float(params["a0"]), float(params["a1"]), float(params["a2"])],
        dtype=np.float64,
    )
    return reduced * widths


def _independent_bands(k_points: np.ndarray, params: dict[str, object]) -> np.ndarray:
    """Closed-form WSM eigenvalues without calling the QXTI model module."""
    gamma = float(params["gamma"])
    tx = float(params["tx"])
    ty = float(params["ty"])
    tz = float(params["tz"])
    mass = float(params["M0"])
    a0 = float(params["a0"])
    a1 = float(params["a1"])
    a2 = float(params["a2"])
    kw_value = params.get("kw")
    kw = np.pi / (2.0 * a0) if kw_value is None else float(kw_value)
    kx, ky, kz = np.asarray(k_points, dtype=np.float64).T
    cos_kw = np.cos(a0 * kw)
    b0 = gamma * (np.cos(2.0 * a0 * kx) - cos_kw) * (
        np.cos(a2 * kz) - cos_kw
    )
    b1 = -(
        mass * (1.0 - np.cos(a2 * kz) ** 2 - np.cos(a1 * ky))
        + 2.0 * tx * (np.cos(a0 * kx) - cos_kw)
    )
    b2 = -2.0 * ty * np.sin(a1 * ky)
    b3 = -2.0 * tz * np.cos(a2 * kz)
    radius = np.sqrt(b1 * b1 + b2 * b2 + b3 * b3)
    return np.column_stack([b0 - radius, b0 + radius])


def _dos_from_bands(
    eigenvalues: np.ndarray, energies: np.ndarray, eta: float
) -> np.ndarray:
    result = np.zeros(energies.size, dtype=np.float64)
    chunk_size = 4096
    for start in range(0, eigenvalues.shape[0], chunk_size):
        chunk = eigenvalues[start : start + chunk_size]
        diff = energies[None, None, :] - chunk[:, :, None]
        result += _kernel(diff, eta, "gaussian").sum(axis=(0, 1))
    return result / eigenvalues.shape[0]


def _independent_dos(
    size: int, params: dict[str, object], energies: np.ndarray, eta: float
) -> np.ndarray:
    points = _native_reduced_grid(size, params)
    return _dos_from_bands(_independent_bands(points, params), energies, eta)


def _qxti_config(
    size: int, energies: np.ndarray, eta: float
) -> QXTIConfig:
    base = QXTIConfig.from_file(PROJECT_ROOT / "inputs" / "inputParams.wsm.cfg")
    return replace(
        base,
        kgrid=replace(
            base.kgrid,
            k_points=(size, size, size),
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


def run_benchmark(
    *,
    grid_sizes: tuple[int, ...] = (7, 11, 15, 21),
    reference_grid: int = 41,
    eta: float = 1.5e-3,
    num_energies: int = 301,
) -> WSMGrid3DResult:
    energies = np.linspace(-0.05, 0.05, num_energies, dtype=np.float64)
    hamiltonian = QXTISimulation(_qxti_config(grid_sizes[0], energies, eta)).build_hamiltonian()
    params = dict(hamiltonian.params)

    expected_bounds = np.asarray(
        [
            [-np.pi / float(params["a0"]), np.pi / float(params["a0"])],
            [-np.pi / float(params["a1"]), np.pi / float(params["a1"])],
            [-np.pi / float(params["a2"]), np.pi / float(params["a2"])],
        ]
    )
    np.testing.assert_allclose(
        np.asarray(hamiltonian.reciprocal_box_bounds()), expected_bounds, atol=1.0e-14
    )

    reference_dos = _independent_dos(reference_grid, params, energies, eta)
    reference_norm = max(float(np.linalg.norm(reference_dos)), np.finfo(float).tiny)
    points: list[WSMGridPoint] = []
    for size in grid_sizes:
        config = _qxti_config(size, energies, eta)
        dataset = compute_dos_spectrum(config, progress=False)["dataset"]
        qxti_dos = np.asarray(dataset["dos"], dtype=np.float64)
        independent_dos = _independent_dos(size, params, energies, eta)
        points.append(
            WSMGridPoint(
                grid=size,
                qxti_dos_integral=float(dataset["integral"]),
                same_grid_relative_l2=float(
                    np.linalg.norm(qxti_dos - independent_dos) / reference_norm
                ),
                qxti_vs_reference_relative_l2=float(
                    np.linalg.norm(qxti_dos - reference_dos) / reference_norm
                ),
                independent_vs_reference_relative_l2=float(
                    np.linalg.norm(independent_dos - reference_dos) / reference_norm
                ),
            )
        )

    positive_window = (energies >= 0.004) & (energies <= 0.012)
    exponent = float(
        np.polyfit(
            np.log(energies[positive_window]),
            np.log(reference_dos[positive_window]),
            1,
        )[0]
    )
    qxti_errors = np.asarray([p.qxti_vs_reference_relative_l2 for p in points])
    errors_decrease = bool(np.all(np.diff(qxti_errors) < 0.0))
    final = points[-1]
    passed = bool(
        errors_decrease
        and final.same_grid_relative_l2 <= 1.0e-11
        and final.qxti_vs_reference_relative_l2 <= 1.0e-2
        and abs(final.qxti_dos_integral - 2.0) <= 2.0e-2
        and 1.7 <= exponent <= 2.3
    )
    return WSMGrid3DResult(
        points=points,
        qxti_errors_decrease=errors_decrease,
        low_energy_dos_exponent=exponent,
        final_same_grid_error=final.same_grid_relative_l2,
        final_reference_error=final.qxti_vs_reference_relative_l2,
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", default="7,11,15,21")
    parser.add_argument("--reference-grid", type=int, default=41)
    parser.add_argument("--eta", type=float, default=1.5e-3)
    parser.add_argument("--num-energies", type=int, default=301)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "wsm_grid_3d.json"
    )
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    grids = tuple(int(item.strip()) for item in args.grids.split(",") if item.strip())
    result = run_benchmark(
        grid_sizes=grids,
        reference_grid=args.reference_grid,
        eta=args.eta,
        num_energies=args.num_energies,
    )
    payload = {
        "benchmark": "GRID-3D two-band Weyl DOS",
        "configuration": {
            "grid_sizes": list(grids),
            "reference_grid": args.reference_grid,
            "eta_ha": args.eta,
            "energy_window_ha": [-0.05, 0.05],
            "num_energies": args.num_energies,
        },
        "result": asdict(result),
        "passed": result.passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.no_register:
        rows = [
            {
                "grid": f"{point.grid}^3",
                "same-grid relative L2": point.same_grid_relative_l2,
                "QXTI vs reference": point.qxti_vs_reference_relative_l2,
                "independent vs reference": point.independent_vs_reference_relative_l2,
                "DOS integral": point.qxti_dos_integral,
            }
            for point in result.points
        ]
        record_result(
            {
                "id": "wsm_dos_grid_3d",
                "title": "Three-dimensional WSM grid and DOS convergence",
                "passed": result.passed,
                "scope": "Three-dimensional reciprocal bounds, independent mesh construction, bulk DOS normalization and convergence for the two-band WSM.",
                "independent_reference": "Separately coded closed-form WSM eigenvalues on an independent reduced 3-D mesh",
                "reference_type": "in-repository closed-form reference; not external software",
                "implementation": [
                    "Resolve the WSM parameters through QXTI, but construct the reference grid independently in reduced coordinates and evaluate the two eigenvalues from $B_0$ and the norm of $(B_1,B_2,B_3)$ without calling the model module.",
                    "Construct the Gaussian reference DOS directly from those eigenvalues, compare same-size grids to isolate implementation error, and use a separate 41^3 calculation to measure discretization convergence.",
                ],
                "reference_provenance": [
                    "The Hamiltonian form is an anisotropic adaptation of Eq. 11 of McCormick, Kimchi and Trivedi, Phys. Rev. B 95, 075133 (2017).",
                    "The current M0=0.014 and a2=12 Bohr are project-specific adjusted parameters, not a parameter table copied unchanged from that paper.",
                    "The exact 2x2 eigenvalue formula, reciprocal bounds, two-state DOS sum rule, 41^3 reference grid and numerical thresholds define this verification benchmark; they are not published comparison data.",
                ],
                "production_code_changes": [
                    "Add a parameter-aware `default_lattice(params)` to `models/wsm_two_weyl.py`, so a0, a1 and a2 determine both the Hamiltonian and reciprocal cell.",
                    "Change `CustomHamiltonian.default_lattice()` to prefer a callable lattice provider over static `DEFAULT_LATTICE`. Previously the static metadata silently ignored an input-dependent a2 and could integrate the correct Hamiltonian over the wrong 3-D Brillouin zone.",
                    "Add a regression test with a0=6, a1=7 and a2=12 that checks real-space lengths and bounds $[-pi/a_i,pi/a_i]$.",
                ],
                "results": rows,
                "error_methodology": [
                    "QXTI independently builds a shifted Cartesian $N^3$ grid while the reference builds a reduced $[-1/2,1/2)^3$ grid and converts each axis using its own lattice constant.",
                    "At each N, compare the complete DOS arrays with an L2 norm normalized by the independent $41^3$ reference DOS.",
                    "Verify every reciprocal bound against plus/minus pi/a_i and integrate the DOS by the composite trapezoidal rule.",
                    "Fit log(g(E)) versus log(E) over 0.004--0.012 Ha; an isolated linear Weyl cone predicts an exponent near two.",
                ],
                "acceptance_criteria": [
                    "QXTI-to-reference errors decrease at every refinement.",
                    "Final same-grid relative L2 error <= 1e-11.",
                    "Final QXTI-to-41^3-reference relative L2 error <= 1%.",
                    "Final DOS integral differs from two states by <= 0.02.",
                    "Fitted low-energy DOS exponent is between 1.7 and 2.3.",
                ],
                "conclusion": "QXTI's anisotropic 3-D reciprocal cell, weights and DOS converge to an independently generated WSM reference.",
                "limitations": [
                    "The reference is an independent closed-form evaluator, not a second external software package.",
                    "This benchmark does not verify Weyl-node chirality, surface Fermi arcs or optical response.",
                ],
                "artifact": str(args.output),
            }
        )
    print(json.dumps(payload, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
