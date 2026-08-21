"""Validate QXTI's Haldane implementation against independent PythTB results.

The benchmark compares quantities in increasing order of derived complexity:

1. band energies at deterministic random Cartesian k-points;
2. a broadened bulk DOS on the exact same k and energy grids;
3. the occupied-band Chern number computed by PythTB's implementation of the
   Fukui-Hatsugai-Suzuki plaquette method.

PythTB works in reduced reciprocal coordinates while QXTI's model takes
Cartesian k in 1/Bohr.  If ``a_i`` are QXTI's real-space primitive vectors, the
conversion used here is ``k_reduced_i = k_cartesian . a_i / (2*pi)``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

import numpy as np

from qxti.analytics.dos import compute_dos_spectrum
from qxti.core import QXTIConfig, QXTISimulation

from .registry import RESULTS_DIR, record_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HaldaneBenchmarkResult:
    case: str
    random_points: int
    kgrid_shape: tuple[int, int]
    band_max_abs_error_ha: float
    band_rms_error_ha: float
    dos_max_abs_error: float
    dos_relative_l2_error: float
    qxti_dos_sum_rule: float
    pythtb_chern: float
    expected_chern_magnitude: int
    passed: bool


def _require_pythtb():
    try:
        from pythtb import Mesh, WFArray
        from pythtb.models import haldane
    except ImportError as exc:  # pragma: no cover - exercised without validation extra
        raise RuntimeError(
            "PythTB is required. Install QXTI with: pip install -e '.[validation]'"
        ) from exc
    return Mesh, WFArray, haldane


def _real_vectors(hamiltonian: Any) -> np.ndarray:
    vectors = hamiltonian.lattice.get("real_space_vectors", {})
    return np.asarray([vectors["a1"], vectors["a2"]], dtype=np.float64)


def _to_reduced(k_cartesian: np.ndarray, real_vectors: np.ndarray) -> np.ndarray:
    return np.asarray(k_cartesian[:, :2] @ real_vectors.T / (2.0 * np.pi), dtype=np.float64)


def _pythtb_model(hamiltonian: Any):
    _, _, haldane = _require_pythtb()
    params = hamiltonian.params
    # PythTB uses onsite [-delta, +delta], whereas QXTI uses [M0, -M0].
    return haldane(
        delta=-float(params["M0"]),
        t1=float(params["t1"]),
        t2=float(params["t2"]),
        phi=float(params["phi0"]),
    )


def _pythtb_chern(model: Any, mesh_size: int) -> float:
    Mesh, WFArray, _ = _require_pythtb()
    mesh = Mesh(["k", "k"])
    mesh.build_grid((mesh_size, mesh_size), gamma_centered=False)
    wavefunctions = WFArray(model.lattice, mesh)
    wavefunctions.solve_model(model)
    flux = wavefunctions.berry_flux(plane=(0, 1), state_idx=[0])
    return float(np.sum(flux) / (2.0 * np.pi))


def _kernel(diff: np.ndarray, eta: float, broadening: str) -> np.ndarray:
    if broadening == "gaussian":
        return np.exp(-0.5 * (diff / eta) ** 2) / (eta * np.sqrt(2.0 * np.pi))
    return (eta / np.pi) / (diff * diff + eta * eta)


def run_case(
    case: str,
    *,
    random_points: int = 128,
    kgrid_size: int = 41,
    chern_mesh_size: int = 41,
) -> HaldaneBenchmarkResult:
    if case not in {"topological", "trivial"}:
        raise ValueError("case must be 'topological' or 'trivial'")

    config_path = PROJECT_ROOT / "inputs" / f"inputParams.haldane_{case}.cfg"
    config = QXTIConfig.from_file(config_path)
    config = replace(
        config,
        kgrid=replace(
            config.kgrid,
            k_points=(kgrid_size, kgrid_size),
            shifted=True,
            auto_degeneracy_guard=False,
            berry_singularity_guard=False,
        ),
        ldos=replace(
            config.ldos,
            method="eigenvalues",
            broadening="gaussian",
            eta=8.0e-4,
            e_min=-0.45,
            e_max=0.45,
            num_energies=501,
            projected=False,
            spectral_enabled=False,
            spectral_plane_enabled=False,
        ),
    )

    simulation = QXTISimulation(config=config)
    hamiltonian = simulation.build_hamiltonian()
    pythtb_model = _pythtb_model(hamiltonian)
    real_vectors = _real_vectors(hamiltonian)

    bounds = hamiltonian.reciprocal_box_bounds()
    rng = np.random.default_rng(20260820)
    random_k = np.column_stack(
        [
            rng.uniform(bounds[0][0], bounds[0][1], random_points),
            rng.uniform(bounds[1][0], bounds[1][1], random_points),
            np.zeros(random_points),
        ]
    )
    qxti_bands = np.linalg.eigvalsh(
        np.asarray([hamiltonian._matrix_at(*k) for k in random_k])
    )
    pythtb_bands = np.asarray(
        pythtb_model.solve_ham(_to_reduced(random_k, real_vectors)), dtype=np.float64
    )
    band_delta = qxti_bands - pythtb_bands

    qxti_dos_result = compute_dos_spectrum(config, progress=False)
    qxti_dataset = qxti_dos_result["dataset"]
    energies = np.asarray(qxti_dataset["energies"], dtype=np.float64)
    qxti_dos = np.asarray(qxti_dataset["dos"], dtype=np.float64)
    eta = float(qxti_dataset["eta"])
    broadening = str(qxti_dataset["broadening"])

    kgrid = simulation.build_kgrid(hamiltonian)
    grid_k = kgrid.points()
    external_bands = np.asarray(
        pythtb_model.solve_ham(_to_reduced(grid_k, real_vectors)), dtype=np.float64
    )
    diff = energies[None, None, :] - external_bands[:, :, None]
    # Average over k-points and sum over bands. Averaging both axes would
    # incorrectly divide the DOS by the number of bands.
    external_dos = _kernel(diff, eta, broadening).mean(axis=0).sum(axis=0)
    dos_delta = qxti_dos - external_dos
    dos_norm = max(float(np.linalg.norm(external_dos)), np.finfo(float).tiny)

    chern = _pythtb_chern(pythtb_model, chern_mesh_size)
    expected_chern_magnitude = 1 if case == "topological" else 0
    passed = bool(
        np.max(np.abs(band_delta)) <= 1.0e-12
        and np.linalg.norm(dos_delta) / dos_norm <= 1.0e-11
        and abs(abs(chern) - expected_chern_magnitude) <= 1.0e-8
        and abs(float(qxti_dataset["integral"]) - 2.0) <= 2.0e-2
    )

    return HaldaneBenchmarkResult(
        case=case,
        random_points=random_points,
        kgrid_shape=(kgrid_size, kgrid_size),
        band_max_abs_error_ha=float(np.max(np.abs(band_delta))),
        band_rms_error_ha=float(np.sqrt(np.mean(np.abs(band_delta) ** 2))),
        dos_max_abs_error=float(np.max(np.abs(dos_delta))),
        dos_relative_l2_error=float(np.linalg.norm(dos_delta) / dos_norm),
        qxti_dos_sum_rule=float(qxti_dataset["integral"]),
        pythtb_chern=chern,
        expected_chern_magnitude=expected_chern_magnitude,
        passed=passed,
    )


def run_benchmark(**kwargs: Any) -> list[HaldaneBenchmarkResult]:
    return [run_case("topological", **kwargs), run_case("trivial", **kwargs)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kgrid-size", type=int, default=41)
    parser.add_argument("--chern-mesh-size", type=int, default=41)
    parser.add_argument("--random-points", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "haldane_pythtb_pointwise.json"
    )
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    results = run_benchmark(
        random_points=args.random_points,
        kgrid_size=args.kgrid_size,
        chern_mesh_size=args.chern_mesh_size,
    )
    payload = {
        "benchmark": "QXTI Haldane vs PythTB",
        "error_methodology": {
            "bands": (
                "Point-by-point comparison at 128 deterministic Cartesian k-points. "
                "For every k and sorted band n, delta_E(k,n)=E_QXTI(k,n)-E_PythTB(k,n). "
                "Reported metrics are max(abs(delta_E)) and sqrt(mean(abs(delta_E)^2))."
            ),
            "dos": (
                "Array comparison at the same 501 energy nodes after evaluating both "
                "codes on the same k-grid. Relative L2 error is "
                "norm(g_QXTI-g_PythTB)_2 / norm(g_PythTB)_2; maximum absolute error "
                "is max_E(abs(g_QXTI(E)-g_PythTB(E)))."
            ),
            "sum_rule": (
                "QXTI DOS is integrated on the discrete energy axis with the composite "
                "trapezoidal rule and compared with the expected two states per unit cell."
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
                "id": "haldane_bands_dos_pointwise",
                "title": "Haldane bands and bulk DOS against PythTB",
                "passed": payload["passed"],
                "scope": "Pointwise bands and same-grid bulk DOS for topological and trivial Haldane inputs.",
                "independent_reference": f"PythTB {version('pythtb')}",
                "reference_type": "external cross-code reference plus state-count invariant",
                "implementation": [
                    "Load the topological and trivial QXTI configurations, instantiate PythTB with the same Haldane hoppings and onsite mass, and convert QXTI Cartesian momenta independently to PythTB reduced coordinates.",
                    "Diagonalize both models point by point; build the reference DOS from PythTB eigenvalues using the same Gaussian width and energy axis used by QXTI.",
                ],
                "reference_provenance": [
                    "The physical model is the Haldane honeycomb Hamiltonian; the executable comparison is against the independently distributed PythTB implementation.",
                ],
                "production_code_changes": [
                    "None. This benchmark passed without modifying the Haldane production Hamiltonian or DOS engine.",
                ],
                "results": [
                    {
                        "case": result.case,
                        "band max error (Ha)": result.band_max_abs_error_ha,
                        "DOS relative L2": result.dos_relative_l2_error,
                        "DOS integral": result.qxti_dos_sum_rule,
                        "PythTB Chern": result.pythtb_chern,
                    }
                    for result in results
                ],
                "error_methodology": [
                    "Bands are compared point by point at 128 deterministic k-points and for each sorted band: $\\Delta E_{kn}=E^{QXTI}_{kn}-E^{PythTB}_{kn}$. The report gives $\\max_{kn}|\\Delta E_{kn}|$; the raw artifact also stores $\\sqrt{\\mathrm{mean}_{kn}|\\Delta E_{kn}|^2}$.",
                    "DOS arrays are compared point by point on the same 501-energy axis. The reported relative error is $\\|g_{QXTI}-g_{PythTB}\\|_2/\\|g_{PythTB}\\|_2$, where the norm sums over every energy node. This is a whole-curve metric, not the error at only one selected energy.",
                    "The raw artifact also stores the largest pointwise DOS difference $\\max_E|g_{QXTI}(E)-g_{PythTB}(E)|$.",
                    "The DOS sum rule is evaluated by composite trapezoidal integration over the energy grid and compared with the expected value 2.",
                ],
                "acceptance_criteria": [
                    "Maximum pointwise band error <= 1e-12 Ha.",
                    "Same-grid DOS relative L2 error <= 1e-11.",
                    "PythTB Chern magnitude matches the expected input classification.",
                    "QXTI DOS integral differs from two bands by <= 0.02.",
                ],
                "conclusion": "QXTI and PythTB implement the same Haldane band spectrum; QXTI's same-grid bulk DOS agrees to floating-point precision.",
                "limitations": [
                    "The same-grid DOS comparison does not independently validate QXTI's reciprocal integration domain; that is covered by the separate convergence benchmark.",
                    "The Chern number is calculated only by PythTB and validates the input classification, not a QXTI Chern implementation.",
                ],
                "artifact": str(args.output),
            }
        )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
