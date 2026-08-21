from __future__ import annotations

import pytest


pytestmark = pytest.mark.scientific_validation


@pytest.mark.external_validation
def test_haldane_matches_pythtb() -> None:
    pytest.importorskip("pythtb")
    from validation.benchmark_haldane_pythtb import run_benchmark

    results = run_benchmark(random_points=32, kgrid_size=21, chern_mesh_size=21)
    assert all(result.passed for result in results), results


@pytest.mark.external_validation
def test_haldane_dos_converges_on_independent_meshes() -> None:
    pytest.importorskip("pythtb")
    from validation.benchmark_haldane_grid_convergence import run_benchmark

    results = run_benchmark(
        grid_sizes=(11, 21, 41),
        reference_grid=81,
        eta=6.0e-3,
        num_energies=201,
    )
    assert all(result.errors_decrease for result in results), results
    assert all(result.points[-1].qxti_vs_pythtb_relative_l2 < 0.03 for result in results)


@pytest.mark.external_validation
def test_haldane_operators_match_independent_references() -> None:
    pytest.importorskip("pythtb")
    from validation.benchmark_haldane_operators import run_benchmark

    results = run_benchmark(random_points=32)
    assert all(result.passed for result in results), results


@pytest.mark.external_validation
def test_graphene_gapless_contract_and_guard() -> None:
    pytest.importorskip("pythtb")
    from validation.benchmark_graphene_gapless import run_benchmark

    result = run_benchmark(random_points=32, grid_size=12)
    assert result.passed, result


def test_wsm_three_dimensional_grid_and_dos() -> None:
    from validation.benchmark_wsm_grid_3d import run_benchmark

    result = run_benchmark()
    assert result.passed, result


def test_taas_eight_band_assembly_and_symmetries() -> None:
    from validation.benchmark_taas_multiband_8 import run_benchmark

    result = run_benchmark(random_points=16)
    assert result.passed, result


def test_tblg_implementation_and_truncation_convergence() -> None:
    from validation.benchmark_tblg_truncation import run_benchmark

    result = run_benchmark(random_points=4, points_per_segment=9)
    assert result.passed, result
