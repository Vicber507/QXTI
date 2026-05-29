from __future__ import annotations

import numpy as np

from qxti.solvers import RKF45Solver


def _constant_matrix_rhs_factory(amplitude: float):
    def derivative(_t: float, _y: np.ndarray) -> np.ndarray:
        return np.full((2, 2), amplitude, dtype=np.complex128)

    return derivative


def _run_constant_rhs(amplitude: float) -> RKF45Solver:
    solver = RKF45Solver(
        tolerance=1.0e-4,
        h_min=1.0e-12,
        h_max=1.0,
        max_iterations=10000,
        enforce_hermiticity=False,
        enforce_trace=False,
    )
    y0 = np.zeros((2, 2), dtype=np.complex128)
    times, states = solver.solve(
        _constant_matrix_rhs_factory(amplitude),
        0.0,
        1.0,
        y0,
        0.2,
    )

    assert np.isclose(times[-1], 1.0)
    expected = np.full((2, 2), amplitude, dtype=np.complex128)
    np.testing.assert_allclose(states[-1], expected, rtol=1.0e-6, atol=1.0e-10)
    return solver


def test_rkf45_error_budget_scales_with_solution_amplitude() -> None:
    small_solver = _run_constant_rhs(1.0e-6)
    large_solver = _run_constant_rhs(1.0)

    assert small_solver.last_scale < 1.0e-3
    assert small_solver.last_allowed_error < 1.0e-6
    assert large_solver.last_scale > 1.0e-1
    assert large_solver.last_allowed_error > small_solver.last_allowed_error * 1.0e4
    assert bool(small_solver.converged)
    assert bool(large_solver.converged)
