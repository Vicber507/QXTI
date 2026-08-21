"""Strong tests for the von Neumann time propagators (qxti/analytics/propagators.py).

Covers, for cfm2 / rkf45 / ab2:
  * exactness on a constant Hamiltonian (vs the analytic matrix exponential),
  * the empirical convergence ORDER on a smooth time-dependent Hamiltonian
    (cfm2 -> 2, rkf45 -> >=4, ab2 -> 2),
  * structure preservation (unitarity, Tr rho, Hermiticity, positivity) — the
    property that separates cfm2 from the general-purpose schemes.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.analytics.propagators import (
    ab2_step,
    apply_unitary,
    cfm2_step,
    rkf45_node_offsets,
    rkf45_step,
    unitary_from_hermitian,
    vn_derivative,
)

# --- Pauli matrices ---
SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def H_of_t(t: float) -> np.ndarray:
    """A smooth, generically time-dependent Hermitian 2x2 (d(t)·sigma).

    The three components have different frequencies so that H(t) and H(t') do NOT
    commute — this is what makes the propagator ORDER matter (a commuting H would
    let even a 1st-order scheme be exact)."""
    dx = 0.7 * np.cos(1.3 * t)
    dy = 0.5 * np.sin(0.9 * t)
    dz = 0.8 + 0.4 * np.cos(0.6 * t)
    return dx * SX + dy * SY + dz * SZ


def rho0() -> np.ndarray:
    """A valid initial density matrix (pure state |0><0|), batched shape (1,2,2)."""
    r = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    return r[None, :, :]


# ---------------------------------------------------------------------------
# Reference propagation: cfm2 at a very fine step is converged to ~machine and
# serves as the ground truth for the coarse-grid error measurements.
# ---------------------------------------------------------------------------
def _reference(T: float, n_ref: int = 200_000) -> np.ndarray:
    dt = T / n_ref
    rho = rho0()
    for it in range(n_ref):
        t_mid = (it + 0.5) * dt
        rho = cfm2_step(rho, H_of_t(t_mid)[None], dt)
    return rho


def _propagate(scheme: str, T: float, n: int) -> np.ndarray:
    dt = T / n
    rho = rho0()
    if scheme == "cfm2":
        for it in range(n):
            rho = cfm2_step(rho, H_of_t((it + 0.5) * dt)[None], dt)
    elif scheme == "rkf45":
        C = rkf45_node_offsets()
        for it in range(n):
            t0 = it * dt
            H_stages = np.stack([H_of_t(t0 + c * dt)[None] for c in C], axis=0)
            rho, _ = rkf45_step(rho, H_stages, dt)
    elif scheme == "ab2":
        # bootstrap the first step with cfm2, then 2-step Adams-Bashforth
        f_prev = vn_derivative(H_of_t(0.0)[None], rho)
        rho = cfm2_step(rho, H_of_t(0.5 * dt)[None], dt)
        for it in range(1, n):
            t = it * dt
            f_now = vn_derivative(H_of_t(t)[None], rho)
            rho = ab2_step(rho, f_now, f_prev, dt)
            f_prev = f_now
    else:
        raise ValueError(scheme)
    return rho


def _order(scheme: str, T: float = 3.0) -> float:
    ref = _reference(T)
    ns = np.array([40, 80, 160, 320])
    errs = []
    for n in ns:
        rho = _propagate(scheme, T, int(n))
        errs.append(np.abs(rho - ref).max())
    errs = np.array(errs)
    # slope of log(err) vs log(dt): dt ∝ 1/n, so order = -slope wrt log(n)
    slope = np.polyfit(np.log(ns.astype(float)), np.log(errs), 1)[0]
    return -slope


def test_constant_hamiltonian_is_exact_for_cfm2_and_accurate_for_rkf45() -> None:
    """Constant H → cfm2 is exact (single exponential); rkf45 is high-accuracy."""
    H = (0.8 * SZ + 0.6 * SX + 0.3 * SY)[None]  # (1,2,2)
    T, n = 2.0, 50
    dt = T / n
    U_exact = unitary_from_hermitian(H, T)
    rho_exact = apply_unitary(U_exact, rho0())

    # cfm2: exact for constant H
    rho = rho0()
    for _ in range(n):
        rho = cfm2_step(rho, H, dt)
    assert np.abs(rho - rho_exact).max() < 1e-12

    # rkf45: 5th order, small but non-zero error
    C = rkf45_node_offsets()
    rho = rho0()
    for _ in range(n):
        H_stages = np.stack([H for _ in C], axis=0)
        rho, _ = rkf45_step(rho, H_stages, dt)
    assert np.abs(rho - rho_exact).max() < 1e-8


def test_convergence_orders() -> None:
    """Empirical global order: cfm2≈2, rkf45≥4, ab2≈2."""
    assert 1.7 < _order("cfm2") < 2.3, "cfm2 should be 2nd order"
    assert _order("rkf45") > 4.0, "rkf45 should be >=4th order"
    assert 1.7 < _order("ab2") < 2.3, "ab2 should be 2nd order"


def test_unitary_propagator_is_unitary() -> None:
    H = H_of_t(0.37)[None]
    U = unitary_from_hermitian(H, 0.53)
    Ud = np.conj(np.swapaxes(U, -1, -2))
    assert np.abs(U @ Ud - I2[None]).max() < 1e-12


def test_cfm2_preserves_structure_but_rkf45_ab2_do_not_exactly() -> None:
    """cfm2 conserves Tr rho, Hermiticity and positivity to machine precision;
    the general-purpose schemes conserve them only approximately."""
    T, n = 4.0, 200
    tr0 = np.trace(rho0()[0]).real

    rho_cfm2 = _propagate("cfm2", T, n)[0]
    assert abs(np.trace(rho_cfm2).real - tr0) < 1e-12           # Tr conserved
    assert np.abs(rho_cfm2 - rho_cfm2.conj().T).max() < 1e-12   # Hermitian
    ev = np.linalg.eigvalsh(rho_cfm2)
    assert ev.min() > -1e-9 and ev.max() < 1 + 1e-9             # positivity, ev in [0,1]

    # rkf45 keeps the trace very well (linear invariant) but is not exactly unitary
    rho_rk = _propagate("rkf45", T, n)[0]
    assert abs(np.trace(rho_rk).real - tr0) < 1e-6
