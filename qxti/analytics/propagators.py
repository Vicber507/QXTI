"""Time propagators for the velocity-gauge von Neumann equation.

The non-perturbative ``tddm`` engine integrates, independently for every k-point,

    dρ/dt = −i [ H(k + A(t)), ρ ]   ( + relaxation, handled by the caller )

This module provides the *coherent* one-step propagators used to advance ρ over a
grid interval ``[t, t+dt]``.  They are all VECTORISED over a block of k-points:
``rho`` has shape ``(Kb, nb, nb)`` and every ``H`` argument ``(Kb, nb, nb)``.

Three schemes, selectable from the input file (``[cmd] tddm_propagator``):

* ``"cfm2"``  — 2nd-order commutator-free Magnus, a.k.a. the **exponential
  midpoint rule**: freeze ``H`` at the interval midpoint ``t+dt/2`` and apply the
  exact matrix exponential.  Unitary by construction; the QXTI default.  This is
  the classic geometric integrator for the Schrödinger/Liouville problem
  (Blanes, Casas, Oteo, Ros, *Phys. Rep.* 470 (2009) 151; Hochbruck & Ostermann,
  *Acta Numer.* 19 (2010) 209; called "exponential midpoint rule / 2nd-order
  Magnus" in Gómez-Pueyo, Marques, Rubio, Castro, *JCTC* 14 (2018) 3040).

* ``"rkf45"`` — one fixed-step Runge–Kutta–Fehlberg 4(5) step on the grid (the
  5th-order solution).  General-purpose; NOT unitarity-preserving.

* ``"ab2"``   — 2-step Adams–Bashforth (explicit linear multistep, 2nd order),
  bootstrapped by one cfm2 step.  NOT unitarity-preserving.

Only ``cfm2`` is structure-preserving (unitary → conserves ``Tr ρ`` and keeps the
eigenvalues of ρ in ``[0, 1]``); ``rkf45``/``ab2`` are offered as reference/general
alternatives.  See ``docs/INTEGRATORS.md``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

PROPAGATORS = ("cfm2", "rkf45", "ab2")

# Runge–Kutta–Fehlberg 4(5) Butcher tableau (Fehlberg 1969).  ``C`` are the stage
# node offsets within the step (used to sample H at t + C[j]*dt).
RKF45_C = np.array([0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0])
RKF45_A = (
    (),
    (1.0 / 4.0,),
    (3.0 / 32.0, 9.0 / 32.0),
    (1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0),
    (439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0),
    (-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0),
)
# 5th-order weights (the propagated solution).
RKF45_B5 = np.array([16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0])
# 4th-order weights (embedded, for the error estimate |y5 - y4|).
RKF45_B4 = np.array([25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0])


def unitary_from_hermitian(H: ComplexArray, dt: float) -> ComplexArray:
    """Return the exact propagator ``U = exp(-i H dt)`` for Hermitian ``H``.

    Vectorised: ``H`` is ``(..., nb, nb)`` Hermitian, ``U`` the same shape.
    Uses the eigendecomposition, so ``U`` is unitary to machine precision.
    """
    E, V = np.linalg.eigh(H)
    phase = np.exp(-1.0j * E * dt)                      # (..., nb)
    # U = V diag(phase) V†
    return np.einsum("...in,...n,...jn->...ij", V, phase, V.conj())


def apply_unitary(U: ComplexArray, rho: ComplexArray) -> ComplexArray:
    """Return ``U ρ U†`` (batched over the leading axis)."""
    Ud = np.conj(np.swapaxes(U, -1, -2))
    return U @ rho @ Ud


def vn_derivative(H: ComplexArray, rho: ComplexArray) -> ComplexArray:
    """Right-hand side of the coherent von Neumann equation, ``-i [H, ρ]``."""
    return -1.0j * (H @ rho - rho @ H)


def cfm2_step(rho: ComplexArray, H_mid: ComplexArray, dt: float) -> ComplexArray:
    """One exponential-midpoint (2nd-order commutator-free Magnus) step.

    ``H_mid`` is ``H`` evaluated at the interval midpoint ``t + dt/2``.
    Unitary → conserves ``Tr ρ`` and positivity.
    """
    U = unitary_from_hermitian(H_mid, dt)
    return apply_unitary(U, rho)


def rkf45_step(rho: ComplexArray, H_stages: ComplexArray, dt: float):
    """One fixed-step RKF45 step of ``dρ/dt = -i[H(t), ρ]``.

    ``H_stages`` is ``H`` at the six Fehlberg node times, shape ``(6, Kb, nb, nb)``
    (i.e. ``H(t + RKF45_C[j] * dt)``).  Returns ``(rho_next, err)`` where ``err`` is
    the embedded 4(5) local-error estimate ``max|y5 - y4|`` (for diagnostics).
    """
    k = [None] * 6
    k[0] = vn_derivative(H_stages[0], rho)
    for j in range(1, 6):
        acc = rho.copy()
        for i, aij in enumerate(RKF45_A[j]):
            if aij != 0.0:
                acc = acc + (dt * aij) * k[i]
        k[j] = vn_derivative(H_stages[j], acc)
    y5 = rho.copy()
    y4 = rho.copy()
    for j in range(6):
        if RKF45_B5[j] != 0.0:
            y5 = y5 + (dt * RKF45_B5[j]) * k[j]
        if RKF45_B4[j] != 0.0:
            y4 = y4 + (dt * RKF45_B4[j]) * k[j]
    err = float(np.abs(y5 - y4).max()) if y5.size else 0.0
    return y5, err


def ab2_step(rho: ComplexArray, f_now: ComplexArray, f_prev: ComplexArray, dt: float) -> ComplexArray:
    """One 2-step Adams–Bashforth step.

    ``f_now = -i[H(t), ρ]`` and ``f_prev = -i[H(t-dt), ρ_{prev}]`` are the RHS at the
    current and previous grid points.  ``ρ_{n+1} = ρ_n + dt(3/2 f_n − 1/2 f_{n-1})``.
    """
    return rho + dt * (1.5 * f_now - 0.5 * f_prev)


def rkf45_node_offsets() -> FloatArray:
    """Stage node offsets (fractions of dt) at which ``H`` must be sampled for RKF45."""
    return RKF45_C.copy()
