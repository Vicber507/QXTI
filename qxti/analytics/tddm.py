"""TDDM — full non-perturbative time-domain density-matrix engine (velocity gauge).

This is the third response engine, alongside the two perturbative ones:

- ``pfddm`` (perturbative frequency-domain DM)  -> :mod:`qxti.analytics.theory_response`
- ``ptddm`` (perturbative time-domain DM)        -> :class:`qxti.response.cmd.CMD`
- ``tddm``  (this)                               -> full non-perturbative solve

Unlike the perturbative engines (which expand the density matrix order-by-order in
the field), TDDM integrates the FULL single-particle density matrix in time with
the exact field, so it is valid *outside* the perturbative regime.

Physics — velocity gauge, per-k INDEPENDENT (Peierls substitution ``k -> k+A(t)``):

    dρ_k/dt = −i [ H(k+A(t)), ρ_k ] − R[ρ_k ; t]
    ρ_k(0)  = V(k) diag(f_n(k)) V(k)†                    (occupied projector, A(t0)=0)
    J(t)    = −Σ_k w_k Tr[ v(k+A(t)) ρ_k(t) ] − J_DC,    v = ∂H/∂k|_{k+A(t)}

For a tight-binding model the Peierls substitution already contains the full band
structure, so there is NO extra diamagnetic or anomalous-velocity term to add.
Because each k evolves independently (no covariant k-gradient), the solve is
embarrassingly parallel and streamable over k-blocks with NO halo (lighter than the
mesh engine).

Correctness essentials (see docs/vault ``Concept - Response Engines``):
  * the current velocity is evaluated at the SHIFTED momentum ``k+A(t)`` and the
    equilibrium (A=0) current ``J_DC`` is subtracted (removes the ω=0 offset);
  * relaxation (T1/T2) and the intra/inter split are done in the INSTANTANEOUS
    eigenbasis of ``H(k+A(t))`` (Houston basis), relaxing toward the *actual*
    occupations ``f_n(k+A(t))`` — not toward zero;
  * integrator: exponential-MIDPOINT in the instantaneous eigenbasis (2nd order in
    dt, exact for the fast interband phase, unitary, conserves Tr ρ and
    hermiticity), Strang-split with the relaxation half-steps.

χ⁽ˢ⁾ for ``-xtp`` is extracted by field-amplitude scaling + polynomial fit.
"""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from qxti.core.config import QXTIConfig
from qxti.analytics.propagators import (
    ab2_step,
    apply_unitary,
    rkf45_node_offsets,
    rkf45_step,
    unitary_from_hermitian,
    vn_derivative,
)
from qxti.analytics.theory_response import (
    build_k_integration_weights,
    _model_h_batch,
    _mesh_parallel_settings,
    _resolve_distribution,
)
import qxti.utils.memory as _mem
from qxti.utils.progress import AtomicCounter as _AtomicCounter, LiveProgress as _LiveProgress

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


# ---------------------------------------------------------------------------
# Live progress — a single ETA line refreshed every few seconds
# ---------------------------------------------------------------------------
# The solve advances one *time step* at a time inside each k-block, so progress is
# tracked in ``k·time-step`` work units (``nk * Nt`` total) instead of per whole
# block.  A block is one opaque ~minutes-long call, so per-block reporting only
# lights up at the very end; counting time-steps lets the ETA appear within a
# couple of seconds and refresh continuously — the SAME for every model and for
# the thread / process / serial paths.  The rendering itself is the shared
# :class:`qxti.utils.progress.LiveProgress`, reused by every response engine.
_PROGRESS_REPORT_EVERY = 16          # flush the step counter this often (cheap)
_PROGRESS_INTERVAL_S = 2.0           # refresh the printed line at most this fast

# Below this band count the per-time-step numpy work is on tiny (Kb, nb, nb) arrays
# dominated by Python glue + the GIL, so a ThreadPool stalls at ~1 core; use a
# ProcessPool instead (true parallelism, identical result).  At/above it the numpy
# steps release the GIL long enough that threads scale and avoid per-process
# spawn/pickle overhead.  Measured crossover is small (nb=2 clearly needs processes).
_TDDM_THREAD_MIN_NB = 16


# Set in every ProcessPool worker by the pool initializer: a ``(counter, lock)``
# pair of Manager proxies the worker bumps as it advances through time steps.
_WORKER_PROGRESS: tuple | None = None


def _init_tddm_worker(counter, lock) -> None:
    global _WORKER_PROGRESS
    _WORKER_PROGRESS = (counter, lock)


# ---------------------------------------------------------------------------
# Vectorized Hamiltonian / velocity evaluation at arbitrary (shifted) k-points
# ---------------------------------------------------------------------------
def _make_h_evaluator(hamiltonian: Any) -> Callable[[FloatArray], ComplexArray]:
    """Return ``Hf(kpts (M,3)) -> (M, nb, nb)`` for arbitrary k.

    Priority:
      1. the model's explicit ``H_batch`` (fastest, trusted);
      2. AUTO-vectorization: if the model's scalar ``H`` happens to be numpy-native
         (accepts array k), use it — but only after VERIFYING it reproduces the
         per-k result to machine precision, so the answer is guaranteed identical;
      3. the per-k Python loop (safe, slow — for models that force scalar k).
    """
    nb = int(hamiltonian.basis_size)
    matrix_at = hamiltonian._matrix_at

    h_batch = _model_h_batch(hamiltonian)
    if h_batch is not None:
        return lambda kp: np.asarray(h_batch(np.asarray(kp, dtype=np.float64)), dtype=np.complex128)

    auto = _auto_vectorized_evaluator(hamiltonian, nb, matrix_at)
    if auto is not None:
        return auto

    def _fallback(kp: FloatArray) -> ComplexArray:
        kp = np.asarray(kp, dtype=np.float64)
        out = np.empty((kp.shape[0], nb, nb), dtype=np.complex128)
        for i in range(kp.shape[0]):
            out[i] = matrix_at(float(kp[i, 0]), float(kp[i, 1]), float(kp[i, 2]))
        return out

    return _fallback


def _auto_vectorized_evaluator(hamiltonian: Any, nb: int, matrix_at: Callable):
    """If the model's scalar ``H`` accepts array k AND reproduces the per-k result
    bit-for-bit, return a vectorized evaluator; else ``None`` (caller falls back).

    This lets ANY numpy-native model be evaluated in one shot with no per-model
    code, and can NEVER change the result (it is verified against the scalar path).
    Models that force ``float(kx)`` / build ``np.array([kx,ky])`` simply fail the
    probe and fall back safely.
    """
    fn = getattr(hamiltonian, "user_function", None)
    if not callable(fn):
        return None
    params = dict(getattr(hamiltonian, "params", {}) or {})

    def _to_batch(out, n):
        out = np.asarray(out)
        if out.shape == (n, nb, nb):
            return out.astype(np.complex128)
        if out.shape == (nb, nb, n):          # some models return (nb,nb,N)
            return np.moveaxis(out, 2, 0).astype(np.complex128)
        return None

    def _vec(kp):
        kp = np.asarray(kp, dtype=np.float64)
        b = _to_batch(fn(kp[:, 0], kp[:, 1], kp[:, 2], params), kp.shape[0])
        if b is None:
            raise ValueError("model H did not vectorize to (N,nb,nb)")
        return b

    # probe on a few generic points and require machine-precision agreement.
    probe = np.array([[0.137, 0.241, 0.0], [-0.313, 0.052, 0.11],
                      [0.42, -0.19, -0.07], [0.011, 0.33, 0.021]], dtype=np.float64)
    try:
        vb = _vec(probe)
        ref = np.stack([matrix_at(float(probe[i, 0]), float(probe[i, 1]), float(probe[i, 2]))
                        for i in range(probe.shape[0])])
        if vb.shape == ref.shape and np.allclose(vb, ref, rtol=1e-11, atol=1e-13):
            return _vec
    except Exception:
        return None
    return None


def _velocity_batch(Hf: Callable, kA: FloatArray, dim: int, dk: float) -> list[ComplexArray]:
    """Central-difference velocity v_a = ∂H/∂k_a at the (shifted) points ``kA``."""
    vels: list[ComplexArray] = []
    for a in range(dim):
        step = np.zeros(3, dtype=np.float64)
        step[a] = dk
        vels.append((Hf(kA + step) - Hf(kA - step)) / (2.0 * dk))
    return vels


def _vector_potential_from_field(E_t: FloatArray, dt: float) -> FloatArray:
    """DEPRECATED — trapezoidal A(t) = −∫_{t0}^t E dt' (O(dt²) accurate), A(t0)=0.

    The tddm engine NO LONGER uses this: it now drives with the ANALYTIC vector
    potential ``laser_system.vector_potential(t)`` (consistent with E=−dA/dt after the
    laser envelope-derivative fix, and with no quadrature error — required for the
    high-order propagators).  Kept only for a few standalone tools that still import
    it; do not use in new code.  See ``docs/INTEGRATORS.md``.
    """
    A_t = np.zeros_like(E_t)
    A_t[1:] = -np.cumsum(0.5 * (E_t[1:] + E_t[:-1]) * dt, axis=0)
    return A_t


# ---------------------------------------------------------------------------
# Core per-block time propagation (velocity gauge, exponential-MIDPOINT + Strang)
# ---------------------------------------------------------------------------
def _propagate_block(
    Hf: Callable,
    kpts_block: FloatArray,
    w_block: FloatArray,
    A_t: FloatArray,          # (Nt, 3) vector potential at the grid times
    dt: float,
    dim: int,
    dk_vel: float,
    distribution,
    mu: float,
    temperature: float,
    gamma_pop: float,         # 1/T1 (0 => no population relaxation)
    gamma_coh: float,         # 1/T2 (0 => no dephasing)
    progress_cb: Callable[[int], None] | None = None,
    pop_stride: int = 0,      # >0 => record band populations n(k,t) every pop_stride steps
    propagator: str = "cfm2", # "cfm2" (exp-midpoint, default) | "rkf45" | "ab2"
    A_mid: FloatArray | None = None,     # (Nt,3) A(t+dt/2), analytic (cfm2 midpoint / relax basis)
    A_stages: FloatArray | None = None,  # (Nt,6,3) A at RKF45 stage nodes (only for rkf45)
) -> tuple:
    """Integrate ρ_k(t) for one block of k-points and return the block's partial
    current contribution.

    Returns ``(J_total (Nt, 3), J_intra (Nt, 3), J_dc (3,), pops)`` — currents are
    already summed over the block's k-points with the BZ weights; ``pops`` is the
    NON-perturbative band populations ``n(k, t)`` for THIS block (kept per-k, not
    summed) sampled every ``pop_stride`` grid times, shape ``(n_frames, Kb, nb)`` in
    the instantaneous (Houston) eigenbasis, or ``None`` when ``pop_stride <= 0``.
    """
    Nt = int(A_t.shape[0])
    Kb = int(kpts_block.shape[0])
    nb = int(Hf(kpts_block[:1]).shape[-1])
    diag = np.arange(nb)

    # --- initial equilibrium ρ0 in the orbital basis (A=0) ---
    H0 = Hf(kpts_block)                                   # (Kb, nb, nb)
    E0, V0 = np.linalg.eigh(H0)                           # (Kb,nb), (Kb,nb,nb)
    f0 = np.asarray(distribution(E0, mu, temperature), dtype=np.float64)
    if f0.ndim == 0:
        f0 = np.broadcast_to(f0, E0.shape).copy()
    # ρ0 = V0 diag(f0) V0†
    rho = np.einsum("kin,kn,kjn->kij", V0, f0.astype(np.complex128), V0.conj())

    # --- equilibrium (DC) current: J_DC = −Σ w Tr[v(k) ρ0] ---
    v0 = _velocity_batch(Hf, kpts_block, dim, dk_vel)
    J_dc = np.zeros(3, dtype=np.float64)
    for a in range(dim):
        tr = np.einsum("kij,kji->k", v0[a], rho).real
        J_dc[a] = -np.sum(w_block * tr)

    J_total = np.zeros((Nt, 3), dtype=np.float64)
    J_intra = np.zeros((Nt, 3), dtype=np.float64)

    relax = (gamma_pop > 0.0) or (gamma_coh > 0.0)
    dpop_half = np.exp(-0.5 * dt * gamma_pop) if gamma_pop > 0.0 else 1.0
    dcoh_half = np.exp(-0.5 * dt * gamma_coh) if gamma_coh > 0.0 else 1.0

    # A(t) at interval MIDPOINTS -> exponential-midpoint propagation (2nd order in
    # dt: evaluating H at t+dt/2 cancels the leading local error, important for the
    # amplitude of the higher harmonics).  A_mid is passed in ANALYTIC (the exact
    # vector potential at t+dt/2); the averaging below is a legacy fallback only.
    if A_mid is None:
        A_mid = A_t.copy()
        A_mid[:-1] = 0.5 * (A_t[:-1] + A_t[1:])
    prop = str(propagator).lower()
    if prop == "rkf45" and A_stages is None:
        raise ValueError("rkf45 propagator requires A_stages (A at the Fehlberg nodes).")
    f_prev = None            # AB2 history: coherent RHS at the previous grid point

    # progress accounting: report work in k·time-step units (Kb per step) so the
    # caller can aggregate a global fraction across all blocks/workers.
    pending_steps = 0
    pops = [] if pop_stride and pop_stride > 0 else None

    for it in range(Nt):
        # ---- current at this grid time t_it: TOTAL is exact (no eigh); the
        #      intra/inter split uses the midpoint eigenbasis (O(dt) on the split
        #      only, the total current is unaffected). ----
        kA = kpts_block + A_t[it][None, :]               # shifted momentum (grid time)
        vA = _velocity_batch(Hf, kA, dim, dk_vel)
        kAm = kpts_block + A_mid[it][None, :]            # shifted momentum (midpoint)
        Em, Vm = np.linalg.eigh(Hf(kAm))                 # instantaneous eigenbasis @ midpoint
        Vmd = Vm.conj()
        rho_diag = np.einsum("kin,kij,kjn->kn", Vmd, rho, Vm).real   # populations
        if pops is not None and (it % pop_stride == 0):
            pops.append(rho_diag.astype(np.float32).copy())          # n(k,t) this frame
        for a in range(dim):
            tr = np.einsum("kij,kji->k", vA[a], rho).real            # Tr[v_a ρ]
            J_total[it, a] = -np.sum(w_block * tr)
            vband = np.einsum("kin,kij,kjn->kn", Vmd, vA[a], Vm).real  # group velocity
            J_intra[it, a] = -np.sum(w_block * np.sum(vband * rho_diag, axis=1))

        # ---- coherent advance of ρ over [t_it, t_it+dt] with the selected propagator,
        #      Strang-split with the T1/T2 relaxation half-steps (in the midpoint basis) ----
        if relax:
            f_inst = np.asarray(distribution(Em, mu, temperature), dtype=np.float64)
            if f_inst.ndim == 0:
                f_inst = np.broadcast_to(f_inst, Em.shape).copy()

        if prop == "cfm2":
            # 2nd-order commutator-free Magnus (exponential midpoint), in the Vm basis:
            rt = np.einsum("kin,kij,kjm->knm", Vmd, rho, Vm)   # ρ̃ = Vm† ρ Vm
            if relax:
                _relax_half_(rt, f_inst, dpop_half, dcoh_half, diag)
            ph = np.exp(-1j * Em * dt)                         # exp(−i E_mid dt)
            rt *= ph[:, :, None] * ph.conj()[:, None, :]
            if relax:
                _relax_half_(rt, f_inst, dpop_half, dcoh_half, diag)
            rho = np.einsum("kin,knm,kjm->kij", Vm, rt, Vmd)   # back to orbital basis
        else:
            # rkf45 / ab2: coherent step in the ORBITAL basis; relaxation is Strang-split
            # around it (half-step, in the midpoint eigenbasis) to match cfm2's structure.
            if relax:
                rho = _relax_half_orbital(rho, Vm, Vmd, f_inst, dpop_half, dcoh_half, diag)
            if prop == "rkf45":
                H_st = np.stack(
                    [np.asarray(Hf(kpts_block + A_stages[it, j][None, :]))
                     for j in range(A_stages.shape[1])], axis=0)   # (6, Kb, nb, nb)
                rho, _ = rkf45_step(rho, H_st, dt)
            elif prop == "ab2":
                Hg = np.asarray(Hf(kA))                           # H at the grid time t_it
                f_now = vn_derivative(Hg, rho)                    # -i[H(t_it), ρ]
                if f_prev is None:                                # bootstrap: one cfm2 step
                    rho = apply_unitary(unitary_from_hermitian(np.asarray(Hf(kAm)), dt), rho)
                else:
                    rho = ab2_step(rho, f_now, f_prev, dt)
                f_prev = f_now
            else:
                raise ValueError(f"Unknown tddm propagator '{propagator}'.")
            if relax:
                rho = _relax_half_orbital(rho, Vm, Vmd, f_inst, dpop_half, dcoh_half, diag)

        if progress_cb is not None:
            pending_steps += 1
            if pending_steps >= _PROGRESS_REPORT_EVERY:
                progress_cb(Kb * pending_steps)
                pending_steps = 0

    if progress_cb is not None and pending_steps:
        progress_cb(Kb * pending_steps)

    pops_arr = np.stack(pops, axis=0) if pops else None   # (n_frames, Kb, nb) or None
    return J_total, J_intra, J_dc, pops_arr


def _relax_half_(rt: ComplexArray, f_inst: FloatArray, dpop_half, dcoh_half, diag) -> None:
    """Apply a half-step of T1/T2 relaxation IN PLACE, in the eigenbasis.

    Populations relax toward ``f_inst`` (T1), coherences toward 0 (T2).
    """
    d = rt[..., diag, diag].real                          # (Kb, nb) populations
    new_d = f_inst + (d - f_inst) * dpop_half             # T1 half-step
    if dcoh_half != 1.0:
        rt *= dcoh_half                                   # T2 on all elements...
    rt[..., diag, diag] = new_d                           # ...then overwrite diagonal


def _relax_half_orbital(rho, Vm, Vmd, f_inst, dpop_half, dcoh_half, diag):
    """Return ρ after a T1/T2 relaxation half-step, transforming to the midpoint
    eigenbasis and back (used by the rkf45/ab2 propagators, whose coherent step
    lives in the orbital basis)."""
    diag_idx = np.arange(rho.shape[-1])
    rt = np.einsum("kin,kij,kjm->knm", Vmd, rho, Vm)
    _relax_half_(rt, f_inst, dpop_half, dcoh_half, diag_idx)
    return np.einsum("kin,knm,kjm->kij", Vm, rt, Vmd)


def _h_is_vectorized(hamiltonian: Any) -> bool:
    """True if H can be evaluated for many k in one numpy call (H_batch or a
    verified auto-vectorized scalar H) -> a ThreadPool suffices.  False for
    scalar-only models (per-k Python loop) -> a ProcessPool gives real speedup."""
    if _model_h_batch(hamiltonian) is not None:
        return True
    return _auto_vectorized_evaluator(hamiltonian, int(hamiltonian.basis_size),
                                      hamiltonian._matrix_at) is not None


def _tddm_block_worker(payload: tuple):
    """Process-pool worker: rebuild the model's H from its source file (picklable),
    then propagate one k-block.  Returns ``(J_total, J_intra, J_dc)`` for the block.

    Runs in a SEPARATE process, so the per-k Python H-evaluation of scalar-only
    models parallelises for real (no GIL) — automatic speedup for ANY model, with
    no per-model vectorization, and the SAME result (blocks are independent).
    """
    import importlib.util
    import numpy as np
    from qxti.analytics.tddm import _propagate_block
    from qxti.analytics.theory_response import _resolve_distribution

    (source_file, function_name, params, nb, kblock, wblock, A_t, dt, dim, dk_vel,
     dist_name, mu, temp, gpop, gcoh, pop_stride, propagator, A_mid, A_stages) = payload

    spec = importlib.util.spec_from_file_location("_tddm_model", source_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Hfn = getattr(mod, function_name)
    hb = getattr(mod, "H_batch", None)
    if callable(hb):
        def Hf(kp):
            return np.asarray(hb(np.asarray(kp, dtype=float), params), dtype=np.complex128)
    else:
        def Hf(kp):
            kp = np.asarray(kp, dtype=float)
            out = np.empty((kp.shape[0], nb, nb), dtype=np.complex128)
            for i in range(kp.shape[0]):
                out[i] = Hfn(float(kp[i, 0]), float(kp[i, 1]), float(kp[i, 2]), params)
            return out

    dist = _resolve_distribution(dist_name)

    # Bump the shared (Manager) counter as this block advances through time steps,
    # so the parent can render a live ETA across all worker processes.
    prog = _WORKER_PROGRESS
    if prog is not None:
        counter, lock = prog

        def progress_cb(inc: int) -> None:
            with lock:
                counter.value += int(inc)
    else:
        progress_cb = None

    return _propagate_block(Hf, kblock, wblock, A_t, dt, dim, dk_vel,
                            dist, mu, temp, gpop, gcoh, progress_cb=progress_cb,
                            pop_stride=pop_stride, propagator=propagator,
                            A_mid=A_mid, A_stages=A_stages)


# ---------------------------------------------------------------------------
# Time-domain current from the full solve (streamed over k-blocks)
# ---------------------------------------------------------------------------
def _tddm_current_time(
    hamiltonian: Any,
    kgrid: Any,
    weights: FloatArray,
    A_t: FloatArray,
    dt: float,
    dim: int,
    ccfg: Any,
    *,
    n_workers: int | None,
    reserve_gb: float,
    progress: bool,
    progress_label: str = "tddm",
    pop_stride: int = 0,
    propagator: str = "cfm2",
    A_mid: FloatArray | None = None,
    A_stages: FloatArray | None = None,
) -> tuple:
    """Return ``(J_total (Nt,3), J_intra (Nt,3), populations)`` from the solve.

    ``populations`` is ``None`` unless ``pop_stride > 0``, in which case it is the
    band occupations ``n(k, t)`` sampled every ``pop_stride`` grid times, shape
    ``(n_frames, nk, nb)`` (per-k, NOT summed) — the density-matrix diagonal.

    Streams over contiguous k-blocks (no halo — the velocity gauge has no k
    coupling), running blocks concurrently on a ThreadPool while keeping
    ``reserve_gb`` free.  The result is independent of ``n_workers``.
    """
    from qxti.analytics.mesh_response import default_worker_count

    Hf = _make_h_evaluator(hamiltonian)
    k_points = np.asarray(kgrid.points(), dtype=np.float64)
    nk = int(k_points.shape[0])
    nb = int(hamiltonian.basis_size)
    Nt = int(A_t.shape[0])
    distribution = _resolve_distribution(ccfg.distribution)
    mu = float(ccfg.fermi_level)
    temperature = float(ccfg.temperature)
    dk_vel = 1.0e-4
    gamma_pop = 0.0 if ccfg.population_time <= 0 or not np.isfinite(ccfg.population_time) else 1.0 / float(ccfg.population_time)
    gamma_coh = 0.0 if ccfg.coherence_time <= 0 or not np.isfinite(ccfg.coherence_time) else 1.0 / float(ccfg.coherence_time)

    workers = int(n_workers) if (n_workers and n_workers > 0) else default_worker_count()
    vectorized = _h_is_vectorized(hamiltonian)
    # Choose the pool by REAL per-step cost, not by whether the model exposes
    # H_batch.  Each time step runs numpy on tiny (Kb, nb, nb) arrays; for small nb
    # the per-step work is dominated by Python glue + the GIL, so a ThreadPool
    # stalls at ~1 core even when split into many blocks (measured: Haldane nb=2 ->
    # ~1.3 cores on threads vs ~all cores on processes, identical result).  Use a
    # ProcessPool for scalar models AND small-nb models; keep threads only for
    # large-nb vectorized models whose numpy steps release the GIL long enough.
    use_procpool = (workers > 1 and nk >= 2
                    and (not vectorized or nb < _TDDM_THREAD_MIN_NB))
    # Per-k live footprint: a handful of (nb,nb) complex arrays + eigh workspace.
    bytes_per_k = nb * nb * 16 * 12
    # Split into >= workers blocks (subject to the per-worker RAM budget) so EVERY
    # worker runs -- pick_block_count sized by RAM only, so a grid that fit in RAM
    # became ONE block and ran serial regardless of the pool (the parallelism bug).
    per_block, n_blocks = _mem.split_units(
        nk, bytes_per_k, workers, reserve_gb=reserve_gb, min_units_per_block=1)
    # split k into contiguous blocks of <= per_block points
    starts = list(range(0, nk, per_block))
    blocks = [(s, min(s + per_block, nk)) for s in starts]

    mode = "processes" if (use_procpool and len(blocks) > 1) else "threads"
    if progress:
        print(f"[{progress_label}] full non-perturbative solve: {nk} k-points, {Nt} time steps, "
              f"{len(blocks)} block(s) x <= {per_block} k, up to {workers} {mode} "
              f"(H {'vectorized' if vectorized else 'per-k'}), >={reserve_gb:g} GB RAM kept free. "
              f"Live progress with ETA below (refreshes every ~{int(_PROGRESS_INTERVAL_S)}s).", flush=True)
    _mem.ensure_headroom(min(per_block, nk) * bytes_per_k * max(1, workers),
                         reserve_gb=reserve_gb, label=f"{progress_label} block")

    total_work = nk * Nt                        # k·time-step work units
    live = _LiveProgress(progress_label, total_work) if progress else None
    J_total = np.zeros((Nt, 3), dtype=np.float64)
    J_intra = np.zeros((Nt, 3), dtype=np.float64)
    J_dc = np.zeros(3, dtype=np.float64)

    pop_blocks: list = []          # per-block populations, in block (k) order

    def _run(block, progress_cb=None):
        lo, hi = block
        return _propagate_block(
            Hf, k_points[lo:hi], weights[lo:hi], A_t, dt, dim, dk_vel,
            distribution, mu, temperature, gamma_pop, gamma_coh,
            progress_cb=progress_cb, pop_stride=pop_stride,
            propagator=propagator, A_mid=A_mid, A_stages=A_stages)

    def _accumulate(res):
        jt, ji, jdc, pops = res
        J_total[...] += jt
        J_intra[...] += ji
        J_dc[...] += jdc
        if pops is not None:
            pop_blocks.append(pops)

    def _drain(futures, get_done):
        """Poll ``get_done()`` while ``futures`` run, refreshing the ETA line."""
        from concurrent.futures import wait, FIRST_COMPLETED
        pending = set(futures)
        while pending:
            _done, pending = wait(pending, timeout=_PROGRESS_INTERVAL_S,
                                  return_when=FIRST_COMPLETED)
            if live is not None:
                live.update(get_done())

    ran_serial = False
    if use_procpool and len(blocks) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        source_file = str(getattr(hamiltonian, "_module_path", "") or "")
        function_name = str(getattr(hamiltonian, "function_name", "H"))
        hparams = dict(getattr(hamiltonian, "params", {}) or {})
        payloads = [
            (source_file, function_name, hparams, nb, k_points[lo:hi], weights[lo:hi],
             A_t, dt, dim, dk_vel, ccfg.distribution, mu, temperature, gamma_pop, gamma_coh,
             pop_stride, propagator, A_mid, A_stages)
            for (lo, hi) in blocks
        ]
        try:
            with _mp.Manager() as mgr:
                counter = mgr.Value("q", 0)
                lock = mgr.Lock()

                def _get_done():
                    try:
                        with lock:
                            return int(counter.value)
                    except Exception:
                        return 0

                with ProcessPoolExecutor(max_workers=workers,
                                         initializer=_init_tddm_worker,
                                         initargs=(counter, lock)) as ex:
                    futures = [ex.submit(_tddm_block_worker, p) for p in payloads]
                    _drain(futures, _get_done)
                    results = [f.result() for f in futures]  # re-raises worker errors
            for res in results:
                _accumulate(res)
        except Exception as exc:  # spawn/pickling issue -> safe (serial) fallback
            if live is not None:
                live.close()
            if progress:
                print(f"[{progress_label}] ProcessPool unavailable ({type(exc).__name__}: "
                      f"{str(exc)[:80]}); running serially.", flush=True)
            J_total[...] = 0.0; J_intra[...] = 0.0; J_dc[...] = 0.0
            ran_serial = True
    elif workers <= 1 or len(blocks) <= 1:
        ran_serial = True
    else:
        counter = _AtomicCounter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run, b, counter.add) for b in blocks]
            _drain(futures, lambda: counter.value)
            for f in futures:
                _accumulate(f.result())

    if ran_serial:
        # single-threaded: the callback both counts and refreshes the line inline.
        counter = _AtomicCounter()

        def _serial_cb(inc):
            counter.add(inc)
            if live is not None:
                live.update(counter.value)

        for b in blocks:
            _accumulate(_run(b, _serial_cb))

    if live is not None:
        live.update(total_work, force=True)
        live.close()

    # subtract the equilibrium (DC) current from every time sample
    J_total -= J_dc[None, :]
    populations = np.concatenate(pop_blocks, axis=1) if pop_blocks else None  # (n_frames, nk, nb)
    return J_total, J_intra, populations


# ---------------------------------------------------------------------------
# Dataset assembly (identical schema to theory_response.compute_hhg_spectrum)
# ---------------------------------------------------------------------------
def _assemble_hhg_dataset(current_time, current_time_intra, field_time, freq, t_axis,
                          dim, Nt, max_order):
    current_spectrum = np.fft.fft(current_time, axis=0)
    current_magnitude = np.abs(current_spectrum)
    current_total_magnitude = np.sqrt(np.sum(current_magnitude ** 2, axis=1))
    current_time_inter = current_time - current_time_intra
    spectrum_intra = np.fft.fft(current_time_intra, axis=0)
    spectrum_inter = current_spectrum - spectrum_intra
    zeros3 = np.zeros((Nt, 3), dtype=np.float64)
    dataset = {
        "omega_axis": freq,
        "current_spectrum": current_spectrum,
        "current_magnitude": current_magnitude,
        "current_total_magnitude": current_total_magnitude,
        "current_time": current_time,
        "current_time_total": current_time,
        "current_spectrum_total": current_spectrum,
        "polarization_time": zeros3,
        "equilibrium_current_time": zeros3,
        "equilibrium_polarization_time": zeros3,
        "time_axis": t_axis,
        "electric_field_time": field_time,
        "current_decomposition_available": True,
        "current_time_intraband": current_time_intra,
        "current_time_interband": current_time_inter,
        "current_spectrum_intraband": spectrum_intra,
        "current_spectrum_interband": spectrum_inter,
        "current_total_magnitude_intraband": np.sqrt(np.sum(np.abs(spectrum_intra) ** 2, axis=1)),
        "current_total_magnitude_interband": np.sqrt(np.sum(np.abs(spectrum_inter) ** 2, axis=1)),
        "orders": np.asarray(tuple(range(1, max_order + 1))),
    }
    return dataset, current_spectrum


# ---------------------------------------------------------------------------
# Public API — HHG spectrum
# ---------------------------------------------------------------------------
def compute_hhg_spectrum_tddm(
    config: QXTIConfig,
    *,
    max_order: int | None = None,
    progress: bool = True,
    extra_k_weight_mask: FloatArray | None = None,
    field_scale: float = 1.0,
    pop_stride: int = 0,
) -> dict[str, Any]:
    """Full non-perturbative HHG current spectrum J(ω) (velocity gauge).

    Returns the SAME dict/dataset schema as
    :func:`qxti.analytics.theory_response.compute_hhg_spectrum`, so graphics and the
    engine comparison work unchanged.  ``field_scale`` multiplies the pulse
    amplitude (used by the susceptibility amplitude-scan).
    """
    from qxti.core.simulation import QXTISimulation

    t_start = time.perf_counter()
    sim = QXTISimulation(config=config)
    hamiltonian = sim.build_hamiltonian()
    kgrid = sim.build_kgrid(hamiltonian)
    laser_system = sim.build_laser_system()
    timegrid = sim.build_timegrid(laser_system)
    dim = int(hamiltonian.dimension)

    ccfg = config.cmd
    if max_order is None:
        max_order = int(ccfg.max_order)

    Nt = int(timegrid.Nt)
    dt = (float(timegrid.tf) - float(timegrid.t0)) / max(Nt - 1, 1)
    t_axis = float(timegrid.t0) + np.arange(Nt) * dt
    E_t = np.array([laser_system.electric_field(t) for t in t_axis], dtype=np.float64)
    if field_scale != 1.0:
        E_t = E_t * field_scale

    # Velocity-gauge vector potential, ANALYTIC.  The laser now satisfies E = −dA/dt
    # exactly (envelope-derivative fix), so the analytic A(t) is consistent with the
    # SAME E(t) the length-gauge engines use — with NO O(dt²) trapezoidal-integration
    # error.  We evaluate it at the grid times, the interval midpoints (cfm2), and the
    # Fehlberg stage nodes (rkf45), which is what lets a high-order propagator reach
    # its formal order (a numerically-integrated A would cap the scheme at 2nd order).
    propagator = str(getattr(ccfg, "tddm_propagator", "cfm2")).lower()

    def _A_at(times: FloatArray) -> FloatArray:
        A = np.array([laser_system.vector_potential(float(t)) for t in times], dtype=np.float64)
        return A * field_scale if field_scale != 1.0 else A

    A_t = _A_at(t_axis)                                 # A at the grid times
    A_mid = _A_at(t_axis + 0.5 * dt)                    # A at the interval midpoints
    A_stages = None
    if propagator == "rkf45":
        C = rkf45_node_offsets()                        # 6 Fehlberg node offsets
        A_stages = np.stack([_A_at(t_axis + float(c) * dt) for c in C], axis=1)  # (Nt,6,3)
    freq = np.fft.fftfreq(Nt, d=dt) * 2.0 * np.pi

    weights = build_k_integration_weights(
        config, hamiltonian=hamiltonian, kgrid=kgrid, extra_k_weight_mask=extra_k_weight_mask)
    n_workers, reserve_gb = _mesh_parallel_settings(config)

    J_total3, J_intra3, populations = _tddm_current_time(
        hamiltonian, kgrid, weights, A_t, dt, dim, ccfg,
        n_workers=n_workers, reserve_gb=reserve_gb, progress=progress, pop_stride=pop_stride,
        propagator=propagator, A_mid=A_mid, A_stages=A_stages)

    current_time = np.zeros((Nt, 3), dtype=np.float64)
    current_time[:, :dim] = J_total3[:, :dim]
    current_time_intra = np.zeros((Nt, 3), dtype=np.float64)
    current_time_intra[:, :dim] = J_intra3[:, :dim]
    field_time = np.zeros((Nt, 3), dtype=np.float64)
    field_time[:, :dim] = E_t[:, :dim]

    dataset, current_spectrum = _assemble_hhg_dataset(
        current_time, current_time_intra, field_time, freq, t_axis, dim, Nt, max_order)

    runtime = time.perf_counter() - t_start
    if progress:
        from qxti.utils.progress import format_duration
        print(f"[tddm] non-perturbative HHG current spectrum done "
              f"(elapsed {format_duration(runtime)}).", flush=True)
    result = {
        "omega_axis": freq,
        "J_total": current_spectrum[:, :dim],
        "J_order": {},
        "harmonic_peaks": {},
        "omega0": float(config.laser.omega),
        "max_order": max_order,
        "runtime_seconds": runtime,
        "dimension": dim,
        "method": "tddm",
        "dataset": dataset,
    }
    if populations is not None:
        shp = tuple(int(kgrid.shape[a]) for a in range(3))
        nfr, _, nbp = populations.shape
        kmesh = np.asarray(kgrid.points(), dtype=np.float64).reshape(shp[0], shp[1], shp[2], 3)
        result["populations"] = {
            # n(kx, ky, band) at each sampled frame (non-perturbative, Houston basis)
            "frames": populations.reshape(nfr, shp[0], shp[1], nbp),
            "frame_times": t_axis[::max(1, pop_stride)][:nfr],
            "kx_grid": kmesh[:, :, 0, 0], "ky_grid": kmesh[:, :, 0, 1],
            "stride": int(pop_stride),
        }
    return result


# ---------------------------------------------------------------------------
# Public API — susceptibility via field-amplitude scaling
# ---------------------------------------------------------------------------
def _default_amplitude_ladder(config: QXTIConfig, max_order: int) -> list[float]:
    ladder = tuple(getattr(config.cmd, "tddm_amplitude_ladder", ()) or ())
    if ladder:
        return [float(c) for c in ladder]
    # need at least max_order+1 amplitudes to fit a degree-(max_order) polynomial
    n = max(max_order + 1, 4)
    return list(np.geomspace(0.25, 2.0, n))


def compute_susceptibility_spectrum_tddm(
    config: QXTIConfig,
    omega_axis: FloatArray,
    orders,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    """χ⁽ˢ⁾ tensors from the non-perturbative solve, by field-amplitude scaling.

    For each output frequency the harmonic amplitude scales as
    ``J_i(sω; E0) = Σ_s χ⁽ˢ⁾_{i,j..j} E0^s + …``; running the full solve at several
    amplitudes and fitting a polynomial in E0 recovers χ⁽ˢ⁾ per order.  Emits the
    SAME dataset schema as
    :func:`qxti.analytics.theory_response.compute_susceptibility_spectrum`.
    """
    t_start = time.perf_counter()
    orders = tuple(int(s) for s in orders)
    max_order = max(orders)

    # base build (dimension, drive direction, laser frequency) — one build reused.
    from qxti.core.simulation import QXTISimulation
    sim = QXTISimulation(config=config)
    hamiltonian = sim.build_hamiltonian()
    dim = int(hamiltonian.dimension)
    omega0 = float(config.laser.omega)

    ladder = _default_amplitude_ladder(config, max_order)
    if progress:
        print(f"[tddm-xtp] susceptibility by amplitude scaling: {len(ladder)} amplitudes "
              f"{['%.3g' % c for c in ladder]}, orders {orders}. Each amplitude is a full solve.",
              flush=True)

    # Run the full solve at each amplitude; collect the harmonic peaks per direction.
    # peaks[c][a] = complex current at s*omega0 (per output direction a), for amplitude c.
    per_amp = []
    for ci, c in enumerate(ladder):
        res = compute_hhg_spectrum_tddm(config, max_order=max_order, progress=False, field_scale=c)
        freq = np.asarray(res["omega_axis"], dtype=np.float64)
        spec = np.asarray(res["dataset"]["current_spectrum"], dtype=np.complex128)  # (Nt,3)
        per_amp.append((c, freq, spec))
        if progress:
            print(f"[tddm-xtp] amplitude {ci+1}/{len(ladder)} (scale={c:.3g}) done.", flush=True)

    freq = per_amp[0][1]
    amps = np.array([c for (c, _, _) in per_amp], dtype=np.float64)

    def _peak_index(w):
        return int(np.argmin(np.abs(freq - w)))

    # Fit, per output frequency sω and per direction a, a polynomial in E0; the
    # coefficient of E0^s is χ⁽ˢ⁾ for the (diagonal) input direction of config.laser.
    dataset: dict[str, Any] = {
        "scan_type": "laser_frequency_sweep",
        "orders": np.asarray(orders),
        "dimension": dim,
        "direction_labels": ["x", "y", "z"][:dim],
        "laser_omega_axis": np.asarray(omega_axis, dtype=np.float64),
        "output_frequency_rule": "omega_out = order * omega_laser",
        "engine": "tddm",
    }
    # input (drive) direction from the single-laser polarization (diagonal component)
    in_axis = int(np.argmax(np.abs(sim.build_laser_system().electric_field(0.0)[:dim])))

    for s in orders:
        # Tensor shape convention matches the mesh path:
        # order 1 -> (nw, dim, dim); order s>=2 -> (nw, dim, [dim]*s).
        if s == 1:
            chi = np.full((omega_axis.size, dim, dim), np.nan, dtype=np.complex128)
        else:
            chi = np.full((omega_axis.size, dim) + (dim,) * s, np.nan, dtype=np.complex128)
        # For a monochromatic drive at omega0, the harmonic sits at s*omega0.
        ip = _peak_index(s * omega0)
        power_index = (len(amps) - 1) - s   # np.polyfit is highest-power-first
        for a in range(dim):
            J_a = np.array([spec[ip, a] for (_, _, spec) in per_amp], dtype=np.complex128)
            # J(E0) = Σ_p χ⁽ᵖ⁾ E0^p ; fit a polynomial in E0 and read the E0^s coeff.
            # Fit real and imag separately (robust complex least-squares).
            cr = np.polyfit(amps, J_a.real, deg=len(amps) - 1)
            ci = np.polyfit(amps, J_a.imag, deg=len(amps) - 1)
            if 0 <= power_index < len(cr):
                chi_val = cr[power_index] + 1j * ci[power_index]
            else:
                chi_val = np.nan
            comp = (slice(None), a) + (in_axis,) * (s if s >= 2 else 1)
            chi[comp] = chi_val
        # store (chi and a sigma alias = chi for schema completeness)
        dataset[f"chi_order_{s}_tensor"] = chi
        dataset[f"sigma_order_{s}_tensor"] = chi
        avail = np.asarray([(a,) + (in_axis,) * (s if s >= 2 else 1) for a in range(dim)],
                           dtype=np.int16)
        dataset[f"chi_order_{s}_available_indices"] = avail
        dataset[f"sigma_order_{s}_available_indices"] = avail

    runtime = time.perf_counter() - t_start
    return {"dataset": dataset, "runtime_seconds": runtime, "orders": orders, "method": "tddm"}
