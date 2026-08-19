"""Mesh-vectorized perturbative response — arbitrary order, any Hamiltonian.

This is the FAST equivalent of the per-k recursion in ``rho_analytic.rho_order_s``.
Instead of solving the A1 recursion independently at every k-point (which costs
~7^(s-1) diagonalizations PER k-point because each order takes a nested
finite-difference covariant k-gradient), it solves the SAME length-gauge
recursion VECTORIZED over the whole Brillouin-zone mesh:

    * one batched ``eigh`` over all k-points,
    * ρ^(1)(k, ω) built on the full mesh at once,
    * one Wilson-link covariant mesh gradient (``np.roll`` between grid
      neighbours) per order to get ρ^(2), ρ^(3), …

Cost is therefore ``O(orders × Nk)`` instead of ``O(Nk × 7^(s-1))`` — orders of
magnitude faster for high harmonic orders, with the SAME physics.

Covariant derivative D_k ρ = ∂_k ρ − i[A, ρ] is obtained IN ONE SHOT from the
Wilson-transported finite difference, so NO separate commutator is added (matches
the ``rho_order_s`` / CMD convention; adding it double-counts the connection and
cancels the intraband/population channel).

Discretization note
-------------------
``rho_order_s`` differentiates with a fixed step ``dk_grad`` (evaluating H at
k ± dk_grad); this module differentiates across the k-GRID neighbours (spacing set
by the grid).  The two are different discretizations of the same continuous
∂_k, so they agree in the converged (fine-grid) limit and, on a uniform grid with
``dk_grad`` = grid spacing, agree essentially point-for-point.  See
``docs/MESH_RESPONSE.md`` and ``tests/test_mesh_response.py``.

All quantities in atomic units (ℏ = e = 1).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from qxti.utils import memory as _mem

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_KB_AU = 3.1668114e-6  # Boltzmann constant in Hartree / K


def default_worker_count() -> int:
    """Best default core count for the mesh engine.

    Single cross-platform source of truth (SLURM allocation on a cluster, CPU
    affinity on Linux, performance cores on local macOS).  See
    :func:`qxti.utils.parallel.resolve_worker_count`.
    """
    try:
        from qxti.utils.parallel import resolve_worker_count
        return resolve_worker_count()
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


def _fermi(E: FloatArray, mu: float, T_au: float) -> FloatArray:
    if T_au < 1e-15:
        return (E <= mu + 1e-14).astype(float)
    return 1.0 / (np.exp(np.clip((E - mu) / T_au, -700, 700)) + 1.0)


def _dfde(f: FloatArray, T_au: float) -> FloatArray:
    if T_au < 1e-15:
        return np.zeros_like(f)
    return -f * (1.0 - f) / T_au


def _build_H_mesh(H_func: Callable, kpts: FloatArray) -> ComplexArray:
    """Evaluate H at every k-point -> (nk, nb, nb).  One pass, then batched."""
    first = np.asarray(H_func(float(kpts[0, 0]), float(kpts[0, 1]), float(kpts[0, 2])),
                       dtype=np.complex128)
    nb = first.shape[0]
    out = np.empty((kpts.shape[0], nb, nb), dtype=np.complex128)
    out[0] = first
    for i in range(1, kpts.shape[0]):
        out[i] = H_func(float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2]))
    return out


def _pad_band_mesh(kpts, shape, bounds, halo, dim):
    """Extend the k-mesh by ``halo`` REAL out-of-box layers on each active axis.

    The covariant-gradient stencil reads neighbours via ``np.roll``, which wraps
    PERIODICALLY at the box edges — correct only if the reciprocal box is a lattice
    cell (``H(k+box)=H(k)``).  For axis-aligned rectangular boxes on non-orthogonal
    (e.g. hexagonal) lattices the wraparound links inequivalent k-points, injecting a
    boundary gradient error that DIVERGES ∝N.  Padding with REAL out-of-box k (same
    spacing, Cartesian extrapolation, no wrap) makes the stencil read true neighbours
    at the boundary.  For a genuinely periodic box the padded points ARE the periodic
    images (``H`` periodic), so this reduces to the wrapped result — correct either way.

    Returns ``(kpts_pad, shape_pad, bounds_pad, interior_mask)`` with the interior
    (original grid) flagged so the BZ sum still runs over exactly one cell.
    """
    n = [int(shape[a]) for a in range(3)]
    km = np.asarray(kpts, dtype=np.float64).reshape(n[0], n[1], n[2], 3)
    axvals = [km[:, 0, 0, 0].copy(), km[0, :, 0, 1].copy(), km[0, 0, :, 2].copy()]
    padded, bounds_pad, interior = [], [], []
    for a in range(3):
        v = axvals[a]
        if a < dim and n[a] > 1:
            dk = (v[-1] - v[0]) / (n[a] - 1)
            below = v[0] - dk * np.arange(halo, 0, -1)
            above = v[-1] + dk * np.arange(1, halo + 1)
            vp = np.concatenate([below, v, above])
            interior.append((halo, halo + n[a]))
            bounds_pad.append((float(vp[0] - 0.5 * dk), float(vp[-1] + 0.5 * dk)))
        else:
            vp = v
            interior.append((0, n[a]))
            bd = bounds[a] if a < len(bounds) else (0.0, 0.0)
            bounds_pad.append((float(bd[0]), float(bd[1])))
        padded.append(vp)
    KX, KY, KZ = np.meshgrid(padded[0], padded[1], padded[2], indexing="ij")
    kpts_pad = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], axis=1)
    shape_pad = tuple(len(p) for p in padded)
    mask = np.zeros(shape_pad, dtype=bool)
    (a0, b0), (a1, b1), (a2, b2) = interior
    mask[a0:b0, a1:b1, a2:b2] = True
    return kpts_pad, shape_pad, tuple(bounds_pad), mask.ravel()


class BandData:
    """Frequency/field-independent band data on the k-mesh, computed ONCE.

    Reused across a whole (ω, drive-direction) sweep by ``harmonic_currents`` so
    the expensive batched ``eigh`` and velocity build happen only a single time.

    ``halo`` > 0 pads the mesh with real out-of-box layers (see ``_pad_band_mesh``)
    so the covariant gradient does not wrap at the BZ box edge; ``self.interior``
    then flags the original grid cells for the BZ sum.
    """

    __slots__ = ("energies", "U", "Udag", "U_mesh", "vel", "A", "f", "dfde",
                 "eps", "valid", "inv_eps", "dks", "shape", "nb", "nk", "dim", "diag",
                 "interior")

    def __init__(self, H_func, kpts, shape, bounds, *, mu=0.0, T_au=0.0,
                 dimension=3, dk_vel=1e-4, distribution=None, h_batch=None, halo=0):
        dim = int(dimension)
        interior = None
        if halo and int(halo) > 0:
            kpts, shape, bounds, interior = _pad_band_mesh(kpts, shape, bounds, int(halo), dim)
        nk = kpts.shape[0]
        # ``h_batch(kpts)->(nk,nb,nb)`` (vectorized) avoids the per-k Python loop in
        # ``_build_H_mesh`` — essential at large grids (150^3 = millions of points).
        build = (lambda kp: np.asarray(h_batch(kp), dtype=np.complex128)) if h_batch is not None \
            else (lambda kp: _build_H_mesh(H_func, kp))
        Hm = build(kpts)
        nb = Hm.shape[1]
        energies, U = np.linalg.eigh(Hm)
        Udag = np.conj(np.swapaxes(U, -1, -2))
        del Hm
        vel = []
        for a in range(dim):
            sh = np.zeros(3)
            sh[a] = dk_vel
            Hp = build(kpts + sh)
            Hmn = build(kpts - sh)
            vel.append(Udag @ ((Hp - Hmn) / (2 * dk_vel)) @ U)
            del Hp, Hmn
        # Occupation: honor the CONFIGURED distribution (e.g. valence_occupation,
        # which fills by band index to match antelope/UniformValence) rather than
        # silently forcing an energy Fermi step.  ``distribution(E, mu, T)`` must
        # match the signature of ``_fermi``; None -> energy Fermi (the per-k
        # ``rho_order_s`` reference, so the machine-precision tests still hold).
        f = _fermi(energies, mu, T_au) if distribution is None \
            else np.asarray(distribution(energies, mu, T_au), dtype=np.float64)
        # df/dε for the intraband/Drude source.  For a sharp (0/1) occupation this
        # is exactly 0 (f(1-f)=0); for Fermi-Dirac it is the analytic derivative.
        dfde = _dfde(f, T_au)
        eps = energies[:, :, None] - energies[:, None, :]
        offdiag = ~np.eye(nb, dtype=bool)
        valid = offdiag[None] & (np.abs(eps) > 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_eps = np.where(valid, 1.0 / eps, 0.0)
        self.energies = energies; self.U = U; self.Udag = Udag
        self.U_mesh = U.reshape(*shape, nb, nb)
        self.vel = vel
        self.A = [1j * vel[a] * inv_eps for a in range(dim)]
        self.f = f; self.dfde = dfde; self.eps = eps
        self.valid = valid; self.inv_eps = inv_eps
        self.dks = [(float(bounds[a][1]) - float(bounds[a][0])) / shape[a] for a in range(dim)]
        self.shape = tuple(int(s) for s in shape)
        self.nb = nb; self.nk = nk; self.dim = dim
        self.diag = np.arange(nb)
        self.interior = interior   # bool mask (nk,) of original-grid cells, or None


def precompute_band_data(H_func, kpts, shape, bounds, *, mu=0.0, T_au=0.0,
                         dimension=3, dk_vel=1e-4, distribution=None,
                         h_batch=None, halo=0) -> BandData:
    """Diagonalize H and build velocities/Berry connection on the mesh ONCE.

    ``distribution(E, mu, T) -> f`` sets the band occupation (default: energy
    Fermi step).  Pass the config's resolved distribution so orders >= 2 use the
    SAME filling as order 1 and as the configured engine (e.g. valence_occupation).
    ``h_batch(kpts)->(nk,nb,nb)`` is an optional VECTORIZED Hamiltonian builder
    (skips the per-k Python loop; needed for large grids).
    ``halo`` > 0 pads the mesh with real out-of-box layers so the covariant gradient
    does not wrap at the BZ box edge; pass ``(max_order-1)*(grad_stencil//2)``.
    """
    return BandData(H_func, kpts, shape, bounds, mu=mu, T_au=T_au,
                    dimension=dimension, dk_vel=dk_vel, distribution=distribution,
                    h_batch=h_batch, halo=halo)


def _wilson_cov_grad(R, b, U_mesh, Udag, mesh_shape, nflat, nb, dk, grad_stencil):
    """Wilson-link covariant k-gradient D_k ρ = ∂_k ρ − i[A_k, ρ] along axis ``b``.

    Each neighbour ρ(k+m·Δk) is parallel-transported into k's eigengauge via the
    Wilson link W = U(k)† U(k+m·Δk) before the finite difference.
    ``grad_stencil=2`` -> 2-point central difference, truncation error O(Δk²).
    ``grad_stencil=4`` -> 5-point 4th-order stencil, error O(Δk⁴); reaches the ±2
    neighbours (upstream halo must be 2·(max_order−1) so the interior stays exact).
    """
    Rm = R.reshape(*mesh_shape, nb, nb)

    def transported(shift):
        W = Udag @ np.roll(U_mesh, -shift, axis=b).reshape(nflat, nb, nb)
        Rs = np.roll(Rm, -shift, axis=b).reshape(nflat, nb, nb)
        return W @ Rs @ np.conj(np.swapaxes(W, -1, -2))

    if grad_stencil >= 4:
        return (-transported(2) + 8.0 * transported(1)
                - 8.0 * transported(-1) + transported(-2)) / (12.0 * dk)
    return (transported(1) - transported(-1)) / (2.0 * dk)


def harmonic_currents(band: BandData, weights, E_field, omega, max_order, *,
                      gamma=1e-3, gamma_pop=None, progress_cb=None,
                      grad_stencil=2) -> dict[int, ComplexArray]:
    """BZ-summed J^(s)_i = Σ_k w_k Tr[v_i ρ^(s)] from precomputed band data.

    ρ^(s) is the length-gauge A1 recursion with the one-shot Wilson covariant
    gradient (no double-count).  Reuses ``band`` across the whole ω/direction
    sweep — the eigh/velocity are NOT recomputed here.

    ``progress_cb(s)``, if given, is called after each order s>=2 is built (used by
    the single-shot HHG path to report per-order progress/ETA).
    """
    E = np.asarray(E_field, dtype=np.complex128)
    if gamma_pop is None:
        gamma_pop = gamma
    nb, nk, dim, diag = band.nb, band.nk, band.dim, band.diag
    eps, valid, A, vel = band.eps, band.valid, band.A, band.vel
    f = band.f; dfde = band.dfde
    fmn = f[:, None, :] - f[:, :, None]

    ow1_coh = complex(omega + 1j * gamma)
    ow1_pop = complex(omega + 1j * gamma_pop)
    rho1 = np.zeros((nk, nb, nb), dtype=np.complex128)
    for c in range(dim):
        if abs(E[c]) < 1e-40:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(valid, (A[c] * fmn) / (ow1_coh - eps), 0.0)
        r[:, diag, diag] += (-1j) * dfde * np.real(vel[c][:, diag, diag]) / ow1_pop
        rho1 += E[c] * r
    rhos = {1: rho1}

    Gamma = np.full((nb, nb), gamma)
    np.fill_diagonal(Gamma, gamma_pop)
    iGamma = 1j * Gamma[None]
    shape, U_mesh, Udag, dks = band.shape, band.U_mesh, band.Udag, band.dks

    def cov_grad(R, b):
        return _wilson_cov_grad(R, b, U_mesh, Udag, shape, nk, nb,
                                dks[b], grad_stencil)

    for s in range(2, max_order + 1):
        src = np.zeros((nk, nb, nb), dtype=np.complex128)
        for b in range(dim):
            if abs(E[b]) < 1e-40:
                continue
            src += E[b] * cov_grad(rhos[s - 1], b)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = s * omega + iGamma - eps
            rhos[s] = np.where(np.abs(denom) > 0, src / denom, 0.0)
        if progress_cb is not None:
            progress_cb(s)

    # BZ sum runs over the ORIGINAL grid only; padded (halo) cells exist solely to
    # feed the covariant gradient real neighbours at the box edge.
    interior = band.interior
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    currents: dict[int, ComplexArray] = {}
    for s in range(1, max_order + 1):
        Js = np.zeros(3, dtype=np.complex128)
        for i in range(dim):
            tr = np.einsum("kmn,knm->k", vel[i], rhos[s], optimize=True)
            if interior is not None:
                tr = tr[interior]
            Js[i] = np.sum(tr * w)
        currents[s] = Js
    return currents


# Per-plane peak live footprint of the streaming recursion, in units of
# (nb*nb complex128) arrays (band build + r/rho/src/cov-grad temporaries). Kept
# a touch conservative so block sizing errs small; the RAM guard checks the true
# concurrent peak against live RAM before launching.
_LIVE_ARRAYS_PER_PLANE = 14.0

# Below this estimated per-run work (nk × orders × nb³ ~ diagonalisation cost) the
# mesh stays on threads: a fast mesh isn't worth the ProcessPool startup (spawn
# re-imports the model per worker; fork on Linux is cheap so nearly everything real
# clears this bar).  Above it, blocks run in separate processes for true parallelism.
_MESH_PROCPOOL_MIN_WORK = 5_000_000


def _plan_stream(n0, bytes_per_plane, halo, n_workers_req, reserve_gb, cap):
    """Decide (mode, n_workers, block_planes) for the memory-safe stream.

    mode 'single' -> whole mesh in one pass (roll wraps periodically, no halo).
    mode 'block'  -> contiguous interior blocks of ``block_planes`` (+halo each
    side); up to ``n_workers`` processed CONCURRENTLY, so the block is sized for
    ``n_workers × (block+2·halo) × bytes_per_plane ≤ budget``.

    When a single pass would exceed the budget we prefer BLOCKING (slower but
    safe) over failing; only a truly impossible case (1 plane won't fit) falls
    through, where the caller's headroom check raises with a clear message.
    """
    budget = _mem.memory_budget_bytes(reserve_gb)
    w_cap = max(1, int(n_workers_req))
    single_fits = (cap >= n0) and (budget <= 0 or n0 * bytes_per_plane <= budget)
    # Single pass ONLY when there is nothing to parallelise (one worker) or the RAM
    # budget is unusable.  With >1 worker we split into >= w blocks so every core
    # runs, even when the whole mesh would fit in one pass -- otherwise a grid that
    # fits in RAM (the common case) ran serial on 1 core.  Block (real out-of-box
    # halo) and single (periodic roll) are numerically identical (verified 7e-15,
    # incl. hexagonal boxes), so this is safe; the blocks then run on a ProcessPool.
    if (w_cap <= 1 or budget <= 0) and single_fits:
        return "single", 1, n0
    for w in range(w_cap, 0, -1):
        fit = int((budget / w) // bytes_per_plane) - 2 * halo
        if cap:
            fit = min(fit, int(cap))
        if fit >= 1:
            bs = max(1, min(fit, -(-n0 // w)))    # >= w blocks so workers stay busy
            if w == 1 and bs >= n0 and single_fits:
                return "single", 1, n0            # only 1 worker fits -> single pass, no halo cost
            return "block", w, bs
    return "block", 1, max(1, min(int(cap) if cap else 1, 1))


def _mesh_block_currents(kb_int, w_int, P, C):
    """Band build + perturbative recursion on ONE contiguous k-block.

    Module-level (NOT a closure) so it can run in a separate PROCESS.  ``kb_int`` /
    ``w_int`` are the block's own interior k-points / weights (shape ``P*n1*n2``);
    the block is padded with a real out-of-box halo so the covariant gradient never
    wraps at the box edge (identical to the whole-BZ pass, verified 7e-15).  ``C``
    is the shared read-only context (built once per process / per call).  Returns a
    length-F list of ``({s: Js(3,)}, {s: Ji(3,)} | None)``.
    """
    n1, n2, halo, dim, nb, max_order = C.n1, C.n2, C.halo, C.dim, C.nb, C.max_order
    if halo > 0:
        kb, bshape, _, sel = _pad_band_mesh(kb_int, (P, n1, n2), C.bounds, halo, dim)
    else:                                    # max_order == 1: no gradient, no halo
        kb, bshape, sel = kb_int, (P, n1, n2), None
    npad = kb.shape[0]
    Hm = C.build(kb)
    energies, U = np.linalg.eigh(Hm)
    Udag = np.conj(np.swapaxes(U, -1, -2)); del Hm
    vel = []
    for a in range(dim):
        sh = np.zeros(3); sh[a] = C.dk_vel
        vel.append(Udag @ ((C.build(kb + sh) - C.build(kb - sh)) / (2 * C.dk_vel)) @ U)
    f = _fermi(energies, C.mu, C.T_au) if C.distribution is None \
        else np.asarray(C.distribution(energies, C.mu, C.T_au), dtype=np.float64)
    dfde = _dfde(f, C.T_au)
    eps = energies[:, :, None] - energies[:, None, :]
    offdiag = ~np.eye(nb, dtype=bool)
    valid = offdiag[None] & (np.abs(eps) > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_eps = np.where(valid, 1.0 / eps, 0.0)
    A = [1j * vel[a] * inv_eps for a in range(dim)]
    diag = np.arange(nb)
    fmn = f[:, None, :] - f[:, :, None]
    U_mesh = U.reshape(*bshape, nb, nb)

    ow1_coh = complex(C.omega + 1j * C.gamma)
    ow1_pop = complex(C.omega + 1j * C.gamma_pop)
    Gamma = np.full((nb, nb), C.gamma); np.fill_diagonal(Gamma, C.gamma_pop)
    iGamma = 1j * Gamma[None]
    r_c = []
    for c in range(dim):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(valid, (A[c] * fmn) / (ow1_coh - eps), 0.0)
        r[:, diag, diag] += (-1j) * dfde * np.real(vel[c][:, diag, diag]) / ow1_pop
        r_c.append(r)
    vint = [vel[i] if sel is None else vel[i][sel] for i in range(dim)]
    wint = w_int
    vint_diag = [vint[i][:, diag, diag] for i in range(dim)] if C.return_intraband else None

    def cov_grad(R, b):
        return _wilson_cov_grad(R, b, U_mesh, Udag, bshape, npad, nb, C.dks[b], C.grad_stencil)

    def _trace_J(rho_s):
        Js = np.zeros(3, dtype=np.complex128)
        ri = rho_s if sel is None else rho_s[sel]
        for i in range(dim):
            tr = np.einsum("kmn,knm->k", vint[i], ri, optimize=True)
            Js[i] = np.sum(tr * wint)
        return Js

    def _trace_intra(rho_s):
        Js = np.zeros(3, dtype=np.complex128)
        ri_d = (rho_s if sel is None else rho_s[sel])[:, diag, diag]
        for i in range(dim):
            Js[i] = np.sum(np.einsum("kn,kn->k", vint_diag[i], ri_d, optimize=True) * wint)
        return Js

    results = []
    for ef in range(C.nF):
        E = C.E_arr[ef]
        rho = np.zeros((npad, nb, nb), dtype=np.complex128)
        for c in range(dim):
            if abs(E[c]) >= 1e-40:
                rho += E[c] * r_c[c]
        Js_by_order = {1: _trace_J(rho)}
        Ji_by_order = {1: _trace_intra(rho)} if C.return_intraband else None
        for s in range(2, max_order + 1):
            src = np.zeros((npad, nb, nb), dtype=np.complex128)
            for b in range(dim):
                if abs(E[b]) >= 1e-40:
                    src += E[b] * cov_grad(rho, b)
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = s * C.omega + iGamma - eps
                rho = np.where(np.abs(denom) > 0, src / denom, 0.0)
            Js_by_order[s] = _trace_J(rho)
            if C.return_intraband:
                Ji_by_order[s] = _trace_intra(rho)
        results.append((Js_by_order, Ji_by_order))
    return results


# Set in each ProcessPool worker by the pool initializer: the read-only mesh
# context (H builder rebuilt from the model source, physics params) reused for
# every k-block that worker processes.
_MESH_WORKER_CTX = None


def _mesh_build_from_spec(source_file, function_name, params, h_batch_name):
    """Rebuild the ``kpts -> (nk, nb, nb)`` H builder inside a worker process, from
    the model's SOURCE FILE (callables aren't picklable across spawn)."""
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("_mesh_model", source_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hb = getattr(mod, h_batch_name, None) if h_batch_name else None
    if callable(hb):
        def build(kp):
            return np.asarray(hb(np.asarray(kp, dtype=float), params), dtype=np.complex128)
        return build
    Hfn = getattr(mod, function_name)

    def _hf(kx, ky, kz):
        return Hfn(kx, ky, kz, params)

    def build(kp):
        return _build_H_mesh(_hf, np.asarray(kp, dtype=np.float64))
    return build


def _mesh_proc_init(payload):
    """ProcessPool initializer: rebuild the shared mesh context ONCE per worker."""
    global _MESH_WORKER_CTX
    from types import SimpleNamespace  # noqa: PLC0415
    from qxti.analytics.theory_response import _resolve_distribution  # noqa: PLC0415
    (source_file, function_name, params, h_batch_name, dist_name, n1, n2, halo, bounds,
     dim, dk_vel, mu, T_au, nb, omega, gamma, gamma_pop, dks, grad_stencil,
     return_intraband, max_order, E_arr, nF) = payload
    build = _mesh_build_from_spec(source_file, function_name, params, h_batch_name)
    dist = _resolve_distribution(dist_name) if dist_name else None
    _MESH_WORKER_CTX = SimpleNamespace(
        n1=n1, n2=n2, halo=halo, bounds=bounds, dim=dim, build=build, dk_vel=dk_vel,
        distribution=dist, mu=mu, T_au=T_au, nb=nb, omega=omega, gamma=gamma,
        gamma_pop=gamma_pop, dks=dks, grad_stencil=grad_stencil,
        return_intraband=return_intraband, max_order=max_order, E_arr=E_arr, nF=nF)


def _mesh_proc_worker(task):
    """ProcessPool task: compute one k-block using this worker's rebuilt context."""
    kb_int, w_int, P = task
    return _mesh_block_currents(kb_int, w_int, P, _MESH_WORKER_CTX)


def harmonic_currents_meshed(
    H_func: Callable,
    kpts: FloatArray,
    shape: tuple[int, int, int],
    bounds: tuple[tuple[float, float], ...],
    weights: FloatArray,
    E_field,
    omega: float,
    max_order: int,
    *,
    gamma: float = 1e-3,
    gamma_pop: float | None = None,
    mu: float = 0.0,
    T_au: float = 0.0,
    dimension: int = 3,
    dk_vel: float = 1e-4,
    distribution=None,
    n_workers: int | None = None,
    reserve_gb: float = 1.0,
    max_block_planes: int | None = None,
    h_batch: Callable | None = None,
    return_intraband: bool = False,
    progress_cb: Callable | None = None,
    grad_stencil: int = 2,
    model_spec: dict | None = None,
) -> dict[int, ComplexArray]:
    """Memory-safe + multi-core BZ-summed harmonic currents from ``H_func``.

    ``model_spec`` (optional): ``{source_file, function_name, params, h_batch_name,
    dist_name}`` describing the model so each k-block can run in a separate PROCESS
    (true parallelism -- the mesh's vectorized per-block numpy is GIL-bound, so a
    ThreadPool stalls at ~1 core regardless of nb).  When omitted the blocks run on
    a ThreadPool (unchanged behaviour).

    Same result as ``precompute_band_data`` + ``harmonic_currents`` (bit-exact),
    but streams the k-mesh in blocks of planes (halo = ``max_order-1`` -> the
    covariant-gradient interior is exact) and runs blocks CONCURRENTLY on a
    ``ThreadPoolExecutor``.  Block thickness is chosen from the RAM available at
    run time so ``n_workers`` concurrent blocks always leave ``reserve_gb`` free
    (Linux/macOS/Windows).  Falls back to a single full-mesh pass when it fits.

    ``n_workers`` : max cores to use (None -> all logical CPUs; 1 -> serial).
    ``reserve_gb``: RAM to keep free.  ``h_batch``: optional vectorized builder.

    ``E_field`` may be a single complex 3-vector OR an ``(F, 3)`` stack of drive
    amplitudes; with a stack, band data is built ONCE per block and reused across
    all F fields (efficient tensor fits), and the return is a length-F list.
    Returns ``{s: J^(s)}`` (single field) or ``[{s: J^(s)}, ...]`` (F fields).
    """
    n0, n1, n2 = int(shape[0]), int(shape[1]), int(shape[2])
    dim = int(dimension)
    nb = None
    E_in = np.asarray(E_field, dtype=np.complex128)
    single_field = (E_in.ndim == 1)
    E_arr = E_in.reshape(1, -1) if single_field else E_in       # (F, 3)
    nF = E_arr.shape[0]
    if gamma_pop is None:
        gamma_pop = gamma
    dks = [(float(bounds[a][1]) - float(bounds[a][0])) / shape[a] for a in range(dim)]
    # 2-point stencil reaches ±1 per gradient application -> halo max_order-1.
    # 5-point (grad_stencil=4) reaches ±2 -> halo doubles so the interior stays exact.
    halo = (int(max_order) - 1) * (int(grad_stencil) // 2)
    n_workers = default_worker_count() if n_workers is None else max(1, int(n_workers))

    build = (lambda kp: np.asarray(h_batch(kp), dtype=np.complex128)) if h_batch is not None \
        else (lambda kp: _build_H_mesh(H_func, kp))

    kmesh = np.asarray(kpts, dtype=np.float64).reshape(n0, n1, n2, 3)
    wmesh = np.asarray(weights, dtype=np.float64).reshape(n0, n1, n2)

    nb = int(build(kmesh[:1, :1, :1].reshape(1, 3)).shape[1])
    # Each block is padded with a real out-of-box halo on every active axis so the
    # covariant gradient never wraps at the box edge; size the plane by the PADDED
    # transverse extent (axes 1,2) so the RAM guard stays honest.
    n1p = n1 + (2 * halo if dim >= 2 else 0)
    n2p = n2 + (2 * halo if dim >= 3 else 0)
    bytes_per_plane = n1p * n2p * nb * nb * 16.0 * _LIVE_ARRAYS_PER_PLANE
    cap = int(max_block_planes) if max_block_planes else n0
    mode, n_workers, bs = _plan_stream(n0, bytes_per_plane, halo, n_workers, reserve_gb, cap)
    single = mode == "single"

    from types import SimpleNamespace  # noqa: PLC0415
    # Read-only per-block context, shared by the thread/serial path (built here) and
    # rebuilt inside each process worker (see _mesh_proc_init).
    C = SimpleNamespace(
        n1=n1, n2=n2, halo=halo, bounds=bounds, dim=dim, build=build, dk_vel=dk_vel,
        distribution=distribution, mu=mu, T_au=T_au, nb=nb, omega=omega, gamma=gamma,
        gamma_pop=gamma_pop, dks=dks, grad_stencil=grad_stencil,
        return_intraband=return_intraband, max_order=max_order, E_arr=E_arr, nF=nF)

    def _band_and_current(p_lo, p_hi):
        P = p_hi - p_lo
        kb_int = kmesh[p_lo:p_hi].reshape(P * n1 * n2, 3)
        w_int = wmesh[p_lo:p_hi].reshape(P * n1 * n2)
        return _mesh_block_currents(kb_int, w_int, P, C)

    # ---- build the block list (interior plane ranges) ------------------------
    # Each block is the interior [lo,hi); ``_band_and_current`` pads it with the
    # real out-of-box halo internally (no periodic wrap at the box edge).
    jobs = []
    if single:
        jobs.append((0, n0))
    else:
        lo = 0
        while lo < n0:
            hi = min(lo + bs, n0)
            jobs.append((lo, hi))
            lo = hi

    # ---- one concurrency-aware RAM check, then run ---------------------------
    # Peak = (#blocks running at once) × (largest padded block).  ThreadPoolExecutor
    # caps concurrency at n_workers, so this bounds the true peak footprint.
    max_block_planes_live = max((hi - lo for (lo, hi) in jobs), default=n0) + 2 * halo
    concurrent = 1 if (single or n_workers <= 1) else min(n_workers, len(jobs))
    _mem.ensure_headroom(concurrent * max_block_planes_live * bytes_per_plane,
                         reserve_gb=reserve_gb, label="mesh-stream (concurrent peak)")

    totals = [{s: np.zeros(3, dtype=np.complex128) for s in range(1, max_order + 1)}
              for _ in range(nF)]
    intras = [{s: np.zeros(3, dtype=np.complex128) for s in range(1, max_order + 1)}
              for _ in range(nF)] if return_intraband else None
    done = [0]

    def _accumulate(part):                        # part = list of F (total, intra) tuples
        for ef in range(nF):
            tot_ef, intra_ef = part[ef]
            for s in tot_ef:
                totals[ef][s] += tot_ef[s]
                if return_intraband:
                    intras[ef][s] += intra_ef[s]
        done[0] += 1
        if progress_cb is not None:
            progress_cb(done[0], len(jobs))

    # The mesh's per-block numpy is GIL-bound (few big vectorized calls + Python
    # glue), so a ThreadPool stalls at ~1 core for ANY nb (measured nb=2 and nb=8).
    # When the caller supplies model_spec, run the blocks in separate PROCESSES
    # (true parallelism); each block is self-contained, so the result is identical.
    # Gate on estimated work so short meshes don't pay the pool-startup tax (spawn
    # re-imports the model per worker on macOS/Windows; fork on Linux is cheap, so
    # on the cluster this fires for essentially every real run).
    _est_work = int(n0) * int(n1) * int(n2) * int(max_order) * int(nb) ** 3
    use_procpool = (model_spec is not None and not single
                    and n_workers > 1 and len(jobs) > 1
                    and _est_work >= _MESH_PROCPOOL_MIN_WORK)
    if concurrent <= 1:
        for job in jobs:
            _accumulate(_band_and_current(*job))
    elif use_procpool:
        payload = (model_spec["source_file"], model_spec["function_name"],
                   model_spec["params"], model_spec.get("h_batch_name"),
                   model_spec.get("dist_name"), n1, n2, halo, bounds, dim, dk_vel,
                   mu, T_au, nb, omega, gamma, gamma_pop, dks, grad_stencil,
                   return_intraband, max_order, E_arr, nF)
        tasks = [(kmesh[lo:hi].reshape((hi - lo) * n1 * n2, 3),
                  wmesh[lo:hi].reshape((hi - lo) * n1 * n2), hi - lo) for (lo, hi) in jobs]
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_mesh_proc_init,
                                     initargs=(payload,)) as ex:
                for part in ex.map(_mesh_proc_worker, tasks):
                    _accumulate(part)
        except Exception:   # spawn/pickling issue -> reset partial sums, fall back to threads
            for ef in range(nF):
                for s in range(1, max_order + 1):
                    totals[ef][s][:] = 0.0
                    if return_intraband:
                        intras[ef][s][:] = 0.0
            done[0] = 0
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for part in ex.map(lambda j: _band_and_current(*j), jobs):
                    _accumulate(part)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for part in ex.map(lambda j: _band_and_current(*j), jobs):
                _accumulate(part)
    out_tot = totals[0] if single_field else totals
    if return_intraband:
        out_intra = intras[0] if single_field else intras
        return out_tot, out_intra
    return out_tot


def mesh_harmonic_currents(
    H_func: Callable,
    kpts: FloatArray,
    shape: tuple[int, int, int],
    bounds: tuple[tuple[float, float], ...],
    weights: FloatArray,
    E_field,
    omega: float,
    max_order: int,
    *,
    gamma: float = 1e-3,
    gamma_pop: float | None = None,
    mu: float = 0.0,
    T_au: float = 0.0,
    dimension: int = 3,
    dk_vel: float = 1e-4,
    distribution=None,
    n_workers: int | None = None,
    reserve_gb: float = 1.0,
    h_batch: Callable | None = None,
    grad_stencil: int = 2,
) -> dict[int, ComplexArray]:
    """BZ-summed harmonic currents J^(s)_i = Σ_k w_k Tr[v_i ρ^(s)(k, s·ω)].

    Thin wrapper over the memory-safe + multi-core streaming engine
    :func:`harmonic_currents_meshed` (single full-mesh pass when it fits).
    """
    return harmonic_currents_meshed(
        H_func, kpts, shape, bounds, weights, E_field, omega, max_order,
        gamma=gamma, gamma_pop=gamma_pop, mu=mu, T_au=T_au, dimension=dimension,
        dk_vel=dk_vel, distribution=distribution, n_workers=n_workers,
        reserve_gb=reserve_gb, h_batch=h_batch, grad_stencil=grad_stencil)


def perk_harmonic_currents(
    H_func: Callable,
    kpts: FloatArray,
    weights: FloatArray,
    E_field,
    omega: float,
    max_order: int,
    *,
    gamma: float = 1e-3,
    mu: float = 0.0,
    T_au: float = 0.0,
    dimension: int = 3,
    dk_grad: float = 1e-3,
    dk_vel: float = 1e-4,
    distribution=None,
) -> dict[int, ComplexArray]:
    """Reference: the SAME J^(s) via the per-k recursion ``rho_order_s``.

    Slow (~7^(s-1) diagonalizations/k) — used only to validate the fast mesh path.
    ``distribution`` (None -> energy Fermi) honors the configured filling so a
    head-to-head comparison uses the SAME occupation as the mesh.
    """
    from qxti.analytics.rho_analytic import rho_order_s, _velocity_band

    E = np.asarray(E_field, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    dim = int(dimension)
    acc = {s: np.zeros(3, dtype=np.complex128) for s in range(1, max_order + 1)}
    for ik in range(kpts.shape[0]):
        kx, ky, kz = float(kpts[ik, 0]), float(kpts[ik, 1]), float(kpts[ik, 2])
        vel = _velocity_band(H_func, kx, ky, kz, dk_vel)
        rhos = rho_order_s(H_func, kx, ky, kz, E, omega, gamma, mu, T_au,
                           max_order=max_order, dk_grad=dk_grad, dk_vel=dk_vel,
                           distribution=distribution)
        for s in range(1, max_order + 1):
            rho_s = rhos.get(s)
            if rho_s is None:
                continue
            for i in range(dim):
                acc[s][i] += w[ik] * np.trace(vel[i] @ rho_s)
    return acc


def time_domain_currents(band: BandData, weights, E_t, dt, max_order, *,
                         gamma=1e-3, gamma_pop=None, k_chunk=None,
                         return_intraband=False, progress_cb=None) -> dict:
    """Order-by-order PERTURBATIVE response to an ARBITRARY field E(t).

    This is the multi-frequency generalization of :func:`harmonic_currents`.  The
    closed-form path assumes a single carrier ω and returns ρ^(s)(sω); here we
    integrate the SAME length-gauge recursion in the time domain with the FULL
    field E(t) (any number of lasers / frequencies), so every mixing product
    (ω_i±ω_j±…) appears automatically in the FFT of the current — no enumeration.

    Physics (identical operators to ``harmonic_currents``; H_0 diagonal in the
    band basis makes the propagator a per-element frequency denominator):

        S^(N)(t) = Σ_α E_α(t) [D_k ρ^(N-1)(t)]_α          (Wilson covariant grad)
        ρ^(N)(ω) = FFT_t[S^(N)(t)] / (ω − ω_mn + iΓ_mn)    (Γ: T1 diag, T2 offdiag)
        ρ^(N)(t) = IFFT_ω[ρ^(N)(ω)]

    with ρ^(0) = diag(f).  Reduces to the closed form at every sω peak when E(t)
    is monochromatic (validated to machine level — see the study script).

    Parameters
    ----------
    band     : precomputed :class:`BandData` (eigh/velocity/occupation on the mesh).
    weights  : (nk,) BZ quadrature weights.
    E_t      : (Nt, dim) real electric field sampled on a uniform time grid.
    dt       : time step (a.u.).
    max_order: highest perturbative order.
    gamma    : coherence dephasing 1/T2 (off-diagonal). gamma_pop: 1/T1 (diagonal).
    k_chunk  : k-points processed at once in the frequency solve (memory knob).

    Returns
    -------
    dict with ``freq`` (Nt, angular), ``J_omega`` {s: (Nt, dim) complex spectrum},
    ``J_t`` {s: (Nt, dim) complex time-domain current}, one per order s=1..max_order.
    """
    E_t = np.asarray(E_t, dtype=np.complex128)
    Nt, dim_in = E_t.shape
    if gamma_pop is None:
        gamma_pop = gamma
    nb, nk, dim, diag = band.nb, band.nk, band.dim, band.diag
    eps, valid, A, vel = band.eps, band.valid, band.A, band.vel
    f, dfde = band.f, band.dfde
    shape, U_mesh, Udag, dks = band.shape, band.U_mesh, band.Udag, band.dks
    fmn = f[:, None, :] - f[:, :, None]

    freq = 2.0 * np.pi * np.fft.fftfreq(Nt, d=dt)          # angular frequency grid
    Gamma = np.full((nb, nb), gamma)
    np.fill_diagonal(Gamma, gamma_pop)
    iGamma = 1j * Gamma                                     # (nb, nb)
    # BZ sum runs over the INTERIOR (original grid); padded halo cells (band built
    # with halo>0) only feed the covariant gradient real neighbours at the box edge.
    interior = band.interior
    w_k = np.asarray(weights, dtype=np.float64).reshape(-1)
    vel_int = vel if interior is None else [vel[i][interior] for i in range(dim)]
    if k_chunk is None:
        k_chunk = max(1, int(2_000_000 // (nb * nb * max(Nt, 1))))

    # The propagator 1/(−ω − ε_mn + iΓ_mn) is ORDER-INDEPENDENT: build it ONCE
    # (chunked to bound the temporary) so every order is a multiply, not a divide.
    # Sign of ω follows numpy's ifft convention x(t)=Σ X(ω)e^{+iωt}: the physical
    # response at output frequency Ω (e^{-iΩt}) sits in the −Ω bin, so (Ω − ε_mn + iΓ)
    # becomes (−ω − ε_mn + iΓ), reproducing the closed form (sω − ε + iγ) at each sω.
    inv_denom = np.empty((nk, nb, nb, Nt), dtype=np.complex128)
    for k0 in range(0, nk, k_chunk):
        k1 = min(k0 + k_chunk, nk)
        inv_denom[k0:k1] = 1.0 / (-freq[None, None, None, :]
                                  - eps[k0:k1, :, :, None]
                                  + iGamma[None, :, :, None])

    # v_i weighted by the BZ measure, TRANSPOSED (m<->n) so the trace Tr[v_i ρ] =
    # Σ_mn v_i[m,n] ρ[n,m] becomes a plain index-aligned contraction that tensordot
    # routes through BLAS (an m<->n-transposed einsum falls back to a slow loop).
    vwT = [(vel_int[i] * w_k[:, None, None]).swapaxes(1, 2).copy() for i in range(dim)]

    def _current(rho_omega, s):
        # J^(s)_i(ω) = i^(s-1) Σ_k w_k Tr[v_i ρ^(s)(ω)] (physical phase -> J(t) real).
        phase = 1j ** (s - 1)
        ro = rho_omega if interior is None else rho_omega[interior]
        Jw = np.empty((Nt, 3), dtype=np.complex128)
        for i in range(dim):
            Jw[:, i] = phase * np.tensordot(vwT[i], ro, axes=([0, 1, 2], [0, 1, 2]))
        return Jw, np.fft.ifft(Jw, axis=0)

    # diagonal (band-group) velocity weighted by the BZ measure, for the intraband
    # current J_intra = Σ_k w_k Σ_n v_nn ρ_nn.  Interband = total − intra.
    vdw = [(vel_int[i][:, diag, diag] * w_k[:, None]) for i in range(dim)] if return_intraband else None

    def _current_intra(rho_omega, s):
        phase = 1j ** (s - 1)
        Jw = np.empty((Nt, 3), dtype=np.complex128)
        rho_d = (rho_omega if interior is None else rho_omega[interior])[:, diag, diag, :]
        for i in range(dim):
            Jw[:, i] = phase * np.tensordot(vdw[i], rho_d, axes=([0, 1], [0, 1]))
        return Jw, np.fft.ifft(Jw, axis=0)

    # Wilson links are time- AND order-independent -> precompute ONCE per direction.
    active_dirs = [b for b in range(dim) if np.any(np.abs(E_t[:, b]) > 1e-40)]
    links = {}
    for b in active_dirs:
        Up = np.roll(U_mesh, -1, axis=b).reshape(nk, nb, nb)
        Un = np.roll(U_mesh, +1, axis=b).reshape(nk, nb, nb)
        wp = Udag @ Up
        wm = Udag @ Un
        links[b] = (wp, np.conj(np.swapaxes(wp, -1, -2)),
                    wm, np.conj(np.swapaxes(wm, -1, -2)))

    def _transport(Rx, wl, wr):
        # wl @ Rx @ wr for Rx (nk,nb,nb,Nt), via BLAS batched matmul (much faster
        # than a 3-operand einsum with an uncontracted time axis).
        s1 = (wl @ Rx.reshape(nk, nb, nb * Nt)).reshape(nk, nb, nb, Nt)
        s2 = np.matmul(np.swapaxes(s1, 2, 3), wr[:, None])   # (nk,nb,Nt,nb)
        return np.swapaxes(s2, 2, 3)

    def cov_grad_t(R, b):                                  # R: (nk,nb,nb,Nt)
        wp, wpd, wm, wmd = links[b]
        Rm = R.reshape(*shape, nb, nb, Nt)
        Rp = np.roll(Rm, -1, axis=b).reshape(nk, nb, nb, Nt)
        Rn = np.roll(Rm, +1, axis=b).reshape(nk, nb, nb, Nt)
        return (_transport(Rp, wp, wpd) - _transport(Rn, wm, wmd)) / (2.0 * dks[b])

    # ---- order 1: source operators are time-independent, scaled by E_α(t) ----
    E_w = np.fft.fft(E_t, axis=0)                          # (Nt, dim)
    num = np.zeros((nk, nb, nb, Nt), dtype=np.complex128)
    for c in range(dim):
        Sc = np.where(valid, A[c] * fmn, 0.0 + 0.0j)
        Sc[:, diag, diag] += (-1j) * dfde * np.real(vel[c][:, diag, diag])
        num += Sc[:, :, :, None] * E_w[None, None, None, :, c]
    rho_omega = num * inv_denom
    del num
    J_omega, J_t = {}, {}
    J_omega_intra, J_t_intra = ({}, {}) if return_intraband else (None, None)
    J_omega[1], J_t[1] = _current(rho_omega, 1)
    if return_intraband:
        J_omega_intra[1], J_t_intra[1] = _current_intra(rho_omega, 1)
    if progress_cb is not None:
        progress_cb(1)

    # ---- orders ≥2: only the PREVIOUS order's ρ(t) is kept (bounded memory) ----
    rho_t_prev = np.fft.ifft(rho_omega, axis=-1) if max_order >= 2 else None
    for s in range(2, max_order + 1):
        S_t = np.zeros((nk, nb, nb, Nt), dtype=np.complex128)
        for b in active_dirs:
            S_t += cov_grad_t(rho_t_prev, b) * E_t[None, None, None, :, b]
        rho_omega = np.fft.fft(S_t, axis=-1) * inv_denom
        del S_t
        J_omega[s], J_t[s] = _current(rho_omega, s)
        if return_intraband:
            J_omega_intra[s], J_t_intra[s] = _current_intra(rho_omega, s)
        if s < max_order:
            rho_t_prev = np.fft.ifft(rho_omega, axis=-1)
        if progress_cb is not None:
            progress_cb(s)
    out = {"freq": freq, "J_omega": J_omega, "J_t": J_t}
    if return_intraband:
        out["J_omega_intra"] = J_omega_intra
        out["J_t_intra"] = J_t_intra
    return out


# Per-plane peak footprint of the multi-laser time-domain solve, in units of
# (nb*nb*Nt complex128) arrays: inv_denom + rho_omega + rho_t + source + a couple of
# temporaries.  Kept a touch high so block sizing errs small (the RAM guard checks
# the true concurrent peak before launching).
_TD_ARRAYS_PER_PLANE = 6.0


def time_domain_currents_meshed(
    H_func, kpts, shape, bounds, weights, E_t, dt, max_order, *,
    gamma=1e-3, gamma_pop=None, mu=0.0, T_au=0.0, dimension=3, dk_vel=1e-4,
    distribution=None, n_workers=None, reserve_gb=1.0, h_batch=None,
    return_intraband=False, progress_cb=None,
):
    """Memory-safe MULTI-LASER (arbitrary ``E(t)``) BZ-summed currents.

    Same physics as ``precompute_band_data`` + :func:`time_domain_currents` on the
    whole grid, but STREAMS the k-mesh in halo-padded blocks and sums the per-block
    BZ current — so the ``(nk, nb, nb, Nt)`` frequency-domain tensors NEVER exist for
    the whole grid.  That all-at-once allocation (``inv_denom``/``ρ(ω)`` sized ``nk ×
    nb² × Nt``) is what makes the single-shot path OOM at large ``nk`` (petabytes at
    140³), which is why multi-laser runs fell back to ptddm.  Each block is
    self-contained (real out-of-box halo for the covariant gradient), so the result
    is identical to the whole-grid solve (verified in tests).

    Returns ``{"J_t": {s: (Nt, dim)}}`` (+ ``"J_t_intra"`` when requested), each
    already summed over the Brillouin zone.
    """
    n0, n1, n2 = int(shape[0]), int(shape[1]), int(shape[2])
    dim = int(dimension)
    E_t = np.asarray(E_t, dtype=np.complex128)
    Nt = int(E_t.shape[0])
    halo = max(0, int(max_order) - 1)          # time_domain_currents uses a 2-point stencil
    n_workers = default_worker_count() if n_workers is None else max(1, int(n_workers))

    build = (lambda kp: np.asarray(h_batch(kp), dtype=np.complex128)) if h_batch is not None \
        else (lambda kp: _build_H_mesh(H_func, kp))
    kmesh = np.asarray(kpts, dtype=np.float64).reshape(n0, n1, n2, 3)
    wmesh = np.asarray(weights, dtype=np.float64).reshape(n0, n1, n2)
    nb = int(build(kmesh[:1, :1, :1].reshape(1, 3)).shape[1])

    n1p = n1 + (2 * halo if dim >= 2 else 0)
    n2p = n2 + (2 * halo if dim >= 3 else 0)
    bytes_per_plane = n1p * n2p * nb * nb * Nt * 16.0 * _TD_ARRAYS_PER_PLANE
    mode, n_workers, bs = _plan_stream(n0, bytes_per_plane, halo, n_workers, reserve_gb, cap=n0)

    if mode == "single":
        jobs = [(0, n0)]
    else:
        # A covariant-gradient block needs >= 2 interior planes along the streamed
        # axis: _pad_band_mesh forms the real out-of-box halo by EXTRAPOLATING from
        # the block's own spacing, and a 1-plane block has no spacing -> no axis-0
        # halo -> the gradient degenerates and every order >= 2 silently vanishes.
        # Enforce >= min(2, n0); if that block no longer fits RAM the guard below
        # raises a clear error instead of returning wrong (zero) higher harmonics.
        if halo > 0:
            bs = max(int(bs), min(2, n0))
        jobs, lo = [], 0
        while lo < n0:
            hi = min(lo + bs, n0)
            jobs.append((lo, hi))
            lo = hi
    # Blocks run SERIALLY here, so only ONE (padded) block is live at a time.
    _mem.ensure_headroom((max(hi - lo for lo, hi in jobs) + 2 * halo) * bytes_per_plane,
                         reserve_gb=reserve_gb, label="td-multilaser block")

    J_t = {s: np.zeros((Nt, dim), dtype=np.complex128) for s in range(1, max_order + 1)}
    J_t_intra = ({s: np.zeros((Nt, dim), dtype=np.complex128) for s in range(1, max_order + 1)}
                 if return_intraband else None)

    for bi, (lo, hi) in enumerate(jobs):
        block_kpts = kmesh[lo:hi].reshape((hi - lo) * n1 * n2, 3)
        block_w = wmesh[lo:hi].reshape((hi - lo) * n1 * n2)
        band = precompute_band_data(
            H_func, block_kpts, (hi - lo, n1, n2), bounds, mu=mu, T_au=T_au,
            dimension=dim, dk_vel=dk_vel, distribution=distribution, h_batch=h_batch, halo=halo)
        td = time_domain_currents(band, block_w, E_t, dt, max_order, gamma=gamma,
                                  gamma_pop=gamma_pop, return_intraband=return_intraband)
        for s in range(1, max_order + 1):
            J_t[s] += np.asarray(td["J_t"][s], dtype=np.complex128)[:, :dim]
            if return_intraband:
                J_t_intra[s] += np.asarray(td["J_t_intra"][s], dtype=np.complex128)[:, :dim]
        del band, td
        if progress_cb is not None:
            progress_cb(bi + 1, len(jobs))

    out = {"J_t": J_t}
    if return_intraband:
        out["J_t_intra"] = J_t_intra
    return out


def uniform_mp_grid(bounds, shape):
    """Shifted Monkhorst-Pack k-points (C-ordered) + weights for a box ``bounds``."""
    axes = []
    for a in range(3):
        lo, hi = float(bounds[a][0]), float(bounds[a][1])
        n = int(shape[a])
        axes.append(lo + (np.arange(n) + 0.5) * (hi - lo) / n)
    KX, KY, KZ = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    kpts = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], axis=1)
    dim = sum(1 for a in range(3) if shape[a] > 1)
    V = 1.0
    for a in range(3):
        if shape[a] > 1:
            V *= (float(bounds[a][1]) - float(bounds[a][0]))
    w = np.full(kpts.shape[0], V / kpts.shape[0])
    return kpts, w
