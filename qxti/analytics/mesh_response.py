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

from typing import Callable

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_KB_AU = 3.1668114e-6  # Boltzmann constant in Hartree / K


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


class BandData:
    """Frequency/field-independent band data on the k-mesh, computed ONCE.

    Reused across a whole (ω, drive-direction) sweep by ``harmonic_currents`` so
    the expensive batched ``eigh`` and velocity build happen only a single time.
    """

    __slots__ = ("energies", "U", "Udag", "U_mesh", "vel", "A", "f", "dfde",
                 "eps", "valid", "inv_eps", "dks", "shape", "nb", "nk", "dim", "diag")

    def __init__(self, H_func, kpts, shape, bounds, *, mu=0.0, T_au=0.0,
                 dimension=3, dk_vel=1e-4, distribution=None):
        dim = int(dimension)
        nk = kpts.shape[0]
        Hm = _build_H_mesh(H_func, kpts)
        nb = Hm.shape[1]
        energies, U = np.linalg.eigh(Hm)
        Udag = np.conj(np.swapaxes(U, -1, -2))
        del Hm
        vel = []
        for a in range(dim):
            sh = np.zeros(3)
            sh[a] = dk_vel
            Hp = _build_H_mesh(H_func, kpts + sh)
            Hmn = _build_H_mesh(H_func, kpts - sh)
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


def precompute_band_data(H_func, kpts, shape, bounds, *, mu=0.0, T_au=0.0,
                         dimension=3, dk_vel=1e-4, distribution=None) -> BandData:
    """Diagonalize H and build velocities/Berry connection on the mesh ONCE.

    ``distribution(E, mu, T) -> f`` sets the band occupation (default: energy
    Fermi step).  Pass the config's resolved distribution so orders >= 2 use the
    SAME filling as order 1 and as the configured engine (e.g. valence_occupation).
    """
    return BandData(H_func, kpts, shape, bounds, mu=mu, T_au=T_au,
                    dimension=dimension, dk_vel=dk_vel, distribution=distribution)


def harmonic_currents(band: BandData, weights, E_field, omega, max_order, *,
                      gamma=1e-3, gamma_pop=None) -> dict[int, ComplexArray]:
    """BZ-summed J^(s)_i = Σ_k w_k Tr[v_i ρ^(s)] from precomputed band data.

    ρ^(s) is the length-gauge A1 recursion with the one-shot Wilson covariant
    gradient (no double-count).  Reuses ``band`` across the whole ω/direction
    sweep — the eigh/velocity are NOT recomputed here.
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
        Up = np.roll(U_mesh, -1, axis=b).reshape(nk, nb, nb)
        Un = np.roll(U_mesh, +1, axis=b).reshape(nk, nb, nb)
        wp = Udag @ Up
        wm = Udag @ Un
        Rm = R.reshape(*shape, nb, nb)
        Rp = np.roll(Rm, -1, axis=b).reshape(nk, nb, nb)
        Rn = np.roll(Rm, +1, axis=b).reshape(nk, nb, nb)
        return (wp @ Rp @ np.conj(np.swapaxes(wp, -1, -2))
                - wm @ Rn @ np.conj(np.swapaxes(wm, -1, -2))) / (2.0 * dks[b])

    for s in range(2, max_order + 1):
        src = np.zeros((nk, nb, nb), dtype=np.complex128)
        for b in range(dim):
            if abs(E[b]) < 1e-40:
                continue
            src += E[b] * cov_grad(rhos[s - 1], b)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = s * omega + iGamma - eps
            rhos[s] = np.where(np.abs(denom) > 0, src / denom, 0.0)

    w = np.asarray(weights, dtype=np.float64).reshape(nk)
    currents: dict[int, ComplexArray] = {}
    for s in range(1, max_order + 1):
        Js = np.zeros(3, dtype=np.complex128)
        for i in range(dim):
            tr = np.einsum("kmn,knm->k", vel[i], rhos[s], optimize=True)
            Js[i] = np.sum(tr * w)
        currents[s] = Js
    return currents


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
) -> dict[int, ComplexArray]:
    """BZ-summed harmonic currents J^(s)_i = Σ_k w_k Tr[v_i ρ^(s)(k, s·ω)].

    Parameters
    ----------
    H_func      : H(kx,ky,kz) -> (nb,nb) hermitian, atomic units.
    kpts        : (nk, 3) k-points, C-ordered to match ``shape``.
    shape       : (nkx, nky, nkz) mesh shape (for the covariant gradient rolls).
    bounds      : per-axis (lo, hi) of the reciprocal box (for grid spacing).
    weights     : (nk,) BZ quadrature weights.
    E_field     : complex 3-vector — amplitude of the e^{-iωt} drive.
    max_order   : highest order s to compute.
    gamma       : coherence dephasing 1/T2 (off-diagonal denominators).
    gamma_pop   : population dephasing 1/T1 (diagonal); default = gamma.
    Returns
    -------
    {s: J^(s)} for s = 1..max_order, each a complex 3-vector.
    """
    band = precompute_band_data(H_func, kpts, shape, bounds, mu=mu, T_au=T_au,
                                dimension=dimension, dk_vel=dk_vel,
                                distribution=distribution)
    return harmonic_currents(band, weights, E_field, omega, max_order,
                             gamma=gamma, gamma_pop=gamma_pop)


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
