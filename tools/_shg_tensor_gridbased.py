#!/usr/bin/env python3
"""Grid-based analytic chi^(2)/sigma^(2) FULL tensor at a single omega (with k mask).

Extends the diagonal-input `_order2_gridbased` path to off-diagonal input
components sigma_{i,jk} (j!=k), which the SHG circular dichroism needs (chi_xzx).
Uses the SAME converged covariant-gradient BZ integral as the susceptibility
engine (vectorized np.roll gradient), NOT the per-k HHG recursion.

sigma_{i,bc} = conj( sum_k w_k  Tr[ J_i . (D_b rho1_c) . 1/(2w - eps) ] ),
symmetrized over the input pair (b,c). Returns sigma[i,j,k] (dim^3), complex.
"""
from __future__ import annotations

import numpy as np

from qxti.analytics.theory_response import _resolve_distribution


def order2_full_tensor(hamiltonian, kgrid, omega, weights, ccfg):
    dim = int(hamiltonian.dimension)
    nb = int(hamiltonian.basis_size)
    shape = tuple(int(kgrid.shape[a]) for a in range(3))
    k_points = np.asarray(kgrid.points(), dtype=np.float64)
    nk = k_points.shape[0]
    Hf = hamiltonian._matrix_at
    bounds = hamiltonian.reciprocal_box_bounds()
    dks = [(float(bounds[a][1]) - float(bounds[a][0])) / shape[a] for a in range(dim)]
    dist = _resolve_distribution(ccfg.distribution)
    mu = float(ccfg.fermi_level)
    T = float(ccfg.temperature)
    gamma = 0.0 if ccfg.coherence_time <= 0 else 1.0 / float(ccfg.coherence_time)
    dk = 1.0e-4

    def Hbatch(kc):
        return np.array([Hf(float(k[0]), float(k[1]), float(k[2])) for k in kc], dtype=np.complex128)

    energies, U = np.linalg.eigh(Hbatch(k_points))
    Udag = np.conj(np.transpose(U, (0, 2, 1)))
    vel = []
    for a in range(dim):
        sh = np.zeros(3); sh[a] = dk
        dH = (Hbatch(k_points + sh) - Hbatch(k_points - sh)) / (2 * dk)
        vel.append(Udag @ dH @ U)

    f = np.asarray(dist(energies, mu, T), dtype=np.float64)
    eps = energies[:, :, None] - energies[:, None, :]
    fmn = f[:, None, :] - f[:, :, None]
    offdiag = ~np.eye(nb, dtype=bool)
    valid = offdiag[None] & (np.abs(eps) > 1e-20)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_eps = np.where(valid, 1.0 / eps, 0.0)
        dfde = (-f * (1.0 - f) / T) if T > 1e-15 else np.zeros_like(f)
    A = [1j * vel[a] * inv_eps for a in range(dim)]
    ow1 = omega + 1j * gamma
    ow2 = 2.0 * omega + 1j * gamma

    # rho1 for each input direction c (response to field along c).
    diagidx = np.arange(nb)
    valid_diag = valid | np.eye(nb, dtype=bool)[None]
    rho1 = []
    for c in range(dim):
        r = np.where(valid, (A[c] * fmn) / (ow1 - eps), 0.0 + 0.0j)
        diag_src = (-1j) * dfde * np.real(np.diagonal(vel[c], axis1=1, axis2=2))
        r[:, diagidx, diagidx] += diag_src / ow1
        r = np.where(valid_diag, r, 0.0)
        rho1.append(r)

    U_mesh = U.reshape(*shape, nb, nb)
    Udag_mesh = Udag.reshape(*shape, nb, nb)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_d2 = np.where(valid, 1.0 / (ow2 - eps), 0.0 + 0.0j)
    w = np.asarray(weights, dtype=np.float64).reshape(nk)

    Tt = np.zeros((dim, dim, dim), dtype=np.complex128)   # T[i,b,c]
    for b in range(dim):
        Up = np.roll(U_mesh, -1, axis=b); Um = np.roll(U_mesh, +1, axis=b)
        Wp = (Udag_mesh @ Up).reshape(nk, nb, nb); Wm = (Udag_mesh @ Um).reshape(nk, nb, nb)
        Wp_d = np.conj(np.swapaxes(Wp, -1, -2)); Wm_d = np.conj(np.swapaxes(Wm, -1, -2))
        Ab = A[b]
        for c in range(dim):
            rc_mesh = rho1[c].reshape(*shape, nb, nb)
            rp = np.roll(rc_mesh, -1, axis=b).reshape(nk, nb, nb)
            rm = np.roll(rc_mesh, +1, axis=b).reshape(nk, nb, nb)
            dpart = (Wp @ rp @ Wp_d - Wm @ rm @ Wm_d) / (2.0 * dks[b])
            comm = Ab @ rho1[c] - rho1[c] @ Ab
            rho2 = (dpart - 1j * comm) * inv_d2
            for i in range(dim):
                Ji = -vel[i]
                tr = np.einsum("kmn,knm->k", Ji, rho2, optimize=True)
                Tt[i, b, c] = np.conj(np.sum(tr * w))

    sig = 0.5 * (Tt + np.transpose(Tt, (0, 2, 1)))   # symmetrize input pair
    return sig


def order2_full_tensor_spectrum(hamiltonian, kgrid, omega_axis, weights, ccfg,
                                *, omega_chunk=6, progress=False):
    """Full sigma^(2)_{ijk}(omega) tensor over an omega array (grid-based, converged).

    Same covariant-gradient BZ integral as ``order2_full_tensor`` but vectorized
    over frequency with omega-chunking to bound memory. Returns sigma of shape
    (n_omega, dim, dim, dim), complex.
    """
    dim = int(hamiltonian.dimension)
    nb = int(hamiltonian.basis_size)
    shape = tuple(int(kgrid.shape[a]) for a in range(3))
    k_points = np.asarray(kgrid.points(), dtype=np.float64)
    nk = k_points.shape[0]
    omega_axis = np.asarray(omega_axis, dtype=np.float64)
    nw = omega_axis.size
    Hf = hamiltonian._matrix_at
    bounds = hamiltonian.reciprocal_box_bounds()
    dks = [(float(bounds[a][1]) - float(bounds[a][0])) / shape[a] for a in range(dim)]
    dist = _resolve_distribution(ccfg.distribution)
    mu = float(ccfg.fermi_level); T = float(ccfg.temperature)
    gamma = 0.0 if ccfg.coherence_time <= 0 else 1.0 / float(ccfg.coherence_time)
    dk = 1.0e-4

    def Hbatch(kc):
        return np.array([Hf(float(k[0]), float(k[1]), float(k[2])) for k in kc], dtype=np.complex128)

    energies, U = np.linalg.eigh(Hbatch(k_points))
    Udag = np.conj(np.transpose(U, (0, 2, 1)))
    vel = []
    for a in range(dim):
        sh = np.zeros(3); sh[a] = dk
        vel.append(Udag @ ((Hbatch(k_points + sh) - Hbatch(k_points - sh)) / (2 * dk)) @ U)
    f = np.asarray(dist(energies, mu, T), dtype=np.float64)
    eps = energies[:, :, None] - energies[:, None, :]
    fmn = f[:, None, :] - f[:, :, None]
    offdiag = ~np.eye(nb, dtype=bool)
    valid = offdiag[None] & (np.abs(eps) > 1e-20)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_eps = np.where(valid, 1.0 / eps, 0.0)
        dfde = (-f * (1.0 - f) / T) if T > 1e-15 else np.zeros_like(f)
    A = [1j * vel[a] * inv_eps for a in range(dim)]
    diagidx = np.arange(nb)
    valid_diag = valid | np.eye(nb, dtype=bool)[None]

    U_mesh = U.reshape(*shape, nb, nb)
    Udag_mesh = Udag.reshape(*shape, nb, nb)
    w = np.asarray(weights, dtype=np.float64).reshape(nk)
    # Wilson-link transports per gradient direction (omega-independent).
    Wp = []; Wm = []
    for b in range(dim):
        Up = np.roll(U_mesh, -1, axis=b); Um = np.roll(U_mesh, +1, axis=b)
        Wp.append((Udag_mesh @ Up).reshape(nk, nb, nb))
        Wm.append((Udag_mesh @ Um).reshape(nk, nb, nb))
    Wp_d = [np.conj(np.swapaxes(x, -1, -2)) for x in Wp]
    Wm_d = [np.conj(np.swapaxes(x, -1, -2)) for x in Wm]

    sig = np.zeros((nw, dim, dim, dim), dtype=np.complex128)
    for wstart in range(0, nw, omega_chunk):
        wstop = min(wstart + omega_chunk, nw)
        wch = omega_axis[wstart:wstop]
        nwc = wch.size
        if progress:
            print(f"[tensor-spec] omega {wstart+1}-{wstop}/{nw}", flush=True)
        ow1 = (wch + 1j * gamma)[None, :, None, None]        # (1,nwc,1,1)
        ow2 = (2.0 * wch + 1j * gamma)[None, :, None, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_d2 = np.where(valid[:, None], 1.0 / (ow2 - eps[:, None]), 0.0)   # (nk,nwc,nb,nb)
        Tt = np.zeros((nwc, dim, dim, dim), dtype=np.complex128)
        for c in range(dim):                                 # input direction c
            with np.errstate(divide="ignore", invalid="ignore"):
                rc = np.where(valid[:, None], (A[c] * fmn)[:, None] / (ow1 - eps[:, None]), 0.0)
            diag_src = (-1j) * dfde * np.real(np.diagonal(vel[c], axis1=1, axis2=2))
            rc[:, :, diagidx, diagidx] += diag_src[:, None] / (wch + 1j * gamma)[None, :, None]
            rc = np.where(valid_diag[:, None], rc, 0.0)
            rc_mesh = rc.reshape(*shape, nwc, nb, nb)
            for b in range(dim):                             # gradient direction b
                rp = np.roll(rc_mesh, -1, axis=b).reshape(nk, nwc, nb, nb)
                rm = np.roll(rc_mesh, +1, axis=b).reshape(nk, nwc, nb, nb)
                dpart = (Wp[b][:, None] @ rp @ Wp_d[b][:, None]
                         - Wm[b][:, None] @ rm @ Wm_d[b][:, None]) / (2.0 * dks[b])
                comm = A[b][:, None] @ rc - rc @ A[b][:, None]
                rho2 = (dpart - 1j * comm) * inv_d2
                for i in range(dim):
                    Ji = -vel[i]
                    tr = np.einsum("kmn,kwnm->wk", Ji, rho2, optimize=True)
                    Tt[:, i, b, c] = np.conj(tr @ w)
        sig[wstart:wstop] = 0.5 * (Tt + np.transpose(Tt, (0, 1, 3, 2)))
    return sig
