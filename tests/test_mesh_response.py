"""Validation of the mesh-vectorized response vs the per-k recursion.

The fast mesh path (``qxti.analytics.mesh_response``) must reproduce the slow
per-k recursion (``rho_analytic.rho_order_s``) BIT-FOR-BIT on a periodic lattice
with the grid spacing used as the finite-difference step, because then both use
the identical discretization — one vectorized, one looped.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.analytics.mesh_response import (
    mesh_harmonic_currents,
    perk_harmonic_currents,
    uniform_mp_grid,
)

_AU = 0.529177210903
_A = 3.4 / _AU


def natphys_H(kx: float, ky: float, kz: float) -> np.ndarray:
    """Periodic 4-band TaAs Weyl model (period 2π/a in every axis)."""
    t, my, mz, delta, a = 0.03, 1.0, 5.0, 0.5, _A
    M = t * (np.cos(kx * a) + my * (1 - np.cos(ky * a)) + mz * (1 - np.cos(kz * a)))
    sy, cy, sz = np.sin(ky * a), np.cos(ky * a), np.sin(kz * a)
    dc = delta * t * cy
    return np.array(
        [[0, M - 1j * t * sy, t * sz, -1j * dc],
         [M + 1j * t * sy, 0, 1j * dc, -t * sz],
         [t * sz, -1j * dc, 0, M - 1j * t * sy],
         [1j * dc, -t * sz, M + 1j * t * sy, 0]],
        dtype=np.complex128,
    )


def _grid(N):
    kmax = np.pi / _A
    bounds = ((-kmax, kmax),) * 3
    shape = (N, N, N)
    kpts, w = uniform_mp_grid(bounds, shape)
    h = 2 * kmax / N
    return kpts, w, shape, bounds, h


def test_mesh_matches_perk_to_machine_precision() -> None:
    """On a periodic lattice with dk_grad = grid spacing, mesh == per-k exactly."""
    kpts, w, shape, bounds, h = _grid(6)
    E = np.array([0.0, 0.0, 1.0e-3], dtype=np.complex128)  # z-drive
    kw = dict(gamma=1.0 / 110.0, mu=0.0, T_au=0.0, dimension=3)

    Jm = mesh_harmonic_currents(natphys_H, kpts, shape, bounds, w, E, 0.02, 3, **kw)
    Jp = perk_harmonic_currents(natphys_H, kpts, w, E, 0.02, 3, dk_grad=h, **kw)

    for s in (1, 2, 3):
        assert np.linalg.norm(Jp[s]) > 0.0  # non-trivial response
        err = np.linalg.norm(Jm[s] - Jp[s]) / np.linalg.norm(Jp[s])
        assert err < 1.0e-9, f"order {s}: mesh vs per-k mismatch {err:.2e}"


def test_mesh_harmonic_currents_scale_as_field_power() -> None:
    """J^(s) must scale as E^s (defining property of the s-th perturbative order)."""
    kpts, w, shape, bounds, _ = _grid(8)
    E = np.array([0.0, 0.0, 2.0e-3], dtype=np.complex128)
    scale = 0.4
    kw = dict(gamma=1.0 / 110.0, mu=0.0, T_au=0.0, dimension=3)

    ref = mesh_harmonic_currents(natphys_H, kpts, shape, bounds, w, E, 0.02, 3, **kw)
    sca = mesh_harmonic_currents(natphys_H, kpts, shape, bounds, w, scale * E, 0.02, 3, **kw)

    for s in (1, 2, 3):
        np.testing.assert_allclose(
            sca[s], (scale ** s) * ref[s], rtol=1.0e-9, atol=1.0e-18
        )


def test_mesh_second_order_is_nonzero_zzz() -> None:
    """Regression: the covariant gradient keeps the SHG z-response (χ_zzz) nonzero.

    A double-counted Berry commutator would cancel the intraband/population
    channel; this guards that the one-shot Wilson gradient is used.
    """
    kpts, w, shape, bounds, _ = _grid(8)
    E = np.array([0.0, 0.0, 1.0e-3], dtype=np.complex128)
    J = mesh_harmonic_currents(natphys_H, kpts, shape, bounds, w, E, 0.02, 2,
                               gamma=1.0 / 110.0, dimension=3)
    assert abs(J[2][2]) > 1.0e-2 * abs(J[1][2])  # χ_zzz comparable to χ_zz


def test_uniform_mp_grid_shape_and_weights() -> None:
    bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    kpts, w = uniform_mp_grid(bounds, (4, 4, 4))
    assert kpts.shape == (64, 3)
    # shifted Monkhorst-Pack: no point on the box edge, weights sum to V_BZ = 8
    assert np.all(np.abs(kpts) < 1.0)
    np.testing.assert_allclose(w.sum(), 8.0, rtol=1e-12)
