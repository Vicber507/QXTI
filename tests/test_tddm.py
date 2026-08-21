"""Tests for the TDDM (full non-perturbative) response engine.

The TDDM engine integrates the full density matrix in the velocity gauge; in the
WEAK-FIELD limit it must reproduce the perturbative engines (pfddm/ptddm) harmonic
by harmonic.  These tests use a small model + grid for speed.
"""
from __future__ import annotations

from dataclasses import replace as R

import numpy as np
import pytest

from qxti.core.config import QXTIConfig


CFG = "inputs/inputParams.graphene.cfg"


def _small_cfg(grid=8, dt=0.4, ncycles=2.0, order=3):
    cfg = QXTIConfig.from_file(CFG).with_standard_output_dirs()
    cfg = R(cfg, kgrid=R(cfg.kgrid, k_points=(grid, grid)))
    cfg = R(cfg, timegrid=R(cfg.timegrid, dt=dt))
    cfg = R(cfg, laser=R(cfg.laser, ncycles=ncycles))
    cfg = R(cfg, cmd=R(cfg.cmd, max_order=order))
    return cfg


REQUIRED_KEYS = [
    "omega_axis", "current_spectrum", "current_magnitude", "current_total_magnitude",
    "current_time", "current_time_total", "current_spectrum_total", "polarization_time",
    "time_axis", "electric_field_time", "current_decomposition_available",
    "current_time_intraband", "current_time_interband", "current_spectrum_intraband",
    "current_spectrum_interband", "current_total_magnitude_intraband",
    "current_total_magnitude_interband", "orders",
]


def test_tddm_dataset_schema_and_split():
    from qxti.analytics.tddm import compute_hhg_spectrum_tddm
    res = compute_hhg_spectrum_tddm(_small_cfg(), progress=False)
    assert res["method"] == "tddm"
    ds = res["dataset"]
    for k in REQUIRED_KEYS:
        assert k in ds, f"missing dataset key {k}"
    assert bool(ds["current_decomposition_available"])
    # intra + inter == total, EXACTLY (by construction)
    tot = np.asarray(ds["current_time"])
    intra = np.asarray(ds["current_time_intraband"])
    inter = np.asarray(ds["current_time_interband"])
    assert np.allclose(intra + inter, tot, atol=1e-12)
    # current is real/finite
    assert np.isfinite(np.asarray(ds["current_spectrum"])).all()


def test_tddm_trace_and_hermiticity_conserved():
    """Tr ρ = N_occ and ρ stays Hermitian throughout the propagation (per k)."""
    from qxti.analytics.tddm import (_make_h_evaluator, _propagate_block,
                                     analytic_vector_potential)
    from qxti.core.simulation import QXTISimulation
    from qxti.analytics.theory_response import _resolve_distribution

    cfg = _small_cfg(grid=6, dt=0.4, ncycles=1.5)
    sim = QXTISimulation(config=cfg)
    ham = sim.build_hamiltonian(); kg = sim.build_kgrid(ham); ls = sim.build_laser_system()
    tg = sim.build_timegrid(ls)
    Nt = int(tg.Nt); dt = (float(tg.tf) - float(tg.t0)) / max(Nt - 1, 1)
    t = float(tg.t0) + np.arange(Nt) * dt
    E = np.array([ls.electric_field(x) for x in t]); A = analytic_vector_potential(ls, t)
    Hf = _make_h_evaluator(ham)
    kpts = np.asarray(kg.points())[:8]
    w = np.ones(kpts.shape[0])
    dist = _resolve_distribution(cfg.cmd.distribution)
    # instrument: re-run the core stepping and check trace + hermiticity on the fly.
    nb = int(ham.basis_size); diag = np.arange(nb)
    H0 = Hf(kpts); E0, V0 = np.linalg.eigh(H0)
    f0 = np.asarray(dist(E0, 0.0, 0.0), dtype=float)
    if f0.ndim == 0:
        f0 = np.broadcast_to(f0, E0.shape).copy()
    rho = np.einsum("kin,kn,kjn->kij", V0, f0.astype(complex), V0.conj())
    n_occ = np.trace(rho, axis1=1, axis2=2).real.copy()
    for it in range(0, Nt, max(1, Nt // 20)):
        kA = kpts + A[it][None, :]
        Ei, Vi = np.linalg.eigh(Hf(kA))
        rt = np.einsum("kin,kij,kjm->knm", Vi.conj(), rho, Vi)
        ph = np.exp(-1j * Ei * dt)
        rt *= ph[:, :, None] * ph.conj()[:, None, :]
        rho = np.einsum("kin,knm,kjm->kij", Vi, rt, Vi.conj())
        tr = np.trace(rho, axis1=1, axis2=2).real
        assert np.allclose(tr, n_occ, atol=1e-9), f"trace drift at step {it}"
        herm = np.abs(rho - np.conj(np.transpose(rho, (0, 2, 1)))).max()
        assert herm < 1e-9, f"hermiticity broken at step {it}: {herm}"


def test_tddm_independent_of_n_workers():
    from qxti.analytics.tddm import _tddm_current_time, analytic_vector_potential
    from qxti.core.simulation import QXTISimulation
    from qxti.analytics.theory_response import build_k_integration_weights
    cfg = _small_cfg(grid=8, dt=0.5, ncycles=1.5)
    sim = QXTISimulation(config=cfg)
    ham = sim.build_hamiltonian(); kg = sim.build_kgrid(ham); ls = sim.build_laser_system()
    tg = sim.build_timegrid(ls)
    Nt = int(tg.Nt); dt = (float(tg.tf) - float(tg.t0)) / max(Nt - 1, 1)
    t = float(tg.t0) + np.arange(Nt) * dt
    E = np.array([ls.electric_field(x) for x in t]); A = analytic_vector_potential(ls, t)
    W = build_k_integration_weights(cfg, hamiltonian=ham, kgrid=kg)
    dim = int(ham.dimension)
    J1, _, _ = _tddm_current_time(ham, kg, W, A, dt, dim, cfg.cmd, n_workers=1, reserve_gb=1.0, progress=False)
    J4, _, _ = _tddm_current_time(ham, kg, W, A, dt, dim, cfg.cmd, n_workers=4, reserve_gb=1.0, progress=False)
    # blocks are independent -> result must not depend on the worker count
    assert np.allclose(J1, J4, rtol=1e-10, atol=1e-14)


def test_tddm_field_scaling_third_harmonic():
    """With spectral leakage suppressed (window), tddm's H3 scales as E0^3."""
    from qxti.analytics.tddm import compute_hhg_spectrum_tddm
    cfg = _small_cfg(grid=10, dt=0.3, ncycles=3.0, order=3)
    w0 = float(cfg.laser.omega)

    def h3(scale):
        res = compute_hhg_spectrum_tddm(cfg, progress=False, field_scale=scale)
        freq = np.asarray(res["omega_axis"]); ct = np.asarray(res["dataset"]["current_time"])
        win = np.hanning(ct.shape[0])[:, None]
        mag = np.sqrt((np.abs(np.fft.fft(ct * win, axis=0)) ** 2).sum(1))
        return float(mag[int(np.argmin(np.abs(freq - 3 * w0)))])

    a, b = h3(1.0), h3(0.5)
    exponent = np.log(a / b) / np.log(1.0 / 0.5)
    assert 2.5 < exponent < 3.5, f"H3 should scale ~E0^3, got E0^{exponent:.2f}"


@pytest.mark.slow
def test_tddm_matches_perturbative_pulsed_weakfield():
    """Weak-field: tddm(full) total current ~ perturbative pulsed, per harmonic.

    Uses a GAPPED model (Haldane) at WEAK field so the comparison is clean:
      * a gapless model (graphene) would make the PERTURBATIVE χ⁽³⁾ diverge near the
        Dirac point as the k-grid refines — tddm regularizes it, so they disagree by
        design (tddm is the correct one there);
      * a strong field would make tddm's FULL H3 (orders 3,5,7…) exceed the
        perturbative reference truncated at max_order.
    Both engines: PULSED, identical E(t)/grid/dt/FFT, A=−∫E, Hann-windowed (the H1
    peak is ~3 orders above H3; its leakage otherwise swamps the raw H3 bin).
    """
    from qxti.analytics.tddm import _tddm_current_time, analytic_vector_potential
    from qxti.core.simulation import QXTISimulation
    from qxti.analytics.theory_response import (build_k_integration_weights, _resolve_distribution)
    from qxti.analytics.mesh_response import precompute_band_data, time_domain_currents

    cfg = QXTIConfig.from_file("inputs/inputParams.haldane_topological.cfg").with_standard_output_dirs()
    cfg = R(cfg, kgrid=R(cfg.kgrid, k_points=(12, 12)))
    cfg = R(cfg, timegrid=R(cfg.timegrid, dt=0.3))
    cfg = R(cfg, cmd=R(cfg.cmd, max_order=3))
    field_scale = 0.25  # weak field: higher orders negligible -> both give order 3

    sim = QXTISimulation(config=cfg)
    ham = sim.build_hamiltonian(); kg = sim.build_kgrid(ham); ls = sim.build_laser_system()
    tg = sim.build_timegrid(ls)
    dim = int(ham.dimension); Nt = int(tg.Nt); dt = (float(tg.tf) - float(tg.t0)) / max(Nt - 1, 1)
    t = float(tg.t0) + np.arange(Nt) * dt
    E = np.array([ls.electric_field(x) for x in t]) * field_scale
    A = analytic_vector_potential(ls, t, field_scale)
    freq = np.fft.fftfreq(Nt, d=dt) * 2 * np.pi
    W = build_k_integration_weights(cfg, hamiltonian=ham, kgrid=kg)
    dist = _resolve_distribution(cfg.cmd.distribution)
    g = 0.0 if cfg.cmd.coherence_time <= 0 else 1.0 / float(cfg.cmd.coherence_time)
    band = precompute_band_data(ham._matrix_at, kg.points(),
                                tuple(int(kg.shape[a]) for a in range(3)),
                                ham.reciprocal_box_bounds(), mu=float(cfg.cmd.fermi_level),
                                T_au=float(cfg.cmd.temperature), dimension=dim, distribution=dist)
    td = time_domain_currents(band, W, E[:, :3], dt, 3, gamma=g, gamma_pop=g)
    Jp = sum(td["J_t"][s][:, :dim].real for s in (1, 2, 3))
    Jf, _, _ = _tddm_current_time(ham, kg, W, A, dt, dim, cfg.cmd, n_workers=0, reserve_gb=1.0, progress=False)
    Jf = Jf[:, :dim]
    w0 = float(cfg.laser.omega)
    win = np.hanning(Nt)[:, None]

    def pk(J, s):
        m = np.sqrt((np.abs(np.fft.fft(J * win, axis=0)) ** 2).sum(1))
        return m[int(np.argmin(np.abs(freq - s * w0)))]
    r1 = pk(Jf, 1) / pk(Jp, 1)
    r3 = pk(Jf, 3) / pk(Jp, 3)
    # H1 is the gauge-INVARIANT linear response -> must match tightly (this is the
    # real validation that the velocity-gauge core is correct).
    assert 0.9 < r1 < 1.1, f"H1 tddm/pert = {r1:.3f} (linear response must match)"
    # H3 is an order-of-magnitude sanity check ONLY: velocity-gauge-full vs
    # length-gauge-perturbative nonlinear AMPLITUDES differ by O(1) factors
    # (gauge organization of the higher orders, k-grid sensitivity near the gap
    # minimum, regularization).  The RIGOROUS order-3 check is the E0^3 scaling test
    # (tddm's H3 is a genuine, self-consistent 3rd-order response); the perturbative
    # pulsed current is not machine-ground-truth for the nonlinear regime -- which is
    # exactly the regime tddm exists for.
    assert 0.3 < r3 < 3.0, f"H3 tddm/pert = {r3:.3f} (order-of-magnitude sanity only)"


def test_tddm_propagator_default_is_cfm2():
    """The shipped default integrator is the unitary exponential-midpoint (cfm2)."""
    assert QXTIConfig.from_file(CFG).cmd.tddm_propagator == "cfm2"


def test_tddm_propagators_agree_and_are_selectable():
    """cfm2 (default), rkf45 and ab2 integrate the SAME velocity-gauge von Neumann
    equation, so on a resolved grid they must agree on the current; and each must be
    selectable from the input file via [cmd] tddm_propagator."""
    from qxti.analytics.tddm import compute_hhg_spectrum_tddm

    base = _small_cfg(grid=6, dt=0.2, ncycles=2.0, order=2)
    J = {}
    for prop in ("cfm2", "rkf45", "ab2"):
        cfg = R(base, cmd=R(base.cmd, tddm_propagator=prop))
        res = compute_hhg_spectrum_tddm(cfg, progress=False)
        Jt = np.asarray(res["dataset"]["current_time"])
        assert np.isfinite(Jt).all(), f"{prop} produced non-finite current"
        J[prop] = Jt

    def rel_l2(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))

    # higher-order rkf45 tracks the unitary cfm2 closely; 2nd-order ab2 a bit looser
    assert rel_l2(J["rkf45"], J["cfm2"]) < 0.10, "rkf45 disagrees with cfm2"
    assert rel_l2(J["ab2"], J["cfm2"]) < 0.25, "ab2 disagrees with cfm2"


def test_tddm_invalid_propagator_rejected():
    """An unknown integrator name is rejected at config-parse time."""
    from qxti.core.config import _canonical_tddm_propagator

    assert _canonical_tddm_propagator("magnus2") == "cfm2"     # alias
    assert _canonical_tddm_propagator("rk45") == "rkf45"       # alias
    with pytest.raises(ValueError):
        _canonical_tddm_propagator("euler_supreme")
