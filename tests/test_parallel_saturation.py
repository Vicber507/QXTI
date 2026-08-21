"""Parallel-saturation regression tests.

Guards the fix for "reserved N cores, used 1": block sizing must cut the work into
>= workers pieces (not just the largest block that fits RAM), and the tddm result
must stay independent of the worker/pool choice."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

import qxti.utils.memory as mem


def test_split_units_gives_at_least_workers_blocks_when_ram_ample():
    # tiny bytes/unit -> RAM never the limit -> must split into >= workers blocks
    per_block, n_blocks = mem.split_units(1000, bytes_per_unit=8.0, workers=10, reserve_gb=1.0)
    assert n_blocks >= 10, (per_block, n_blocks)
    # ~ceil(1000/10) per block, so ~10 blocks (not one 1000-unit block)
    assert per_block <= 1000 // 10 + 1


def test_split_units_single_worker_is_one_block():
    per_block, n_blocks = mem.split_units(1000, bytes_per_unit=8.0, workers=1, reserve_gb=1.0)
    assert n_blocks == 1
    assert per_block == 1000


def test_split_units_ram_tight_makes_more_not_fewer_blocks():
    # absurd bytes/unit -> RAM caps to ~1 unit/block -> MANY blocks (never < workers)
    per_block, n_blocks = mem.split_units(1000, bytes_per_unit=1e15, workers=4, reserve_gb=0.0)
    assert per_block >= 1
    assert n_blocks >= 4


def _tddm_current(cfg, ham, kgrid, weights, dim, A_t, dt, n_workers):
    from qxti.analytics.tddm import _tddm_current_time
    Jt, _, _ = _tddm_current_time(
        ham, kgrid, weights, A_t, dt, dim, cfg.cmd,
        n_workers=n_workers, reserve_gb=1.0, progress=False)
    return Jt


def test_tddm_result_independent_of_worker_count():
    """The physical current must not depend on how many workers/blocks are used."""
    from qxti.core import QXTIConfig, QXTISimulation
    from qxti.analytics.theory_response import build_k_integration_weights

    cfg = QXTIConfig.from_file("inputs/inputParams.haldane_topological.cfg")
    cfg = replace(cfg, kgrid=replace(cfg.kgrid, k_points=(8, 8)))  # 64 k, small & fast
    sim = QXTISimulation(config=cfg)
    ham = sim.build_hamiltonian()
    kgrid = sim.build_kgrid(ham)
    weights = build_k_integration_weights(cfg, hamiltonian=ham, kgrid=kgrid)
    dim = int(ham.dimension)

    Nt, dt = 200, 0.08
    t = np.arange(Nt) * dt
    # synthetic smooth vector potential (worker-count invariance is field-independent;
    # no laser and no integral needed here).
    A_t = np.zeros((Nt, 3))
    A_t[:, 0] = 3.5e-4 * np.sin(0.09 * t) * np.exp(-((t - Nt * dt / 2) / (Nt * dt / 5)) ** 2)

    J_serial = _tddm_current(cfg, ham, kgrid, weights, dim, A_t, dt, n_workers=1)
    J_par = _tddm_current(cfg, ham, kgrid, weights, dim, A_t, dt, n_workers=4)  # ProcessPool (nb=2)

    rel = np.max(np.abs(J_par - J_serial)) / (np.max(np.abs(J_serial)) + 1e-30)
    assert rel < 1e-10, f"tddm result changed with worker count: rel diff {rel:.2e}"


def test_mesh_procpool_result_matches_serial():
    """The mesh (pfddm) ProcessPool must give the same currents as the serial pass."""
    import qxti.analytics.mesh_response as MR
    from qxti.core import QXTIConfig
    from qxti.analytics.theory_response import compute_hhg_spectrum

    cfg = QXTIConfig.from_file("inputs/inputParams.haldane_topological.cfg")
    cfg = replace(cfg, kgrid=replace(cfg.kgrid, k_points=(24, 24)))

    J_serial = compute_hhg_spectrum(
        replace(cfg, cmd=replace(cfg.cmd, n_workers=1)), max_order=3, progress=False)["J_total"]

    orig = MR._MESH_PROCPOOL_MIN_WORK
    MR._MESH_PROCPOOL_MIN_WORK = 0     # force the ProcessPool even for this tiny grid
    try:
        J_proc = compute_hhg_spectrum(
            replace(cfg, cmd=replace(cfg.cmd, n_workers=4)), max_order=3, progress=False)["J_total"]
    finally:
        MR._MESH_PROCPOOL_MIN_WORK = orig

    rel = np.max(np.abs(J_proc - J_serial)) / (np.max(np.abs(J_serial)) + 1e-30)
    assert rel < 1e-10, f"mesh ProcessPool changed the result: rel diff {rel:.2e}"


def test_cmd_fork_result_matches_serial(monkeypatch):
    """ptddm/CMD fork ProcessPool must give the same orders as the serial solve."""
    import pytest
    monkeypatch.setenv("QXTI_FORCE_FORK", "1")   # exercise the fork path off-Linux too
    import qxti.response.cmd as C
    if not C._cmd_fork_supported():
        pytest.skip("fork start method unavailable on this platform")
    from qxti.core import QXTIConfig, QXTISimulation

    cfg = QXTIConfig.from_file("inputs/inputParams.haldane_topological.cfg")
    cfg = replace(cfg, kgrid=replace(cfg.kgrid, k_points=(6, 6)),
                  laser=replace(cfg.laser, ncycles=3.0), cmd=replace(cfg.cmd, max_order=3))

    def solve(nw):
        sim = QXTISimulation(config=cfg)
        ham = sim.build_hamiltonian()
        cmd = sim.build_cmd(ham)
        cmd._n_workers = nw
        cmd.progress_enabled = False
        return cmd.solve_time_domain_in_memory()

    o_serial = solve(1)   # serial
    o_fork = solve(4)     # fork ProcessPool
    for s in o_serial:
        rel = (np.max(np.abs(np.asarray(o_fork[s]) - np.asarray(o_serial[s])))
               / (np.max(np.abs(o_serial[s])) + 1e-30))
        assert rel < 1e-10, f"CMD fork order {s} differs from serial: rel diff {rel:.2e}"


def test_cmd_streaming_currents_match_in_memory():
    """The streaming ptddm path (no full-rho in RAM) must give the SAME observed
    currents J^(s)(t) as tracing v.rho from the in-memory density matrix."""
    from qxti.core import QXTIConfig, QXTISimulation

    cfg = QXTIConfig.from_file("inputs/inputParams.haldane_topological.cfg")
    # complex128 scratch so the check isolates the streaming ALGORITHM from the
    # reduced-precision disk scratch (the default complex64/float16 scratch makes
    # higher orders differ from the complex128 in-memory path -- precision, not a bug).
    cfg = replace(cfg, kgrid=replace(cfg.kgrid, k_points=(6, 6)),
                  laser=replace(cfg.laser, ncycles=3.0),
                  cmd=replace(cfg.cmd, max_order=3, basis="band",
                              rho_storage_dtype="complex128",
                              scratch_rho_storage_dtype="complex128"))
    sim = QXTISimulation(config=cfg)
    ham = sim.build_hamiltonian()
    dim = int(ham.dimension)
    directions = ("x", "y", "z")[:dim]
    nk = int(np.asarray(sim.build_kgrid(ham).points()).shape[0])
    weights = np.full(nk, 1.0 / nk)   # any consistent weights (they cancel in the check)

    # streaming: J^(s)(t) accumulated on the fly, full rho never in RAM
    cmd_s = sim.build_cmd(ham)
    cmd_s.progress_enabled = False
    acc = cmd_s.solve_currents_streaming(weights)

    # in-memory reference: trace the stored rho with the SAME weights + operators
    cmd_m = sim.build_cmd(ham)
    cmd_m.progress_enabled = False
    orders = cmd_m.solve_time_domain_in_memory()      # {s: (nk, nt, nb, nb)} band basis

    J_ref = {}
    for s in range(1, int(cfg.cmd.max_order) + 1):
        rho = np.asarray(orders[s], dtype=np.complex128)      # (nk, nt, nb, nb)
        Js = np.zeros((rho.shape[1], 3), dtype=np.float64)
        for axis, d in enumerate(directions):
            v = np.asarray(cmd_m.band_gauge_frame.current(d), dtype=np.complex128)  # (nk, nb, nb)
            Js[:, axis] = np.real(np.einsum("k,kmn,ktnm->t", weights, v, rho, optimize=True))
        J_ref[s] = Js
    # Normalise the tolerance by the GLOBAL current scale: some orders are exactly
    # zero by symmetry (e.g. the 2nd-harmonic current in centrosymmetric Haldane), so
    # a per-order relative error would divide machine noise by ~0.
    scale = max(float(np.max(np.abs(v))) for v in J_ref.values()) + 1e-300
    for s in J_ref:
        abs_err = float(np.max(np.abs(acc.current_time(s) - J_ref[s])))
        assert abs_err < 1e-10 * scale, (
            f"order {s}: streaming vs in-memory current differ by {abs_err:.2e} "
            f"(global scale {scale:.2e})")


def test_multilaser_streaming_matches_all_k():
    """The memory-safe block-streamed MULTI-LASER path must equal the whole-grid
    time-domain solve (incl. the mixing frequencies) to machine precision.

    Uses ``valence_occupation`` + a field strong enough that orders 2/3 are clearly
    NONZERO, and FORCES thin blocks (so _plan_stream would pick a 1-plane block).  A
    1-plane block can't form the axis-0 halo, which used to silently zero the higher
    orders -- the assertions below (order 3 nonzero AND matching the whole-grid pass)
    guard that regression."""
    import qxti.analytics.mesh_response as MR
    from qxti.core import QXTIConfig, QXTISimulation
    from qxti.analytics.theory_response import build_k_integration_weights, _resolve_distribution

    cfg = QXTIConfig.from_file("inputs/inputParams.haldane_topological.cfg")
    cfg = replace(cfg, kgrid=replace(cfg.kgrid, k_points=(12, 12)))
    sim = QXTISimulation(config=cfg)
    ham = sim.build_hamiltonian()
    kg = sim.build_kgrid(ham)
    w = build_k_integration_weights(cfg, hamiltonian=ham, kgrid=kg)
    kp = kg.points()
    shape = tuple(int(kg.shape[a]) for a in range(3))
    bnd = ham.reciprocal_box_bounds()
    dim = int(ham.dimension)
    dist = _resolve_distribution("valence_occupation")
    mo, Nt, dt = 3, 400, 0.08
    t = np.arange(Nt) * dt
    env = np.exp(-((t - Nt * dt / 2) / (Nt * dt / 5)) ** 2)
    E = np.zeros((Nt, 3))
    E[:, 0] = 1.6e-3 * (np.cos(0.09 * t) + 0.7 * np.cos(0.13 * t)) * env  # two carriers, mixing
    if dim >= 2:
        E[:, 1] = 1.6e-3 * 0.5 * np.cos(0.11 * t) * env

    band = MR.precompute_band_data(ham._matrix_at, kp, shape, bnd, dimension=dim,
                                   halo=mo - 1, distribution=dist)
    ref = MR.time_domain_currents(band, w, E, dt, mo, gamma=1e-3)
    # sanity: the reference actually HAS nonzero higher-order content (else 0==0)
    assert float(np.max(np.abs(ref["J_t"][3]))) > 1e-6, "test setup has no order-3 signal"

    orig = MR._plan_stream
    MR._plan_stream = lambda n0, bpp, halo, nw, rg, cap: ("block", 4, 1)  # force 1-plane -> exercises the >=2 guard
    try:
        blk = MR.time_domain_currents_meshed(
            ham._matrix_at, kp, shape, bnd, w, E, dt, mo,
            gamma=1e-3, mu=0.0, T_au=0.0, distribution=dist, dimension=dim)
    finally:
        MR._plan_stream = orig

    # normalise by order 1 (always well-defined); the forbidden even harmonic can be
    # a 0/0 NaN in the base solve, so compare NaN-safe.
    scale = float(np.max(np.abs(np.nan_to_num(ref["J_t"][1])))) + 1e-300
    for s in range(1, mo + 1):
        d = np.nan_to_num(np.asarray(blk["J_t"][s])) - np.nan_to_num(np.asarray(ref["J_t"][s])[:, :dim])
        abs_err = float(np.max(np.abs(d)))
        assert abs_err < 1e-8 * scale, f"multi-laser order {s} streamed vs all-k differ: {abs_err:.2e}"
    # the ODD order-3 mixing content must survive the block streaming (the bug zeroed it)
    assert float(np.max(np.abs(np.nan_to_num(blk["J_t"][3])))) > 1e-6, "order 3 vanished (block bug)"
