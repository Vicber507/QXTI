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
    Jt, _ = _tddm_current_time(
        ham, kgrid, weights, A_t, dt, dim, cfg.cmd,
        n_workers=n_workers, reserve_gb=1.0, progress=False)
    return Jt


def test_tddm_result_independent_of_worker_count():
    """The physical current must not depend on how many workers/blocks are used."""
    from qxti.core import QXTIConfig, QXTISimulation
    from qxti.analytics.tddm import _vector_potential_from_field
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
    E = np.zeros((Nt, 3))
    E[:, 0] = 3.5e-4 * np.sin(0.09 * t) * np.exp(-((t - Nt * dt / 2) / (Nt * dt / 5)) ** 2)
    A_t = _vector_potential_from_field(E, dt)

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
