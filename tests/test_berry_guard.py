"""Tests for the Berry-connection-aware degeneracy guard.

The plain degeneracy guard only reacts to an EXACT band touching (gap ~ 0). The
Berry guard additionally scans max_k |v_mn|/gap (the below-gap intraband /
anomalous driver) and nudges the point count when it finds an anomalous spike,
so a grid that clears the exact-degeneracy floor but lands on a NEAR-node point
still gets rebuilt to the least-singular nearby grid.
"""
from __future__ import annotations

from dataclasses import replace as R

import numpy as np

from qxti.core.config import QXTIConfig
from qxti.core.simulation import QXTISimulation


def _sim(cfg_path, **kgrid_over):
    base = QXTIConfig.from_file(cfg_path).with_standard_output_dirs()
    base = R(base, kgrid=R(base.kgrid, **kgrid_over))
    sim = QXTISimulation(config=base)
    return sim, sim.build_hamiltonian()


def test_berry_guard_config_defaults_and_parse():
    cfg = QXTIConfig.from_file("inputs/inputParams.graphene.cfg")
    assert cfg.kgrid.berry_singularity_guard is True
    assert cfg.kgrid.berry_guard_ratio == 12.0


def test_berry_diagnostic_shapes_and_signs():
    sim, ham = _sim("inputs/inputParams.graphene.cfg", k_points=(16, 16), shifted=True)
    bounds = ham.reciprocal_box_bounds()
    axes = [np.linspace(b[0], b[1], 16, endpoint=False) + 0.5 * (b[1] - b[0]) / 16 for b in bounds]
    axes = axes + [np.array([0.0])] * (3 - len(axes))
    bmax, bp90, worst = sim._grid_berry_diagnostics(ham, [np.asarray(a) for a in axes])
    assert bmax >= bp90 >= 0.0
    assert worst is not None
    assert np.isfinite(bmax)  # exact-degeneracy floor keeps it finite


def test_healthy_shifted_grid_not_over_nudged():
    """A well-behaved shifted 2-D grid must keep its requested point count."""
    for cfg_path in ("inputs/inputParams.graphene.cfg",
                     "inputs/inputParams.haldane_topological.cfg"):
        sim, ham = _sim(cfg_path, k_points=(24, 24), shifted=True)
        kg = sim.build_kgrid(ham)
        assert np.size(kg.kx_values) == 24
        assert np.size(kg.ky_values) == 24


def test_berry_guard_off_leaves_grid_alone():
    sim, ham = _sim("inputs/inputParams.graphene.cfg",
                    k_points=(24, 24), shifted=True, berry_singularity_guard=False)
    kg = sim.build_kgrid(ham)
    assert np.size(kg.kx_values) == 24
