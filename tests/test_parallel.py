"""Worker-count resolution must be correct and cross-platform.

These guard the fix that made cluster runs use the *whole* SLURM allocation
instead of a local heuristic fraction of the node.
"""
from __future__ import annotations

import os

import pytest

from qxti.utils import parallel
from qxti.response.cmd import _contiguous_chunks


_ENV_KEYS = (
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_JOB_CPUS_PER_NODE",
    "QXTI_NUM_WORKERS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield


def test_explicit_request_wins_over_everything(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
    monkeypatch.setenv("QXTI_NUM_WORKERS", "32")
    # config n_workers = 7 must win (the "o en su defecto el input params" rule)
    assert parallel.resolve_worker_count(7) == 7


def test_slurm_allocation_is_used_in_full():
    os.environ["SLURM_CPUS_PER_TASK"] = "48"
    # NOT halved, NOT a local heuristic: the whole allocation.
    assert parallel.resolve_worker_count() == 48


def test_cap_is_respected(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
    assert parallel.resolve_worker_count(cap=10) == 10
    # request still wins, but cap bounds it
    assert parallel.resolve_worker_count(100, cap=16) == 16


def test_env_override(monkeypatch):
    monkeypatch.setenv("QXTI_NUM_WORKERS", "12")
    assert parallel.resolve_worker_count() == 12


def test_slurm_per_task_beats_env(monkeypatch):
    # priority: request > QXTI_NUM_WORKERS > SLURM
    monkeypatch.setenv("QXTI_NUM_WORKERS", "12")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
    assert parallel.resolve_worker_count() == 12  # env beats SLURM


def test_slurm_job_cpus_per_node_parsing(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "128(x2)")
    assert parallel.slurm_cpu_allocation() == 128


def test_slurm_cpus_on_node_fallback(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "36")
    assert parallel.slurm_cpu_allocation() == 36


def test_no_slurm_returns_none():
    assert parallel.slurm_cpu_allocation() is None


def test_always_at_least_one(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "0")  # degenerate
    assert parallel.resolve_worker_count() >= 1
    assert parallel.resolve_worker_count(-5) >= 1


def test_available_cpus_positive():
    assert parallel.available_cpus() >= 1


def test_configure_thread_env_pins_blas(monkeypatch):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    parallel.configure_thread_env()
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"


def test_configure_thread_env_respects_override(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("QXTI_BLAS_THREADS", "4")
    parallel.configure_thread_env(force=True)
    assert os.environ["OMP_NUM_THREADS"] == "4"


def test_configure_runtime_env_sets_cache(monkeypatch):
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    parallel.configure_runtime_env()
    assert os.environ["MPLCONFIGDIR"].endswith("qxti_cache")


def test_parallel_plan_mentions_source(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
    plan = parallel.parallel_plan()
    assert "64" in plan and "SLURM" in plan


def test_cmd_chunks_remain_valid_when_workers_exceed_remaining_points():
    chunks = _contiguous_chunks(start=1, stop=9, n_workers=168)

    assert chunks == [(index, index + 1) for index in range(1, 9)]
    assert all(lo < hi for lo, hi in chunks)
    assert [index for lo, hi in chunks for index in range(lo, hi)] == list(range(1, 9))
