"""Cross-platform worker-count resolution and thread-environment setup.

Single source of truth for *how many parallel workers QXTI should use*, so the
same code path uses **all** the cores it is entitled to on a laptop, a
workstation, and a SLURM cluster node — without ever silently falling back to a
fraction of the machine.

Resolution priority (highest first):

1. an explicit positive request (e.g. ``[cmd] n_workers`` in the ``.cfg``);
2. the ``QXTI_NUM_WORKERS`` environment variable;
3. the SLURM allocation for this task
   (``SLURM_CPUS_PER_TASK`` → ``SLURM_CPUS_ON_NODE`` → ``SLURM_JOB_CPUS_PER_NODE``);
4. otherwise **all usable local cores** — the maximum (Linux
   ``os.sched_getaffinity(0)`` respects cgroups / ``taskset`` / containers;
   other OSes use ``os.cpu_count()``).

Opt-in: on Apple Silicon, ``QXTI_MAC_PERF_CORES=1`` restricts the default to the
performance-core count (efficiency cores + the GIL can make extra threads
slower).  It is NOT the default — the default is "use every core".

Key rules: on a cluster an allocation of ``N`` means ``N``; locally ``n_workers
= 0`` means *all* cores.  This module never halves the count (the old per-engine
heuristics did — that is why cluster runs used a fraction of the node and Macs
used only their performance cores).
"""
from __future__ import annotations

import os
import platform
import re

__all__ = [
    "resolve_worker_count",
    "available_cpus",
    "slurm_cpu_allocation",
    "configure_thread_env",
    "configure_runtime_env",
    "parallel_plan",
]


def slurm_cpu_allocation() -> int | None:
    """Cores SLURM gave THIS task, or ``None`` when not under SLURM.

    ``SLURM_CPUS_PER_TASK`` is the per-process allocation (what one QXTI process
    may use).  Fall back to the per-node counts, parsing the leading integer of
    forms like ``"64"`` or ``"64(x2)"``.
    """
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v and v.strip().isdigit():
        return max(1, int(v))
    for key in ("SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        val = os.environ.get(key)
        if val:
            m = re.match(r"\s*(\d+)", val)
            if m:
                return max(1, int(m.group(1)))
    return None


def available_cpus() -> int:
    """Usable logical CPUs, honouring the affinity mask when the OS exposes it."""
    getaff = getattr(os, "sched_getaffinity", None)
    if getaff is not None:  # Linux: respects taskset / cgroups / SLURM binding
        try:
            n = len(getaff(0))
            if n > 0:
                return n
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _macos_performance_cores() -> int | None:
    if platform.system() != "Darwin":
        return None
    try:
        import subprocess  # noqa: PLC0415

        out = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        n = int(out.stdout.strip())
        return n if n > 0 else None
    except Exception:
        return None


def _env_worker_count() -> int | None:
    v = os.environ.get("QXTI_NUM_WORKERS")
    if v and v.strip().isdigit() and int(v) > 0:
        return int(v)
    return None


def _perf_core_optin() -> bool:
    """Opt-in to the Apple-Silicon performance-cores-only heuristic."""
    return os.environ.get("QXTI_MAC_PERF_CORES", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _auto_worker_count() -> int:
    """Best default when nothing explicit was requested.

    Uses the MAXIMUM cores available ("usa el máximo de núcleos de la PC"): the
    full SLURM allocation on a cluster, otherwise every usable local core on
    mac/win/linux.  On Apple Silicon you can opt into performance-cores-only
    (sometimes faster — efficiency cores + the GIL) with ``QXTI_MAC_PERF_CORES=1``.
    """
    slurm = slurm_cpu_allocation()
    if slurm:
        return slurm  # cluster: use the WHOLE allocation, never a fraction
    if _perf_core_optin():  # opt-in Apple-Silicon heuristic (not the default)
        perf = _macos_performance_cores()
        if perf:
            return perf
    return available_cpus()  # DEFAULT: all usable local cores (the maximum)


def resolve_worker_count(
    requested: int | None = None, *, cap: int | None = None
) -> int:
    """Return the worker count to use (always ``>= 1``).

    ``requested`` is the value from the config (``n_workers``); a positive value
    wins over everything (this is the "o en su defecto el colocado en los input
    params" rule).  ``cap`` bounds the result (e.g. by the number of frequencies
    or a RAM budget).
    """
    if requested is not None and int(requested) > 0:
        n = int(requested)
    else:
        n = _env_worker_count() or _auto_worker_count()
    if cap is not None and int(cap) > 0:
        n = min(n, int(cap))
    return max(1, n)


def parallel_plan(requested: int | None = None, *, cap: int | None = None) -> str:
    """Human-readable one-liner: how many workers and WHY (for logs)."""
    n = resolve_worker_count(requested, cap=cap)
    if requested and int(requested) > 0:
        src = "config n_workers"
    elif _env_worker_count():
        src = "QXTI_NUM_WORKERS env"
    elif slurm_cpu_allocation():
        src = "SLURM allocation"
    elif _perf_core_optin() and _macos_performance_cores():
        src = "macOS performance cores (opt-in)"
    else:
        src = "all usable cores"
    return f"{n} workers (source: {src}; usable CPUs={available_cpus()})"


def configure_thread_env(force: bool = False) -> None:
    """Pin the BLAS/OpenMP thread pools (default 1) to avoid oversubscription.

    QXTI parallelises over k with a Python ``ThreadPoolExecutor``; if BLAS also
    spawns threads inside every NumPy call the machine runs ``workers × BLAS``
    threads and thrashes.  Pin BLAS to one thread so the k-loop owns the
    parallelism.  Override with ``QXTI_BLAS_THREADS`` (e.g. for a few very large
    dense diagonalisations).  Call this BEFORE importing NumPy for full effect.
    """
    n = os.environ.get("QXTI_BLAS_THREADS", "1")
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if force or var not in os.environ:
            os.environ[var] = n


def configure_runtime_env() -> None:
    """Set a cross-platform matplotlib cache dir and pin BLAS threads.

    Replaces the old hard-coded ``/private/tmp`` (macOS-only) with a temp dir
    that exists on Linux/Windows/macOS and honours ``$TMPDIR`` (which SLURM sets
    per-job).  Idempotent; safe to call from any entry point.
    """
    configure_thread_env()
    import tempfile  # noqa: PLC0415

    cache = os.path.join(tempfile.gettempdir(), "qxti_cache")
    os.environ.setdefault("MPLCONFIGDIR", cache)
    os.environ.setdefault("XDG_CACHE_HOME", cache)
