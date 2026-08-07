"""QXTI package.

Pin the BLAS/OpenMP thread pools to 1 thread AS EARLY AS POSSIBLE — before any
submodule imports NumPy — so the k-loop worker pool owns the parallelism and BLAS
never oversubscribes the machine (``workers × BLAS threads``).

Without this, running QXTI from a script/benchmark that does not go through
``main.py`` leaves BLAS multi-threaded: with ``n_workers = 1`` one process spreads
across every core via BLAS (looks fast), but with ``n_workers = N`` you get
``N × cores`` threads fighting for the cores and it gets SLOWER as you add workers.
Pinning here makes the behaviour automatic and identical for every entry point.

Override with ``QXTI_BLAS_THREADS`` (or set ``OMP_NUM_THREADS`` etc. yourself — an
already-set value is respected, so SLURM/user settings win).
"""
from qxti.utils.parallel import configure_thread_env as _configure_thread_env

_configure_thread_env()
