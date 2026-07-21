from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray

from qxti.grids import KGrid, TimeGrid
from qxti.physics import BandGaugeFrame, Hamiltonian, LaserSystem, OperatorFactory
from qxti.solvers import Solver
from qxti.utils.io_utils import (
    is_float16_complex_dtype,
    normalize_complex_storage_dtype,
    open_array_npy,
    read_complex_slice,
    save_array_npy,
    write_complex_to_float16_memmap,
)
from qxti.utils.progress import ProgressTimer, format_bytes, format_duration

from .distributions import T1T2Relaxation, bose_einstein, fermi_dirac, full_occupation, maxwell_boltzmann, valence_occupation


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_DEFAULT_WORKERS_CACHE: int | None = None


def _default_worker_count() -> int:
    """Best default thread count for the GIL-bound parallel k-loop.

    Delegates to :func:`qxti.utils.parallel.resolve_worker_count`, the single
    cross-platform source of truth: it uses the SLURM allocation on a cluster
    (all of it, never a fraction), the CPU-affinity mask on Linux, and the
    performance-core count on local Apple Silicon.  See
    ``docs/vault/Concept - Memory and Parallelism`` and ``Cluster and SLURM``.
    """
    global _DEFAULT_WORKERS_CACHE
    if _DEFAULT_WORKERS_CACHE is not None:
        return _DEFAULT_WORKERS_CACHE
    from qxti.utils.parallel import resolve_worker_count  # noqa: PLC0415

    _DEFAULT_WORKERS_CACHE = resolve_worker_count()
    return _DEFAULT_WORKERS_CACHE


class CMD:
    """Recursive perturbative density-matrix solver in the length gauge.

    Solves the standard length-gauge Liouville–von-Neumann equation order by
    order in the applied field amplitude:

        d rho^(s) / dt = -(i*omega_{nm} + gamma_{nm}) * rho^(s)_{nm}
                         + E(t) · [D_k rho^(s-1)]_{nm}

    where in the internal band basis the covariant k-derivative is

        D_k rho = nabla_k rho - i [A, rho]

    with ``A`` the Berry-connection matrix.  All energies are in Hartree and
    lengths in Bohr (atomic units).

    **Computational workflow.**

    1. Build the band-gauge frame once (eigenvalues, eigenvectors, Berry
       connection) — this is the most expensive one-time step.
    2. Solve the equilibrium order ``rho^(0)`` (diagonal Fermi–Dirac matrix).
    3. For s = 1 … max_order, compute the source term from ``rho^(s-1)``
       (intraband k-gradient + interband Berry-commutator) and propagate
       ``rho^(s)`` forward in time using the trapezoidal exponential integrator.

    **Time propagation.**  For each driven order the ODE reduces to

        rho^(s)[t+1] = P(dt) * rho^(s)[t] + (dt/2) * (P(dt)*src[t] + src[t+1])

    where ``P(dt) = exp(-(damping + i*omega)*dt)`` is the exact element-wise
    propagator.  On uniform time grids this recurrence is solved by FFT linear
    convolution (``O(Nt log Nt)`` numpy calls instead of an ``O(Nt)`` Python
    loop), which gives a large speedup for production-size time series.

    **Parallelisation.**  The k-loop is embarrassingly parallel; it is
    distributed over ``n_workers`` threads via ``ThreadPoolExecutor``.  NumPy
    releases the GIL for most array operations so threads gain near-linear
    speedup when the per-k computation is dominated by numpy work (large ``Nb``
    or large ``Nt``).

    Parameters
    ----------
    hamiltonian:
        Tight-binding Hamiltonian providing ``H(k)`` and ``diagonalize(k)``.
    laser_system:
        Electric field ``E(t)`` and vector potential ``A(t)`` of all pulses.
    kgrid:
        Reciprocal-space grid on which the density matrix is resolved.
    timegrid:
        Temporal discretisation (uniform spacing recommended for FFT path).
    operator_factory:
        Builds velocity, dipole, and current operators from the Hamiltonian.
    solver:
        Numerical ODE integrator (RKF45 or Adams–Bashforth).  Currently used
        for the full non-perturbative path; the perturbative path uses its own
        exponential integrator.
    max_order:
        Highest perturbative order to compute (s = 1 … max_order).
    population_time:
        T₁ longitudinal relaxation time (∞ → no population decay).
    coherence_time:
        T₂ transverse relaxation time (∞ → no decoherence).
    temperature:
        Electronic temperature in Hartree for the equilibrium distribution.
    fermi_level:
        Chemical potential in Hartree.
    distribution:
        Equilibrium distribution name: ``fermi_dirac``, ``valence_occupation``,
        ``full_occupation``, ``maxwell_boltzmann``, or ``bose_einstein``.
    basis:
        ``"band"`` (diagonal band basis, recommended) or ``"working"`` (orbital
        basis — useful when the eigenvectors are not smooth across k).
    gauge:
        Only ``"length"`` is currently implemented.
    include_intraband:
        Include the k-gradient (intraband / Bloch oscillation) source term.
    include_interband:
        Include the Berry-commutator (interband / optical transition) source
        term.
    include_dephasing:
        Apply T₁/T₂ relaxation to the source term.
    rho_storage_dtype:
        On-disk dtype for ``rho_order_*.npy`` files (``"complex128"`` or
        ``"complex64"``).  Computations always use ``complex128`` internally.
    n_workers:
        Number of threads for the parallel k-loop.  ``None`` → use all logical
        CPUs (``os.cpu_count()``).  Set ``1`` to disable parallelism.
    """

    # Switch from Python-loop time propagation to FFT convolution above this threshold.
    _FFT_NT_THRESHOLD: int = 64
    # Fraction of physical RAM reserved for the OS page cache when computing
    # the memory-safe worker count.  Lower values are more conservative.
    _RAM_CACHE_FRACTION: float = 0.40

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        laser_system: LaserSystem,
        kgrid: KGrid,
        timegrid: TimeGrid,
        operator_factory: OperatorFactory,
        solver: Solver,
        max_order: int,
        population_time: float,
        coherence_time: float,
        temperature: float,
        fermi_level: float,
        distribution: str,
        basis: str,
        gauge: str,
        include_intraband: bool,
        include_interband: bool,
        include_dephasing: bool,
        rho_storage_dtype: str | np.dtype = "complex128",
        n_workers: int | None = None,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.laser_system = laser_system
        self.kgrid = kgrid
        self.timegrid = timegrid
        self.operator_factory = operator_factory
        self.solver = solver
        self.max_order = int(max_order)
        self.population_time = float(population_time)
        self.coherence_time = float(coherence_time)
        self.temperature = float(temperature)
        self.fermi_level = float(fermi_level)
        self.distribution_name = self._normalize_distribution(distribution)
        self.distribution = self._resolve_distribution(self.distribution_name)
        self.basis = self._normalize_basis(basis)
        self.gauge = self._normalize_gauge(gauge)
        self.include_intraband = bool(include_intraband)
        self.include_interband = bool(include_interband)
        self.include_dephasing = bool(include_dephasing)
        self.rho_storage_dtype = normalize_complex_storage_dtype(rho_storage_dtype)

        if self.max_order < 0:
            raise ValueError("max_order must be non-negative.")
        if self.population_time <= 0.0 and not np.isinf(self.population_time):
            raise ValueError("population_time must be strictly positive or infinite.")
        if self.coherence_time <= 0.0 and not np.isinf(self.coherence_time):
            raise ValueError("coherence_time must be strictly positive or infinite.")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if self.operator_factory.hamiltonian is not self.hamiltonian:
            raise ValueError("operator_factory must be built from the same Hamiltonian instance.")

        self.gamma_population = 0.0 if np.isinf(self.population_time) else 1.0 / self.population_time
        self.gamma_coherence = 0.0 if np.isinf(self.coherence_time) else 1.0 / self.coherence_time
        self._has_population_relaxation = self.gamma_population > 0.0
        self._has_coherence_relaxation = self.gamma_coherence > 0.0
        self.relaxation_model = T1T2Relaxation(
            T1=self.population_time,
            T2=self.coherence_time,
        )
        self._diag_indices = np.diag_indices(self.hamiltonian.basis_size)
        self._offdiag_mask = ~np.eye(self.hamiltonian.basis_size, dtype=bool)
        self.band_gauge_frame = BandGaugeFrame(
            hamiltonian=self.hamiltonian,
            kgrid=self.kgrid,
        )
        self._time_domain_cache: dict[int, ComplexArray] | None = None
        self._frequency_domain_cache: dict[int, ComplexArray] | None = None
        # Worker count for the parallel k-point loop.  None → auto.
        # The k-loop is a GIL-bound ThreadPoolExecutor (NumPy releases the GIL for
        # the heavy ops, but the per-k Python glue does not), so beyond the number
        # of PERFORMANCE cores adding threads only adds GIL/efficiency-core
        # contention and gets SLOWER. The auto default therefore uses the
        # performance-core count, not all logical CPUs.
        self._n_workers: int = int(_default_worker_count() if n_workers is None else max(1, n_workers))
        # Pre-compute the k-independent damping matrix once; reused every time step.
        self._damping_cache: ComplexArray = self._damping_matrix()
        # Console progress (set False to silence, e.g. inside parallel workers).
        self.progress_enabled: bool = True
        # Use the gauge-invariant covariant k-gradient (parallel transport /
        # Wilson links) instead of computing grad_k rho and the Berry-connection
        # commutator separately. The separate form is not individually
        # gauge-invariant and breaks symmetries for off-diagonal rho at orders
        # >= 2; the covariant form fixes this. See _covariant_gradient_for_k_index.
        self.use_covariant_gradient: bool = True

    def rho_equilibrium(self, k: NDArray[np.float64]) -> ComplexArray:
        """Return the equilibrium density matrix at one k-point."""

        rho_band = self._rho_equilibrium_band(self._k_index(k))

        if self.basis == "band":
            return self.hamiltonian.validate_matrix(rho_band)
        return self._transform_one_matrix_from_band_basis(rho_band, self._k_index(k))

    def compute_rho_order(self, order: int) -> ComplexArray:
        """Return one density-matrix order with shape ``(Nk, Nt, Nb, Nb)``."""

        all_orders = self.compute_all_orders()
        if order not in all_orders:
            raise ValueError(f"Order {order} is not available.")
        return all_orders[order]

    def compute_all_orders(self) -> dict[int, ComplexArray]:
        """Return the full dictionary of density-matrix orders."""

        return self.solve_time_domain_in_memory()

    def solve_time_domain_in_memory(self) -> dict[int, ComplexArray]:
        """Solve all density-matrix orders and keep them in memory.

        This is the legacy/convenience path. It is useful for small grids and
        tests, but large ``Nk * Nt * Nb * Nb`` runs should prefer
        :meth:`solve_time_domain`.
        """

        if self.gauge != "length":
            raise NotImplementedError(
                "CMD currently implements the recursive perturbative equation "
                "only in length gauge."
            )
        if self._time_domain_cache is not None:
            return self._time_domain_cache

        target_times = np.asarray(self.timegrid.generate(), dtype=float)
        k_points = np.asarray(self.kgrid.points(), dtype=float)
        orders_band = self._solve_orders_in_band_basis(k_points, target_times)

        if self.basis == "band":
            result = orders_band
        else:
            result = {
                order: self._transform_tensor_from_band_basis(tensor, k_points)
                for order, tensor in orders_band.items()
            }

        self._time_domain_cache = result
        self._frequency_domain_cache = None
        return result

    def solve_time_domain_legacy(self) -> dict[int, ComplexArray]:
        """Backward-compatible name for :meth:`solve_time_domain_in_memory`."""

        return self.solve_time_domain_in_memory()

    def solve_frequency_domain(self) -> dict[int, ComplexArray]:
        """FFT-transform the time-domain density matrices along the time axis."""

        if self._frequency_domain_cache is not None:
            return self._frequency_domain_cache

        rho_orders = self.solve_time_domain_in_memory()
        window = np.asarray(
            self.timegrid.apply_window(np.ones(self.timegrid.Nt, dtype=float)),
            dtype=float,
        )
        nfft = self.timegrid.Nt * self.timegrid.padding_factor if self.timegrid.zero_padding else self.timegrid.Nt

        transformed: dict[int, ComplexArray] = {}
        for order, tensor in rho_orders.items():
            weighted = tensor * window[np.newaxis, :, np.newaxis, np.newaxis]
            transformed[order] = np.asarray(np.fft.fft(weighted, n=nfft, axis=1), dtype=np.complex128)

        self._frequency_domain_cache = transformed
        return transformed

    def solve_time_domain(
        self,
        output_dir: str | Path,
        order_observe_callbacks: dict[int, object] | None = None,
        *,
        reclaim_intermediate_orders: bool = False,
        skip_final_order_disk_write: bool = False,
    ) -> dict[int, Path]:
        """Solve density-matrix orders sequentially and save each one to disk.

        This is the primary low-memory time-domain path. It keeps only the
        previous order in the internal band basis, because that is the only
        tensor needed to build the next perturbative source term. Use
        :meth:`solve_time_domain_in_memory` for the legacy all-orders-in-RAM
        behavior.

        Parameters
        ----------
        output_dir:
            Directory where ``rho_order_*.npy`` scratch files are written.
        order_observe_callbacks:
            Optional mapping from order index to a ``(ik, rho_series) ->
            None`` callable.  When provided for the last driven order, the
            disk write for that order is skipped (streaming observable mode):
            the density matrix is never materialised on disk, saving one full
            ``(Nk, Nt, Nb, Nb)`` file (~36 GB for large grids).  Intermediate
            orders still write to disk (they are needed as source terms).
        reclaim_intermediate_orders:
            Delete scratch files for old orders immediately after the next
            order has been built from them.  This keeps peak scratch close to
            the algorithmic minimum, but should only be enabled when later
            post-processing does not need every saved order.
        skip_final_order_disk_write:
            Compute the last driven order without saving ``rho^(max_order)``
            when no later stage needs that tensor on disk.
        """

        if self.gauge != "length":
            raise NotImplementedError(
                "CMD currently implements the recursive perturbative equation "
                "only in length gauge."
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        target_times = np.asarray(self.timegrid.generate(), dtype=float)
        k_points = np.asarray(self.kgrid.points(), dtype=float)
        field_series = self._field_series(target_times)
        total_order_solves = max(0, self.max_order * len(k_points))
        completed_solves = 0
        saved_paths: dict[int, Path] = {}
        can_stream_saved_band_orders = self.basis == "band"
        progress_timer = ProgressTimer(total=total_order_solves)
        callbacks = order_observe_callbacks or {}
        last_order_observed = callbacks.get(self.max_order) is not None

        nb = self.hamiltonian.basis_size
        bytes_per_elem = (
            4 if is_float16_complex_dtype(self.rho_storage_dtype) else
            np.dtype(self.rho_storage_dtype).itemsize
        )
        bytes_per_order = int(len(k_points)) * int(len(target_times)) * int(nb) * int(nb) * int(bytes_per_elem)
        last_order_written = not (bool(skip_final_order_disk_write) or last_order_observed)
        if reclaim_intermediate_orders:
            if self.max_order <= 0:
                active_order_scratch = 0
            elif last_order_written:
                active_order_scratch = 1 if self.max_order == 1 else 2
            else:
                active_order_scratch = 0 if self.max_order == 1 else 1 if self.max_order == 2 else 2
        else:
            active_order_scratch = self.max_order if last_order_written else max(0, self.max_order - 1)
        total_scratch = int(bytes_per_order) * int(active_order_scratch)  # order0 ≈ 0
        ram_bytes = self._estimate_physical_ram()

        self._emit_progress(
            f"CMD streaming start: {self.max_order} driven orders, {len(k_points)} k-points, "
            f"{total_order_solves} order/k-point solves total. "
            f"Output dtype on disk: {self.rho_storage_dtype.name}."
        )
        self._emit_progress(
            f"CMD scratch estimate: "
            f"~{format_bytes(bytes_per_order)} per driven order × {active_order_scratch} active orders "
            f"= ~{format_bytes(total_scratch)} peak scratch on disk. "
            f"Physical RAM: {format_bytes(ram_bytes)}."
        )
        if total_scratch > ram_bytes:
            self._emit_progress(
                f"[WARNING] Scratch size ({format_bytes(total_scratch)}) exceeds physical RAM "
                f"({format_bytes(ram_bytes)}). The simulation will use the OS page cache "
                f"(slow I/O). Consider reducing k_points or setting "
                f"rho_storage_dtype = float16_complex to halve file sizes."
            )

        eq_observe = (
            callbacks.get(0)
            if can_stream_saved_band_orders
            else None
        )

        if can_stream_saved_band_orders:
            order0_path = output_path / "rho_order_0.npy"
            order0_start = time.perf_counter()
            equilibrium_writer = open_array_npy(
                order0_path,
                shape=(len(k_points), 1, self.hamiltonian.basis_size, self.hamiltonian.basis_size),
                dtype=self.rho_storage_dtype,
            )
            for ik in range(len(k_points)):
                rho0_band = self._rho_equilibrium_band(ik)
                if is_float16_complex_dtype(equilibrium_writer.dtype):
                    write_complex_to_float16_memmap(equilibrium_writer, ik, rho0_band[np.newaxis])
                else:
                    equilibrium_writer[ik, 0] = rho0_band
                if eq_observe is not None:
                    eq_observe(ik, rho0_band)
            equilibrium_writer.flush()
            saved_paths[0] = order0_path
            self._close_array_memmap(equilibrium_writer)
            previous_order_band = self._load_saved_order_for_recursion(saved_paths[0])
            self._emit_progress(
                f"CMD saved order 0: '{saved_paths[0].name}' "
                f"({format_bytes(saved_paths[0].stat().st_size)}, "
                f"{format_duration(time.perf_counter() - order0_start)})."
            )
        else:
            equilibrium_band = self._equilibrium_tensor_band(k_points, target_times)
            saved_paths[0] = self._save_order_tensor(
                output_path,
                order=0,
                tensor_band=equilibrium_band,
            )
            self._emit_progress(f"CMD saved order 0: '{saved_paths[0].name}'.")
            previous_order_band = equilibrium_band

        for order in range(1, self.max_order + 1):
            self._emit_progress(
                f"CMD order {order}/{self.max_order}: building source terms."
            )
            order_start = time.perf_counter()
            observe_cb = callbacks.get(order)

            # Last driven order with an observe_callback: skip disk write
            # (streaming observable mode).  Intermediate orders still write to
            # disk because they are needed as source terms for the next order.
            is_last_order = order == self.max_order
            use_streaming_for_this_order = is_last_order and observe_cb is not None
            discard_output_for_this_order = (
                is_last_order
                and bool(skip_final_order_disk_write)
                and observe_cb is None
            )
            skip_disk_write_for_this_order = (
                use_streaming_for_this_order
                or discard_output_for_this_order
            )

            if can_stream_saved_band_orders:
                order_path = None if skip_disk_write_for_this_order else output_path / f"rho_order_{order}.npy"
                ck_path = output_path / f".checkpoint_order_{order}.json" if not skip_disk_write_for_this_order else None
                old_previous_order_band = previous_order_band
                current_order_band, completed_solves = self._solve_single_order_band(
                    k_points,
                    target_times,
                    field_series,
                    previous_order_band,
                    order=order,
                    completed_solves=completed_solves,
                    total_order_solves=total_order_solves,
                    progress_timer=progress_timer,
                    output_path=order_path,
                    observe_callback=observe_cb,
                    discard_output=discard_output_for_this_order,
                    checkpoint_path=ck_path,
                )
                next_previous_order_band: np.ndarray | None = None
                if not skip_disk_write_for_this_order:
                    saved_paths[order] = order_path
                    self._emit_progress(
                        f"CMD saved order {order}: '{saved_paths[order].name}' "
                        f"({format_bytes(saved_paths[order].stat().st_size)}, "
                        f"{format_duration(time.perf_counter() - order_start)}, "
                        f"ETA {progress_timer.eta_text()})."
                    )
                    if order < self.max_order:
                        self._close_array_memmap(current_order_band)
                        next_previous_order_band = self._load_saved_order_for_recursion(saved_paths[order])
                else:
                    if use_streaming_for_this_order:
                        self._emit_progress(
                            f"CMD order {order} streamed (no disk write), "
                            f"{format_duration(time.perf_counter() - order_start)}, "
                            f"ETA {progress_timer.eta_text()}."
                        )
                    else:
                        self._emit_progress(
                            f"CMD order {order} computed and discarded (no disk write), "
                            f"{format_duration(time.perf_counter() - order_start)}, "
                            f"ETA {progress_timer.eta_text()}."
                        )
                if order < self.max_order and next_previous_order_band is not None:
                    previous_order_band = next_previous_order_band
                if ck_path is not None:
                    try:
                        ck_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                self._close_array_memmap(old_previous_order_band)
                self._close_array_memmap(current_order_band)
                if reclaim_intermediate_orders:
                    reclaim_order = order - 1
                    reclaim_path = saved_paths.pop(reclaim_order, None)
                    if reclaim_path is not None:
                        self._remove_order_scratch_files(output_path, reclaim_order, reclaim_path)
                        self._emit_progress(
                            f"CMD reclaimed scratch order {reclaim_order}: removed '{reclaim_path.name}'."
                        )
            else:
                current_order_band, completed_solves = self._solve_single_order_band(
                    k_points,
                    target_times,
                    field_series,
                    previous_order_band,
                    order=order,
                    completed_solves=completed_solves,
                    total_order_solves=total_order_solves,
                    progress_timer=progress_timer,
                    observe_callback=observe_cb,
                    discard_output=discard_output_for_this_order,
                )
                if not skip_disk_write_for_this_order:
                    saved_paths[order] = self._save_order_tensor(
                        output_path,
                        order=order,
                        tensor_band=current_order_band,
                    )
                    self._emit_progress(
                        f"CMD saved order {order}: '{saved_paths[order].name}' "
                        f"({format_bytes(saved_paths[order].stat().st_size)}, "
                        f"{format_duration(time.perf_counter() - order_start)}, "
                        f"ETA {progress_timer.eta_text()})."
                    )
                else:
                    message = "streamed" if use_streaming_for_this_order else "computed and discarded"
                    self._emit_progress(
                        f"CMD order {order} {message} (no disk write), "
                        f"{format_duration(time.perf_counter() - order_start)}, "
                        f"ETA {progress_timer.eta_text()}."
                    )
                previous_order_band = current_order_band

        self._time_domain_cache = None
        self._frequency_domain_cache = None
        return saved_paths

    def save_time_domain_orders(self, output_dir: str | Path) -> dict[int, Path]:
        """Alias for the primary low-memory :meth:`solve_time_domain` path."""

        return self.solve_time_domain(output_dir)

    def solve_time_domain_to_directory(self, output_dir: str | Path) -> dict[int, Path]:
        """Backward-compatible alias for :meth:`solve_time_domain`."""

        return self.solve_time_domain(output_dir)

    def save_density_matrices(self, output_dir: str) -> None:
        """Save all available time-domain density matrices as ``.npy`` files."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if self._time_domain_cache is None:
            self.solve_time_domain(output_path)
            return
        for order, tensor in self.solve_time_domain_in_memory().items():
            save_array_npy(output_path / f"rho_order_{order}.npy", tensor, dtype=self.rho_storage_dtype)

    def _solve_orders_in_band_basis(
        self,
        k_points: FloatArray,
        target_times: FloatArray,
    ) -> dict[int, ComplexArray]:
        equilibrium = self._equilibrium_tensor_band(k_points, target_times)
        orders_band: dict[int, ComplexArray] = {0: equilibrium}
        total_order_solves = max(0, self.max_order * len(k_points))
        completed_solves = 0
        field_series = self._field_series(target_times)
        progress_timer = ProgressTimer(total=total_order_solves)

        self._emit_progress(
            f"CMD starting: {self.max_order} driven orders, {len(k_points)} k-points, "
            f"{total_order_solves} order/k-point solves total."
        )

        for order in range(1, self.max_order + 1):
            self._emit_progress(
                f"CMD order {order}/{self.max_order}: building source terms."
            )
            orders_band[order], completed_solves = self._solve_single_order_band(
                k_points,
                target_times,
                field_series,
                orders_band[order - 1],
                order=order,
                completed_solves=completed_solves,
                total_order_solves=total_order_solves,
                progress_timer=progress_timer,
            )
            self._emit_progress(
                f"CMD order {order}/{self.max_order} completed."
            )

        return orders_band

    def _equilibrium_tensor_band(
        self,
        k_points: FloatArray,
        target_times: FloatArray,
    ) -> ComplexArray:
        nk = len(k_points)
        nt = len(target_times)
        nb = self.hamiltonian.basis_size

        equilibrium_tensor = np.empty((nk, nt, nb, nb), dtype=np.complex128)
        for ik in range(nk):
            rho0_band = self._rho_equilibrium_band(ik)
            equilibrium_tensor[ik] = np.broadcast_to(rho0_band, (nt, nb, nb)).copy()
        return equilibrium_tensor

    def _scratch_file_matches(
        self,
        output_path: Path | None,
        *,
        nk: int,
        nt: int,
        nb: int,
    ) -> bool:
        """Return whether an existing scratch ``.npy`` matches the run dimensions.

        Checks the on-disk shape without loading the data, supporting both the
        native-complex layout ``(Nk, Nt, Nb, Nb)`` and the float16_complex
        layout ``(Nk, Nt, Nb, Nb, 2)``.  Used to decide whether a checkpoint can
        be safely resumed.
        """
        if output_path is None:
            return False
        path = Path(output_path)
        if not path.exists():
            return False
        try:
            existing = np.load(path, mmap_mode="r")
        except Exception:
            return False
        try:
            shape = existing.shape
            expected_native = (nk, nt, nb, nb)
            expected_f16 = (nk, nt, nb, nb, 2)
            return shape == expected_native or shape == expected_f16
        finally:
            del existing

    @staticmethod
    def _estimate_physical_ram() -> int:
        """Return an estimate of the physical RAM in bytes."""
        try:
            return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, ValueError, OSError):
            pass
        try:
            import resource  # noqa: PLC0415
            return int(resource.getrlimit(resource.RLIMIT_AS)[1])
        except Exception:
            return 8 * 1024 ** 3  # conservative 8 GiB fallback

    @staticmethod
    def _close_array_memmap(array: np.ndarray | None) -> None:
        current = array
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            mmap_obj = getattr(current, "_mmap", None)
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except (AttributeError, BufferError, OSError, ValueError):
                    pass
            current = getattr(current, "base", None)

    @staticmethod
    def _remove_order_scratch_files(output_dir: Path, order: int, rho_path: Path) -> None:
        try:
            rho_path.unlink(missing_ok=True)
        except OSError:
            pass
        checkpoint_path = output_dir / f".checkpoint_order_{order}.json"
        try:
            checkpoint_path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _load_saved_order_for_recursion(path: Path) -> np.ndarray:
        return np.load(path, mmap_mode="r")

    @staticmethod
    def _expand_compact_series(series: np.ndarray, nt: int) -> np.ndarray:
        values = np.asarray(series, dtype=np.complex128)
        if values.ndim == 2:
            return np.broadcast_to(values[np.newaxis, :, :], (nt, values.shape[0], values.shape[1]))
        if values.ndim == 3:
            if values.shape[0] == nt:
                return values
            if values.shape[0] == 1:
                return np.broadcast_to(values, (nt, values.shape[1], values.shape[2]))
        raise ValueError(
            f"Stored rho series must have shape (Nb, Nb), (1, Nb, Nb), or ({nt}, Nb, Nb); got {values.shape}."
        )

    def _rho_series_from_storage_slice(
        self,
        tensor: np.ndarray,
        key,
        *,
        nt: int,
    ) -> np.ndarray:
        return self._expand_compact_series(read_complex_slice(tensor, key), nt)

    def _effective_n_workers_for_memmap(
        self,
        nk: int,
        nt: int,
        nb: int,
        *,
        using_memmap: bool,
    ) -> int:
        """Return a RAM-safe worker count when writing to or reading from a large memmap.

        When multiple threads access widely separated regions of the same large
        memory-mapped file simultaneously, the OS page cache fills up and the
        system starts thrashing (paging to/from disk rapidly).  This method
        limits ``n_workers`` so the total active page-cache footprint stays
        within ``_RAM_CACHE_FRACTION`` of physical memory.

        **Rule of thumb:** each worker needs roughly 3 kx-rows of
        ``(Nky, Nt, Nb, Nb)`` active pages for the gradient stencil plus the
        write buffer.  If the product ``n_workers × bytes_per_worker`` exceeds
        available page-cache budget, the count is reduced accordingly.
        """
        requested = max(1, min(nk, self._n_workers))
        if not using_memmap or requested <= 1:
            return requested

        # 3 kx-rows with all Nky points, full time series, both read+write.
        nky = int(self.kgrid.shape[1]) if len(self.kgrid.shape) >= 2 else 1
        bytes_per_worker = int(3 * nky * nt * nb * nb * 16)  # factor-of-2 for R+W

        available = int(self._estimate_physical_ram() * self._RAM_CACHE_FRACTION)
        max_workers = max(1, available // max(bytes_per_worker, 1))
        effective = min(requested, max_workers)

        if effective < requested:
            self._emit_progress(
                f"[CMD] Memory guard: n_workers reduced {requested} → {effective}. "
                f"Each worker needs ~{bytes_per_worker / 1e9:.1f} GB of page cache; "
                f"available for cache ~{available / 1e9:.1f} GB "
                f"({self._RAM_CACHE_FRACTION*100:.0f}% of physical RAM). "
                f"Reduce k_points or increase RAM for full parallelism."
            )
        return effective

    def _solve_single_order_band(
        self,
        k_points: FloatArray,
        target_times: FloatArray,
        field_series: FloatArray,
        previous_order_band: np.ndarray,
        *,
        order: int,
        completed_solves: int,
        total_order_solves: int,
        progress_timer: ProgressTimer | None = None,
        output_path: Path | None = None,
        observe_callback=None,
        discard_output: bool = False,
        checkpoint_path: Path | None = None,
    ) -> tuple[ComplexArray | None, int]:
        """Solve one perturbative density-matrix order for all k-points.

        Each k-point is solved independently using the per-k-point FFT
        integrator (see :meth:`_solve_linear_order_band_on_grid`).  When
        ``_n_workers > 1``, k-points are distributed across threads in
        contiguous chunks via :class:`~concurrent.futures.ThreadPoolExecutor`.

        NumPy releases the GIL for FFT and element-wise operations, so threads
        achieve real concurrency on numpy-heavy workloads.  Progress is
        reported at ~10 % milestones.

        When ``checkpoint_path`` is provided, a small JSON file records the last
        completed k-point index so the solve can resume from that point after a
        crash instead of starting from k=0.
        """
        nk = len(k_points)
        nt = len(target_times)
        nb = self.hamiltonian.basis_size

        # --- Checkpoint: check for a previous partial run ------------------
        import json as _json  # noqa: PLC0415

        # Metadata written with every checkpoint so a resumed run can verify
        # that the existing scratch file matches the current run dimensions.
        checkpoint_meta = {
            "order": order,
            "nt": int(nt),
            "nk": int(nk),
            "nb": int(nb),
            "dtype": self.rho_storage_dtype.name,
        }

        resume_from_k: int = 0
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            try:
                ck = _json.loads(Path(checkpoint_path).read_text())
                completed_k = int(ck.get("completed_k", 0))
                # The checkpoint is only valid if the run dimensions match what
                # is on disk. If the user changed k_points, the time grid, or
                # the storage dtype between runs, the partial scratch file is
                # incompatible and must be discarded (restart this order).
                meta_ok = all(
                    int(ck.get(key, -1)) == checkpoint_meta[key]
                    for key in ("nt", "nk", "nb")
                ) and str(ck.get("dtype", "")) == checkpoint_meta["dtype"]
                scratch_ok = self._scratch_file_matches(output_path, nk=nk, nt=nt, nb=nb)
                if meta_ok and scratch_ok and 0 < completed_k < nk:
                    resume_from_k = completed_k
                    self._emit_progress(
                        f"CMD checkpoint found: resuming order {order} from k-point "
                        f"{resume_from_k}/{nk} (skipping {resume_from_k} already solved k-points)."
                    )
                else:
                    resume_from_k = 0
                    if not (meta_ok and scratch_ok):
                        self._emit_progress(
                            f"CMD checkpoint for order {order} is incompatible with the "
                            f"current run (grid/time/dtype changed). Ignoring it and "
                            f"restarting this order from scratch."
                        )
            except Exception:
                resume_from_k = 0

        # Pure streaming mode: no dense (Nk, Nt, Nb, Nb) array is allocated.
        # The solved density matrix is passed to observe_callback k-by-k and
        # discarded immediately, saving both memory and disk I/O.
        streaming_only = output_path is None and (observe_callback is not None or discard_output)

        if streaming_only:
            solved = None
        elif output_path is None:
            solved = np.empty((nk, nt, nb, nb), dtype=np.complex128)
        else:
            solved = open_array_npy(
                output_path,
                shape=(nk, nt, nb, nb),
                dtype=self.rho_storage_dtype,
                mode="r+" if resume_from_k > 0 else "w+",
            )

        previous_order_mesh = previous_order_band.reshape(*self.kgrid.shape, *previous_order_band.shape[1:])
        connection_cache = tuple(
            self.band_gauge_frame.connection(direction)
            for direction in ("x", "y", "z")
        )
        # Eigenvectors reshaped to the k-mesh, used by the covariant-gradient
        # parallel transport. None when covariant gradient is disabled or when
        # the grid is too coarse (any active axis with < 2 points) — in that
        # case parallel transport cannot be formed and the separate
        # gradient + Berry-commutator (with single-point fallback) is used.
        vectors_mesh = (
            self.band_gauge_frame.eigenvectors.reshape(*self.kgrid.shape, nb, nb)
            if self.use_covariant_gradient and self._grid_supports_covariant_gradient()
            else None
        )


        import threading as _threading

        # Shared state for throttled parallel progress reporting.
        # A lock guards the counter and the last-emit timestamp so that
        # exactly one thread emits per heartbeat interval, regardless of
        # how many are running concurrently.
        _par_counter: list[int] = [0]
        _par_last_emit: list[float] = [time.perf_counter()]
        _par_lock = _threading.Lock()
        _PAR_HEARTBEAT_S: float = 15.0  # emit at most once per 15 seconds

        def _solve_k_range(start: int, stop: int) -> None:
            """Process k-points [start, stop) as one thread task.

            Chunking k-points reduces ThreadPoolExecutor scheduling overhead
            vs. one-task-per-k-point, and keeps each task long enough for the
            GIL-release benefit of NumPy to dominate.  A time-based heartbeat
            inside the loop emits periodic progress so long runs do not appear
            frozen when all threads are busy.
            """
            for ik in range(start, stop):
                omega_matrix = self.band_gauge_frame.omega_matrix(ik)
                source_components = self._driving_components_for_k_index(
                    previous_order_band,
                    previous_order_mesh,
                    connection_cache,
                    ik,
                    nt=nt,
                    vectors_mesh=vectors_mesh,
                )
                source_series_k = self._field_weighted_source_series(
                    field_series,
                    source_components,
                )
                solved_series = self._solve_linear_order_band_on_grid(
                    target_times,
                    source_series_k,
                    omega_matrix,
                )
                # Each ik writes to a distinct row — no lock needed.
                if solved is not None:
                    if is_float16_complex_dtype(solved.dtype):
                        write_complex_to_float16_memmap(solved, ik, solved_series)
                    else:
                        solved[ik] = solved_series
                # Call the observable accumulator (streaming path).
                if observe_callback is not None:
                    observe_callback(ik, np.asarray(solved_series, dtype=np.complex128))

                # Heartbeat: emit a progress line at most once per interval.
                # Also write a checkpoint file so a crash can be resumed.
                now = time.perf_counter()
                with _par_lock:
                    _par_counter[0] += 1
                    done = _par_counter[0]
                    if progress_timer is not None:
                        progress_timer.advance()
                    if now - _par_last_emit[0] >= _PAR_HEARTBEAT_S:
                        _par_last_emit[0] = now
                        elapsed = (
                            format_duration(progress_timer.elapsed_seconds)
                            if progress_timer else "unknown"
                        )
                        eta = progress_timer.eta_text() if progress_timer else "unknown"
                        self._emit_progress(
                            f"CMD progress: order {order}/{self.max_order}, "
                            f"{done + resume_from_k}/{nk} k-points "
                            f"({100 * (done + resume_from_k) // nk}%), "
                            f"global {completed_solves + done}/{total_order_solves}, "
                            f"elapsed {elapsed}, ETA {eta}."
                        )
                        if checkpoint_path is not None:
                            try:
                                Path(checkpoint_path).write_text(
                                    _json.dumps({**checkpoint_meta, "completed_k": resume_from_k + done})
                                )
                            except OSError:
                                pass

        n_workers = self._effective_n_workers_for_memmap(
            nk, nt, nb, using_memmap=(output_path is not None)
        )
        # Emit at most ~10 progress messages per order (milestones every 10 %).
        milestone_interval = max(1, nk // 10)

        # k-points to actually solve: skip already-completed ones on resume.
        remaining_k_start = resume_from_k
        remaining_nk = nk - resume_from_k

        if remaining_nk <= 0:
            self._emit_progress(
                f"CMD order {order}: all {nk} k-points already completed (checkpoint). Skipping."
            )
            completed_solves += nk
            if progress_timer is not None:
                progress_timer.advance(nk)
        elif n_workers > 1:
            # Distribute remaining k-points across workers in contiguous chunks.
            chunk_size = max(1, (remaining_nk + n_workers - 1) // n_workers)
            chunks = [
                (remaining_k_start + i * chunk_size,
                 min(remaining_k_start + (i + 1) * chunk_size, nk))
                for i in range(min(n_workers, remaining_nk))
            ]
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(lambda ab: _solve_k_range(*ab), chunks))
            completed_solves += remaining_nk
            elapsed = format_duration(progress_timer.elapsed_seconds) if progress_timer else "unknown"
            eta = progress_timer.eta_text() if progress_timer else "unknown"
            self._emit_progress(
                f"CMD order {order}/{self.max_order}: {nk} k-points completed "
                f"({n_workers} workers, {len(chunks)} chunks), "
                f"global {completed_solves}/{total_order_solves}, "
                f"elapsed {elapsed}, ETA {eta}."
            )
            # Final checkpoint: mark order as complete.
            if checkpoint_path is not None:
                try:
                    Path(checkpoint_path).write_text(
                        _json.dumps({**checkpoint_meta, "completed_k": nk})
                    )
                except OSError:
                    pass
        else:
            # Sequential k-loop with throttled milestone reporting.
            for ik in range(remaining_k_start, nk):
                _solve_k_range(ik, ik + 1)
                completed_solves += 1
                if (ik + 1) % milestone_interval == 0 or ik + 1 == nk:
                    elapsed = format_duration(progress_timer.elapsed_seconds) if progress_timer else "unknown"
                    eta = progress_timer.eta_text() if progress_timer else "unknown"
                    self._emit_progress(
                        f"CMD progress: order {order}/{self.max_order}, "
                        f"k-point {ik + 1}/{nk} ({100 * (ik + 1) // nk}%), "
                        f"global {completed_solves}/{total_order_solves}, "
                        f"elapsed {elapsed}, ETA {eta}."
                    )
                    if checkpoint_path is not None:
                        try:
                            Path(checkpoint_path).write_text(
                                _json.dumps({**checkpoint_meta, "completed_k": ik + 1})
                            )
                        except OSError:
                            pass

        if solved is not None and isinstance(solved, np.memmap):
            solved.flush()
        return solved, completed_solves

    def _solve_linear_order_band_on_grid(
        self,
        target_times: FloatArray,
        source_series: ComplexArray,
        omega_matrix: ComplexArray,
    ) -> ComplexArray:
        """Propagate one driven density-matrix order on the discrete time grid.

        Implements the trapezoidal exponential integrator

            rho[t+1] = P(dt) * rho[t] + (dt/2) * (P(dt)*src[t] + src[t+1])

        where ``P(dt) = exp(-(damping + i*omega) * dt)`` is the exact element-wise
        propagator and ``rho[0] = 0`` (driven orders start from vacuum).

        **Uniform grid (fast path).**  When consecutive time-steps are equal the
        recurrence is solved via FFT linear convolution instead of a Python loop.
        This replaces ``O(Nt)`` Python iterations with ``O(Nt log Nt)`` numpy FFT
        calls, yielding a large speedup for ``Nt > _FFT_NT_THRESHOLD``.

        **Non-uniform or short grid (fallback).**  Falls back to direct iteration
        with propagator caching on dt change.
        """
        nt = len(target_times)
        nb = self.hamiltonian.basis_size
        rho_series = np.zeros((nt, nb, nb), dtype=np.complex128)
        if nt <= 1:
            return rho_series

        # _damping_cache is built once at construction — same matrix for all k.
        lambda_matrix = self._damping_cache + 1.0j * np.asarray(omega_matrix, dtype=np.complex128)
        dts = np.diff(target_times.astype(float))

        if np.any(dts <= 0.0):
            raise ValueError("target_times must be strictly increasing.")

        uniform = bool(dts.size > 0 and np.allclose(dts, dts[0], rtol=1e-10, atol=1e-15))

        if uniform and nt > self._FFT_NT_THRESHOLD:
            # --- FFT convolution path ---
            dt = float(dts[0])
            half_dt = 0.5 * dt
            propagator = np.exp(-lambda_matrix * dt)  # (Nb, Nb)

            # Effective input after trapezoidal source integration:
            #   u[t] = half_dt * (P * src[t] + src[t+1])  shape (Nt-1, Nb, Nb)
            u = half_dt * (propagator[np.newaxis] * source_series[:-1] + source_series[1:])

            # Impulse response: h[k, n, m] = P[n,m]^k = exp(-lambda[n,m] * k * dt)
            # Computed via direct exponentiation (numerically stable, avoids complex power).
            k_vals = np.arange(nt - 1, dtype=np.float64)
            h = np.exp(
                -lambda_matrix[np.newaxis] * (k_vals[:, np.newaxis, np.newaxis] * dt)
            )  # (Nt-1, Nb, Nb)

            # Linear convolution via FFT.  Zero-pad to the next power of 2 that is
            # at least 2*(Nt-1) to avoid circular aliasing.
            n_fft = 1 << max(1, (2 * (nt - 1)).bit_length())
            conv = np.fft.ifft(
                np.fft.fft(h, n=n_fft, axis=0) * np.fft.fft(u, n=n_fft, axis=0),
                axis=0,
            )
            # rho[t+1] = (h ★ u)[t]  →  rho[1:] = conv[0 : Nt-1]
            rho_series[1:] = conv[:nt - 1]

        else:
            # --- Direct trapezoidal integrator (non-uniform / short grid) ---
            # Cache the propagator and recompute only when dt changes.
            prev_dt: float = -1.0
            propagator: ComplexArray = np.zeros_like(lambda_matrix)
            for it in range(nt - 1):
                dt = float(dts[it])
                if dt != prev_dt:
                    propagator = np.exp(-lambda_matrix * dt)
                    prev_dt = dt
                rho_series[it + 1] = (
                    propagator * rho_series[it]
                    + 0.5 * dt * (propagator * source_series[it] + source_series[it + 1])
                )

        return rho_series

    def save_density_matrix_dat(
        self,
        output_path: str | Path,
        tensor: ComplexArray,
        *,
        order: int,
        domain: str,
        axis_values: FloatArray | None = None,
        axis_label: str = "time",
    ) -> Path:
        """Save one density-matrix tensor as a flat ``.dat`` table."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        nk, nt, nb, _ = tensor.shape
        k_points = np.asarray(self.kgrid.points(), dtype=float)
        if axis_values is None:
            axis = np.asarray(self.timegrid.generate(), dtype=float)
        else:
            axis = np.asarray(axis_values, dtype=float)
        if nt != len(axis):
            axis = np.arange(nt, dtype=float)

        with path.open("w", encoding="ascii") as handle:
            handle.write(f"# domain={domain} order={order}\n")
            handle.write(f"# shape=({nk},{nt},{nb},{nb})\n")
            handle.write(f"# ik it kx ky kz {axis_label} row col real imag\n")
            for ik in range(nk):
                kx, ky, kz = self._k_components(k_points[ik])
                for it in range(nt):
                    axis_value = float(axis[it])
                    for row in range(nb):
                        for col in range(nb):
                            value = tensor[ik, it, row, col]
                            handle.write(
                                f"{ik} {it} {kx:.16e} {ky:.16e} {kz:.16e} "
                                f"{axis_value:.16e} {row} {col} "
                                f"{value.real:.16e} {value.imag:.16e}\n"
                            )

        return path

    def _order_equation_of_motion(
        self,
        t: float,
        rho: ComplexArray,
        source_times: FloatArray,
        driving_components: ComplexArray,
        omega_matrix: ComplexArray,
    ) -> ComplexArray:
        derivative = -1.0j * omega_matrix * rho

        if self.include_dephasing:
            derivative = derivative + self._dephasing_term(rho)

        field = self._field_vector(self.laser_system.electric_field(t))
        source_components = self._interpolate_complex_tensor(
            t,
            source_times,
            driving_components,
        )
        active_dim = min(self.hamiltonian.dimension, source_components.shape[0])
        derivative = derivative + np.tensordot(
            field[:active_dim],
            source_components[:active_dim],
            axes=(0, 0),
        )

        return np.asarray(derivative, dtype=np.complex128)

    def _build_driving_components(
        self,
        previous_order: ComplexArray,
        k_points: FloatArray,
    ) -> ComplexArray:
        nk, nt, nb, _ = previous_order.shape
        components = np.zeros((nk, nt, 3, nb, nb), dtype=np.complex128)
        gradient_components = self._k_gradient_components(previous_order)
        connection_commutator = self._connection_commutator_components(previous_order)

        if self.include_intraband:
            components += gradient_components

        if self.include_interband:
            components += connection_commutator

        return components

    def _driving_components_for_k_index(
        self,
        previous_order: np.ndarray,
        previous_order_mesh: np.ndarray,
        connection_cache: tuple[ComplexArray, ComplexArray, ComplexArray],
        index: int,
        *,
        nt: int,
        vectors_mesh: np.ndarray | None = None,
    ) -> ComplexArray:
        rho_series = self._rho_series_from_storage_slice(previous_order, index, nt=nt)
        nb = rho_series.shape[1]
        components = np.zeros((nt, 3, nb, nb), dtype=np.complex128)

        # Gauge-invariant path: the covariant gradient D_k rho = grad_k rho -
        # i[A, rho] is computed in one shot by parallel transport, which is
        # exactly gauge-invariant and preserves crystal symmetries (no separate
        # plain gradient + Berry commutator that individually break gauge).
        if (
            self.use_covariant_gradient
            and self.include_intraband
            and self.include_interband
            and vectors_mesh is not None
        ):
            components += self._covariant_gradient_for_k_index(
                previous_order_mesh, vectors_mesh, index, nt=nt
            )
            return components

        if self.include_intraband:
            components += self._k_gradient_components_for_k_index(
                previous_order_mesh,
                index,
                nt=nt,
            )

        if self.include_interband:
            components += self._connection_commutator_components_for_k_index(
                rho_series,
                connection_cache,
                index,
            )

        return components

    def _grid_supports_covariant_gradient(self) -> bool:
        """Return whether every active k-axis has >= 2 points (needed for transport)."""
        axes = (self.kgrid.kx_values, self.kgrid.ky_values, self.kgrid.kz_values)
        for axis in range(self.hamiltonian.dimension):
            if len(axes[axis]) < 2:
                return False
        return True

    def _covariant_gradient_for_k_index(
        self,
        rho_mesh: np.ndarray,
        vectors_mesh: np.ndarray,
        index: int,
        *,
        nt: int,
    ) -> ComplexArray:
        """Gauge-invariant covariant k-gradient via parallel transport.

        Computes ``D_k rho = grad_k rho - i[A, rho]`` for all active Cartesian
        directions using Wilson links ``W = U(k)^dag U(k')`` between neighbouring
        k-points: the neighbour's density matrix is parallel-transported to the
        band basis of the current k, ``W rho(k') W^dag``, before taking the finite
        difference. The transport cancels the arbitrary eigenvector phases, so the
        result is exactly gauge-invariant and respects the crystal symmetries even
        for off-diagonal rho (orders >= 2), where the separate plain-gradient +
        Berry-commutator form fails.
        """
        nb = self.hamiltonian.basis_size
        grad = np.zeros((nt, 3, nb, nb), dtype=np.complex128)
        multi_index = list(np.unravel_index(index, self.kgrid.shape))
        u0_dag = vectors_mesh[tuple(multi_index)].conj().T

        for axis, grid_values in enumerate(
            (self.kgrid.kx_values, self.kgrid.ky_values, self.kgrid.kz_values)
        ):
            if axis >= self.hamiltonian.dimension:
                break
            coords = np.asarray(grid_values, dtype=float)
            if coords.size < 2:
                continue
            grad[:, axis] = self._covariant_gradient_series(
                rho_mesh, vectors_mesh, multi_index, axis, coords, u0_dag, nt
            )
        return grad

    def _covariant_gradient_series(
        self,
        rho_mesh: np.ndarray,
        vectors_mesh: np.ndarray,
        multi_index: list[int],
        axis: int,
        coords: FloatArray,
        u0_dag: ComplexArray,
        nt: int,
    ) -> ComplexArray:
        """Finite-difference covariant derivative along one axis (parallel transport).

        Same finite-difference coefficients as :meth:`_gradient_series_at_grid_index`
        (handles non-uniform grids and edges), but each neighbour contributes the
        transported density matrix ``W rho(k') W^dag`` instead of ``rho(k')``.
        """
        position = multi_index[axis]
        n = int(coords.size)

        def take(p: int) -> ComplexArray:
            local = list(multi_index)
            local[axis] = p
            rho_j = self._rho_series_from_storage_slice(rho_mesh, tuple(local), nt=nt)
            wilson = u0_dag @ vectors_mesh[tuple(local)]  # W = U(k)^dag U(k')
            # (W rho_j W^dag) for every time step. Plain matmul broadcasts over the
            # time axis and avoids einsum's per-call path search (the hot path).
            return (wilson @ rho_j) @ wilson.conj().T

        if n == 2:
            delta = float(coords[1] - coords[0])
            if delta == 0.0:
                raise ValueError("coordinates must be strictly monotonic.")
            return np.asarray((take(1) - take(0)) / delta, dtype=np.complex128)

        if position == 0:
            s1 = float(coords[1] - coords[0])
            s2 = float(coords[2] - coords[1])
            c0 = -(2.0 * s1 + s2) / (s1 * (s1 + s2))
            c1 = (s1 + s2) / (s1 * s2)
            c2 = -s1 / (s2 * (s1 + s2))
            return np.asarray(c0 * take(0) + c1 * take(1) + c2 * take(2), dtype=np.complex128)

        if position == n - 1:
            s1 = float(coords[-2] - coords[-3])
            s2 = float(coords[-1] - coords[-2])
            c0 = s2 / (s1 * (s1 + s2))
            c1 = -(s1 + s2) / (s1 * s2)
            c2 = (2.0 * s2 + s1) / (s2 * (s1 + s2))
            return np.asarray(c0 * take(n - 3) + c1 * take(n - 2) + c2 * take(n - 1), dtype=np.complex128)

        sl = float(coords[position] - coords[position - 1])
        sr = float(coords[position + 1] - coords[position])
        cp = -sr / (sl * (sl + sr))
        cc = (sr - sl) / (sl * sr)
        cn = sl / (sr * (sl + sr))
        return np.asarray(
            cp * take(position - 1) + cc * take(position) + cn * take(position + 1),
            dtype=np.complex128,
        )

    def _field_series(self, target_times: FloatArray) -> FloatArray:
        field = np.asarray(self.laser_system.electric_field(target_times), dtype=float)
        return np.atleast_2d(field)

    def _field_weighted_source_series(
        self,
        field_series: FloatArray,
        source_components: ComplexArray,
    ) -> ComplexArray:
        active_dim = min(self.hamiltonian.dimension, source_components.shape[1], field_series.shape[1])
        if active_dim <= 0:
            raise ValueError("No active field/source dimensions are available for CMD.")
        # sum_a field[t,a] * source[t,a,b,c]; broadcast-sum avoids einsum's
        # per-call contraction-path search (this is on the per-k hot path).
        weighted = (
            field_series[:, :active_dim, None, None]
            * source_components[:, :active_dim]
        ).sum(axis=1)
        return np.asarray(weighted, dtype=np.complex128)

    def _k_gradient_components(self, tensor: ComplexArray) -> ComplexArray:
        nk, nt, nb, _ = tensor.shape
        reshaped = tensor.reshape(*self.kgrid.shape, nt, nb, nb)
        gradients = np.zeros((nk, nt, 3, nb, nb), dtype=np.complex128)

        for axis, grid_values in enumerate(
            (self.kgrid.kx_values, self.kgrid.ky_values, self.kgrid.kz_values)
        ):
            if axis >= self.hamiltonian.dimension:
                break
            if len(grid_values) < 2:
                continue
            edge_order = 2 if len(grid_values) >= 3 else 1
            component = np.gradient(
                reshaped,
                np.asarray(grid_values, dtype=float),
                axis=axis,
                edge_order=edge_order,
            )
            gradients[:, :, axis] = np.asarray(component, dtype=np.complex128).reshape(nk, nt, nb, nb)

        return gradients

    def _k_gradient_components_for_k_index(
        self,
        tensor_mesh: np.ndarray,
        index: int,
        *,
        nt: int,
    ) -> ComplexArray:
        sample_series = self._rho_series_from_storage_slice(
            tensor_mesh,
            tuple(np.unravel_index(index, self.kgrid.shape)),
            nt=nt,
        )
        nb = sample_series.shape[1]
        gradients = np.zeros((nt, 3, nb, nb), dtype=np.complex128)
        multi_index = list(np.unravel_index(index, self.kgrid.shape))

        for axis, grid_values in enumerate(
            (self.kgrid.kx_values, self.kgrid.ky_values, self.kgrid.kz_values)
        ):
            if axis >= self.hamiltonian.dimension:
                break
            if len(grid_values) < 2:
                continue
            gradients[:, axis] = self._gradient_series_at_grid_index(
                tensor_mesh,
                multi_index,
                axis,
                np.asarray(grid_values, dtype=float),
                nt=nt,
            )

        return gradients

    def _connection_commutator_components(self, tensor: ComplexArray) -> ComplexArray:
        nk, nt, nb, _ = tensor.shape
        components = np.zeros((nk, nt, 3, nb, nb), dtype=np.complex128)

        for axis, direction in enumerate(("x", "y", "z")):
            if axis >= self.hamiltonian.dimension:
                break
            connection = self.band_gauge_frame.connection(direction)
            left = np.einsum(
                "kij,ktjl->ktil",
                connection,
                tensor,
                optimize=True,
            )
            right = np.einsum(
                "ktij,kjl->ktil",
                tensor,
                connection,
                optimize=True,
            )
            components[:, :, axis] = -1.0j * (left - right)

        return np.asarray(components, dtype=np.complex128)

    def _connection_commutator_components_for_k_index(
        self,
        rho_series: ComplexArray,
        connection_cache: tuple[ComplexArray, ComplexArray, ComplexArray],
        index: int,
    ) -> ComplexArray:
        nt = rho_series.shape[0]
        nb = rho_series.shape[1]
        components = np.zeros((nt, 3, nb, nb), dtype=np.complex128)

        for axis in range(self.hamiltonian.dimension):
            connection = connection_cache[axis][index]
            left = np.matmul(connection[np.newaxis, :, :], rho_series)
            right = np.matmul(rho_series, connection[np.newaxis, :, :])
            components[:, axis] = -1.0j * (left - right)

        return np.asarray(components, dtype=np.complex128)

    @staticmethod
    def _gradient_series_at_grid_index(
        tensor_mesh: np.ndarray,
        multi_index: list[int],
        axis: int,
        coordinates: FloatArray,
        *,
        nt: int,
    ) -> ComplexArray:
        position = multi_index[axis]
        if len(coordinates) < 2:
            raise ValueError("coordinates must contain at least two points.")

        def take(axis_position: int) -> ComplexArray:
            local_index = list(multi_index)
            local_index[axis] = axis_position
            values = read_complex_slice(tensor_mesh, tuple(local_index))
            return CMD._expand_compact_series(values, nt)

        if len(coordinates) == 2:
            delta = float(coordinates[1] - coordinates[0])
            if delta == 0.0:
                raise ValueError("coordinates must be strictly monotonic.")
            return np.asarray((take(1) - take(0)) / delta, dtype=np.complex128)

        if position == 0:
            step_1 = float(coordinates[1] - coordinates[0])
            step_2 = float(coordinates[2] - coordinates[1])
            if step_1 == 0.0 or step_2 == 0.0:
                raise ValueError("coordinates must be strictly monotonic.")
            coeff_0 = -(2.0 * step_1 + step_2) / (step_1 * (step_1 + step_2))
            coeff_1 = (step_1 + step_2) / (step_1 * step_2)
            coeff_2 = -step_1 / (step_2 * (step_1 + step_2))
            return np.asarray(
                coeff_0 * take(0) + coeff_1 * take(1) + coeff_2 * take(2),
                dtype=np.complex128,
            )

        if position == len(coordinates) - 1:
            step_1 = float(coordinates[-2] - coordinates[-3])
            step_2 = float(coordinates[-1] - coordinates[-2])
            if step_1 == 0.0 or step_2 == 0.0:
                raise ValueError("coordinates must be strictly monotonic.")
            coeff_0 = step_2 / (step_1 * (step_1 + step_2))
            coeff_1 = -(step_1 + step_2) / (step_1 * step_2)
            coeff_2 = (2.0 * step_2 + step_1) / (step_2 * (step_1 + step_2))
            return np.asarray(
                coeff_0 * take(len(coordinates) - 3)
                + coeff_1 * take(len(coordinates) - 2)
                + coeff_2 * take(len(coordinates) - 1),
                dtype=np.complex128,
            )

        step_left = float(coordinates[position] - coordinates[position - 1])
        step_right = float(coordinates[position + 1] - coordinates[position])
        if step_left == 0.0 or step_right == 0.0:
            raise ValueError("coordinates must be strictly monotonic.")
        coeff_prev = -step_right / (step_left * (step_left + step_right))
        coeff_curr = (step_right - step_left) / (step_left * step_right)
        coeff_next = step_left / (step_right * (step_left + step_right))
        return np.asarray(
            coeff_prev * take(position - 1)
            + coeff_curr * take(position)
            + coeff_next * take(position + 1),
            dtype=np.complex128,
        )

    def _rho_equilibrium_band(self, index: int) -> ComplexArray:
        """Return the equilibrium diagonal density matrix for k-point ``index``.

        Issues a warning when any occupation falls outside ``[0, 1]``, which
        would indicate a Fermi-level or temperature configuration that is
        physically inconsistent with the chosen distribution.
        """
        energies = self.band_gauge_frame.energies[index]
        occupations = np.asarray(
            self.distribution(energies, self.fermi_level, self.temperature),
            dtype=float,
        )
        # Fermi-Dirac occupations must lie in [0, 1] by definition. Other
        # distributions (Maxwell-Boltzmann, Bose-Einstein) can exceed 1.
        if self.distribution_name in {"fermi_dirac", "valence_occupation", "full_occupation"}:
            if np.any(occupations < -1e-9) or np.any(occupations > 1.0 + 1e-9):
                import warnings
                bad = occupations[(occupations < -1e-9) | (occupations > 1.0 + 1e-9)]
                warnings.warn(
                    f"[CMD] Fermi-Dirac occupations at k-index {index} contain "
                    f"{len(bad)} value(s) outside [0, 1]: {bad}. "
                    "Check fermi_level and temperature settings.",
                    stacklevel=3,
                )
        return np.diag(occupations).astype(np.complex128)

    def _omega_matrix(self, kx: float, ky: float, kz: float) -> ComplexArray:
        energies = np.asarray(self.hamiltonian.eigenvalues(kx, ky, kz), dtype=float)
        omega = energies[:, np.newaxis] - energies[np.newaxis, :]
        return np.asarray(omega, dtype=np.complex128)

    def _transform_tensor_from_band_basis(
        self,
        tensor: ComplexArray,
        k_points: FloatArray,
    ) -> ComplexArray:
        del k_points
        return self.band_gauge_frame.transform_from_band_basis(tensor)

    def _transform_one_matrix_from_band_basis(
        self,
        matrix: ComplexArray,
        index: int,
    ) -> ComplexArray:
        unitary = self.band_gauge_frame.eigenvectors[index]
        return np.asarray(unitary @ matrix @ unitary.conj().T, dtype=np.complex128)

    def _save_order_tensor(
        self,
        output_dir: Path,
        *,
        order: int,
        tensor_band: ComplexArray,
    ) -> Path:
        tensor = (
            tensor_band
            if self.basis == "band"
            else self._transform_tensor_from_band_basis(tensor_band, self.kgrid.points())
        )
        path = output_dir / f"rho_order_{order}.npy"
        save_array_npy(path, np.asarray(tensor), dtype=self.rho_storage_dtype)
        return path

    @staticmethod
    def _resample_density_trajectory(
        times: list[float] | NDArray[np.float64],
        states: list[ComplexArray] | NDArray[np.complex128],
        target_times: NDArray[np.float64],
    ) -> ComplexArray:
        source_times = np.asarray(times, dtype=float)
        source_states = np.asarray(states, dtype=np.complex128)
        if source_states.shape[0] != source_times.shape[0]:
            raise ValueError("The solver returned inconsistent time/state lengths.")

        time_tolerance = max(1.0e-12, 1.0e-10 * np.max(np.abs(target_times)))
        if source_times[0] > target_times[0] + time_tolerance or source_times[-1] < target_times[-1] - time_tolerance:
            raise RuntimeError(
                "The solver trajectory does not cover the requested target-time window. "
                f"Available range: [{source_times[0]:.16e}, {source_times[-1]:.16e}], "
                f"requested range: [{target_times[0]:.16e}, {target_times[-1]:.16e}]."
            )

        if source_states.shape[0] == target_times.shape[0] and np.allclose(source_times, target_times):
            return np.asarray(source_states, dtype=np.complex128)

        nt = len(target_times)
        nb = source_states.shape[1]
        resampled = np.empty((nt, nb, nb), dtype=np.complex128)

        for row in range(nb):
            for col in range(nb):
                series = source_states[:, row, col]
                real_part = np.interp(target_times, source_times, np.real(series))
                imag_part = np.interp(target_times, source_times, np.imag(series))
                resampled[:, row, col] = real_part + 1.0j * imag_part

        return resampled

    @staticmethod
    def _interpolate_complex_tensor(
        t: float,
        source_times: FloatArray,
        tensor_series: ComplexArray,
    ) -> ComplexArray:
        times = np.asarray(source_times, dtype=float)
        values = np.asarray(tensor_series, dtype=np.complex128)
        if values.shape[0] != times.shape[0]:
            raise ValueError("tensor_series must have the same leading length as source_times.")
        if values.shape[0] == 1:
            return np.asarray(values[0], dtype=np.complex128)

        return np.asarray(
            CMD._linear_interpolate_series(t, times, values),
            dtype=np.complex128,
        )

    def _dephasing_term(self, rho: ComplexArray) -> ComplexArray:
        derivative = np.zeros_like(rho, dtype=np.complex128)

        if self._has_population_relaxation:
            derivative[self._diag_indices] = -self.gamma_population * rho[self._diag_indices]

        if self._has_coherence_relaxation:
            derivative[self._offdiag_mask] = -self.gamma_coherence * rho[self._offdiag_mask]

        return derivative

    def _damping_matrix(self) -> ComplexArray:
        damping = np.zeros((self.hamiltonian.basis_size, self.hamiltonian.basis_size), dtype=np.complex128)
        if self._has_population_relaxation:
            damping[self._diag_indices] = self.gamma_population
        if self._has_coherence_relaxation:
            damping[self._offdiag_mask] = self.gamma_coherence
        return damping

    @staticmethod
    def _linear_interpolate_series(
        t: float,
        source_times: FloatArray,
        series: ComplexArray,
    ) -> ComplexArray:
        times = np.asarray(source_times, dtype=float)
        values = np.asarray(series, dtype=np.complex128)
        if values.shape[0] != times.shape[0]:
            raise ValueError("series must have the same leading length as source_times.")
        if values.shape[0] == 1 or t <= times[0]:
            return np.asarray(values[0], dtype=np.complex128)
        if t >= times[-1]:
            return np.asarray(values[-1], dtype=np.complex128)

        upper = int(np.searchsorted(times, t, side="right"))
        lower = upper - 1
        t_lower = float(times[lower])
        t_upper = float(times[upper])
        if t_upper <= t_lower:
            return np.asarray(values[lower], dtype=np.complex128)

        weight = (t - t_lower) / (t_upper - t_lower)
        return np.asarray(
            (1.0 - weight) * values[lower] + weight * values[upper],
            dtype=np.complex128,
        )

    @staticmethod
    def _normalize_basis(basis: str) -> str:
        key = basis.strip().lower()
        if key in {"orbital", "working", "original"}:
            return "working"
        if key in {"band", "bands"}:
            return "band"
        raise ValueError("basis must be 'orbital'/'working' or 'band'.")

    @staticmethod
    def _normalize_gauge(gauge: str) -> str:
        key = gauge.strip().lower()
        if key not in {"velocity", "length"}:
            raise ValueError("gauge must be 'velocity' or 'length'.")
        return key

    @staticmethod
    def _normalize_distribution(distribution: str) -> str:
        key = distribution.strip().lower()
        aliases = {
            "fermi": "fermi_dirac",
            "fermi_dirac": "fermi_dirac",
            "fd": "fermi_dirac",
            "maxwell": "maxwell_boltzmann",
            "maxwell_boltzmann": "maxwell_boltzmann",
            "mb": "maxwell_boltzmann",
            "bose": "bose_einstein",
            "bose_einstein": "bose_einstein",
            "be": "bose_einstein",
            "valence": "valence_occupation",
            "valence_occupation": "valence_occupation",
            "filled_valence": "valence_occupation",
            "semiconductor": "valence_occupation",
            "full": "full_occupation",
            "full_occupation": "full_occupation",
            "identity": "full_occupation",
        }
        if key not in aliases:
            raise ValueError(
                "distribution must be fermi_dirac, maxwell_boltzmann, "
                "bose_einstein, valence_occupation, or full_occupation."
            )
        return aliases[key]

    @staticmethod
    def _resolve_distribution(distribution: str):
        if distribution == "fermi_dirac":
            return fermi_dirac
        if distribution == "maxwell_boltzmann":
            return maxwell_boltzmann
        if distribution == "bose_einstein":
            return bose_einstein
        if distribution == "valence_occupation":
            return valence_occupation
        if distribution == "full_occupation":
            return full_occupation
        raise ValueError(f"Unsupported distribution '{distribution}'.")

    @staticmethod
    def _k_components(k_point: NDArray[np.float64]) -> tuple[float, float, float]:
        k_vector = np.asarray(k_point, dtype=float)
        if k_vector.shape != (3,):
            raise ValueError("k must have shape (3,).")
        return float(k_vector[0]), float(k_vector[1]), float(k_vector[2])

    def _k_index(self, k_point: NDArray[np.float64]) -> int:
        k_vector = np.asarray(k_point, dtype=float)
        if k_vector.shape != (3,):
            raise ValueError("k must have shape (3,).")
        matches = np.where(np.all(np.isclose(self.kgrid.points(), k_vector[np.newaxis, :], atol=1.0e-12), axis=1))[0]
        if matches.size == 0:
            raise ValueError("The requested k-point is not present in the configured KGrid.")
        return int(matches[0])

    @staticmethod
    def _field_vector(values: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (3,):
            raise ValueError("LaserSystem must return 3 Cartesian components.")
        return vector

    def _emit_progress(self, message: str) -> None:
        # Per-instance progress can be silenced (e.g. inside parallel sweep
        # workers, where the per-k-point progress would spam the console and
        # the meaningful progress is the per-frequency global counter).
        if getattr(self, "progress_enabled", True):
            print(f"[CMD] {message}")
