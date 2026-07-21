from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qxti.core.config import CMDConfig, LaserConfig, QXTIConfig
from qxti.core.simulation import QXTISimulation
from qxti.data import save_dataset_npz
from qxti.response import SusceptibilityTensorCalculator, XTP
from qxti.response.cmd import CMD
from qxti.utils.progress import ProgressTimer, format_bytes, format_duration


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
_AU_TO_EV = 27.211386245988


def _resolve_scan_solver_config(config: QXTIConfig) -> CMDConfig:
    """Return the solver config used for susceptibility probes."""
    scan_solver = config.susceptibility_solver
    if scan_solver != type(scan_solver)():
        return scan_solver
    return config.cmd


def _build_probe_laser_config(laser_config: LaserConfig, direction: str, omega: float) -> LaserConfig:
    """Return a single-pulse laser config polarized along one Cartesian axis."""
    if direction == "x":
        thetaz, phiz, phix = 0.0, 0.0, 0.0
    elif direction == "y":
        thetaz, phiz, phix = 0.0, 0.0, 0.5 * np.pi
    elif direction == "z":
        thetaz, phiz, phix = -0.5 * np.pi, 0.0, 0.0
    else:
        raise ValueError(f"Unsupported Cartesian probe direction {direction!r}.")
    return replace(
        laser_config,
        omega=float(omega),
        ellip=0.0,
        phix=float(phix),
        thetaz=float(thetaz),
        phiz=float(phiz),
        pulses=[],
    )


def _resolve_susceptibility_runtime_solver_config(config: QXTIConfig) -> CMDConfig:
    """Return a low-memory solver config for one susceptibility probe."""
    solver_config = replace(_resolve_scan_solver_config(config), n_workers=1)
    # Susceptibility workers now use temporary rho scratch on disk instead of
    # keeping every order in RAM. complex64 is a good scratch compromise:
    # native complex memmaps can be consumed directly by XTP without inflating
    # back to dense in-memory arrays.
    if solver_config.rho_storage_dtype == "complex128":
        solver_config = replace(solver_config, rho_storage_dtype="complex64")
    return solver_config


def _close_rho_order_memmaps(rho_orders: dict[int, np.ndarray]) -> None:
    """Close any memmaps backing one rho-order mapping."""
    for tensor in rho_orders.values():
        CMD._close_array_memmap(tensor)


def _compute_frequency_tensor_values(
    xtp_by_direction: dict[str, XTP],
    *,
    input_omega: float,
    orders: tuple[int, ...],
    direction_labels: tuple[str, ...],
    dimension: int,
    eps: float,
) -> dict[str, Any]:
    """Compute one frequency's susceptibility/conductivity tensor rows.

    Returns a dict (not the full dataset) with the per-order tensor rows and the
    sampled FFT frequencies, so it can be returned cheaply from a worker process.
    """
    calculator = SusceptibilityTensorCalculator(xtp_by_direction, eps=eps)
    result: dict[str, Any] = {
        "chi": {},
        "sigma": {},
        "chi_sampled": {},
        "sigma_sampled": {},
        "bz_mask": xtp_by_direction[direction_labels[0]].bz_mask_summary(),
    }

    if 1 in orders:
        omega_axis, chi_tensor = calculator.chi1()
        omega_index = XTP._nearest_frequency_index(
            np.asarray(omega_axis, dtype=np.float64), float(input_omega), prefer_positive=True
        )
        result["chi"][1] = np.asarray(chi_tensor[omega_index, :dimension, :dimension], dtype=np.complex128)
        result["chi_sampled"][1] = float(omega_axis[omega_index])

        sigma_row = np.full((dimension, dimension), np.nan + 1.0j * np.nan, dtype=np.complex128)
        sigma_sampled = np.nan
        for input_axis, direction in enumerate(direction_labels):
            xtp = xtp_by_direction[direction]
            sigma_omega_axis, sigma_column = xtp.linear_conductivity(input_direction=direction, eps=eps)
            sigma_omega_index = XTP._nearest_frequency_index(
                np.asarray(sigma_omega_axis, dtype=np.float64), float(input_omega), prefer_positive=True
            )
            sigma_row[:, input_axis] = np.asarray(sigma_column[sigma_omega_index, :dimension], dtype=np.complex128)
            sigma_sampled = float(sigma_omega_axis[sigma_omega_index])
        result["sigma"][1] = sigma_row
        result["sigma_sampled"][1] = sigma_sampled

    for order in orders:
        if order == 1:
            continue
        tensor_shape = (dimension,) * (order + 1)
        chi_row = np.full(tensor_shape, np.nan + 1.0j * np.nan, dtype=np.complex128)
        sigma_row = np.full(tensor_shape, np.nan + 1.0j * np.nan, dtype=np.complex128)
        target_output_omega = float(order * input_omega)
        chi_sampled = np.nan
        sigma_sampled = np.nan
        for input_axis, direction in enumerate(direction_labels):
            xtp = xtp_by_direction[direction]
            omega_axis, chi_column, _meta = xtp.effective_susceptibility_spectrum(
                order=order, input_direction=direction, input_omega=input_omega, eps=eps
            )
            omega_index = XTP._nearest_frequency_index(
                np.asarray(omega_axis, dtype=np.float64), target_output_omega, prefer_positive=True
            )
            chi_row[(slice(None),) + (input_axis,) * order] = np.asarray(
                chi_column[omega_index, :dimension], dtype=np.complex128
            )
            chi_sampled = float(omega_axis[omega_index])

            sigma_omega_axis, sigma_column, _meta = xtp.effective_conductivity_spectrum(
                order=order, input_direction=direction, input_omega=input_omega, eps=eps
            )
            sigma_omega_index = XTP._nearest_frequency_index(
                np.asarray(sigma_omega_axis, dtype=np.float64), target_output_omega, prefer_positive=True
            )
            sigma_row[(slice(None),) + (input_axis,) * order] = np.asarray(
                sigma_column[sigma_omega_index, :dimension], dtype=np.complex128
            )
            sigma_sampled = float(sigma_omega_axis[sigma_omega_index])
        result["chi"][order] = chi_row
        result["sigma"][order] = sigma_row
        result["chi_sampled"][order] = chi_sampled
        result["sigma_sampled"][order] = sigma_sampled

    return result


def _susceptibility_frequency_worker(payload: tuple) -> tuple[int, dict[str, Any]]:
    """Top-level worker: solve all Cartesian probes at one laser frequency.

    Runs in a separate process. The internal CMD k-loop is forced to a single
    thread because parallelism comes from the process pool (one frequency per
    process), avoiding core oversubscription.
    """
    config, index, input_omega, orders, eps, max_order, dimension, direction_labels, show_cmd_progress = payload
    simulation = QXTISimulation(config=config)
    hamiltonian = simulation.build_hamiltonian()
    solver_config = _resolve_susceptibility_runtime_solver_config(config)

    xtp_by_direction: dict[str, XTP] = {}
    open_rho_orders: list[dict[int, np.ndarray]] = []
    temp_dirs: list[tempfile.TemporaryDirectory[str]] = []

    try:
        for direction in direction_labels:
            if show_cmd_progress:
                omega_ev = float(input_omega) * _AU_TO_EV
                print(
                    f"[QXTI] Susceptibility probe {index + 1}: solving direction '{direction}' "
                    f"at omega_laser={omega_ev:.4f} eV with temporary scratch.",
                    flush=True,
                )
            cmd = simulation.build_cmd(
                hamiltonian,
                cmd_config=solver_config,
                laser_system=simulation.build_laser_system(
                    _build_probe_laser_config(config.laser, direction, input_omega)
                ),
                max_order=max_order,
            )
            cmd.progress_enabled = bool(show_cmd_progress)

            temp_dir = tempfile.TemporaryDirectory(prefix=f"qxti_susc_f{index:03d}_{direction}_")
            temp_dirs.append(temp_dir)
            rho_order_paths = cmd.solve_time_domain(temp_dir.name)
            rho_orders = simulation._load_saved_rho_order_paths(rho_order_paths, nt=cmd.timegrid.Nt)
            open_rho_orders.append(rho_orders)
            xtp_by_direction[direction] = simulation.build_xtp(cmd, rho_orders)

        values = _compute_frequency_tensor_values(
            xtp_by_direction,
            input_omega=float(input_omega),
            orders=tuple(orders),
            direction_labels=tuple(direction_labels),
            dimension=int(dimension),
            eps=float(eps),
        )
        return int(index), values
    finally:
        for rho_orders in open_rho_orders:
            _close_rho_order_memmaps(rho_orders)
        for temp_dir in temp_dirs:
            temp_dir.cleanup()


@dataclass(slots=True)
class SusceptibilityScanRunner:
    """Run a dedicated laser-frequency sweep for susceptibility tensors."""

    config: QXTIConfig

    @classmethod
    def from_file(cls, config_path: str | Path) -> SusceptibilityScanRunner:
        # Standardize outputs to outputs/<model_name>/xtp so any entry point
        # (CLI, runner, graphics) reads/writes the same paths.
        return cls(config=QXTIConfig.from_file(config_path).with_standard_output_dirs())

    def run(self) -> dict[str, Path]:
        """Dispatch on ``[xtp] susceptibility_method``: simulation, theory, or both."""
        method = str(getattr(self.config.xtp, "susceptibility_method", "simulation")).lower()
        if method not in {"simulation", "theory", "both"}:
            raise ValueError(
                f"susceptibility_method must be 'simulation', 'theory', or 'both' (got '{method}')."
            )

        outputs: dict[str, Path] = {}
        sim_runtime = theory_runtime = None

        if method in {"theory", "both"}:
            theory_out, theory_runtime = self._run_theory()
            outputs.update(theory_out)

        if method in {"simulation", "both"}:
            t0 = time.perf_counter()
            sim_out = self._run_simulation()
            sim_runtime = time.perf_counter() - t0
            outputs.update(sim_out)

        if method == "both":
            self._report_timing(sim_runtime, theory_runtime, outputs)

        return outputs

    def _run_theory(self) -> tuple[dict[str, Path], float]:
        """Compute the analytical susceptibility tensors (all orders) and save them."""
        from qxti.analytics.theory_response import compute_susceptibility_spectrum

        omega_axis = np.asarray(self._resolve_laser_omega_axis(), dtype=np.float64)
        orders = self._resolve_orders()
        self._emit_progress(
            f"Theory engine: computing analytical susceptibility tensors for orders "
            f"{orders} on {omega_axis.size} frequencies (no time propagation). "
            "Order 1 is a single fast k-grid pass; orders >=2 use the closed-form "
            "rho^(s) per frequency/direction. Progress with ETA below."
        )
        result = compute_susceptibility_spectrum(self.config, omega_axis, orders, progress=True)

        # In 'both' mode the simulation owns xtp_susceptibility.npz, so theory
        # writes a companion file; otherwise it writes the standard name so the
        # susceptibility graphics find it transparently.
        method = str(getattr(self.config.xtp, "susceptibility_method", "simulation")).lower()
        out_dir = Path(self.config.xtp.susceptibility_output_dir) / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = "xtp_susceptibility_theory.npz" if method == "both" else "xtp_susceptibility.npz"
        out_path = out_dir / out_name
        from qxti.data import save_dataset_npz
        save_dataset_npz(out_path, result["dataset"])
        plot_hint = (
            f"python qxti/graphics/graphics.py <config> --family susceptibility"
            + (f"  (theory dataset: {out_name})" if method == "both" else "")
        )
        self._emit_progress(
            f"Theory susceptibility tensors saved as '{out_path.name}' "
            f"(computed in {format_duration(result['runtime_seconds'])}). Plot with: {plot_hint}"
        )
        return {"xtp_susceptibility_theory_data": out_path}, float(result["runtime_seconds"])

    def _report_timing(self, sim_runtime: float, theory_runtime: float, outputs: dict[str, Path]) -> None:
        speedup = (sim_runtime / theory_runtime) if theory_runtime and theory_runtime > 0 else float("inf")
        self._emit_progress(
            "=== Comparacion de tiempos (mismos parametros) ===\n"
            f"  Simulacion (CMD time-domain): {format_duration(sim_runtime)}\n"
            f"  Teoria (Kubo analitico):      {format_duration(theory_runtime)}\n"
            f"  Speedup teoria/simulacion:    {speedup:.1f}x mas rapida la teoria\n"
            "  Datos: xtp_susceptibility.npz (simulacion) y "
            "xtp_susceptibility_theory.npz (teoria)."
        )

    def _run_simulation(self) -> dict[str, Path]:
        xtp_cfg = self.config.xtp
        if not xtp_cfg.susceptibility_enabled:
            raise ValueError(
                "The dedicated XTP susceptibility workflow is disabled. "
                "Set xtp.susceptibility_enabled = true in the dedicated input."
            )
        if self.config.laser.pulses:
            raise ValueError(
                "The XTP susceptibility workflow currently supports only the single-pulse [laser] input style."
            )

        requested_orders = self._resolve_orders()
        laser_omega_axis = self._resolve_laser_omega_axis()
        simulation = QXTISimulation(config=self.config)
        hamiltonian = simulation.build_hamiltonian()
        direction_labels = XTP._direction_labels(hamiltonian.dimension)
        solver_config = self._solver_config()

        self._emit_progress(
            "XTP susceptibility sweep enabled: running a dedicated laser-frequency sweep "
            f"for orders {requested_orders} on {len(laser_omega_axis)} frequencies. "
            "This mode keeps the workflow self-contained: no rho_order files or CMD/XTP datasets "
            "are saved persistently, but large runs may use temporary rho scratch during each "
            "frequency probe to stay within RAM."
        )

        dataset = self._initialize_dataset(
            orders=requested_orders,
            laser_omega_axis=laser_omega_axis,
            direction_labels=direction_labels,
            dimension=hamiltonian.dimension,
        )

        frequency_timer = ProgressTimer(total=len(laser_omega_axis))
        max_order = max(requested_orders)
        n_workers = self._resolve_n_workers(
            nfreq=len(laser_omega_axis),
            ndir=len(direction_labels),
            max_order=max_order,
        )

        # Build one payload per laser frequency (the natural parallel unit, since
        # every Cartesian probe of a frequency is needed together to assemble chi).
        payloads = [
            (
                self.config,
                index,
                float(laser_omega),
                requested_orders,
                xtp_cfg.susceptibility_eps,
                max_order,
                hamiltonian.dimension,
                direction_labels,
                False,
            )
            for index, laser_omega in enumerate(laser_omega_axis)
        ]

        nfreq_total = len(laser_omega_axis)

        def _emit_frequency_progress(index: int) -> None:
            frequency_timer.advance()
            omega_ev = float(laser_omega_axis[index]) * _AU_TO_EV
            self._emit_progress(
                f"Susceptibility sweep: frequency {frequency_timer.completed}/{nfreq_total} done "
                f"(omega_laser={omega_ev:.4f} eV), {self._runtime_suffix(frequency_timer)}."
            )

        if n_workers <= 1:
            self._emit_progress(
                f"Susceptibility sweep: running serially over {nfreq_total} frequencies (n_workers=1)."
            )
            serial_payloads = [
                (*payload[:-1], True)
                for payload in payloads
            ]
            for index, values in (_susceptibility_frequency_worker(p) for p in serial_payloads):
                self._write_frequency_result(dataset, index, values)
                _emit_frequency_progress(index)
        else:
            self._emit_progress(
                f"Susceptibility sweep: parallelizing {nfreq_total} frequencies "
                f"over {n_workers} processes (each process uses 1 thread for its k-loop)."
            )
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for index, values in executor.map(_susceptibility_frequency_worker, payloads):
                    self._write_frequency_result(dataset, index, values)
                    _emit_frequency_progress(index)

        output_dir = Path(xtp_cfg.susceptibility_output_dir) / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "xtp_susceptibility.npz"
        save_start = time.perf_counter()
        dataset_path = save_dataset_npz(output_path, dataset)
        self._emit_progress(
            f"XTP susceptibility dataset saved as '{dataset_path.name}' "
            f"({format_bytes(dataset_path.stat().st_size)}) "
            f"in {format_duration(time.perf_counter() - save_start)}."
        )
        outputs: dict[str, Path] = {"xtp_susceptibility_data": dataset_path}

        # Plots are NOT generated here. To keep the workflow uniform with the
        # rest of QXTI (data first, plots second), the susceptibility/conductivity
        # graphics are produced from the saved dataset by the graphics entry point:
        #   python qxti/graphics/graphics.py <config> --family susceptibility
        #   python qxti/graphics/plot_susceptibility_tensor.py <config>
        # (both honor [xtp] susceptibility_plot_enabled).
        config_name = self.config.source_path.name if self.config.source_path is not None else "<config>"
        self._emit_progress(
            "Susceptibility data saved. To generate the chi/sigma plots run: "
            f"python qxti/graphics/graphics.py {config_name} --family susceptibility "
            f"(or python qxti/graphics/plot_susceptibility_tensor.py {config_name})."
        )

        return outputs

    def _solver_config(self):
        scan_solver = self.config.susceptibility_solver
        if self._has_nondefault_solver_config(scan_solver):
            return scan_solver
        return self.config.cmd

    def _resolve_orders(self) -> tuple[int, ...]:
        unique_orders: list[int] = []
        for raw_order in self.config.xtp.susceptibility_orders:
            order = int(raw_order)
            if order <= 0 or order in unique_orders:
                continue
            unique_orders.append(order)
        if not unique_orders:
            raise ValueError("xtp.susceptibility_orders must contain at least one positive order.")
        return tuple(unique_orders)

    def _resolve_laser_omega_axis(self) -> RealArray:
        xtp_cfg = self.config.xtp
        if xtp_cfg.susceptibility_omega_values:
            omega_axis = np.asarray(xtp_cfg.susceptibility_omega_values, dtype=np.float64)
        else:
            if xtp_cfg.susceptibility_num_frequencies <= 0:
                raise ValueError("xtp.susceptibility_num_frequencies must be strictly positive.")
            if xtp_cfg.susceptibility_num_frequencies == 1:
                omega_axis = np.asarray([xtp_cfg.susceptibility_omega_min], dtype=np.float64)
            else:
                if xtp_cfg.susceptibility_omega_max <= xtp_cfg.susceptibility_omega_min:
                    raise ValueError(
                        "xtp.susceptibility_omega_max must be larger than "
                        "xtp.susceptibility_omega_min when susceptibility_num_frequencies > 1."
                    )
                omega_axis = np.linspace(
                    xtp_cfg.susceptibility_omega_min,
                    xtp_cfg.susceptibility_omega_max,
                    xtp_cfg.susceptibility_num_frequencies,
                    dtype=np.float64,
                )

        if omega_axis.ndim != 1 or omega_axis.size == 0:
            raise ValueError("xtp susceptibility requires at least one laser frequency.")
        if np.any(omega_axis <= 0.0):
            raise ValueError("xtp susceptibility laser frequencies must be strictly positive.")
        return omega_axis

    def _probe_laser_config(self, *, direction: str, omega: float):
        if direction == "x":
            thetaz = 0.0
            phiz = 0.0
            phix = 0.0
        elif direction == "y":
            thetaz = 0.0
            phiz = 0.0
            phix = 0.5 * np.pi
        elif direction == "z":
            thetaz = -0.5 * np.pi
            phiz = 0.0
            phix = 0.0
        else:
            raise ValueError(f"Unsupported Cartesian probe direction {direction!r}.")

        return replace(
            self.config.laser,
            omega=float(omega),
            ellip=0.0,
            phix=float(phix),
            thetaz=float(thetaz),
            phiz=float(phiz),
            pulses=[],
        )

    def _initialize_dataset(
        self,
        *,
        orders: tuple[int, ...],
        laser_omega_axis: RealArray,
        direction_labels: tuple[str, ...],
        dimension: int,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "scan_type": "laser_frequency_sweep",
            "orders": orders,
            "dimension": int(dimension),
            "direction_labels": direction_labels,
            "laser_omega_axis": np.asarray(laser_omega_axis, dtype=np.float64),
            "cartesian_probe_directions": direction_labels,
            "output_frequency_rule": "omega_out = order * omega_laser",
        }

        num_frequencies = len(laser_omega_axis)
        nan_value = np.nan + 1.0j * np.nan
        for order in orders:
            tensor_shape = (num_frequencies,) + (dimension,) * (order + 1)
            data[f"chi_order_{order}_tensor"] = np.full(
                tensor_shape,
                nan_value,
                dtype=np.complex128,
            )
            data[f"sigma_order_{order}_tensor"] = np.full(
                tensor_shape,
                nan_value,
                dtype=np.complex128,
            )
            data[f"chi_order_{order}_available_indices"] = np.asarray(
                self._available_indices(order=order, dimension=dimension),
                dtype=np.int16,
            )
            data[f"sigma_order_{order}_available_indices"] = np.asarray(
                self._available_indices(order=order, dimension=dimension),
                dtype=np.int16,
            )
            data[f"chi_order_{order}_component_labels"] = [
                XTP._tensor_component_label(indices)
                for indices in self._available_indices(order=order, dimension=dimension)
            ]
            data[f"sigma_order_{order}_component_labels"] = list(data[f"chi_order_{order}_component_labels"])
            data[f"chi_order_{order}_sampled_fft_omega"] = np.full(
                num_frequencies,
                np.nan,
                dtype=np.float64,
            )
            data[f"sigma_order_{order}_sampled_fft_omega"] = np.full(
                num_frequencies,
                np.nan,
                dtype=np.float64,
            )
            data[f"chi_order_{order}_target_output_omega"] = order * np.asarray(
                laser_omega_axis,
                dtype=np.float64,
            )
            data[f"sigma_order_{order}_target_output_omega"] = np.asarray(
                data[f"chi_order_{order}_target_output_omega"],
                dtype=np.float64,
            )
            data[f"chi_order_{order}_normalization_mode"] = (
                "sampled_linear_response"
                if order == 1
                else "sampled_repeated_axis_effective_response"
            )
            data[f"sigma_order_{order}_normalization_mode"] = (
                "sampled_linear_current_response"
                if order == 1
                else "sampled_repeated_axis_effective_current_response"
            )
        return data

    def _write_frequency_result(
        self,
        dataset: dict[str, Any],
        index: int,
        values: dict[str, Any],
    ) -> None:
        """Write one worker's per-frequency tensor rows into the dataset."""
        if "bz_mask" not in dataset and values.get("bz_mask") is not None:
            dataset["bz_mask"] = values["bz_mask"]
        for order, row in values["chi"].items():
            np.asarray(dataset[f"chi_order_{order}_tensor"])[index] = row
            dataset[f"chi_order_{order}_sampled_fft_omega"][index] = values["chi_sampled"][order]
        for order, row in values["sigma"].items():
            np.asarray(dataset[f"sigma_order_{order}_tensor"])[index] = row
            dataset[f"sigma_order_{order}_sampled_fft_omega"][index] = values["sigma_sampled"][order]

    def _resolve_n_workers(self, *, nfreq: int, ndir: int, max_order: int) -> int:
        """Return a RAM-bounded process count for the frequency sweep.

        Each process solves one frequency at a time (its k-loop pinned to a
        single thread) using temporary rho scratch on disk.  Worker count now
        follows the cross-platform resolver — so on a SLURM node it uses the
        whole allocation — capped by (a) the number of frequencies and (b) a
        rough RAM budget so N concurrent CMD solves never exhaust memory.  Set
        ``[xtp] susceptibility_n_workers`` to force a value.
        """
        del ndir, max_order
        from qxti.utils.parallel import resolve_worker_count, available_cpus

        requested = int(self.config.xtp.susceptibility_n_workers)
        cpu = available_cpus()
        # ~3 GB per frequency worker (a full CMD solve + scratch page cache).
        avail_gb = max(1.0, self._available_ram() / (1024.0 ** 3))
        ram_cap = max(1, int(avail_gb // 3.0))
        n = resolve_worker_count(requested if requested > 0 else None, cap=nfreq)
        n = max(1, min(n, ram_cap))
        mode = "user-requested" if requested > 0 else "auto"
        self._emit_progress(
            f"Susceptibility sweep workers: {n} ({mode}; usable CPUs={cpu}, "
            f"nfreq={nfreq}, ram_cap={ram_cap} @~3GB/worker, "
            f"avail={avail_gb:.1f}GB). Each process pins its k-loop to 1 thread."
        )
        return n

    @staticmethod
    def _available_ram() -> int:
        try:
            return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, ValueError, OSError):
            return 8 * 1024 ** 3

    def _accumulate_frequency_point(
        self,
        dataset: dict[str, Any],
        *,
        index: int,
        input_omega: float,
        orders: tuple[int, ...],
        direction_labels: tuple[str, ...],
        xtp_by_direction: dict[str, XTP],
        eps: float,
    ) -> None:
        dimension = int(dataset["dimension"])
        calculator = SusceptibilityTensorCalculator(xtp_by_direction, eps=eps)

        if 1 in orders:
            omega_axis, chi_tensor = calculator.chi1()
            omega_index = XTP._nearest_frequency_index(
                np.asarray(omega_axis, dtype=np.float64),
                float(input_omega),
                prefer_positive=True,
            )
            tensor = np.asarray(dataset["chi_order_1_tensor"], dtype=np.complex128)
            tensor[index] = np.asarray(chi_tensor[omega_index, :dimension, :dimension], dtype=np.complex128)
            dataset["chi_order_1_sampled_fft_omega"][index] = float(omega_axis[omega_index])

            sigma_tensor = np.asarray(dataset["sigma_order_1_tensor"], dtype=np.complex128)
            sigma_sampled_omega = np.asarray(dataset["sigma_order_1_sampled_fft_omega"], dtype=np.float64)
            for input_axis, direction in enumerate(direction_labels):
                xtp = xtp_by_direction[direction]
                sigma_omega_axis, sigma_column = xtp.linear_conductivity(
                    input_direction=direction,
                    eps=eps,
                )
                sigma_omega_index = XTP._nearest_frequency_index(
                    np.asarray(sigma_omega_axis, dtype=np.float64),
                    float(input_omega),
                    prefer_positive=True,
                )
                sigma_tensor[index, :, input_axis] = np.asarray(
                    sigma_column[sigma_omega_index, :dimension],
                    dtype=np.complex128,
                )
                sigma_sampled_omega[index] = float(sigma_omega_axis[sigma_omega_index])

        for order in orders:
            if order == 1:
                continue

            chi_tensor = np.asarray(dataset[f"chi_order_{order}_tensor"], dtype=np.complex128)
            chi_sampled_omega = np.asarray(dataset[f"chi_order_{order}_sampled_fft_omega"], dtype=np.float64)
            sigma_tensor = np.asarray(dataset[f"sigma_order_{order}_tensor"], dtype=np.complex128)
            sigma_sampled_omega = np.asarray(dataset[f"sigma_order_{order}_sampled_fft_omega"], dtype=np.float64)
            target_output_omega = float(order * input_omega)

            for input_axis, direction in enumerate(direction_labels):
                xtp = xtp_by_direction[direction]
                omega_axis, chi_column, _metadata = xtp.effective_susceptibility_spectrum(
                    order=order,
                    input_direction=direction,
                    input_omega=input_omega,
                    eps=eps,
                )
                omega_index = XTP._nearest_frequency_index(
                    np.asarray(omega_axis, dtype=np.float64),
                    target_output_omega,
                    prefer_positive=True,
                )
                chi_tensor[(index, slice(None)) + (input_axis,) * order] = np.asarray(
                    chi_column[omega_index, :dimension],
                    dtype=np.complex128,
                )
                chi_sampled_omega[index] = float(omega_axis[omega_index])

                sigma_omega_axis, sigma_column, _metadata = xtp.effective_conductivity_spectrum(
                    order=order,
                    input_direction=direction,
                    input_omega=input_omega,
                    eps=eps,
                )
                sigma_omega_index = XTP._nearest_frequency_index(
                    np.asarray(sigma_omega_axis, dtype=np.float64),
                    target_output_omega,
                    prefer_positive=True,
                )
                sigma_tensor[(index, slice(None)) + (input_axis,) * order] = np.asarray(
                    sigma_column[sigma_omega_index, :dimension],
                    dtype=np.complex128,
                )
                sigma_sampled_omega[index] = float(sigma_omega_axis[sigma_omega_index])

    @staticmethod
    def _available_indices(order: int, *, dimension: int) -> list[tuple[int, ...]]:
        if order == 1:
            return [
                (output_axis, input_axis)
                for output_axis in range(dimension)
                for input_axis in range(dimension)
            ]
        return [
            (output_axis,) + (input_axis,) * order
            for output_axis in range(dimension)
            for input_axis in range(dimension)
        ]

    @staticmethod
    def _has_nondefault_solver_config(config) -> bool:
        return config != type(config)()

    @staticmethod
    def _runtime_suffix(timer: ProgressTimer) -> str:
        elapsed = format_duration(timer.elapsed_seconds)
        eta = timer.eta_text()
        if eta == "unknown":
            return f"elapsed {elapsed}"
        return f"elapsed {elapsed}, ETA {eta}"

    @staticmethod
    def _emit_progress(message: str) -> None:
        print(f"[QXTI] {message}")
