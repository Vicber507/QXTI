from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qxti.core.config import CMDConfig, LaserConfig, QXTIConfig
from qxti.core.simulation import QXTISimulation
from qxti.data import save_dataset_npz
from qxti.response import SusceptibilityTensorCalculator, XTP
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
    config, index, input_omega, orders, eps, max_order, dimension, direction_labels = payload
    simulation = QXTISimulation(config=config)
    hamiltonian = simulation.build_hamiltonian()
    solver_config = replace(_resolve_scan_solver_config(config), n_workers=1)

    xtp_by_direction: dict[str, XTP] = {}
    for direction in direction_labels:
        cmd = simulation.build_cmd(
            hamiltonian,
            cmd_config=solver_config,
            laser_system=simulation.build_laser_system(
                _build_probe_laser_config(config.laser, direction, input_omega)
            ),
            max_order=max_order,
        )
        # Silence the per-k-point CMD progress inside the worker; the meaningful
        # progress is the per-frequency global counter printed by the main process.
        cmd.progress_enabled = False
        rho_orders = cmd.compute_all_orders()
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


@dataclass(slots=True)
class SusceptibilityScanRunner:
    """Run a dedicated laser-frequency sweep for susceptibility tensors."""

    config: QXTIConfig

    @classmethod
    def from_file(cls, config_path: str | Path) -> SusceptibilityScanRunner:
        return cls(config=QXTIConfig.from_file(config_path))

    def run(self) -> dict[str, Path]:
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
            "This mode solves the response in memory only; it does not save rho_order files "
            "or CMD/XTP datasets to disk."
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
            for index, values in (_susceptibility_frequency_worker(p) for p in payloads):
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

        Each process holds the in-memory density matrices of all orders for the
        current frequency, ``(max_order+1) * Nk * Nt * Nb^2`` complex128 values,
        plus FFT temporaries.  The count is capped so the total stays within a
        fraction of physical RAM, and never exceeds the CPU count or the number
        of frequencies.  ``susceptibility_n_workers = 0`` selects this automatic
        value; a positive value overrides it (still clamped to nfreq).
        """
        del ndir
        requested = int(self.config.xtp.susceptibility_n_workers)
        cpu = os.cpu_count() or 1

        simulation = QXTISimulation(config=self.config)
        hamiltonian = simulation.build_hamiltonian()
        kgrid = simulation.build_kgrid(hamiltonian)
        nk = kgrid.total_points
        nb = hamiltonian.basis_size
        low_omega = float(np.min(self._resolve_laser_omega_axis()))
        laser_system = simulation.build_laser_system(
            _build_probe_laser_config(self.config.laser, "x", low_omega)
        )
        nt = simulation.build_timegrid(laser_system).Nt
        # 3x margin for FFT/operator temporaries created during the solve.
        bytes_per_proc = int((max_order + 1) * nk * nt * nb * nb * 16 * 3)
        ram = self._available_ram()
        by_ram = max(1, int(ram * 0.6 / max(bytes_per_proc, 1)))

        auto = min(cpu, nfreq, by_ram)
        n = auto if requested <= 0 else min(requested, nfreq)
        n = max(1, n)
        self._emit_progress(
            f"Susceptibility sweep workers: {n} "
            f"(cpu={cpu}, nfreq={nfreq}, RAM-cap={by_ram}, "
            f"~{format_bytes(bytes_per_proc)}/process, Nk={nk}, Nt={nt})."
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
