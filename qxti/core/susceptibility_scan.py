from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qxti.core.config import QXTIConfig
from qxti.core.simulation import QXTISimulation
from qxti.data import save_dataset_npz
from qxti.response import SusceptibilityTensorCalculator, XTP
from qxti.utils.progress import ProgressTimer, format_bytes, format_duration


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


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

        total_probe_runs = len(laser_omega_axis) * len(direction_labels)
        probe_timer = ProgressTimer(total=total_probe_runs)
        frequency_timer = ProgressTimer(total=len(laser_omega_axis))

        for index, laser_omega in enumerate(laser_omega_axis):
            frequency_start = time.perf_counter()
            self._emit_progress(
                f"Susceptibility frequency {index + 1}/{len(laser_omega_axis)}: "
                f"omega_laser={laser_omega:.6f} a.u., running Cartesian probes."
            )

            xtp_by_direction: dict[str, XTP] = {}
            for direction in direction_labels:
                probe_start = time.perf_counter()
                cmd = simulation.build_cmd(
                    hamiltonian,
                    cmd_config=solver_config,
                    laser_system=simulation.build_laser_system(
                        self._probe_laser_config(direction=direction, omega=laser_omega)
                    ),
                    max_order=max(requested_orders),
                )
                rho_orders = cmd.compute_all_orders()
                xtp = simulation.build_xtp(cmd, rho_orders)
                xtp_by_direction[direction] = xtp
                if "bz_mask" not in dataset:
                    dataset["bz_mask"] = xtp.bz_mask_summary()

                probe_timer.advance()
                self._emit_progress(
                    f"Susceptibility probe {probe_timer.completed}/{probe_timer.total}: "
                    f"omega_laser={laser_omega:.6f} a.u., direction={direction} "
                    f"completed in {format_duration(time.perf_counter() - probe_start)} "
                    f"({self._runtime_suffix(probe_timer)})."
                )

            self._accumulate_frequency_point(
                dataset,
                index=index,
                input_omega=laser_omega,
                orders=requested_orders,
                direction_labels=direction_labels,
                xtp_by_direction=xtp_by_direction,
                eps=xtp_cfg.susceptibility_eps,
            )
            frequency_timer.advance()
            self._emit_progress(
                f"Susceptibility frequency {index + 1}/{len(laser_omega_axis)} assembled "
                f"in {format_duration(time.perf_counter() - frequency_start)} "
                f"({self._runtime_suffix(frequency_timer)})."
            )

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

        if self.config.xtp.susceptibility_plot_enabled and self.config.source_path is not None:
            self._emit_progress(
                "XTP susceptibility plots enabled: generating susceptibility graphics from the saved dataset."
            )
            from qxti.graphics.graphics import plot_susceptibility_graphics_from_saved_data

            outputs.update(plot_susceptibility_graphics_from_saved_data(self.config.source_path))

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
