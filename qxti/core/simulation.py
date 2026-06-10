from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qxti.core.config import QXTIConfig
from qxti.data import HarmonicData, HamiltonianData, ResponseData, save_dataset_npz
from qxti.grids import FrequencyGrid, KGrid, TimeGrid
from qxti.physics import CustomHamiltonian, Hamiltonian, Laser, LaserSystem, OperatorFactory
from qxti.response import CMD, XTP
from qxti.solvers import AdamsBashforth2Solver, RKF45Solver, Solver
from qxti.utils.io_utils import expand_rho_tensor_time_axis, save_array_npy
from qxti.utils.progress import ProgressTimer, format_bytes, format_duration


ComplexArray = NDArray[np.complex128]
DEFAULT_HAMILTONIAN_PLOTS = (
    "band_structure_2d",
    "band_surface_3d",
    "velocity_2d",
    "velocity_field_3d",
    "velocity_magnitude",
)


@dataclass(slots=True)
class QXTISimulation:
    """Orchestrator for Hamiltonian plots and CMD density-matrix runs."""

    config: QXTIConfig

    @classmethod
    def from_file(cls, config_path: str | Path) -> QXTISimulation:
        return cls(config=QXTIConfig.from_file(config_path))

    def build_hamiltonian(self) -> Hamiltonian:
        hcfg = self.config.hamiltonian
        kwargs: dict[str, Any] = {
            "source_file": hcfg.source_file,
            "function_name": hcfg.function_name,
            "model_name": hcfg.model_name,
            "basis_type": hcfg.basis_type,
            "is_periodic": hcfg.is_periodic,
            "dk_derivative": hcfg.dk_derivative,
            "params": dict(hcfg.params),
            "lattice": dict(hcfg.lattice),
        }
        if hcfg.basis_size is not None:
            kwargs["basis_size"] = hcfg.basis_size
        if hcfg.dimension is not None:
            kwargs["dimension"] = hcfg.dimension
        return CustomHamiltonian(**kwargs)

    def build_kgrid(self, hamiltonian: Hamiltonian) -> KGrid:
        kcfg = self.config.kgrid
        dimension = kcfg.dimension if kcfg.dimension is not None else hamiltonian.dimension

        if kcfg.kx_values or kcfg.ky_values or kcfg.kz_values:
            kx_values = np.asarray(kcfg.kx_values or [0.0], dtype=float)
            ky_values = np.asarray(kcfg.ky_values or [0.0], dtype=float)
            kz_values = np.asarray(kcfg.kz_values or [0.0], dtype=float)
        else:
            if kcfg.k_points:
                if len(kcfg.k_points) != dimension:
                    raise ValueError(
                        f"kgrid.k_points must have exactly {dimension} components for dimension={dimension}."
                    )
                if any(points <= 0 for points in kcfg.k_points):
                    raise ValueError("All entries in kgrid.k_points must be strictly positive.")
                nkx = int(kcfg.k_points[0])
                nky = int(kcfg.k_points[1]) if dimension >= 2 else 1
                nkz = int(kcfg.k_points[2]) if dimension >= 3 else 1
            else:
                points_per_axis = 3 if kcfg.points_per_axis is None else int(kcfg.points_per_axis)
                if points_per_axis <= 0:
                    raise ValueError("kgrid.points_per_axis must be strictly positive.")
                nkx = points_per_axis
                nky = points_per_axis if dimension >= 2 else 1
                nkz = points_per_axis if dimension >= 3 else 1

            bounds = hamiltonian.reciprocal_box_bounds()
            kx_values = np.linspace(bounds[0][0], bounds[0][1], nkx, dtype=float)
            ky_values = (
                np.linspace(bounds[1][0], bounds[1][1], nky, dtype=float)
                if dimension >= 2
                else np.array([0.0], dtype=float)
            )
            kz_values = (
                np.linspace(bounds[2][0], bounds[2][1], nkz, dtype=float)
                if dimension >= 3
                else np.array([0.0], dtype=float)
            )

        return KGrid(
            kx_values=kx_values,
            ky_values=ky_values,
            kz_values=kz_values,
            dimension=dimension,
        )

    def build_timegrid(self, laser_system: LaserSystem) -> TimeGrid:
        tcfg = self.config.timegrid
        if tcfg.dt <= 0.0:
            raise ValueError("timegrid.dt must be strictly positive.")

        if tcfg.t_min is not None and tcfg.t_max is not None and tcfg.Nt is not None:
            return TimeGrid(
                t_min=tcfg.t_min,
                t_max=tcfg.t_max,
                Nt=tcfg.Nt,
                fft_window=tcfg.fft_window,
                zero_padding=tcfg.zero_padding,
                padding_factor=tcfg.padding_factor,
            )

        t_min, t_max = laser_system.temporal_bounds()
        return TimeGrid.from_dt(
            t_min=t_min if tcfg.t_min is None else tcfg.t_min,
            t_max=t_max if tcfg.t_max is None else tcfg.t_max,
            dt=tcfg.dt,
            fft_window=tcfg.fft_window,
            zero_padding=tcfg.zero_padding,
            padding_factor=tcfg.padding_factor,
        )

    def build_laser_system(self) -> LaserSystem:
        lcfg = self.config.laser
        if lcfg.pulses:
            lasers = [self._build_laser_from_mapping(spec) for spec in lcfg.pulses]
        else:
            lasers = [
                Laser(
                    omega=lcfg.omega,
                    E0=lcfg.E0,
                    cep=lcfg.cep,
                    ellip=lcfg.ellip,
                    ncycles=lcfg.ncycles,
                    envname=lcfg.envname,
                    t0=lcfg.t0,
                    phix=lcfg.phix,
                    thetaz=lcfg.thetaz,
                    phiz=lcfg.phiz,
                    ncycles_clipped=lcfg.ncycles_clipped,
                )
            ]
        return LaserSystem(lasers, blaser=lcfg.blaser, alaser=lcfg.alaser)

    def build_cmd(self, hamiltonian: Hamiltonian) -> CMD:
        ccfg = self.config.cmd
        laser_system = self.build_laser_system()
        timegrid = self.build_timegrid(laser_system)
        operator_factory = OperatorFactory(hamiltonian=hamiltonian, basis=ccfg.basis)
        return CMD(
            hamiltonian=hamiltonian,
            laser_system=laser_system,
            kgrid=self.build_kgrid(hamiltonian),
            timegrid=timegrid,
            operator_factory=operator_factory,
            solver=self._build_solver(timegrid),
            max_order=ccfg.max_order,
            population_time=ccfg.population_time,
            coherence_time=ccfg.coherence_time,
            temperature=ccfg.temperature,
            fermi_level=ccfg.fermi_level,
            distribution=ccfg.distribution,
            basis=ccfg.basis,
            gauge=ccfg.gauge,
            include_intraband=ccfg.include_intraband,
            include_interband=ccfg.include_interband,
            include_dephasing=ccfg.include_dephasing,
            rho_storage_dtype=ccfg.rho_storage_dtype,
        )

    def run(self) -> dict[str, Path]:
        hamiltonian = self.build_hamiltonian()
        outputs: dict[str, Path] = {}
        self._emit_progress("Building data products from input configuration.")
        outputs.update(self.generate_hamiltonian_datasets(hamiltonian))
        outputs.update(self.generate_cmd_outputs(hamiltonian))
        self._emit_progress(f"Simulation completed with {len(outputs)} generated files.")
        return outputs

    def generate_hamiltonian_datasets(self, hamiltonian: Hamiltonian) -> dict[str, Path]:
        plot_cfg = self.config.hamiltonian_plots
        if not plot_cfg.enabled:
            return {}

        output_dir = Path(plot_cfg.output_dir) / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        data_builder = HamiltonianData(hamiltonian)
        outputs: dict[str, Path] = {}
        requested_plots = plot_cfg.plots or DEFAULT_HAMILTONIAN_PLOTS

        total_plots = len(requested_plots)
        step_timer = ProgressTimer(total=total_plots)
        for index, plot_name in enumerate(requested_plots, start=1):
            normalized = self._normalize_plot_name(plot_name)
            self._emit_step_started("Hamiltonian dataset", step_timer, f"generating '{normalized}'")
            step_start = time.perf_counter()
            if normalized == "band_structure_2d":
                data = data_builder.band_structure_2d_data(
                    path_type=plot_cfg.path_type,
                    k_min=plot_cfg.k_min,
                    k_max=plot_cfg.k_max,
                    num_points=plot_cfg.nk_path,
                    fixed_kx=plot_cfg.fixed_kx,
                    fixed_ky=plot_cfg.fixed_ky,
                    fixed_kz=plot_cfg.fixed_kz,
                    manual_path=plot_cfg.manual_path or None,
                )
                dataset_path = save_dataset_npz(output_dir / f"{normalized}.npz", data)
                outputs[f"{normalized}_data"] = dataset_path
            elif normalized == "band_surface_3d":
                data = data_builder.band_surface_3d_data(
                    plane=plot_cfg.plane,
                    band_index=plot_cfg.band_index,
                    band_indices=plot_cfg.band_indices,
                    k1_min=plot_cfg.k1_min,
                    k1_max=plot_cfg.k1_max,
                    k2_min=plot_cfg.k2_min,
                    k2_max=plot_cfg.k2_max,
                    num_points_1=plot_cfg.mesh_points_1 or plot_cfg.mesh_points,
                    num_points_2=plot_cfg.mesh_points_2 or plot_cfg.mesh_points,
                    fixed_kx=plot_cfg.fixed_kx,
                    fixed_ky=plot_cfg.fixed_ky,
                    fixed_kz=plot_cfg.fixed_kz,
                )
                dataset_path = save_dataset_npz(output_dir / f"{normalized}.npz", data)
                outputs[f"{normalized}_data"] = dataset_path
            elif normalized == "velocity_2d":
                data = data_builder.velocity_2d_data(
                    path_type=plot_cfg.path_type,
                    k_min=plot_cfg.k_min,
                    k_max=plot_cfg.k_max,
                    num_points=plot_cfg.nk_path,
                    fixed_kx=plot_cfg.fixed_kx,
                    fixed_ky=plot_cfg.fixed_ky,
                    fixed_kz=plot_cfg.fixed_kz,
                    manual_path=plot_cfg.manual_path or None,
                )
                dataset_path = save_dataset_npz(output_dir / f"{normalized}.npz", data)
                outputs[f"{normalized}_data"] = dataset_path
            elif normalized == "velocity_field_3d":
                data = data_builder.velocity_field_3d_data(
                    plane=plot_cfg.plane,
                    band_index=plot_cfg.band_index,
                    band_indices=plot_cfg.band_indices,
                    k1_min=plot_cfg.k1_min,
                    k1_max=plot_cfg.k1_max,
                    k2_min=plot_cfg.k2_min,
                    k2_max=plot_cfg.k2_max,
                    num_points_1=plot_cfg.mesh_points_1 or plot_cfg.mesh_points,
                    num_points_2=plot_cfg.mesh_points_2 or plot_cfg.mesh_points,
                    fixed_kx=plot_cfg.fixed_kx,
                    fixed_ky=plot_cfg.fixed_ky,
                    fixed_kz=plot_cfg.fixed_kz,
                )
                dataset_path = save_dataset_npz(output_dir / f"{normalized}.npz", data)
                outputs[f"{normalized}_data"] = dataset_path
            elif normalized == "velocity_magnitude":
                data = data_builder.velocity_magnitude_data(
                    plane=plot_cfg.plane,
                    band_index=plot_cfg.band_index,
                    band_indices=plot_cfg.band_indices,
                    k1_min=plot_cfg.k1_min,
                    k1_max=plot_cfg.k1_max,
                    k2_min=plot_cfg.k2_min,
                    k2_max=plot_cfg.k2_max,
                    num_points_1=plot_cfg.mesh_points_1 or plot_cfg.mesh_points,
                    num_points_2=plot_cfg.mesh_points_2 or plot_cfg.mesh_points,
                    fixed_kx=plot_cfg.fixed_kx,
                    fixed_ky=plot_cfg.fixed_ky,
                    fixed_kz=plot_cfg.fixed_kz,
                )
                dataset_path = save_dataset_npz(output_dir / f"{normalized}.npz", data)
                outputs[f"{normalized}_data"] = dataset_path
            else:
                raise ValueError(f"Unsupported Hamiltonian plot '{plot_name}'.")
            self._emit_step_completed(
                "Hamiltonian dataset",
                step_timer,
                f"saved '{normalized}.npz' ({format_bytes(dataset_path.stat().st_size)})",
                step_seconds=time.perf_counter() - step_start,
            )

        return outputs

    def generate_cmd_outputs(self, hamiltonian: Hamiltonian) -> dict[str, Path]:
        cmd_cfg = self.config.cmd
        if not cmd_cfg.enabled:
            return {}

        cmd = self.build_cmd(hamiltonian)
        output_dir = Path(cmd_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._emit_progress(
            f"CMD enabled: solving density matrices up to order {cmd_cfg.max_order}."
        )

        outputs: dict[str, Path] = {}
        rho_order_paths = cmd.solve_time_domain(output_dir)
        for order, npy_path in rho_order_paths.items():
            outputs[f"rho_order_{order}"] = npy_path

        if cmd_cfg.save_frequency_domain:
            frequency_dir = output_dir / "frequency"
            frequency_dir.mkdir(parents=True, exist_ok=True)
            for order, npy_path in self._save_frequency_domain_from_saved_orders(
                cmd,
                rho_order_paths,
                frequency_dir,
            ).items():
                outputs[f"rho_frequency_order_{order}"] = npy_path
                self._emit_progress(
                    f"CMD saved frequency order {order}: '{npy_path.name}'."
                )

        rho_orders = self._load_saved_rho_order_paths(rho_order_paths, nt=cmd.timegrid.Nt)
        outputs.update(self.generate_cmd_datasets(cmd, rho_orders))
        outputs.update(self.generate_xtp_datasets(cmd, rho_orders))
        return outputs

    def generate_cmd_datasets(
        self,
        cmd: CMD,
        rho_orders: dict[int, ComplexArray],
    ) -> dict[str, Path]:
        self._emit_progress("CMD data products enabled: generating reusable kx-ky datasets.")
        data_builder = ResponseData(cmd)
        output_dir = Path(self.config.cmd.output_dir) / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        step_timer = ProgressTimer(total=4, min_completed_for_eta=2)

        all_orders = tuple(sorted(rho_orders))

        outputs: dict[str, Path] = {}

        self._emit_step_started("CMD dataset", step_timer, "building population kx-ky dataset")
        step_start = time.perf_counter()
        try:
            kmap_data = data_builder.population_kxky_animation_data(
                orders=all_orders,
                rho_orders=rho_orders,
            )
        except ValueError as exc:
            self._emit_progress(f"CMD kx-ky population data skipped: {exc}")
            kmap_data = None
            self._emit_step_completed(
                "CMD dataset",
                step_timer,
                "population kx-ky dataset skipped",
                step_seconds=time.perf_counter() - step_start,
            )
            step_timer.advance()
        else:
            self._emit_step_completed(
                "CMD dataset",
                step_timer,
                "population kx-ky dataset built",
                step_seconds=time.perf_counter() - step_start,
            )
            self._emit_step_started("CMD dataset", step_timer, "saving population kx-ky dataset")
            step_start = time.perf_counter()
            animation_data_path = save_dataset_npz(
                output_dir / "population_kx_ky_per_band.npz",
                kmap_data,
            )
            outputs["rho_population_kxky_data"] = animation_data_path
            self._emit_step_completed(
                "CMD dataset",
                step_timer,
                f"population kx-ky dataset saved as '{animation_data_path.name}' "
                f"({format_bytes(animation_data_path.stat().st_size)})",
                step_seconds=time.perf_counter() - step_start,
            )
            del kmap_data
        self._emit_step_started("CMD dataset", step_timer, "building coherence kx-ky dataset")
        step_start = time.perf_counter()
        try:
            coherence_kmap_data = data_builder.coherence_kxky_animation_data(
                orders=all_orders,
                component="magnitude",
                rho_orders=rho_orders,
            )
        except ValueError as exc:
            self._emit_progress(f"CMD kx-ky coherence data skipped: {exc}")
            coherence_kmap_data = None
            self._emit_step_completed(
                "CMD dataset",
                step_timer,
                "coherence kx-ky dataset skipped",
                step_seconds=time.perf_counter() - step_start,
            )
            step_timer.advance()
        else:
            self._emit_step_completed(
                "CMD dataset",
                step_timer,
                "coherence kx-ky dataset built",
                step_seconds=time.perf_counter() - step_start,
            )
            self._emit_step_started("CMD dataset", step_timer, "saving coherence kx-ky dataset")
            step_start = time.perf_counter()
            coherence_animation_data_path = save_dataset_npz(
                output_dir / "coherence_kx_ky_per_pair.npz",
                coherence_kmap_data,
            )
            outputs["rho_coherence_kxky_data"] = coherence_animation_data_path
            self._emit_step_completed(
                "CMD dataset",
                step_timer,
                f"coherence kx-ky dataset saved as '{coherence_animation_data_path.name}' "
                f"({format_bytes(coherence_animation_data_path.stat().st_size)})",
                step_seconds=time.perf_counter() - step_start,
            )
            del coherence_kmap_data
        return outputs

    def generate_xtp_datasets(
        self,
        cmd: CMD,
        rho_orders: dict[int, ComplexArray],
    ) -> dict[str, Path]:
        self._emit_progress("XTP data products enabled: generating reusable current-spectrum dataset.")
        step_timer = ProgressTimer(total=2)
        omega_axis = np.asarray(cmd.timegrid.frequency_axis(), dtype=float)
        omega_max = float(np.max(np.abs(omega_axis))) if omega_axis.size else 1.0
        frequencygrid = FrequencyGrid(
            0.0,
            omega_max if omega_max > 0.0 else 1.0,
            len(omega_axis) if omega_axis.size else 1,
        )

        xtp = XTP(
            hamiltonian=cmd.hamiltonian,
            rho_orders=rho_orders,
            kgrid=cmd.kgrid,
            timegrid=cmd.timegrid,
            frequencygrid=frequencygrid,
            operator_factory=cmd.operator_factory,
            directions=self._active_directions(cmd.hamiltonian.dimension),
            orders=sorted(rho_orders),
            band_gauge_frame=cmd.band_gauge_frame if cmd.basis == "band" else None,
            bz_mask_enabled=self.config.xtp.bz_mask_enabled,
            bz_mask_radius_percent=self.config.xtp.bz_mask_radius_percent,
            bz_mask_sigma=self.config.xtp.bz_mask_sigma,
            bz_mask_sigma_percent_legacy=self.config.xtp.bz_mask_sigma_percent_legacy,
        )
        electric_field_time = np.asarray(
            cmd.laser_system.electric_field(cmd.timegrid.generate()),
            dtype=float,
        )
        self._emit_step_started("XTP dataset", step_timer, "building current-spectrum dataset")
        step_start = time.perf_counter()
        data_builder = HarmonicData(xtp, electric_field_time=electric_field_time)
        output_dir = Path(self.config.cmd.output_dir) / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        current_spectrum_data = data_builder.current_spectrum_data()
        self._emit_step_completed(
            "XTP dataset",
            step_timer,
            "current-spectrum dataset built",
            step_seconds=time.perf_counter() - step_start,
        )

        self._emit_step_started("XTP dataset", step_timer, "saving current-spectrum dataset")
        step_start = time.perf_counter()
        current_spectrum_path = save_dataset_npz(
            output_dir / "current_spectrum.npz",
            current_spectrum_data,
        )
        self._emit_step_completed(
            "XTP dataset",
            step_timer,
            f"current-spectrum dataset saved as '{current_spectrum_path.name}' "
            f"({format_bytes(current_spectrum_path.stat().st_size)})",
            step_seconds=time.perf_counter() - step_start,
        )
        return {"xtp_current_spectrum_data": current_spectrum_path}

    def _build_solver(self, timegrid: TimeGrid) -> Solver:
        cmd_cfg = self.config.cmd
        solver_name = cmd_cfg.solver.strip().lower()

        if solver_name in {"rkf45", "rkf", "fehlberg", "runge_kutta_fehlberg"}:
            window_duration = max(float(timegrid.t_max - timegrid.t_min), float(timegrid.dt))
            return RKF45Solver(
                tolerance=cmd_cfg.solver_tolerance,
                max_iterations=cmd_cfg.solver_max_iterations,
                h_min=cmd_cfg.solver_h_min,
                h_max=window_duration if cmd_cfg.solver_h_max is None else cmd_cfg.solver_h_max,
                safety_factor=cmd_cfg.solver_safety_factor,
                min_factor=cmd_cfg.solver_min_factor,
                max_factor=cmd_cfg.solver_max_factor,
                max_rejections=cmd_cfg.solver_max_rejections,
                enforce_hermiticity=True,
                enforce_trace=False,
            )
        if solver_name in {"ab2", "adams_bashforth_2", "adams-bashforth-2"}:
            return AdamsBashforth2Solver(
                tolerance=cmd_cfg.solver_tolerance,
                max_iterations=cmd_cfg.solver_max_iterations,
                enforce_hermiticity=True,
                enforce_trace=False,
            )
        raise ValueError(f"Unsupported CMD solver '{cmd_cfg.solver}'.")

    @staticmethod
    def _save_frequency_domain_from_saved_orders(
        cmd: CMD,
        rho_order_paths: dict[int, Path],
        output_dir: Path,
    ) -> dict[int, Path]:
        window = np.asarray(
            cmd.timegrid.apply_window(np.ones(cmd.timegrid.Nt, dtype=float)),
            dtype=float,
        )
        nfft = cmd.timegrid.Nt * cmd.timegrid.padding_factor if cmd.timegrid.zero_padding else cmd.timegrid.Nt

        saved_paths: dict[int, Path] = {}
        progress_timer = ProgressTimer(total=len(rho_order_paths))
        for order, path in rho_order_paths.items():
            start = time.perf_counter()
            tensor = expand_rho_tensor_time_axis(
                np.load(path, mmap_mode="r"),
                nt=cmd.timegrid.Nt,
            )
            weighted = tensor * window[np.newaxis, :, np.newaxis, np.newaxis]
            transformed = np.asarray(np.fft.fft(weighted, n=nfft, axis=1), dtype=np.complex128)
            npy_path = output_dir / f"rho_frequency_order_{order}.npy"
            save_array_npy(npy_path, transformed, dtype=cmd.rho_storage_dtype)
            saved_paths[order] = npy_path
            progress_timer.advance()
            QXTISimulation._emit_progress(
                f"Frequency save {progress_timer.completed}/{progress_timer.total}: '{npy_path.name}' "
                f"({format_bytes(npy_path.stat().st_size)}) "
                f"in {format_duration(time.perf_counter() - start)} "
                f"(elapsed {format_duration(progress_timer.elapsed_seconds)}, ETA {progress_timer.eta_text()})."
            )
        return saved_paths

    @staticmethod
    def _load_saved_rho_order_paths(
        rho_order_paths: dict[int, Path],
        *,
        nt: int,
    ) -> dict[int, ComplexArray]:
        rho_orders: dict[int, ComplexArray] = {}
        progress_timer = ProgressTimer(total=len(rho_order_paths))
        for order, path in rho_order_paths.items():
            start = time.perf_counter()
            raw_tensor = np.load(path, mmap_mode="r")
            tensor = expand_rho_tensor_time_axis(raw_tensor, nt=nt)
            is_memmap = isinstance(raw_tensor, np.memmap)
            if np.issubdtype(tensor.dtype, np.complexfloating):
                rho_orders[order] = tensor
            else:
                rho_orders[order] = np.asarray(tensor, dtype=np.complex128)
            progress_timer.advance()
            if is_memmap:
                QXTISimulation._emit_progress(
                    f"Rho map {progress_timer.completed}/{progress_timer.total}: memory-mapped '{path.name}' "
                    f"({format_bytes(path.stat().st_size)} on disk) "
                    f"in {format_duration(time.perf_counter() - start)} "
                    f"(elapsed {format_duration(progress_timer.elapsed_seconds)}, ETA {progress_timer.eta_text()})."
                )
            else:
                QXTISimulation._emit_progress(
                    f"Rho load {progress_timer.completed}/{progress_timer.total}: '{path.name}' "
                    f"({format_bytes(path.stat().st_size)}) "
                    f"in {format_duration(time.perf_counter() - start)} "
                    f"(elapsed {format_duration(progress_timer.elapsed_seconds)}, ETA {progress_timer.eta_text()})."
                )
        return rho_orders

    @staticmethod
    def _build_laser_from_mapping(spec: dict[str, Any]) -> Laser:
        payload = dict(spec)
        if "w0" in payload and "omega" not in payload:
            payload["omega"] = payload.pop("w0")
        if "phase" in payload and "cep" not in payload:
            payload["cep"] = payload.pop("phase")
        if "ellipticity" in payload and "ellip" not in payload:
            payload["ellip"] = payload.pop("ellipticity")
        if "num_cycles" in payload and "ncycles" not in payload:
            payload["ncycles"] = payload.pop("num_cycles")
        if "envelope" in payload and "envname" not in payload:
            payload["envname"] = payload.pop("envelope")
        return Laser(**payload)

    @staticmethod
    def _normalize_plot_name(plot_name: str) -> str:
        key = plot_name.strip().lower()
        aliases = {
            "bands_2d": "band_structure_2d",
            "bandas_2d": "band_structure_2d",
            "band_structure_2d": "band_structure_2d",
            "bands_3d": "band_surface_3d",
            "bandas_3d": "band_surface_3d",
            "band_surface_3d": "band_surface_3d",
            "velocities_2d": "velocity_2d",
            "velocidades_2d": "velocity_2d",
            "velocity_2d": "velocity_2d",
            "velocities_3d": "velocity_field_3d",
            "velocidades_3d": "velocity_field_3d",
            "velocity_field_3d": "velocity_field_3d",
            "velocity_magnitude": "velocity_magnitude",
            "modulo_velocidad": "velocity_magnitude",
            "modulo_de_velocidad": "velocity_magnitude",
        }
        return aliases.get(key, key)

    @staticmethod
    def _active_directions(dimension: int) -> list[str]:
        if dimension == 1:
            return ["x"]
        if dimension == 2:
            return ["x", "y"]
        return ["x", "y", "z"]

    def _emit_step_started(
        self,
        label: str,
        timer: ProgressTimer,
        message: str,
    ) -> None:
        self._emit_progress(
            f"{label} step {timer.completed + 1}/{timer.total}: {message} "
            f"(elapsed {format_duration(timer.elapsed_seconds)}, ETA {timer.eta_text()})."
        )

    def _emit_step_completed(
        self,
        label: str,
        timer: ProgressTimer,
        message: str,
        *,
        step_seconds: float,
    ) -> None:
        timer.advance()
        self._emit_progress(
            f"{label} step {timer.completed}/{timer.total}: {message} "
            f"in {format_duration(step_seconds)} "
            f"(elapsed {format_duration(timer.elapsed_seconds)}, ETA {timer.eta_text()})."
        )

    @staticmethod
    def _emit_progress(message: str) -> None:
        print(f"[QXTI] {message}")
