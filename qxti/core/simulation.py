from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Thread
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qxti.core.config import CMDConfig, LaserConfig, QXTIConfig
from qxti.data import HarmonicData, HamiltonianData, ResponseData, StreamingCurrentAccumulator, save_dataset_npz
from qxti.grids import FrequencyGrid, KGrid, TimeGrid
from qxti.physics import CustomHamiltonian, Hamiltonian, Laser, LaserSystem, OperatorFactory
from qxti.response import CMD, XTP
from qxti.response.xtp import XTP as _XTPClass
from qxti.solvers import AdamsBashforth2Solver, RKF45Solver, Solver
from qxti.utils.io_utils import expand_rho_tensor_time_axis, save_array_npy
from qxti.utils.progress import DetailedProgressReporter, ProgressTimer, format_bytes, format_duration


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
    _STEP_HEARTBEAT_SECONDS = 15.0

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
        shifted = bool(getattr(kcfg, "shifted", False))

        if kcfg.kx_values or kcfg.ky_values or kcfg.kz_values:
            # Explicit user-provided axes are used verbatim (no auto-shift).
            kx_values = np.asarray(kcfg.kx_values or [0.0], dtype=float)
            ky_values = np.asarray(kcfg.ky_values or [0.0], dtype=float)
            kz_values = np.asarray(kcfg.kz_values or [0.0], dtype=float)
            kgrid = KGrid(
                kx_values=kx_values,
                ky_values=ky_values,
                kz_values=kz_values,
                dimension=dimension,
                shifted=shifted,
            )
            if not shifted:
                self._warn_if_grid_hits_degeneracy(hamiltonian, kgrid)
            return kgrid

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

        def _axis(lo: float, hi: float, n: int) -> NDArray[np.float64]:
            if shifted:
                # Symmetric Monkhorst-Pack: never lands on a BZ edge/degeneracy.
                return KGrid.shifted_axis(lo, hi, n)
            return np.linspace(lo, hi, n, dtype=float)

        kx_values = _axis(bounds[0][0], bounds[0][1], nkx)
        ky_values = (
            _axis(bounds[1][0], bounds[1][1], nky)
            if dimension >= 2
            else np.array([0.0], dtype=float)
        )
        kz_values = (
            _axis(bounds[2][0], bounds[2][1], nkz)
            if dimension >= 3
            else np.array([0.0], dtype=float)
        )

        kgrid = KGrid(
            kx_values=kx_values,
            ky_values=ky_values,
            kz_values=kz_values,
            dimension=dimension,
            shifted=shifted,
        )
        if shifted:
            self._emit_progress(
                "KGrid: using shifted Monkhorst-Pack sampling (offset half a step). "
                "Grid never lands on BZ edges/degeneracies; BZ integrals use uniform weights."
            )
        else:
            self._warn_if_grid_hits_degeneracy(hamiltonian, kgrid)
        return kgrid

    def _warn_if_grid_hits_degeneracy(
        self,
        hamiltonian: Hamiltonian,
        kgrid: KGrid,
        *,
        gap_threshold: float = 1.0e-6,
    ) -> None:
        """Warn when a grid point coincides with a band degeneracy.

        A k-point sitting exactly on a band-touching point (gap ~ 0, e.g. a
        Dirac point) makes the eigenvectors ambiguous and the Berry connection
        singular, which can spuriously break harmonic selection rules (e.g.
        revive the forbidden 2nd harmonic in graphene).  This is cheap: a single
        ``eigvalsh`` per k-point.  Skipped entirely for shifted grids.
        """
        if hamiltonian.basis_size < 2:
            return
        points = kgrid.points()
        min_gap = np.inf
        worst_k = None
        for kx, ky, kz in points:
            energies = np.linalg.eigvalsh(hamiltonian._matrix_at(float(kx), float(ky), float(kz)))
            gap = float(energies[1] - energies[0])
            if gap < min_gap:
                min_gap = gap
                worst_k = (float(kx), float(ky), float(kz))
        if min_gap < gap_threshold:
            self._emit_progress(
                f"[WARNING] KGrid hits a band degeneracy: minimum band gap is "
                f"{min_gap:.2e} a.u. at k={worst_k}. A grid point sits on a "
                f"band-touching point (e.g. a Dirac point), which makes the Berry "
                f"connection singular and can produce SPURIOUS even harmonics "
                f"(e.g. a forbidden 2nd harmonic). FIX: set 'shifted = true' in the "
                f"[kgrid] section to use a Monkhorst-Pack grid that avoids "
                f"degeneracies for any number of points (no extra cost), or change "
                f"the number of k-points."
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

    def build_laser_system(self, laser_config: LaserConfig | None = None) -> LaserSystem:
        lcfg = self.config.laser if laser_config is None else laser_config
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

    def build_cmd(
        self,
        hamiltonian: Hamiltonian,
        *,
        cmd_config: CMDConfig | None = None,
        laser_system: LaserSystem | None = None,
        max_order: int | None = None,
    ) -> CMD:
        ccfg = self.config.cmd if cmd_config is None else cmd_config
        laser_system = self.build_laser_system() if laser_system is None else laser_system
        timegrid = self.build_timegrid(laser_system)
        operator_factory = OperatorFactory(hamiltonian=hamiltonian, basis=ccfg.basis)
        return CMD(
            hamiltonian=hamiltonian,
            laser_system=laser_system,
            kgrid=self.build_kgrid(hamiltonian),
            timegrid=timegrid,
            operator_factory=operator_factory,
            solver=self._build_solver(timegrid),
            max_order=ccfg.max_order if max_order is None else int(max_order),
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
            n_workers=None if ccfg.n_workers <= 0 else ccfg.n_workers,
        )

    def build_xtp(
        self,
        cmd: CMD,
        rho_orders: dict[int, ComplexArray],
    ) -> XTP:
        omega_axis = np.asarray(cmd.timegrid.frequency_axis(), dtype=float)
        omega_max = float(np.max(np.abs(omega_axis))) if omega_axis.size else 1.0
        frequencygrid = FrequencyGrid(
            0.0,
            omega_max if omega_max > 0.0 else 1.0,
            len(omega_axis) if omega_axis.size else 1,
        )

        return XTP(
            hamiltonian=cmd.hamiltonian,
            rho_orders=rho_orders,
            kgrid=cmd.kgrid,
            timegrid=cmd.timegrid,
            frequencygrid=frequencygrid,
            operator_factory=cmd.operator_factory,
            directions=self._active_directions(cmd.hamiltonian.dimension),
            orders=sorted(rho_orders),
            band_gauge_frame=cmd.band_gauge_frame if cmd.basis == "band" else None,
            laser_system=cmd.laser_system,
            bz_mask_enabled=self.config.xtp.bz_mask_enabled,
            bz_mask_radius_percent=self.config.xtp.bz_mask_radius_percent,
            bz_mask_sigma=self.config.xtp.bz_mask_sigma,
            bz_mask_sigma_percent_legacy=self.config.xtp.bz_mask_sigma_percent_legacy,
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

        runtime_cmd_cfg = self._cmd_runtime_config()
        cmd = self.build_cmd(hamiltonian, cmd_config=runtime_cmd_cfg)
        output_dir = Path(cmd_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._emit_progress(
            f"CMD enabled: solving density matrices up to order {cmd_cfg.max_order}."
        )
        if runtime_cmd_cfg.rho_storage_dtype != cmd_cfg.rho_storage_dtype:
            self._emit_progress(
                "CMD scratch storage auto-optimized: using "
                f"{runtime_cmd_cfg.rho_storage_dtype} for temporary rho scratch "
                f"(configured retained-rho dtype: {cmd_cfg.rho_storage_dtype})."
            )

        outputs: dict[str, Path] = {}
        scratch_dir = output_dir if cmd_cfg.keep_rho_orders else (output_dir / ".scratch_rho")
        if not cmd_cfg.keep_rho_orders:
            scratch_dir.mkdir(parents=True, exist_ok=True)
            self._emit_progress(
                "CMD rho retention disabled: large rho_order_*.npy files will be used as temporary scratch and deleted after datasets are saved."
            )

        # ---- Streaming observable path ----
        # When only the HHG spectrum is needed (not rho files, not kx-ky maps)
        # we stream J^(s)(t) on-the-fly during the CMD solve.
        # This eliminates the need to save and re-read the final rho_order file,
        # which can save tens of GB of disk I/O for large grids.
        use_streaming = (
            bool(cmd_cfg.save_xtp_dataset)
            and not bool(cmd_cfg.keep_rho_orders)
            and not bool(cmd_cfg.save_population_dataset)
            and not bool(cmd_cfg.save_coherence_dataset)
            and not bool(cmd_cfg.save_frequency_domain)
            and cmd.basis == "band"
            and cmd.band_gauge_frame is not None
        )
        needs_saved_rho_postprocessing = self._needs_saved_rho_postprocessing(use_streaming=use_streaming)
        reclaim_intermediate_orders = (
            not bool(cmd_cfg.keep_rho_orders)
            and not needs_saved_rho_postprocessing
            and not bool(cmd_cfg.save_frequency_domain)
        )
        skip_final_order_disk_write = (
            not bool(cmd_cfg.keep_rho_orders)
            and not needs_saved_rho_postprocessing
            and not use_streaming
        )

        streaming_acc: StreamingCurrentAccumulator | None = None
        order_observe_callbacks: dict[int, object] | None = None

        if use_streaming:
            weights = self._compute_bz_weights(cmd)
            active_dim = hamiltonian.dimension
            direction_labels = self._active_directions(active_dim)
            streaming_acc = StreamingCurrentAccumulator(
                nt=cmd.timegrid.Nt,
                integration_weights=weights,
                current_operators={
                    d: cmd.band_gauge_frame.current(d) for d in direction_labels
                },
                dipole_operators={
                    d: cmd.band_gauge_frame.connection(d) for d in direction_labels
                },
                active_dimension=active_dim,
            )
            # Accumulate order 0 (equilibrium, constant in time) via the
            # specialised equilibrium callback (avoids (Nt, Nb, Nb) broadcast).
            order_observe_callbacks = {0: streaming_acc.make_equilibrium_callback(0)}
            # Accumulate driven orders; last order uses streaming (no disk write).
            for s in range(1, cmd.max_order + 1):
                order_observe_callbacks[s] = streaming_acc.make_callback(s)
            self._emit_progress(
                "CMD streaming observable mode: J(t) and P(t) accumulated on the fly. "
                f"Last order (rho^({cmd.max_order})) will NOT be written to disk."
            )
        elif skip_final_order_disk_write:
            self._emit_progress(
                "CMD compact-only mode: final rho order will be computed without being written to disk."
            )

        if reclaim_intermediate_orders:
            self._emit_progress(
                "CMD scratch reclamation enabled: obsolete rho scratch files will be deleted order-by-order."
            )

        rho_order_paths = cmd.solve_time_domain(
            scratch_dir,
            order_observe_callbacks=order_observe_callbacks,
            reclaim_intermediate_orders=reclaim_intermediate_orders,
            skip_final_order_disk_write=skip_final_order_disk_write,
        )

        if cmd_cfg.keep_rho_orders:
            for order, npy_path in rho_order_paths.items():
                outputs[f"rho_order_{order}"] = npy_path

        try:
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

            if use_streaming and streaming_acc is not None:
                # Build the harmonic dataset directly from accumulated arrays.
                outputs.update(
                    self._generate_xtp_datasets_from_streaming(cmd, streaming_acc, output_dir)
                )
            elif needs_saved_rho_postprocessing:
                rho_orders = self._load_saved_rho_order_paths(rho_order_paths, nt=cmd.timegrid.Nt)
                outputs.update(self.generate_cmd_datasets(cmd, rho_orders))
                outputs.update(self.generate_xtp_datasets(cmd, rho_orders))
            else:
                self._emit_progress(
                    "CMD saved-rho postprocessing disabled by config: skipping response/XTP dataset generation."
                )
        finally:
            if not cmd_cfg.keep_rho_orders:
                self._cleanup_rho_order_paths(rho_order_paths)
                self._cleanup_directory_if_empty(scratch_dir)
        return outputs

    @staticmethod
    def _compute_bz_weights(cmd: CMD) -> np.ndarray:
        """Compute BZ integration weights matching XTP._integration_weights."""
        kgrid = cmd.kgrid
        hamiltonian = cmd.hamiltonian
        try:
            bounds = tuple(hamiltonian.reciprocal_box_bounds())
        except ValueError:
            kx = np.asarray(kgrid.kx_values, dtype=float)
            ky = np.asarray(kgrid.ky_values, dtype=float)
            kz = np.asarray(kgrid.kz_values, dtype=float)
            bounds = (
                (float(kx[0]), float(kx[-1])),
                (float(ky[0]), float(ky[-1])),
                (float(kz[0]), float(kz[-1])),
            )
        periodic = bool(getattr(kgrid, "shifted", False))
        wx = _XTPClass._axis_integration_weights(
            np.asarray(kgrid.kx_values, dtype=float), bounds[0], active=True, periodic=periodic
        )
        wy = _XTPClass._axis_integration_weights(
            np.asarray(kgrid.ky_values, dtype=float),
            bounds[1] if kgrid.dimension >= 2 else (0.0, 0.0),
            active=kgrid.dimension >= 2,
            periodic=periodic,
        )
        wz = _XTPClass._axis_integration_weights(
            np.asarray(kgrid.kz_values, dtype=float),
            bounds[2] if kgrid.dimension >= 3 else (0.0, 0.0),
            active=kgrid.dimension >= 3,
            periodic=periodic,
        )
        return np.asarray(
            (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).reshape(-1),
            dtype=np.float64,
        )

    def _generate_xtp_datasets_from_streaming(
        self,
        cmd: CMD,
        acc: StreamingCurrentAccumulator,
        output_dir: Path,
    ) -> dict[str, Path]:
        """Build current-spectrum dataset from streamed J^(s)(t) arrays."""
        self._emit_progress("XTP streaming: assembling current-spectrum dataset from on-the-fly accumulation.")
        output_data_dir = output_dir / "data"
        output_data_dir.mkdir(parents=True, exist_ok=True)

        nt = cmd.timegrid.Nt
        driven_orders = acc.driven_orders()
        all_orders = acc.available_orders()
        directions = tuple(self._active_directions(cmd.hamiltonian.dimension))

        # Driven current time series: sum of positive orders.
        driven_current = np.zeros((nt, 3), dtype=np.float64)
        driven_polarization = np.zeros((nt, 3), dtype=np.float64)
        for s in driven_orders:
            driven_current += acc.current_time(s)
            driven_polarization += acc.polarization_time(s)

        # Total current: sum over all orders (including equilibrium if accumulated).
        total_current = np.zeros((nt, 3), dtype=np.float64)
        for s in all_orders:
            total_current += acc.current_time(s)

        equilibrium_current = acc.current_time(0) if 0 in all_orders else np.zeros((nt, 3), dtype=np.float64)
        equilibrium_polarization = acc.polarization_time(0) if 0 in all_orders else np.zeros((nt, 3), dtype=np.float64)

        # FFT helper (mirrors HarmonicData._fft_time_signal).
        def _fft(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            vals = np.asarray(signal, dtype=np.complex128)
            window = np.asarray(
                cmd.timegrid.apply_window(np.ones(nt, dtype=float)), dtype=float
            )
            reshape = (nt,) + (1,) * (vals.ndim - 1)
            weighted = vals * window.reshape(reshape)
            nfft = nt * cmd.timegrid.padding_factor if cmd.timegrid.zero_padding else nt
            spectrum = cmd.timegrid.dt * np.fft.fft(weighted, n=nfft, axis=0)
            omega = np.asarray(cmd.timegrid.frequency_axis(), dtype=float)
            return omega, np.asarray(spectrum, dtype=np.complex128)

        # Intraband / interband decomposition of the driven current.
        driven_intra = np.zeros((nt, 3), dtype=np.float64)
        for s in driven_orders:
            driven_intra += acc.current_intra_time(s)
        driven_inter = driven_current - driven_intra

        omega_axis, current_spectrum = _fft(driven_current)
        _, total_current_spectrum = _fft(total_current)
        _, intraband_spectrum = _fft(driven_intra)
        _, interband_spectrum = _fft(driven_inter)
        current_total_magnitude = np.sqrt(np.sum(np.abs(current_spectrum) ** 2, axis=1))

        electric_field_time = np.asarray(
            cmd.laser_system.electric_field(cmd.timegrid.generate()), dtype=float
        )

        data: dict = {
            "omega_axis": omega_axis,
            "current_spectrum": np.asarray(current_spectrum, dtype=np.complex128),
            "current_magnitude": np.abs(current_spectrum).astype(float),
            "current_total_magnitude": current_total_magnitude.astype(float),
            "current_time": driven_current,
            "polarization_time": driven_polarization,
            "current_time_total": total_current,
            "current_spectrum_total": np.asarray(total_current_spectrum, dtype=np.complex128),
            "equilibrium_current_time": equilibrium_current,
            "equilibrium_polarization_time": equilibrium_polarization,
            "time_axis": np.asarray(cmd.timegrid.generate(), dtype=float),
            "directions": directions,
            "orders": driven_orders,
            "all_orders": all_orders,
            "equilibrium_subtracted": bool(0 in all_orders),
            "bz_mask": {
                "enabled": False,
                "shape": "circular" if cmd.kgrid.dimension == 2 else "spherical",
                "radius_percent": 100.0,
                "sigma_input": None,
                "reference_radius": 0.0,
                "radius": 0.0,
                "sigma": 0.0,
            },
            "electric_field_time": electric_field_time,
            # Intraband/interband decomposition (streamed at no extra k-loop cost).
            "current_decomposition_available": True,
            "current_time_intraband": driven_intra,
            "current_time_interband": driven_inter,
            "current_spectrum_intraband": np.asarray(intraband_spectrum, dtype=np.complex128),
            "current_spectrum_interband": np.asarray(interband_spectrum, dtype=np.complex128),
            "current_total_magnitude_intraband": np.sqrt(
                np.sum(np.abs(intraband_spectrum) ** 2, axis=1)
            ).astype(float),
            "current_total_magnitude_interband": np.sqrt(
                np.sum(np.abs(interband_spectrum) ** 2, axis=1)
            ).astype(float),
        }

        step_timer = ProgressTimer(total=1)
        step_start = time.perf_counter()
        with self._step_progress_monitor("XTP streaming dataset", step_timer, "saving current-spectrum dataset"):
            current_spectrum_path = save_dataset_npz(output_data_dir / "current_spectrum.npz", data)
        self._emit_step_completed(
            "XTP streaming dataset",
            step_timer,
            f"saved '{current_spectrum_path.name}' ({format_bytes(current_spectrum_path.stat().st_size)})",
            step_seconds=time.perf_counter() - step_start,
        )
        return {"xtp_current_spectrum_data": current_spectrum_path}

    def generate_cmd_datasets(
        self,
        cmd: CMD,
        rho_orders: dict[int, ComplexArray],
    ) -> dict[str, Path]:
        cmd_cfg = self.config.cmd
        population_enabled = bool(cmd_cfg.save_population_dataset)
        coherence_enabled = bool(cmd_cfg.save_coherence_dataset)
        if not population_enabled and not coherence_enabled:
            self._emit_progress(
                "CMD response datasets disabled by config: skipping population/coherence kx-ky datasets."
            )
            return {}

        self._emit_progress("CMD data products enabled: generating reusable kx-ky datasets.")
        data_builder = ResponseData(cmd)
        output_dir = Path(self.config.cmd.output_dir) / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        step_timer = ProgressTimer(
            total=2 * int(population_enabled) + 2 * int(coherence_enabled),
            min_completed_for_eta=2,
        )

        all_orders = tuple(sorted(rho_orders))
        dataset_time_indices = self._cmd_dataset_time_indices(cmd.timegrid.Nt)
        frame_note = self._cmd_dataset_frame_note(cmd.timegrid.Nt, dataset_time_indices)

        outputs: dict[str, Path] = {}

        if population_enabled:
            step_start = time.perf_counter()
            with self._step_progress_monitor(
                "CMD dataset",
                step_timer,
                f"building population kx-ky dataset{frame_note}",
            ):
                try:
                    kmap_data = data_builder.population_kxky_animation_data(
                        orders=all_orders,
                        time_indices=dataset_time_indices,
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
                else:
                    kmap_data["population_frames"] = np.asarray(kmap_data["population_frames"], dtype=np.float32)
                    if kmap_data.get("equilibrium_population_frame") is not None:
                        kmap_data["equilibrium_population_frame"] = np.asarray(
                            kmap_data["equilibrium_population_frame"],
                            dtype=np.float32,
                        )
                    self._emit_step_completed(
                        "CMD dataset",
                        step_timer,
                        "population kx-ky dataset built",
                        step_seconds=time.perf_counter() - step_start,
                    )
            if kmap_data is not None:
                step_start = time.perf_counter()
                with self._step_progress_monitor(
                    "CMD dataset",
                    step_timer,
                    "saving population kx-ky dataset",
                ):
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
        else:
            self._emit_progress("CMD population dataset disabled by config: skipping population kx-ky dataset.")

        if coherence_enabled:
            step_start = time.perf_counter()
            with self._step_progress_monitor(
                "CMD dataset",
                step_timer,
                f"building coherence kx-ky dataset{frame_note}",
            ):
                try:
                    coherence_kmap_data = data_builder.coherence_kxky_animation_data(
                        orders=all_orders,
                        component="complex",
                        time_indices=dataset_time_indices,
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
                else:
                    if "coherence_frames" in coherence_kmap_data:
                        coherence_kmap_data["coherence_frames_complex"] = np.asarray(
                            coherence_kmap_data.pop("coherence_frames"),
                            dtype=np.complex64,
                        )
                    self._emit_step_completed(
                        "CMD dataset",
                        step_timer,
                        "coherence kx-ky dataset built",
                        step_seconds=time.perf_counter() - step_start,
                    )
            if coherence_kmap_data is not None:
                step_start = time.perf_counter()
                with self._step_progress_monitor(
                    "CMD dataset",
                    step_timer,
                    "saving coherence kx-ky dataset",
                ):
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
        else:
            self._emit_progress("CMD coherence dataset disabled by config: skipping coherence kx-ky dataset.")
        return outputs

    def _cmd_dataset_time_indices(self, nt: int) -> NDArray[np.int_] | None:
        stride = max(1, int(self.config.cmd.dataset_time_stride))
        if stride <= 1 or nt <= 1:
            return None
        indices = np.arange(0, nt, stride, dtype=int)
        if indices[-1] != nt - 1:
            indices = np.concatenate([indices, np.array([nt - 1], dtype=int)])
        return indices

    @staticmethod
    def _cmd_dataset_frame_note(
        nt: int,
        dataset_time_indices: NDArray[np.int_] | None,
    ) -> str:
        if dataset_time_indices is None:
            return ""
        return f" using {len(dataset_time_indices)}/{nt} time frames"

    def generate_xtp_datasets(
        self,
        cmd: CMD,
        rho_orders: dict[int, ComplexArray],
    ) -> dict[str, Path]:
        current_spectrum_enabled = bool(self.config.cmd.save_xtp_dataset)
        if not current_spectrum_enabled:
            self._emit_progress(
                "XTP datasets disabled by config: skipping current-spectrum dataset."
            )
            return {}

        self._emit_progress(
            "XTP data products enabled: generating reusable current-spectrum dataset."
        )
        step_timer = ProgressTimer(total=2)
        xtp = self.build_xtp(cmd, rho_orders)
        electric_field_time = np.asarray(
            cmd.laser_system.electric_field(cmd.timegrid.generate()),
            dtype=float,
        )
        step_start = time.perf_counter()
        output_dir = Path(self.config.cmd.output_dir) / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}

        data_builder = HarmonicData(xtp, electric_field_time=electric_field_time)
        build_reporter = DetailedProgressReporter(
            label="XTP dataset build",
            total=max(1, data_builder.current_spectrum_work_units()),
            emit=self._emit_progress,
            interval_seconds=self._STEP_HEARTBEAT_SECONDS,
            min_completed_for_eta=max(1, min(cmd.kgrid.total_points, 8)),
        )
        with self._step_progress_monitor(
            "XTP dataset",
            step_timer,
            "building current-spectrum dataset",
            interval_seconds=0.0,
        ):
            current_spectrum_data = data_builder.current_spectrum_data(
                progress_callback=build_reporter.advance,
            )
        build_reporter.finish("current-spectrum observables assembled")
        self._emit_step_completed(
            "XTP dataset",
            step_timer,
            "current-spectrum dataset built",
            step_seconds=time.perf_counter() - step_start,
        )

        step_start = time.perf_counter()
        with self._step_progress_monitor("XTP dataset", step_timer, "saving current-spectrum dataset"):
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
        outputs["xtp_current_spectrum_data"] = current_spectrum_path

        return outputs

    def _cmd_runtime_config(self) -> CMDConfig:
        cmd_cfg = self.config.cmd
        if cmd_cfg.keep_rho_orders:
            return cmd_cfg

        scratch_dtype = cmd_cfg.scratch_rho_storage_dtype
        if scratch_dtype == "auto":
            scratch_dtype = self._auto_cmd_scratch_rho_storage_dtype()

        if scratch_dtype == cmd_cfg.rho_storage_dtype:
            return cmd_cfg
        return replace(cmd_cfg, rho_storage_dtype=scratch_dtype)

    def _auto_cmd_scratch_rho_storage_dtype(self) -> str:
        cmd_cfg = self.config.cmd
        scratch_only_mode = (
            not cmd_cfg.keep_rho_orders
            and not cmd_cfg.save_population_dataset
            and not cmd_cfg.save_coherence_dataset
            and not cmd_cfg.save_frequency_domain
        )
        if scratch_only_mode:
            return "float16_complex"
        return cmd_cfg.rho_storage_dtype

    def _needs_saved_rho_postprocessing(self, *, use_streaming: bool = False) -> bool:
        cmd_cfg = self.config.cmd
        return bool(
            cmd_cfg.save_population_dataset
            or cmd_cfg.save_coherence_dataset
            or cmd_cfg.save_frequency_domain
            or (cmd_cfg.save_xtp_dataset and not use_streaming)
        )

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
    def _cleanup_rho_order_paths(rho_order_paths: dict[int, Path]) -> None:
        for path in rho_order_paths.values():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue

    @staticmethod
    def _cleanup_directory_if_empty(directory: Path) -> None:
        try:
            directory.rmdir()
        except OSError:
            pass

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
        self._emit_progress(f"{label} step {timer.completed + 1}/{timer.total}: {message}.")

    def _emit_step_running(
        self,
        label: str,
        timer: ProgressTimer,
        message: str,
    ) -> None:
        suffix = self._step_runtime_suffix(timer)
        self._emit_progress(f"{label} step {timer.completed + 1}/{timer.total}: {message} ({suffix}).")

    def _emit_step_completed(
        self,
        label: str,
        timer: ProgressTimer,
        message: str,
        *,
        step_seconds: float,
    ) -> None:
        timer.advance()
        suffix = self._step_runtime_suffix(timer)
        self._emit_progress(
            f"{label} step {timer.completed}/{timer.total}: {message} "
            f"in {format_duration(step_seconds)} "
            f"({suffix})."
        )

    @staticmethod
    def _step_runtime_suffix(timer: ProgressTimer) -> str:
        elapsed = format_duration(timer.elapsed_seconds)
        eta = timer.eta_text()
        if eta == "unknown":
            return f"elapsed {elapsed}"
        return f"elapsed {elapsed}, ETA {eta}"

    @contextmanager
    def _step_progress_monitor(
        self,
        label: str,
        timer: ProgressTimer,
        message: str,
        *,
        interval_seconds: float | None = None,
    ):
        self._emit_step_started(label, timer, message)
        stop_event = Event()
        heartbeat_thread: Thread | None = None
        heartbeat_interval = self._STEP_HEARTBEAT_SECONDS if interval_seconds is None else float(interval_seconds)
        if heartbeat_interval > 0.0:
            heartbeat_thread = Thread(
                target=self._step_progress_heartbeat,
                args=(label, timer, message, stop_event, heartbeat_interval),
                daemon=True,
            )
            heartbeat_thread.start()
        try:
            yield
        finally:
            stop_event.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.2)

    def _step_progress_heartbeat(
        self,
        label: str,
        timer: ProgressTimer,
        message: str,
        stop_event: Event,
        interval_seconds: float,
    ) -> None:
        while not stop_event.wait(interval_seconds):
            self._emit_step_running(label, timer, message)

    @staticmethod
    def _emit_progress(message: str) -> None:
        print(f"[QXTI] {message}")
