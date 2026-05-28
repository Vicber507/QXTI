from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qxti.core.config import QXTIConfig
from qxti.data import HamiltonianData
from qxti.graphics import HamiltonianGraphics
from qxti.physics import CustomHamiltonian, Hamiltonian


@dataclass(slots=True)
class QXTISimulation:
    """Minimal orchestrator for intrinsic Hamiltonian diagnostics and plots."""

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

    def run(self) -> dict[str, Path]:
        hamiltonian = self.build_hamiltonian()
        outputs: dict[str, Path] = {}
        outputs.update(self.generate_hamiltonian_plots(hamiltonian))
        return outputs

    def generate_hamiltonian_plots(self, hamiltonian: Hamiltonian) -> dict[str, Path]:
        plot_cfg = self.config.hamiltonian_plots
        if not plot_cfg.enabled or not plot_cfg.plots:
            return {}

        output_dir = Path(plot_cfg.output_dir)
        data_builder = HamiltonianData(hamiltonian)
        outputs: dict[str, Path] = {}

        for plot_name in plot_cfg.plots:
            normalized = self._normalize_plot_name(plot_name)
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
                outputs[normalized] = HamiltonianGraphics.plot_band_structure_2d(
                    data,
                    output_dir / f"{normalized}.png",
                )
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
                outputs[normalized] = HamiltonianGraphics.plot_band_surface_3d(
                    data,
                    output_dir / f"{normalized}.png",
                    style=plot_cfg.surface_style,
                )
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
                outputs[normalized] = HamiltonianGraphics.plot_velocity_2d(
                    data,
                    output_dir / f"{normalized}.png",
                )
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
                outputs[normalized] = HamiltonianGraphics.plot_velocity_field_3d(
                    data,
                    output_dir / f"{normalized}.png",
                    stride=plot_cfg.quiver_stride,
                )
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
                outputs[normalized] = HamiltonianGraphics.plot_velocity_magnitude(
                    data,
                    output_dir / f"{normalized}.png",
                )
            else:
                raise ValueError(f"Unsupported Hamiltonian plot '{plot_name}'.")

        return outputs
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
