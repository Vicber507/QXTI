from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep matplotlib/font caches in a writable place for local runs.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

from qxti.core import QXTIConfig
from qxti.data import (
    ResponseData,
    load_dataset_npz,
    load_rho_orders_from_dat,
    load_rho_orders_from_npy,
)
from qxti.graphics.plot_harmonics import HarmonicGraphics, resolve_harmonic_plot_config
from qxti.graphics.plot_hamiltonian import HamiltonianGraphics
from qxti.graphics.plot_response import ResponseGraphics, resolve_response_plot_config


DEFAULT_HAMILTONIAN_PLOTS = (
    "band_structure_2d",
    "band_surface_3d",
    "velocity_2d",
    "velocity_field_3d",
    "velocity_magnitude",
)


def plot_hamiltonian_graphics_from_saved_data(
    config_path: str | Path,
) -> dict[str, Path]:
    config = QXTIConfig.from_file(config_path)
    plot_cfg = config.hamiltonian_plots
    output_dir = Path(plot_cfg.output_dir)
    data_dir = output_dir / "data"
    requested_plots = plot_cfg.plots or DEFAULT_HAMILTONIAN_PLOTS
    outputs: dict[str, Path] = {}

    for plot_name in requested_plots:
        normalized = _normalize_plot_name(plot_name)
        dataset_path = data_dir / f"{normalized}.npz"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Missing Hamiltonian dataset '{dataset_path}'. "
                "Run `python main.py inputParams.cfg` first."
            )
        data = load_dataset_npz(dataset_path)
        print(f"[graphics] plotting Hamiltonian dataset '{dataset_path.name}'.")

        if normalized == "band_structure_2d":
            outputs[normalized] = HamiltonianGraphics.plot_band_structure_2d(
                data,
                output_dir / f"{normalized}.png",
            )
        elif normalized == "band_surface_3d":
            outputs[normalized] = HamiltonianGraphics.plot_band_surface_3d(
                data,
                output_dir / f"{normalized}.png",
                style=plot_cfg.surface_style,
            )
        elif normalized == "velocity_2d":
            outputs[normalized] = HamiltonianGraphics.plot_velocity_2d(
                data,
                output_dir / f"{normalized}.png",
            )
        elif normalized == "velocity_field_3d":
            outputs[normalized] = HamiltonianGraphics.plot_velocity_field_3d(
                data,
                output_dir / f"{normalized}.png",
                stride=plot_cfg.quiver_stride,
            )
        elif normalized == "velocity_magnitude":
            outputs[normalized] = HamiltonianGraphics.plot_velocity_magnitude(
                data,
                output_dir / f"{normalized}.png",
            )
        else:
            raise ValueError(f"Unsupported Hamiltonian plot '{plot_name}'.")

    return outputs


def plot_response_graphics_from_saved_data(
    config_path: str | Path,
    *,
    plot_config: dict[str, object] | None = None,
) -> dict[str, Path]:
    config = QXTIConfig.from_file(config_path)
    output_dir = Path(config.cmd.output_dir)
    resolved_plot_config = resolve_response_plot_config(plot_config)
    outputs: dict[str, Path] = {}

    rho_orders, time_axis, k_points, kx_values, ky_values, kz_values = _load_response_fallback_data(
        config_path
    )
    if not rho_orders:
        raise FileNotFoundError(
            "Missing response datasets and no saved rho_order_*.npy files were found. "
            "Legacy rho_order_*.dat files are still supported if they already exist. "
            "Run `python main.py inputParams.cfg` first."
        )
    print("[graphics] plotting response graphics from saved rho_order tensors.")

    requested_orders = _resolve_requested_orders(resolved_plot_config.get("orders"))
    population_cfg = resolved_plot_config["population"]
    coherence_cfg = resolved_plot_config["coherence"]

    try:
        population_kmap_data = ResponseData.population_kxky_animation_data_from_saved_rho(
            rho_orders,
            time_axis=time_axis,
            kx_values=kx_values,
            ky_values=ky_values,
            kz_values=kz_values,
            orders=requested_orders,
            value_mode=str(population_cfg.get("value_mode", "delta")),
        )
    except ValueError as exc:
        population_kmap_data = None
        print(f"[graphics] skipped population kx-ky data: {exc}")

    try:
        coherence_kmap_data = ResponseData.coherence_kxky_animation_data_from_saved_rho(
            rho_orders,
            time_axis=time_axis,
            kx_values=kx_values,
            ky_values=ky_values,
            kz_values=kz_values,
            orders=requested_orders,
            component=str(coherence_cfg["component"]),
        )
    except ValueError as exc:
        coherence_kmap_data = None
        print(f"[graphics] skipped coherence kx-ky data: {exc}")

    if population_kmap_data is not None and bool(population_cfg["snapshots"]["enabled"]):
        population_snapshot_indices = ResponseGraphics.resolve_snapshot_indices(
            np.asarray(population_kmap_data["time_axis"], dtype=float),
            num_snapshots=int(population_cfg["snapshots"]["num_snapshots"]),
            snapshot_times=list(population_cfg["snapshots"]["snapshot_times"]),
            snapshot_indices=list(population_cfg["snapshots"]["snapshot_indices"]),
        )
        outputs["rho_population_snapshots"] = ResponseGraphics.plot_population_snapshots(
            population_kmap_data,
            output_dir / str(population_cfg["snapshots"]["output_file"]),
            snapshot_indices=population_snapshot_indices,
            cmap=str(population_cfg["snapshots"]["cmap"]),
            center_zero=bool(population_cfg["snapshots"].get("center_zero", False)),
            contrast_percentile=float(population_cfg["snapshots"].get("contrast_percentile", 100.0)),
        )

    if coherence_kmap_data is not None and bool(coherence_cfg["snapshots"]["enabled"]):
        coherence_snapshot_indices = ResponseGraphics.resolve_snapshot_indices(
            np.asarray(coherence_kmap_data["time_axis"], dtype=float),
            num_snapshots=int(coherence_cfg["snapshots"]["num_snapshots"]),
            snapshot_times=list(coherence_cfg["snapshots"]["snapshot_times"]),
            snapshot_indices=list(coherence_cfg["snapshots"]["snapshot_indices"]),
        )
        outputs["rho_coherence_snapshots"] = ResponseGraphics.plot_coherence_snapshots(
            coherence_kmap_data,
            output_dir / str(coherence_cfg["snapshots"]["output_file"]),
            snapshot_indices=coherence_snapshot_indices,
            cmap=str(coherence_cfg["snapshots"]["cmap"]),
            center_zero=bool(coherence_cfg["snapshots"].get("center_zero", False)),
            contrast_percentile=float(coherence_cfg["snapshots"].get("contrast_percentile", 100.0)),
        )

    if population_kmap_data is not None and bool(population_cfg["video"]["enabled"]):
        print("[graphics] static population plots completed; generating population video.")
        try:
            outputs["rho_population_kxky_video"] = ResponseGraphics.animate_population_kxky_maps(
                population_kmap_data,
                output_dir / str(population_cfg["video"]["output_file"]),
                fps=int(population_cfg["video"]["fps"]),
                duration_seconds=None if population_cfg["video"].get("duration_seconds") is None else float(population_cfg["video"]["duration_seconds"]),
                frame_stride=int(population_cfg["video"]["frame_stride"]),
                cmap=str(population_cfg["video"]["cmap"]),
                center_zero=bool(population_cfg["video"].get("center_zero", False)),
                contrast_percentile=float(population_cfg["video"].get("contrast_percentile", 100.0)),
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[graphics] skipped population video: {exc}")

    if coherence_kmap_data is not None and bool(coherence_cfg["video"]["enabled"]):
        print("[graphics] static coherence plots completed; generating coherence video.")
        try:
            outputs["rho_coherence_kxky_video"] = ResponseGraphics.animate_coherence_kxky_maps(
                coherence_kmap_data,
                output_dir / str(coherence_cfg["video"]["output_file"]),
                fps=int(coherence_cfg["video"]["fps"]),
                duration_seconds=None if coherence_cfg["video"].get("duration_seconds") is None else float(coherence_cfg["video"]["duration_seconds"]),
                frame_stride=int(coherence_cfg["video"]["frame_stride"]),
                cmap=str(coherence_cfg["video"]["cmap"]),
                center_zero=bool(coherence_cfg["video"].get("center_zero", False)),
                contrast_percentile=float(coherence_cfg["video"].get("contrast_percentile", 100.0)),
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[graphics] skipped coherence video: {exc}")

    return outputs


def plot_harmonic_graphics_from_saved_data(
    config_path: str | Path,
    *,
    plot_config: dict[str, object] | None = None,
) -> dict[str, Path]:
    config = QXTIConfig.from_file(config_path)
    output_dir = Path(config.cmd.output_dir)
    data_dir = output_dir / "data"
    resolved_plot_config = resolve_harmonic_plot_config(plot_config)
    outputs: dict[str, Path] = {}

    dataset_name = None
    for section_name in ("field_current_time", "current_spectrum", "current_circular_spectrum"):
        section_cfg = resolved_plot_config[section_name]
        if bool(section_cfg["enabled"]):
            dataset_name = str(section_cfg["dataset_file"])
            break
    if dataset_name is None:
        return outputs

    dataset_path = data_dir / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing harmonic dataset '{dataset_path}'. "
            "Run `python main.py inputParams.cfg` first."
        )

    data = load_dataset_npz(dataset_path)
    print(f"[graphics] plotting harmonic dataset '{dataset_path.name}'.")

    field_current_cfg = resolved_plot_config["field_current_time"]
    if bool(field_current_cfg["enabled"]):
        outputs["field_current_time"] = HarmonicGraphics.plot_field_current_time_comparison(
            np.asarray(data["time_axis"], dtype=float),
            np.asarray(data["electric_field_time"], dtype=float),
            np.asarray(data["current_time"], dtype=float),
            output_dir / str(field_current_cfg["output_file"]),
            directions=tuple(str(direction) for direction in field_current_cfg["directions"]),
            include_total=bool(field_current_cfg["include_total"]),
        )

    current_cfg = resolved_plot_config["current_spectrum"]
    if bool(current_cfg["enabled"]):
        outputs["current_spectrum"] = HarmonicGraphics.plot_current_magnitude_spectrum(
            np.asarray(data["omega_axis"], dtype=float),
            np.asarray(data["current_spectrum"], dtype=np.complex128),
            output_dir / str(current_cfg["output_file"]),
            orders=tuple(int(order) for order in data.get("orders", ())),
            directions=tuple(str(direction) for direction in current_cfg["directions"]),
            positive_only=bool(current_cfg["positive_only"]),
            omega_min=None if current_cfg["omega_min"] is None else float(current_cfg["omega_min"]),
            omega_max=None if current_cfg["omega_max"] is None else float(current_cfg["omega_max"]),
            fundamental_omega=float(config.laser.omega),
            use_harmonic_order=bool(current_cfg.get("use_harmonic_order", False)),
            max_harmonic_order=None if current_cfg.get("max_harmonic_order") is None else float(current_cfg["max_harmonic_order"]),
            log_scale=bool(current_cfg["log_scale"]),
        )

    circular_cfg = resolved_plot_config["current_circular_spectrum"]
    if bool(circular_cfg["enabled"]):
        outputs["current_circular_spectrum"] = HarmonicGraphics.plot_circular_current_spectrum(
            np.asarray(data["omega_axis"], dtype=float),
            np.asarray(data["current_spectrum"], dtype=np.complex128),
            output_dir / str(circular_cfg["output_file"]),
            orders=tuple(int(order) for order in data.get("orders", ())),
            positive_only=bool(circular_cfg["positive_only"]),
            omega_min=None if circular_cfg["omega_min"] is None else float(circular_cfg["omega_min"]),
            omega_max=None if circular_cfg["omega_max"] is None else float(circular_cfg["omega_max"]),
            fundamental_omega=float(config.laser.omega),
            use_harmonic_order=bool(circular_cfg.get("use_harmonic_order", False)),
            max_harmonic_order=None if circular_cfg.get("max_harmonic_order") is None else float(circular_cfg["max_harmonic_order"]),
            log_scale=bool(circular_cfg["log_scale"]),
        )

    return outputs


def plot_all_graphics_from_saved_data(
    config_path: str | Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs.update(plot_hamiltonian_graphics_from_saved_data(config_path))
    outputs.update(plot_harmonic_graphics_from_saved_data(config_path))
    outputs.update(plot_response_graphics_from_saved_data(config_path))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate QXTI graphics from saved datasets without recalculating physics."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="inputParams.cfg",
        help="Path to the configuration file. Defaults to inputParams.cfg.",
    )
    parser.add_argument(
        "--family",
        choices=("all", "hamiltonian", "response", "harmonics"),
        default="all",
        help="Choose which graphics family to generate from saved data.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.family == "all":
        outputs = plot_all_graphics_from_saved_data(args.config)
    elif args.family == "hamiltonian":
        outputs = plot_hamiltonian_graphics_from_saved_data(args.config)
    elif args.family == "harmonics":
        outputs = plot_harmonic_graphics_from_saved_data(args.config)
    else:
        outputs = plot_response_graphics_from_saved_data(args.config)

    print(f"Generated {len(outputs)} graphics from saved data in {args.config}:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


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


def _load_response_fallback_data(
    config_path: str | Path,
) -> tuple[dict[int, object], object, object, object, object, object]:
    config = QXTIConfig.from_file(config_path)
    output_dir = Path(config.cmd.output_dir)

    rho_orders = load_rho_orders_from_npy(output_dir)
    if rho_orders:
        from qxti.core import QXTISimulation

        simulation = QXTISimulation.from_file(config_path)
        hamiltonian = simulation.build_hamiltonian()
        cmd = simulation.build_cmd(hamiltonian)
        return (
            rho_orders,
            np.asarray(cmd.timegrid.generate(), dtype=float),
            np.asarray(cmd.kgrid.points(), dtype=float),
            np.asarray(cmd.kgrid.kx_values, dtype=float),
            np.asarray(cmd.kgrid.ky_values, dtype=float),
            np.asarray(cmd.kgrid.kz_values, dtype=float),
        )

    rho_orders_dat, k_points, time_axis = load_rho_orders_from_dat(output_dir)
    if rho_orders_dat:
        kx_values = np.unique(k_points[:, 0])
        ky_values = np.unique(k_points[:, 1])
        kz_values = np.unique(k_points[:, 2])
        return (
            rho_orders_dat,
            time_axis,
            k_points,
            kx_values,
            ky_values,
            kz_values,
        )

    empty = []
    return ({}, empty, empty, empty, empty, empty)


def _resolve_requested_orders(config_value: object) -> tuple[int, ...] | None:
    if config_value is None:
        return None
    if isinstance(config_value, str) and config_value.strip().lower() in {"", "all", "none"}:
        return None
    if isinstance(config_value, int):
        return (int(config_value),)
    if isinstance(config_value, (list, tuple)):
        return tuple(int(item) for item in config_value)
    raise ValueError("response plot config 'orders' must be 'all' or a list/tuple of ints.")


if __name__ == "__main__":
    raise SystemExit(main())
