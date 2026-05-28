from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep matplotlib/font caches in a writable place for local runs.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

from qxti.core import QXTIConfig
from qxti.data import load_dataset_npz
from qxti.graphics.plot_hamiltonian import HamiltonianGraphics
from qxti.graphics.plot_response import ResponseGraphics


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
) -> dict[str, Path]:
    config = QXTIConfig.from_file(config_path)
    output_dir = Path(config.cmd.output_dir)
    data_dir = output_dir / "data"
    outputs: dict[str, Path] = {}

    heatmap_data_path = data_dir / "population_time_heatmap_all_orders.npz"
    if not heatmap_data_path.exists():
        raise FileNotFoundError(
            f"Missing response dataset '{heatmap_data_path}'. "
            "Run `python main.py inputParams.cfg` first."
        )

    heatmap_data = load_dataset_npz(heatmap_data_path)
    print(f"[graphics] plotting response dataset '{heatmap_data_path.name}'.")
    outputs["rho_population_heatmap"] = ResponseGraphics.plot_population_heatmap(
        heatmap_data,
        output_dir / "population_time_heatmap_all_orders.png",
    )

    kmap_data_path = data_dir / "population_kx_ky_per_band.npz"
    if kmap_data_path.exists():
        kmap_data = load_dataset_npz(kmap_data_path)
        try:
            outputs["rho_population_kxky_animation"] = ResponseGraphics.animate_population_kxky_maps(
                kmap_data,
                output_dir / "population_kx_ky_per_band.gif",
                fps=10,
                frame_stride=2,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[graphics] skipped kx-ky animation: {exc}")
    else:
        print(f"[graphics] skipped kx-ky animation because '{kmap_data_path.name}' is missing.")

    return outputs


def plot_all_graphics_from_saved_data(
    config_path: str | Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    outputs.update(plot_hamiltonian_graphics_from_saved_data(config_path))
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
        choices=("all", "hamiltonian", "response"),
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


if __name__ == "__main__":
    raise SystemExit(main())
