from __future__ import annotations

import argparse
import copy
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")


DEFAULT_RESPONSE_PLOT_CONFIG = {
    "orders": "all",
    "population": {
        "video": {
            "enabled": True,
            "output_file": "population_kx_ky_per_band.mp4",
            "fps": 10,
            "frame_stride": 2,
            "cmap": "inferno",
        },
        "snapshots": {
            "enabled": True,
            "output_file": "population_snapshots.png",
            "num_snapshots": 4,
            "snapshot_times": [],
            "snapshot_indices": [],
            "cmap": "inferno",
        },
    },
    "coherence": {
        "component": "magnitude",
        "video": {
            "enabled": True,
            "output_file": "coherence_kx_ky_per_pair.mp4",
            "fps": 10,
            "frame_stride": 2,
            "cmap": "magma",
        },
        "snapshots": {
            "enabled": True,
            "output_file": "coherence_snapshots.png",
            "num_snapshots": 4,
            "snapshot_times": [],
            "snapshot_indices": [],
            "cmap": "magma",
        },
    },
}

# Edit this dictionary directly if you want to change the plotting behavior.
RESPONSE_PLOT_CONFIG = copy.deepcopy(DEFAULT_RESPONSE_PLOT_CONFIG)


HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import animation
    from matplotlib import pyplot as plt
else:
    animation = None
    plt = None


class ResponseGraphics:
    """Plot-only helpers for CMD density-matrix observables."""

    @staticmethod
    def plot_population_heatmap(
        data: dict[str, Any],
        output_path: Path,
        *,
        cmap: str = "inferno",
    ) -> Path:
        band_indices = [str(index) for index in np.asarray(data["band_indices"], dtype=int)]
        return ResponseGraphics._plot_time_heatmap(
            time_axis=np.asarray(data["time_axis"], dtype=float),
            value_map=np.asarray(data["population_map"], dtype=float),
            labels=band_indices,
            output_path=output_path,
            cmap=cmap,
            title=f"Populations from sum of rho^(s), s = {ResponseGraphics._orders_text(data)}",
            ylabel="band index",
            colorbar_label=f"population ({data['aggregation_label']})",
        )

    @staticmethod
    def plot_coherence_heatmap(
        data: dict[str, Any],
        output_path: Path,
        *,
        cmap: str = "magma",
    ) -> Path:
        pair_labels = [str(label) for label in data["pair_labels"]]
        component = str(data["component"])
        return ResponseGraphics._plot_time_heatmap(
            time_axis=np.asarray(data["time_axis"], dtype=float),
            value_map=np.asarray(data["coherence_map"], dtype=float),
            labels=pair_labels,
            output_path=output_path,
            cmap=cmap,
            title=f"Coherences ({component}) from sum of rho^(s), s = {ResponseGraphics._orders_text(data)}",
            ylabel="band pair",
            colorbar_label=f"coherence {component} ({data['aggregation_label']})",
        )

    @staticmethod
    def animate_population_kxky_maps(
        data: dict[str, Any],
        output_path: Path,
        *,
        fps: int = 10,
        frame_stride: int = 1,
        cmap: str = "inferno",
    ) -> Path:
        labels = [f"Band {index}" for index in np.asarray(data["band_indices"], dtype=int)]
        return ResponseGraphics._animate_kxky_maps(
            frames=np.asarray(data["population_frames"], dtype=float),
            labels=labels,
            time_axis=np.asarray(data["time_axis"], dtype=float),
            kx_values=np.asarray(data["kx_values"], dtype=float),
            ky_values=np.asarray(data["ky_values"], dtype=float),
            output_path=output_path,
            fps=fps,
            frame_stride=frame_stride,
            cmap=cmap,
            title_prefix="Population density in kx-ky",
            colorbar_label="population density",
        )

    @staticmethod
    def animate_coherence_kxky_maps(
        data: dict[str, Any],
        output_path: Path,
        *,
        fps: int = 10,
        frame_stride: int = 1,
        cmap: str = "magma",
    ) -> Path:
        labels = [f"Pair {label}" for label in data["pair_labels"]]
        component = str(data["component"])
        return ResponseGraphics._animate_kxky_maps(
            frames=np.asarray(data["coherence_frames"], dtype=float),
            labels=labels,
            time_axis=np.asarray(data["time_axis"], dtype=float),
            kx_values=np.asarray(data["kx_values"], dtype=float),
            ky_values=np.asarray(data["ky_values"], dtype=float),
            output_path=output_path,
            fps=fps,
            frame_stride=frame_stride,
            cmap=cmap,
            title_prefix=f"Coherence {component} in kx-ky",
            colorbar_label=f"coherence {component}",
        )

    @staticmethod
    def plot_population_snapshots(
        data: dict[str, Any],
        output_path: Path,
        *,
        snapshot_indices: list[int] | np.ndarray,
        cmap: str = "inferno",
    ) -> Path:
        labels = [f"Band {index}" for index in np.asarray(data["band_indices"], dtype=int)]
        return ResponseGraphics._plot_kxky_snapshots(
            frames=np.asarray(data["population_frames"], dtype=float),
            labels=labels,
            time_axis=np.asarray(data["time_axis"], dtype=float),
            kx_values=np.asarray(data["kx_values"], dtype=float),
            ky_values=np.asarray(data["ky_values"], dtype=float),
            snapshot_indices=np.asarray(snapshot_indices, dtype=int),
            output_path=output_path,
            cmap=cmap,
            title_prefix="Population density",
            colorbar_label="population density",
        )

    @staticmethod
    def plot_coherence_snapshots(
        data: dict[str, Any],
        output_path: Path,
        *,
        snapshot_indices: list[int] | np.ndarray,
        cmap: str = "magma",
    ) -> Path:
        labels = [f"Pair {label}" for label in data["pair_labels"]]
        component = str(data["component"])
        return ResponseGraphics._plot_kxky_snapshots(
            frames=np.asarray(data["coherence_frames"], dtype=float),
            labels=labels,
            time_axis=np.asarray(data["time_axis"], dtype=float),
            kx_values=np.asarray(data["kx_values"], dtype=float),
            ky_values=np.asarray(data["ky_values"], dtype=float),
            snapshot_indices=np.asarray(snapshot_indices, dtype=int),
            output_path=output_path,
            cmap=cmap,
            title_prefix=f"Coherence {component}",
            colorbar_label=f"coherence {component}",
        )

    @staticmethod
    def resolve_snapshot_indices(
        time_axis: np.ndarray,
        *,
        num_snapshots: int,
        snapshot_times: list[float] | tuple[float, ...] | None = None,
        snapshot_indices: list[int] | tuple[int, ...] | None = None,
    ) -> np.ndarray:
        if snapshot_indices:
            resolved = [int(index) for index in snapshot_indices]
        elif snapshot_times:
            resolved = [int(np.argmin(np.abs(time_axis - float(time)))) for time in snapshot_times]
        else:
            count = max(int(num_snapshots), 1)
            resolved = np.linspace(0, len(time_axis) - 1, count, dtype=int).tolist()
        return np.asarray(sorted(set(resolved)), dtype=int)

    @staticmethod
    def _plot_time_heatmap(
        *,
        time_axis: np.ndarray,
        value_map: np.ndarray,
        labels: list[str],
        output_path: Path,
        cmap: str,
        title: str,
        ylabel: str,
        colorbar_label: str,
    ) -> Path:
        pyplot = ResponseGraphics._require_matplotlib()
        figure, axis = pyplot.subplots(figsize=(9.0, 4.8))
        extent = [float(time_axis[0]), float(time_axis[-1]), -0.5, len(labels) - 0.5]
        image = axis.imshow(
            value_map,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
        )
        axis.set_title(title)
        axis.set_xlabel("time (a.u.)")
        axis.set_ylabel(ylabel)
        axis.set_yticks(np.arange(len(labels), dtype=int))
        axis.set_yticklabels(labels)
        figure.colorbar(image, ax=axis, label=colorbar_label)
        return ResponseGraphics._save(figure, output_path)

    @staticmethod
    def _animate_kxky_maps(
        *,
        frames: np.ndarray,
        labels: list[str],
        time_axis: np.ndarray,
        kx_values: np.ndarray,
        ky_values: np.ndarray,
        output_path: Path,
        fps: int,
        frame_stride: int,
        cmap: str,
        title_prefix: str,
        colorbar_label: str,
    ) -> Path:
        pyplot = ResponseGraphics._require_matplotlib()
        frame_stride = max(int(frame_stride), 1)
        fps = max(int(fps), 1)
        frame_indices = np.arange(0, frames.shape[0], frame_stride, dtype=int)
        n_entities = len(labels)
        ncols = min(2, max(n_entities, 1))
        nrows = int(np.ceil(max(n_entities, 1) / ncols))
        figure, axes = pyplot.subplots(
            nrows,
            ncols,
            figsize=(6.6 * ncols, 5.3 * nrows),
            squeeze=False,
        )
        axes_flat = list(axes.flat)
        images = []
        extent = [float(kx_values[0]), float(kx_values[-1]), float(ky_values[0]), float(ky_values[-1])]

        for axis, label, entity_index in zip(axes_flat, labels, range(n_entities), strict=False):
            entity_frames = frames[:, entity_index, :, :]
            image = axis.imshow(
                entity_frames[frame_indices[0]],
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=float(np.min(entity_frames)),
                vmax=float(np.max(entity_frames)),
            )
            axis.set_xlabel("kx (a.u.)")
            axis.set_ylabel("ky (a.u.)")
            axis.set_title(label)
            figure.colorbar(image, ax=axis, label=colorbar_label, shrink=0.88)
            images.append(image)

        for axis in axes_flat[n_entities:]:
            axis.set_visible(False)

        suptitle = figure.suptitle(
            f"{title_prefix} at t = {time_axis[frame_indices[0]]:.4f} a.u.",
            fontsize=14,
        )

        def _update(frame_position: int):
            frame_index = frame_indices[frame_position]
            for entity_index, image in enumerate(images):
                image.set_data(frames[frame_index, entity_index, :, :])
            suptitle.set_text(f"{title_prefix} at t = {time_axis[frame_index]:.4f} a.u.")
            return tuple(images) + (suptitle,)

        animation_obj = animation.FuncAnimation(
            figure,
            _update,
            frames=len(frame_indices),
            interval=1000.0 / fps,
            blit=False,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = ResponseGraphics._animation_writer(output_path, fps=fps)
        animation_obj.save(output_path, writer=writer, dpi=140)
        plt.close(figure)
        return output_path

    @staticmethod
    def _plot_kxky_snapshots(
        *,
        frames: np.ndarray,
        labels: list[str],
        time_axis: np.ndarray,
        kx_values: np.ndarray,
        ky_values: np.ndarray,
        snapshot_indices: np.ndarray,
        output_path: Path,
        cmap: str,
        title_prefix: str,
        colorbar_label: str,
    ) -> Path:
        pyplot = ResponseGraphics._require_matplotlib()
        n_entities = len(labels)
        n_snapshots = len(snapshot_indices)
        figure, axes = pyplot.subplots(
            n_entities,
            n_snapshots,
            figsize=(4.8 * n_snapshots, 4.4 * n_entities),
            squeeze=False,
        )
        extent = [float(kx_values[0]), float(kx_values[-1]), float(ky_values[0]), float(ky_values[-1])]

        for row, label in enumerate(labels):
            for col, frame_index in enumerate(snapshot_indices):
                axis = axes[row, col]
                image = axis.imshow(
                    frames[frame_index, row, :, :],
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    cmap=cmap,
                )
                axis.set_xlabel("kx (a.u.)")
                axis.set_ylabel("ky (a.u.)")
                axis.set_title(f"{label}, t = {time_axis[frame_index]:.4f} a.u.")
                figure.colorbar(image, ax=axis, label=colorbar_label, shrink=0.82)

        figure.suptitle(f"{title_prefix} snapshots", fontsize=14)
        return ResponseGraphics._save(figure, output_path)

    @staticmethod
    def _orders_text(data: dict[str, Any]) -> str:
        return ", ".join(str(order) for order in data["orders"])

    @staticmethod
    def _animation_writer(output_path: Path, *, fps: int):
        if animation is None:
            raise RuntimeError("matplotlib animation is not available in this environment.")

        suffix = output_path.suffix.lower()
        if suffix == ".gif":
            if animation.writers.is_available("pillow"):
                return animation.PillowWriter(fps=fps)
            raise RuntimeError("PillowWriter is not available; install Pillow to save GIF animations.")
        if suffix == ".mp4":
            if animation.writers.is_available("ffmpeg"):
                return animation.FFMpegWriter(fps=fps)
            raise RuntimeError("FFMpegWriter is not available; install ffmpeg to save MP4 animations.")
        raise ValueError("Animation output_path must end in .gif or .mp4.")

    @staticmethod
    def _save(figure: Any, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        return output_path

    @staticmethod
    def _require_matplotlib() -> Any:
        if not HAS_MATPLOTLIB:
            raise RuntimeError(
                "matplotlib is not installed in this environment. "
                "Install it to generate response plots."
            )
        return plt


def resolve_response_plot_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(RESPONSE_PLOT_CONFIG)
    if overrides:
        _deep_update(config, overrides)
    return config


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CMD response plots from saved data using the plot dictionary in plot_response.py."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="inputParams.cfg",
        help="Path to the configuration file. Defaults to inputParams.cfg.",
    )
    return parser


def _main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from qxti.graphics.graphics import plot_response_graphics_from_saved_data

    parser = _build_parser()
    args = parser.parse_args()
    outputs = plot_response_graphics_from_saved_data(
        args.config,
        plot_config=resolve_response_plot_config(),
    )
    print(f"Generated {len(outputs)} response graphics from saved data in {args.config}:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
