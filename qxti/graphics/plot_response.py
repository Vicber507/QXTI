from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np


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
    def plot_population_heatmap(data: dict[str, Any], output_path: Path) -> Path:
        pyplot = ResponseGraphics._require_matplotlib()
        figure, axis = pyplot.subplots(figsize=(8.8, 4.8))

        population_map = np.asarray(data["population_map"], dtype=float)
        time_axis = np.asarray(data["time_axis"], dtype=float)
        band_indices = np.asarray(data["band_indices"], dtype=int)
        extent = [
            float(time_axis[0]),
            float(time_axis[-1]),
            float(band_indices[0]) - 0.5,
            float(band_indices[-1]) + 0.5,
        ]
        image = axis.imshow(
            population_map,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="inferno",
        )

        orders_text = ", ".join(str(order) for order in data["orders"])
        axis.set_title(f"Populations from sum of rho^(s), s = {orders_text}")
        axis.set_xlabel("time (a.u.)")
        axis.set_ylabel("band index")
        axis.set_yticks(band_indices)
        axis.set_yticklabels([str(index) for index in band_indices])
        figure.colorbar(
            image,
            ax=axis,
            label=f"population ({data['aggregation_label']})",
        )
        return ResponseGraphics._save(figure, output_path)

    @staticmethod
    def animate_population_heatmap(
        data: dict[str, Any],
        output_path: Path,
        *,
        fps: int = 10,
        frame_stride: int = 1,
    ) -> Path:
        pyplot = ResponseGraphics._require_matplotlib()
        frame_stride = max(int(frame_stride), 1)
        fps = max(int(fps), 1)

        population_frames = np.asarray(data["population_frames"], dtype=float)
        time_axis = np.asarray(data["time_axis"], dtype=float)
        band_indices = np.asarray(data["band_indices"], dtype=int)
        k_point_indices = np.asarray(data["k_point_indices"], dtype=int)
        frame_indices = np.arange(0, population_frames.shape[0], frame_stride, dtype=int)

        figure, axis = pyplot.subplots(figsize=(8.6, 4.8))
        image = axis.imshow(
            population_frames[frame_indices[0]],
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=float(np.min(population_frames)),
            vmax=float(np.max(population_frames)),
        )
        axis.set_xlabel("k-point index")
        axis.set_ylabel("band index")
        if len(k_point_indices) <= 9:
            axis.set_xticks(k_point_indices)
        else:
            axis.set_xticks(np.linspace(0, len(k_point_indices) - 1, 6, dtype=int))
        axis.set_yticks(band_indices)
        axis.set_yticklabels([str(index) for index in band_indices])
        title = axis.set_title(f"Populations at t = {time_axis[frame_indices[0]]:.4f} a.u.")
        figure.colorbar(image, ax=axis, label="population")

        def _update(frame_position: int):
            frame_index = frame_indices[frame_position]
            image.set_data(population_frames[frame_index])
            title.set_text(f"Populations at t = {time_axis[frame_index]:.4f} a.u.")
            return image, title

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
    def animate_population_kxky_maps(
        data: dict[str, Any],
        output_path: Path,
        *,
        fps: int = 10,
        frame_stride: int = 1,
    ) -> Path:
        pyplot = ResponseGraphics._require_matplotlib()
        frame_stride = max(int(frame_stride), 1)
        fps = max(int(fps), 1)

        population_frames = np.asarray(data["population_frames"], dtype=float)
        time_axis = np.asarray(data["time_axis"], dtype=float)
        band_indices = np.asarray(data["band_indices"], dtype=int)
        kx_values = np.asarray(data["kx_values"], dtype=float)
        ky_values = np.asarray(data["ky_values"], dtype=float)
        frame_indices = np.arange(0, population_frames.shape[0], frame_stride, dtype=int)

        n_bands = len(band_indices)
        ncols = min(2, n_bands)
        nrows = int(np.ceil(n_bands / ncols))
        figure, axes = pyplot.subplots(
            nrows,
            ncols,
            figsize=(6.6 * ncols, 5.3 * nrows),
            squeeze=False,
        )
        axes_flat = list(axes.flat)
        images = []
        extent = [float(kx_values[0]), float(kx_values[-1]), float(ky_values[0]), float(ky_values[-1])]

        for axis, band_id in zip(axes_flat, band_indices, strict=False):
            band_values = population_frames[:, band_id, :, :]
            image = axis.imshow(
                band_values[frame_indices[0]],
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="inferno",
                vmin=float(np.min(band_values)),
                vmax=float(np.max(band_values)),
            )
            axis.set_xlabel("kx (a.u.)")
            axis.set_ylabel("ky (a.u.)")
            axis.set_title(f"Band {band_id}")
            figure.colorbar(image, ax=axis, label="population density", shrink=0.88)
            images.append(image)

        for axis in axes_flat[n_bands:]:
            axis.set_visible(False)

        suptitle = figure.suptitle(
            f"Population density in kx-ky at t = {time_axis[frame_indices[0]]:.4f} a.u.",
            fontsize=14,
        )

        def _update(frame_position: int):
            frame_index = frame_indices[frame_position]
            for image, band_id in zip(images, band_indices, strict=False):
                image.set_data(population_frames[frame_index, band_id, :, :])
            suptitle.set_text(
                f"Population density in kx-ky at t = {time_axis[frame_index]:.4f} a.u."
            )
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CMD response plots from an inputParams.cfg file."
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
    outputs = plot_response_graphics_from_saved_data(args.config)
    print(f"Generated {len(outputs)} response graphics from saved data in {args.config}:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
