from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(slots=True)
class SusceptibilityTensorPlotter:
    """Plot real and imaginary parts of susceptibility tensor components."""

    x_axis: RealArray
    tensor: ComplexArray
    output_dir: str | Path = "outputs/chi_plots"
    x_label: str = "omega"
    tensor_name: str = "chi"
    direction_labels: tuple[str, ...] = ("x", "y", "z")
    dpi: int = 200

    def __post_init__(self) -> None:
        self.x_axis = np.asarray(self.x_axis, dtype=np.float64)
        self.tensor = np.asarray(self.tensor, dtype=np.complex128)
        self.output_dir = Path(self.output_dir)

        if self.tensor.ndim < 3:
            raise ValueError(
                "tensor must have shape (N, dim, dim, ...) with at least 3 dimensions."
            )

        if self.tensor.shape[0] != self.x_axis.shape[0]:
            raise ValueError(
                "x_axis length must match tensor first dimension."
            )

        self.dimension = int(self.tensor.shape[1])
        self.order = int(self.tensor.ndim - 2)

        expected_shape = (self.x_axis.shape[0],) + (self.dimension,) * (self.order + 1)
        if self.tensor.shape != expected_shape:
            raise ValueError(
                f"Expected tensor shape {expected_shape}, got {self.tensor.shape}."
            )

        if len(self.direction_labels) < self.dimension:
            raise ValueError("direction_labels must contain at least dimension labels.")

    @classmethod
    def from_npy(
        cls,
        *,
        x_axis_path: str | Path,
        tensor_path: str | Path,
        output_dir: str | Path = "outputs/chi_plots",
        x_label: str = "omega",
        tensor_name: str = "chi",
        direction_labels: tuple[str, ...] = ("x", "y", "z"),
    ) -> SusceptibilityTensorPlotter:
        x_axis = np.load(x_axis_path)
        tensor = np.load(tensor_path)
        return cls(
            x_axis=x_axis,
            tensor=tensor,
            output_dir=output_dir,
            x_label=x_label,
            tensor_name=tensor_name,
            direction_labels=direction_labels,
        )

    def plot_component(
        self,
        indices: tuple[int, ...],
        *,
        save: bool = True,
        show: bool = False,
    ) -> Path | None:
        """Plot real and imaginary parts for one tensor component."""

        self._validate_component_indices(indices)

        values = self.tensor[(slice(None),) + indices]
        label = self._component_label(indices)

        fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)

        axes[0].plot(self.x_axis, np.real(values), color="tab:blue", linewidth=1.4)
        axes[0].set_ylabel(f"Re {self.tensor_name}_{label}")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.x_axis, np.imag(values), color="tab:red", linewidth=1.4)
        axes[1].set_xlabel(self.x_label)
        axes[1].set_ylabel(f"Im {self.tensor_name}_{label}")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"{self.tensor_name}_{label}")
        fig.tight_layout()

        output_path = None
        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"{self.tensor_name}_{label}.png"
            fig.savefig(output_path, dpi=self.dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return output_path

    def plot_all_components(
        self,
        *,
        save: bool = True,
        show: bool = False,
    ) -> list[Path]:
        """Plot every tensor component."""

        paths: list[Path] = []

        for indices in self.component_indices():
            path = self.plot_component(indices, save=save, show=show)
            if path is not None:
                paths.append(path)

        return paths

    def plot_grid(
        self,
        *,
        save: bool = True,
        show: bool = False,
    ) -> Path | None:
        """
        Plot all components in one grid.

        Best for chi1 tensors. For higher-order tensors, many panels may be produced.
        """

        indices = list(self.component_indices())
        n_components = len(indices)

        ncols = self.dimension
        nrows = int(np.ceil(n_components / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.0 * ncols, 3.0 * nrows),
            squeeze=False,
            sharex=True,
        )

        for ax, component_indices in zip(axes.flat, indices):
            values = self.tensor[(slice(None),) + component_indices]
            label = self._component_label(component_indices)

            ax.plot(self.x_axis, np.real(values), label="Re", linewidth=1.2)
            ax.plot(self.x_axis, np.imag(values), label="Im", linewidth=1.2)
            ax.set_title(f"{self.tensor_name}_{label}")
            ax.grid(True, alpha=0.3)

        for ax in axes.flat[n_components:]:
            ax.axis("off")

        for ax in axes[-1, :]:
            ax.set_xlabel(self.x_label)

        axes[0, 0].legend()
        fig.tight_layout()

        output_path = None
        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"{self.tensor_name}_grid.png"
            fig.savefig(output_path, dpi=self.dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return output_path

    def component_indices(self) -> Iterable[tuple[int, ...]]:
        """Return all component index combinations excluding the sample axis."""

        return product(range(self.dimension), repeat=self.order + 1)

    def _validate_component_indices(self, indices: tuple[int, ...]) -> None:
        expected = self.order + 1
        if len(indices) != expected:
            raise ValueError(f"indices must contain {expected} entries.")

        for index in indices:
            if index < 0 or index >= self.dimension:
                raise ValueError(
                    f"component index {index} is outside dimension {self.dimension}."
                )

    def _component_label(self, indices: tuple[int, ...]) -> str:
        labels = self.direction_labels[: self.dimension]
        return "".join(labels[index] for index in indices)