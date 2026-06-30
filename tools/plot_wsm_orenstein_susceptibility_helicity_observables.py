#!/usr/bin/env python3
from __future__ import annotations

"""Plot derived helicity observables from the saved WSM Orenstein susceptibility dataset.

Pure post-processing: reads ``xtp_susceptibility.npz``, rotates ``sigma^(1)``
to the helicity basis, and plots

    delta_-(omega) = (sigma_++ - sigma_+-)/2
    delta_+(omega) = (sigma_++ + sigma_+-)/2
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from qxti.data.io import load_dataset_npz
from qxti.graphics.plot_susceptibility_tensor import apply_paper_style, to_helicity_basis


AU_TO_EV = 27.211386245988
DEFAULT_DATASET = Path("outputs/wsm_orenstein_susceptibility/data/xtp_susceptibility.npz")
FALLBACK_DATASET = Path("outputs/good runs/wsm_orenstein_susceptibility/data/xtp_susceptibility.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the saved WSM Orenstein susceptibility dataset and plot derived "
            "helicity observables built from sigma_++ and sigma_+-."
        )
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Path to xtp_susceptibility.npz. If omitted, the standard Orenstein output is used.",
    )
    return parser.parse_args()


def _resolve_dataset_path(raw: str) -> Path:
    if raw:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path
    for candidate in (DEFAULT_DATASET, FALLBACK_DATASET):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No saved WSM Orenstein susceptibility dataset was found.")


def _plot_observable(
    *,
    ax_re,
    ax_im,
    ax_abs,
    omega_ev: np.ndarray,
    values: np.ndarray,
    title: str,
) -> None:
    ax_re.plot(omega_ev, np.real(values), color="#0072B2", linewidth=2.0)
    ax_im.plot(omega_ev, np.imag(values), color="#D55E00", linewidth=2.0)
    ax_abs.plot(omega_ev, np.abs(values), color="#009E73", linewidth=2.0)

    for ax in (ax_re, ax_im, ax_abs):
        ax.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.set_xlim(float(omega_ev.min()), float(omega_ev.max()))

    ax_abs.set_ylim(bottom=0.0)
    ax_re.set_title(title, loc="left", fontsize=12)


def main() -> int:
    args = parse_args()
    apply_paper_style()

    dataset_path = _resolve_dataset_path(args.dataset)
    data = load_dataset_npz(dataset_path)

    omega_axis = np.asarray(data.get("laser_omega_axis", data.get("omega_axis")), dtype=np.float64)
    sigma_cart = np.asarray(data["sigma_order_1_tensor"], dtype=np.complex128)
    dimension = int(sigma_cart.shape[1])
    sigma_hel, hel_labels = to_helicity_basis(sigma_cart, dimension)

    if tuple(hel_labels[:2]) != ("+", "-"):
        raise ValueError("Expected helicity labels (+, -, z) after rotation.")

    sigma_pp = np.asarray(sigma_hel[:, 0, 0], dtype=np.complex128)
    sigma_pm = np.asarray(sigma_hel[:, 0, 1], dtype=np.complex128)
    delta_minus = 0.5 * (sigma_pp - sigma_pm)
    delta_plus = 0.5 * (sigma_pp + sigma_pm)

    omega_ev = omega_axis * AU_TO_EV
    fig, axes = plt.subplots(3, 2, figsize=(11.4, 8.8), sharex=True, squeeze=False)

    _plot_observable(
        ax_re=axes[0, 0],
        ax_im=axes[1, 0],
        ax_abs=axes[2, 0],
        omega_ev=omega_ev,
        values=delta_minus,
        title=r"(a) $\delta_-(\omega)=\frac{\sigma_{++}-\sigma_{+-}}{2}$",
    )
    _plot_observable(
        ax_re=axes[0, 1],
        ax_im=axes[1, 1],
        ax_abs=axes[2, 1],
        omega_ev=omega_ev,
        values=delta_plus,
        title=r"(b) $\delta_+(\omega)=\frac{\sigma_{++}+\sigma_{+-}}{2}$",
    )

    axes[0, 0].set_ylabel(r"$\Re$")
    axes[1, 0].set_ylabel(r"$\Im$")
    axes[2, 0].set_ylabel(r"$|\,\cdot\,|$")
    axes[2, 0].set_xlabel(r"$\omega_\mathrm{laser}\;(\mathrm{eV})$")
    axes[2, 1].set_xlabel(r"$\omega_\mathrm{laser}\;(\mathrm{eV})$")

    fig.suptitle(
        r"WSM Orenstein derived helicity observables from $\sigma^{(1)}$",
        fontsize=14,
    )
    fig.tight_layout()

    output_dir = dataset_path.parent.parent / "derived_helicity_observables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sigma_pp_pm_sum_and_difference_over_2.png"
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)

    print(
        "[orenstein-helicity] Generated derived helicity plots from the saved susceptibility dataset.\n"
        f"  Dataset: {dataset_path}\n"
        f"  Output:  {output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
