#!/usr/bin/env python3
from __future__ import annotations

"""Plot WSM node-mask responses in an elliptical polarization basis.

This is a pure post-processing utility: it reads the already saved node-mask
datasets and rotates the cartesian conductivity tensors to an elliptical basis
in the x-y plane, leaving z unchanged.

The default basis uses one example ellipticity parameter η:

    e1 = cos(η) x + i sin(η) y
    e2 = sin(η) x - i cos(η) y
    ez = z

η = 0°    -> linear x/y basis (up to a harmless phase on e2)
η = 45°   -> circular helicity basis (+, -)
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
from qxti.graphics.plot_susceptibility_tensor import apply_paper_style


AU_TO_EV = 27.211386245988
DEFAULT_BASE_DIR = Path("outputs/wsm_orenstein_helicity_node_masks")
CASE_ORDER = ("full", "no_positive_chirality", "no_negative_chirality")
CASE_COLORS = {
    "full": "#111827",
    "no_positive_chirality": "#0072B2",
    "no_negative_chirality": "#D55E00",
}
CASE_LABELS = {
    "full": "All nodes",
    "no_positive_chirality": r"Without $\chi=+1$ nodes",
    "no_negative_chirality": r"Without $\chi=-1$ nodes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reads saved WSM node-mask conductivity datasets and replots them in an "
            "elliptical polarization basis, without running any simulation."
        )
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Base directory containing full/no_positive/no_negative saved datasets.",
    )
    parser.add_argument(
        "--eta-deg",
        type=float,
        default=22.5,
        help="Ellipticity angle η in degrees. 0° = linear, 45° = circular helicity.",
    )
    return parser.parse_args()


def _elliptical_unitary(eta_rad: float) -> tuple[np.ndarray, tuple[str, ...]]:
    c = float(np.cos(eta_rad))
    s = float(np.sin(eta_rad))
    unitary = np.array(
        [
            [c, s, 0.0],
            [1j * s, -1j * c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    return unitary, ("e1", "e2", "z")


def _rotate_rank2(tensor: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    values = np.asarray(tensor, dtype=np.complex128)
    return np.asarray(
        np.einsum("ai,wij,jb->wab", unitary.conj().T, values, unitary, optimize=True),
        dtype=np.complex128,
    )


def _rotate_rank3(tensor: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    values = np.asarray(np.nan_to_num(tensor, nan=0.0), dtype=np.complex128)
    return np.asarray(
        np.einsum("ai,wijk,jb,kc->wabc", unitary.conj().T, values, unitary, unitary, optimize=True),
        dtype=np.complex128,
    )


def _load_case_tensor(base_dir: Path, case_name: str, filename: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_dataset_npz(base_dir / case_name / "data" / filename)
    omega = np.asarray(dataset["omega_axis"], dtype=np.float64)
    tensor = np.asarray(dataset[key], dtype=np.complex128)
    return omega, tensor


def _plot_drive_response(
    *,
    output_path: Path,
    omega_axis_ev: np.ndarray,
    responses: dict[str, dict[str, np.ndarray]],
    title: str,
    ylabel_prefix: str,
) -> Path:
    component_order = ("e1", "e2", "z")
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 9.8), sharex=True, squeeze=False)
    panel_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for col, component in enumerate(component_order):
        for row, part in enumerate(("real", "imag", "abs")):
            ax = axes[row, col]
            ipanel = row * len(component_order) + col
            for case_name in CASE_ORDER:
                values = np.asarray(responses[case_name][component], dtype=np.complex128)
                if part == "real":
                    y = np.real(values)
                elif part == "imag":
                    y = np.imag(values)
                else:
                    y = np.abs(values)
                ax.plot(
                    omega_axis_ev,
                    y,
                    color=CASE_COLORS[case_name],
                    linewidth=2.0,
                    label=CASE_LABELS[case_name],
                )
            ax.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.set_xlim(float(omega_axis_ev.min()), float(omega_axis_ev.max()))
            ax.set_title(
                f"({panel_labels[ipanel].lower()}) {component} output, {part}",
                loc="left",
                fontsize=11,
            )
            if col == 0:
                if part == "real":
                    ax.set_ylabel(rf"$\Re\,{ylabel_prefix}$")
                elif part == "imag":
                    ax.set_ylabel(rf"$\Im\,{ylabel_prefix}$")
                else:
                    ax.set_ylabel(rf"$|{ylabel_prefix}|$")
            if row == 2:
                ax.set_xlabel(r"$\omega_\mathrm{laser}\;(\mathrm{eV})$")
            if part == "abs":
                ax.set_ylim(bottom=0.0)

    axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    return output_path


def _plot_helicity_delta(
    *,
    output_path: Path,
    omega_axis_ev: np.ndarray,
    deltas: dict[str, np.ndarray],
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.2), sharex=True, squeeze=False)
    ax_re = axes[0, 0]
    ax_im = axes[1, 0]
    ax_abs = axes[2, 0]

    for case_name in CASE_ORDER:
        values = np.asarray(deltas[case_name], dtype=np.complex128)
        style = dict(color=CASE_COLORS[case_name], linewidth=2.0, label=CASE_LABELS[case_name])
        ax_re.plot(omega_axis_ev, np.real(values), **style)
        ax_im.plot(omega_axis_ev, np.imag(values), **style)
        ax_abs.plot(omega_axis_ev, np.abs(values), **style)

    for ax in (ax_re, ax_im, ax_abs):
        ax.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.set_xlim(float(omega_axis_ev.min()), float(omega_axis_ev.max()))
    ax_abs.set_ylim(bottom=0.0)

    ax_re.set_ylabel(r"$\Re\,\delta(\omega)$")
    ax_im.set_ylabel(r"$\Im\,\delta(\omega)$")
    ax_abs.set_ylabel(r"$|\delta(\omega)|$")
    ax_abs.set_xlabel(r"$\omega_\mathrm{laser}\;(\mathrm{eV})$")
    ax_re.legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(
        r"Derived helicity observable $\delta(\omega)=\frac{\sigma_{++}-\sigma_{+-}}{2}$",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    apply_paper_style()

    base_dir = Path(args.base_dir)
    eta_deg = float(args.eta_deg)
    eta_rad = np.deg2rad(eta_deg)
    unitary, basis_labels = _elliptical_unitary(eta_rad)

    omega_order1: np.ndarray | None = None
    omega_order2: np.ndarray | None = None
    order1_responses: dict[str, dict[str, np.ndarray]] = {}
    order2_responses: dict[str, dict[str, np.ndarray]] = {}
    delta_helicity: dict[str, np.ndarray] = {}

    for case_name in CASE_ORDER:
        omega1, sigma1_cart = _load_case_tensor(base_dir, case_name, "sigma_node_mask_case.npz", "sigma_order_1_tensor")
        omega2, sigma2_cart = _load_case_tensor(base_dir, case_name, "sigma2_node_mask_case.npz", "sigma_order_2_tensor")
        _omega_h, sigma1_helicity = _load_case_tensor(
            base_dir,
            case_name,
            "sigma_node_mask_case.npz",
            "sigma_order_1_helicity_tensor",
        )

        if omega_order1 is None:
            omega_order1 = np.asarray(omega1, dtype=np.float64)
        elif not np.allclose(omega_order1, omega1):
            raise ValueError(f"omega axis mismatch in order-1 dataset for case {case_name}.")

        if omega_order2 is None:
            omega_order2 = np.asarray(omega2, dtype=np.float64)
        elif not np.allclose(omega_order2, omega2):
            raise ValueError(f"omega axis mismatch in order-2 dataset for case {case_name}.")

        sigma1_ell = _rotate_rank2(sigma1_cart, unitary)
        sigma2_ell = _rotate_rank3(sigma2_cart, unitary)

        # Elliptical drive: E along the first elliptical basis vector e1.
        order1_responses[case_name] = {
            "e1": np.asarray(sigma1_ell[:, 0, 0], dtype=np.complex128),
            "e2": np.asarray(sigma1_ell[:, 1, 0], dtype=np.complex128),
            "z": np.asarray(sigma1_ell[:, 2, 0], dtype=np.complex128),
        }
        order2_responses[case_name] = {
            "e1": np.asarray(sigma2_ell[:, 0, 0, 0], dtype=np.complex128),
            "e2": np.asarray(sigma2_ell[:, 1, 0, 0], dtype=np.complex128),
            "z": np.asarray(sigma2_ell[:, 2, 0, 0], dtype=np.complex128),
        }
        delta_helicity[case_name] = 0.5 * (
            np.asarray(sigma1_helicity[:, 0, 0], dtype=np.complex128)
            - np.asarray(sigma1_helicity[:, 0, 1], dtype=np.complex128)
        )

    if omega_order1 is None or omega_order2 is None:
        raise FileNotFoundError("No suitable saved node-mask datasets were found.")

    output_dir = base_dir / f"elliptical_basis_eta_{eta_deg:.1f}".replace(".", "p")
    omega1_ev = np.asarray(omega_order1, dtype=np.float64) * AU_TO_EV
    omega2_ev = np.asarray(omega_order2, dtype=np.float64) * AU_TO_EV

    out1 = _plot_drive_response(
        output_path=output_dir / "sigma1_elliptical_drive_e1.png",
        omega_axis_ev=omega1_ev,
        responses=order1_responses,
        title=(
            rf"WSM order-1 response for elliptical drive $e_1$, "
            rf"$\eta = {eta_deg:.1f}^\circ$"
        ),
        ylabel_prefix=r"\sigma^{(1)}",
    )
    out2 = _plot_drive_response(
        output_path=output_dir / "sigma2_elliptical_drive_e1e1.png",
        omega_axis_ev=omega2_ev,
        responses=order2_responses,
        title=(
            rf"WSM SHG response for elliptical drive $e_1 e_1$, "
            rf"$\eta = {eta_deg:.1f}^\circ$"
        ),
        ylabel_prefix=r"\sigma^{(2)}",
    )
    out_delta = _plot_helicity_delta(
        output_path=output_dir / "delta_helicity_sigma_pp_minus_pm_over2.png",
        omega_axis_ev=omega1_ev,
        deltas=delta_helicity,
    )

    # Extra compact z-focused plot for the out-of-plane response.
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)
    for case_name in CASE_ORDER:
        values = np.asarray(order2_responses[case_name]["z"], dtype=np.complex128)
        axes[0].plot(omega2_ev, np.real(values), color=CASE_COLORS[case_name], linewidth=2.0, label=CASE_LABELS[case_name])
        axes[1].plot(omega2_ev, np.abs(values), color=CASE_COLORS[case_name], linewidth=2.0, label=CASE_LABELS[case_name])
    axes[0].set_ylabel(r"$\Re\,\sigma^{(2)}_{z,e_1,e_1}$")
    axes[1].set_ylabel(r"$|\sigma^{(2)}_{z,e_1,e_1}|$")
    axes[1].set_xlabel(r"$\omega_\mathrm{laser}\;(\mathrm{eV})$")
    axes[1].set_ylim(bottom=0.0)
    for ax in axes:
        ax.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.set_xlim(float(omega2_ev.min()), float(omega2_ev.max()))
        ax.legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(
        rf"Out-of-plane SHG under elliptical drive: $\sigma^{{(2)}}_{{z,e_1,e_1}}$, $\eta = {eta_deg:.1f}^\circ$",
        fontsize=13,
    )
    fig.tight_layout()
    out3 = output_dir / "sigma2_z_e1e1_focus.png"
    out3.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out3, dpi=300, facecolor="white")
    plt.close(fig)

    print(
        "[elliptical-basis] Generated post-processed plots from saved data only.\n"
        f"  Basis: e1 = cos(eta) x + i sin(eta) y, e2 = sin(eta) x - i cos(eta) y, eta = {eta_deg:.1f} deg\n"
        f"  Files:\n"
        f"    - {out1}\n"
        f"    - {out2}\n"
        f"    - {out3}\n"
        f"    - {out_delta}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
