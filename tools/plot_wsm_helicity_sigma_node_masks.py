#!/usr/bin/env python3
from __future__ import annotations

"""Graficar sigma^(1) en base de helicidad con mascaras locales en nodos de Weyl.

Genera tres casos para un modelo Weyl cuyo archivo de material exponga
``weyl_nodes_with_chirality(params)``:

1. caso completo,
2. excluyendo volumenes alrededor de los nodos de quiralidad positiva,
3. excluyendo volumenes alrededor de los nodos de quiralidad negativa.

La exclusion se implementa como una mascara multiplicativa sobre los pesos de
integracion en k, reutilizando la misma logica de pesos de BZ que usa XTP.
Los plots se producen con el mismo ``SusceptibilityTensorPlotter`` del repo,
en la base de helicidad ``(+,-,z)``.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.analytics.theory_response import (
    _resolve_distribution,
    build_k_integration_weights,
    compute_linear_response_spectrum,
)
from qxti.core.config import QXTIConfig
from qxti.core.simulation import QXTISimulation
from qxti.data import save_dataset_npz
from qxti.graphics.plot_susceptibility_tensor import (
    SusceptibilityTensorPlotter,
    apply_paper_style,
    to_helicity_basis,
)
from qxti.utils.progress import format_duration

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


AU_TO_EV = 27.211386245988
DEFAULT_CONFIG = Path("inputs/inputParams.wsm_orenstein.cfg")
DEFAULT_OUTPUT_DIR = Path("outputs/wsm_orenstein_helicity_node_masks")
CASE_SPECS = (
    ("full", "full", None),
    ("no_positive_chirality", "remove chi=+1 nodes", +1),
    ("no_negative_chirality", "remove chi=-1 nodes", -1),
)
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
ORDER2_OMEGA_CHUNK = 4


def _resolve_model_path(source_file: str) -> Path:
    source_path = Path(source_file).expanduser()
    if source_path.is_absolute():
        return source_path
    candidates = (
        PROJECT_ROOT / source_path,
        PROJECT_ROOT / "models" / source_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No se encontro el archivo del modelo: {source_file}")


def _load_model_module(config: QXTIConfig):
    module_path = _resolve_model_path(config.hamiltonian.source_file)
    module_name = f"qxti_masked_sigma_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el modelo desde {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module_path


def _resolve_omega_axis(config: QXTIConfig) -> np.ndarray:
    xtp_cfg = config.xtp
    if xtp_cfg.susceptibility_omega_values:
        omega_axis = np.asarray(xtp_cfg.susceptibility_omega_values, dtype=np.float64)
    else:
        nfreq = int(xtp_cfg.susceptibility_num_frequencies)
        if nfreq <= 0:
            raise ValueError("xtp.susceptibility_num_frequencies debe ser positivo.")
        if nfreq == 1:
            omega_axis = np.asarray([xtp_cfg.susceptibility_omega_min], dtype=np.float64)
        else:
            omega_axis = np.linspace(
                float(xtp_cfg.susceptibility_omega_min),
                float(xtp_cfg.susceptibility_omega_max),
                nfreq,
                dtype=np.float64,
            )
    if omega_axis.ndim != 1 or omega_axis.size == 0:
        raise ValueError("El barrido de frecuencia debe contener al menos un punto.")
    if np.any(omega_axis <= 0.0):
        raise ValueError("Las frecuencias del barrido deben ser estrictamente positivas.")
    return np.asarray(omega_axis, dtype=np.float64)


def _min_nonzero_pair_distance(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    distances[distances <= 1.0e-15] = np.inf
    minimum = float(np.min(distances))
    if not np.isfinite(minimum):
        return 0.0
    return minimum


def _resolve_mask_radius(node_positions: np.ndarray, args: argparse.Namespace) -> float:
    if args.mask_radius is not None:
        radius = float(args.mask_radius)
    else:
        min_distance = _min_nonzero_pair_distance(node_positions)
        if min_distance <= 0.0:
            raise ValueError("No se pudo inferir una distancia minima entre nodos para fijar la mascara.")
        radius = float(args.mask_radius_factor) * min_distance
    if radius <= 0.0:
        raise ValueError("La mascara debe tener radio estrictamente positivo.")
    return radius


def _build_local_node_mask(
    k_points: np.ndarray,
    *,
    tagged_nodes: list[dict[str, Any]],
    chirality_to_remove: int | None,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    mask = np.ones(k_points.shape[0], dtype=np.float64)
    if chirality_to_remove is None:
        return mask, np.empty((0, k_points.shape[1]), dtype=np.float64), float(radius)

    centers = np.asarray(
        [np.asarray(node["k"], dtype=np.float64) for node in tagged_nodes if int(node["chirality"]) == chirality_to_remove],
        dtype=np.float64,
    )
    if centers.size == 0:
        return mask, centers.reshape(0, k_points.shape[1]), float(radius)

    distances = np.linalg.norm(k_points[:, None, :] - centers[None, :, :], axis=2)
    effective_radius = float(radius)
    excluded = np.any(distances <= effective_radius, axis=1)
    if not np.any(excluded):
        nearest_distance = float(np.min(distances))
        effective_radius = max(effective_radius, nearest_distance * 1.001)
        excluded = np.any(distances <= effective_radius, axis=1)
    mask[excluded] = 0.0
    return mask, centers, effective_radius


def _save_case_outputs(
    *,
    case_dir: Path,
    omega_axis_au: np.ndarray,
    sigma_cart: np.ndarray,
    sigma_hel: np.ndarray,
    hel_labels: tuple[str, ...],
    cart_labels: tuple[str, ...],
    dpi: int,
) -> None:
    omega_axis_ev = np.asarray(omega_axis_au, dtype=np.float64) * AU_TO_EV
    helicity_dir = case_dir / "order_1" / "helicity"
    plotter = SusceptibilityTensorPlotter(
        x_axis=omega_axis_ev,
        tensor=sigma_hel,
        output_dir=helicity_dir,
        x_label=r"$\omega_\mathrm{laser}\;(\mathrm{eV})$",
        argument_label=r"\omega_\mathrm{laser}",
        tensor_name="sigma1",
        direction_labels=hel_labels,
        dpi=int(dpi),
        include_ev_axis=False,
    )
    plotter.plot_overview()
    plotter.plot_grid()
    plotter.plot_all_components(output_file_template="sigma1_{label}.png")

    dataset = {
        "omega_axis": np.asarray(omega_axis_au, dtype=np.float64),
        "omega_axis_ev": omega_axis_ev,
        "sigma_order_1_tensor": np.asarray(sigma_cart, dtype=np.complex128),
        "sigma_order_1_helicity_tensor": np.asarray(sigma_hel, dtype=np.complex128),
        "direction_labels": list(cart_labels),
        "helicity_labels": list(hel_labels),
    }
    save_dataset_npz(case_dir / "data" / "sigma_node_mask_case.npz", dataset, compressed=True)


def _component_label(labels: tuple[str, ...], i: int, j: int) -> str:
    return f"{labels[i]}{labels[j]}"


def _component_label_multi(labels: tuple[str, ...], indices: tuple[int, ...]) -> str:
    return "".join(labels[index] for index in indices)


def _helicity_unitary(dimension: int) -> tuple[np.ndarray, tuple[str, ...]]:
    if dimension <= 1:
        return np.eye(1, dtype=np.complex128), ("x",)

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    if dimension == 2:
        unitary = np.array(
            [[inv_sqrt2, inv_sqrt2], [1j * inv_sqrt2, -1j * inv_sqrt2]],
            dtype=np.complex128,
        )
        labels: tuple[str, ...] = ("+", "-")
    else:
        unitary = np.array(
            [
                [inv_sqrt2, inv_sqrt2, 0.0],
                [1j * inv_sqrt2, -1j * inv_sqrt2, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.complex128,
        )
        labels = ("+", "-", "z")
    return unitary, labels


def _rotate_rank3_to_helicity(
    tensor: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(np.nan_to_num(tensor, nan=0.0), dtype=np.complex128)
    if values.ndim != 4:
        raise ValueError("rank-3 tensor expected with shape (N, dim, dim, dim).")
    unitary, labels = _helicity_unitary(dimension)
    transformed = np.einsum(
        "ai,wijk,jb,kc->wabc",
        unitary.conj().T,
        values,
        unitary,
        unitary,
        optimize=True,
    )
    return np.asarray(transformed, dtype=np.complex128), labels


def _select_z_containing_rank3_components(
    tensor: np.ndarray,
    labels: tuple[str, ...],
    *,
    relative_threshold: float = 1.0e-12,
) -> list[tuple[int, int, int]]:
    z_index = labels.index("z")
    scale = float(np.max(np.abs(tensor))) if tensor.size else 0.0
    threshold = max(scale * relative_threshold, 1.0e-18)
    selected: list[tuple[int, int, int]] = []
    for indices in np.ndindex((len(labels),) * 3):
        if z_index not in indices:
            continue
        values = np.asarray(tensor[(slice(None),) + indices], dtype=np.complex128)
        if float(np.max(np.abs(values))) > threshold:
            selected.append(tuple(int(index) for index in indices))
    return selected


def _compute_order2_sigma_masked(
    *,
    config: QXTIConfig,
    hamiltonian,
    kgrid,
    omega_axis: np.ndarray,
    weights: np.ndarray,
    progress: bool = True,
    omega_chunk_size: int = ORDER2_OMEGA_CHUNK,
) -> np.ndarray:
    """Compute sparse cartesian sigma^(2)_{i j j}(omega) with bounded memory."""
    dim = int(hamiltonian.dimension)
    nb = int(hamiltonian.basis_size)
    shape = tuple(int(kgrid.shape[axis]) for axis in range(3))
    k_points = np.asarray(kgrid.points(), dtype=np.float64)
    nk = int(k_points.shape[0])
    omega_axis = np.asarray(omega_axis, dtype=np.float64)
    nw = int(omega_axis.size)
    if nw == 0:
        raise ValueError("omega_axis cannot be empty for order-2 conductivity.")

    Hf = hamiltonian._matrix_at
    ccfg = config.susceptibility_solver
    bounds = hamiltonian.reciprocal_box_bounds()
    dks = [(float(bounds[axis][1]) - float(bounds[axis][0])) / max(shape[axis], 1) for axis in range(dim)]
    distribution = _resolve_distribution(ccfg.distribution)
    mu = float(ccfg.fermi_level)
    temperature = float(ccfg.temperature)
    gamma = 0.0 if ccfg.coherence_time <= 0 else 1.0 / float(ccfg.coherence_time)
    dk = 1.0e-4
    omega_chunk_size = max(1, int(omega_chunk_size))

    def _Hbatch(kc: np.ndarray) -> np.ndarray:
        return np.array([Hf(float(k[0]), float(k[1]), float(k[2])) for k in kc], dtype=np.complex128)

    t0 = time.perf_counter()
    if progress:
        print(f"[node-mask] order 2: diagonalizing {nk} k-points once for chunked SHG.", flush=True)

    H0 = _Hbatch(k_points)
    energies, U = np.linalg.eigh(H0)
    Udag = np.conj(np.transpose(U, (0, 2, 1)))

    vel = []
    for axis in range(dim):
        shift = np.zeros(3, dtype=np.float64)
        shift[axis] = dk
        dH = (_Hbatch(k_points + shift) - _Hbatch(k_points - shift)) / (2.0 * dk)
        vel.append(Udag @ dH @ U)

    f = np.asarray(distribution(energies, mu, temperature), dtype=np.float64)
    eps = energies[:, :, None] - energies[:, None, :]
    fmn = f[:, None, :] - f[:, :, None]
    offdiag = ~np.eye(nb, dtype=bool)
    valid = offdiag[None] & (np.abs(eps) > 1.0e-20)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_eps = np.where(valid, 1.0 / eps, 0.0)
        dfde = (-f * (1.0 - f) / temperature) if temperature > 1.0e-15 else np.zeros_like(f)
    A = [1j * vel[axis] * inv_eps for axis in range(dim)]

    U_mesh = U.reshape(*shape, nb, nb)
    Udag_mesh = Udag.reshape(*shape, nb, nb)
    valid_with_diag = valid[:, None] | np.eye(nb, dtype=bool)[None, None]
    sig2 = np.full((nw, dim, dim, dim), np.nan + 1j * np.nan, dtype=np.complex128)

    nchunks_total = dim * int(np.ceil(nw / omega_chunk_size))
    chunk_counter = 0

    for j in range(dim):
        if progress:
            print(f"[node-mask] order 2: input dir {('x','y','z')[j]} ({j+1}/{dim}).", flush=True)

        Up = np.roll(U_mesh, -1, axis=j)
        Um = np.roll(U_mesh, +1, axis=j)
        Wp = Udag_mesh @ Up
        Wm = Udag_mesh @ Um
        Wp_d = np.conj(np.swapaxes(Wp, -1, -2))
        Wm_d = np.conj(np.swapaxes(Wm, -1, -2))
        Aj = A[j].reshape(*shape, nb, nb)[..., None, :, :]
        diag_src = (-1j) * dfde * np.real(np.diagonal(vel[j], axis1=1, axis2=2))
        Ji_blocks = [-vel[i] for i in range(dim)]

        for start in range(0, nw, omega_chunk_size):
            stop = min(start + omega_chunk_size, nw)
            omega_chunk = np.asarray(omega_axis[start:stop], dtype=np.float64)
            nwc = int(omega_chunk.size)
            chunk_counter += 1
            if progress:
                elapsed = time.perf_counter() - t0
                print(
                    f"[node-mask] order 2: chunk {chunk_counter}/{nchunks_total} "
                    f"(dir={('x','y','z')[j]}, freq {start+1}-{stop}/{nw}), elapsed {elapsed:.1f}s.",
                    flush=True,
                )

            ow1 = omega_chunk[:, None, None] + 1j * gamma
            ow2 = 2.0 * omega_chunk[:, None, None] + 1j * gamma
            inv_d2 = np.where(valid[:, None], 1.0 / (ow2[None] - eps[:, None, :, :]), 0.0)

            rho1 = np.zeros((nk, nwc, nb, nb), dtype=np.complex128)
            rho1 += (A[j][:, None, :, :] * fmn[:, None, :, :]) / (ow1[None] - eps[:, None, :, :])
            if np.any(diag_src):
                diagonal_indices = np.arange(nb)
                rho1[:, :, diagonal_indices, diagonal_indices] += diag_src[:, None, :] / ow1[None, :, 0, :]
            rho1 = np.where(valid_with_diag, rho1, 0.0)

            rho1_mesh = rho1.reshape(*shape, nwc, nb, nb)
            rp = np.roll(rho1_mesh, -1, axis=j)
            rm = np.roll(rho1_mesh, +1, axis=j)
            tp = Wp[..., None, :, :] @ rp @ Wp_d[..., None, :, :]
            tm = Wm[..., None, :, :] @ rm @ Wm_d[..., None, :, :]
            dpart = (tp - tm) / (2.0 * dks[j])
            comm = Aj @ rho1_mesh - rho1_mesh @ Aj
            Dj_rho1 = (dpart - 1j * comm).reshape(nk, nwc, nb, nb)
            rho2 = Dj_rho1 * inv_d2

            for i in range(dim):
                tr = np.einsum("kmn,kwnm->wk", Ji_blocks[i], rho2, optimize=True)
                sig2[start:stop, i, j, j] = np.conj(tr @ weights)

    if progress:
        print(f"[node-mask] order 2 done in {format_duration(time.perf_counter() - t0)}.", flush=True)
    return sig2


def _save_case_outputs_order2(
    *,
    case_dir: Path,
    omega_axis_au: np.ndarray,
    sigma2_cart: np.ndarray,
    sigma2_hel: np.ndarray,
    hel_labels: tuple[str, ...],
    cart_labels: tuple[str, ...],
    selected_components: list[tuple[int, int, int]],
    dpi: int,
) -> None:
    omega_axis_ev = np.asarray(omega_axis_au, dtype=np.float64) * AU_TO_EV
    helicity_dir = case_dir / "order_2" / "helicity_with_z"
    plotter = SusceptibilityTensorPlotter(
        x_axis=omega_axis_ev,
        tensor=sigma2_hel,
        output_dir=helicity_dir,
        x_label=r"$\omega_\mathrm{laser}\;(\mathrm{eV})$",
        argument_label=r"\omega_\mathrm{laser}",
        tensor_name="sigma2",
        direction_labels=hel_labels,
        available_components=selected_components,
        dpi=int(dpi),
        include_ev_axis=False,
    )
    if selected_components:
        plotter.plot_overview()
        plotter.plot_grid()
        plotter.plot_all_components(output_file_template="sigma2_{label}.png")

    dataset = {
        "omega_axis": np.asarray(omega_axis_au, dtype=np.float64),
        "omega_axis_ev": omega_axis_ev,
        "sigma_order_2_tensor": np.asarray(sigma2_cart, dtype=np.complex128),
        "sigma_order_2_helicity_tensor": np.asarray(sigma2_hel, dtype=np.complex128),
        "direction_labels": list(cart_labels),
        "helicity_labels": list(hel_labels),
        "sigma_order_2_helicity_selected_components": np.asarray(selected_components, dtype=np.int16),
    }
    save_dataset_npz(case_dir / "data" / "sigma2_node_mask_case.npz", dataset, compressed=True)


def _plot_case_comparison_grid(
    *,
    output_dir: Path,
    omega_axis_ev: np.ndarray,
    case_tensors: dict[str, np.ndarray],
    hel_labels: tuple[str, ...],
) -> Path:
    ncomp = len(hel_labels) * len(hel_labels)
    ncols = min(len(hel_labels), 3)
    nrows = int(np.ceil(ncomp / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        squeeze=False,
        sharex=True,
    )
    panel_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for ipanel, (i, j) in enumerate(np.ndindex((len(hel_labels), len(hel_labels)))):
        ax = axes.flat[ipanel]
        comp = _component_label(hel_labels, i, j)
        for case_name, tensor in case_tensors.items():
            values = np.asarray(tensor[:, i, j], dtype=np.complex128)
            ax.plot(
                omega_axis_ev,
                np.abs(values),
                color=CASE_COLORS[case_name],
                linewidth=2.0,
                label=CASE_LABELS[case_name],
            )
        ax.set_title(rf"({panel_labels[ipanel].lower()}) $\sigma_{{{comp}}}$", loc="left", fontsize=11)
        ax.set_xlim(float(omega_axis_ev.min()), float(omega_axis_ev.max()))
        ax.set_ylim(bottom=0.0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    for ax in axes.flat[ncomp:]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\omega_\mathrm{laser}\;(\mathrm{eV})$")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(r"Helicity conductivity comparison: $|\sigma^{(1)}|$", fontsize=14)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "sigma1_case_comparison_grid.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output


def _plot_case_comparison_components(
    *,
    output_dir: Path,
    omega_axis_ev: np.ndarray,
    case_tensors: dict[str, np.ndarray],
    hel_labels: tuple[str, ...],
) -> list[Path]:
    outputs: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, j in np.ndindex((len(hel_labels), len(hel_labels))):
        comp = _component_label(hel_labels, i, j)
        fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.8), sharex=True, squeeze=False)
        ax_re = axes[0, 0]
        ax_im = axes[1, 0]
        ax_abs = axes[2, 0]

        for case_name, tensor in case_tensors.items():
            values = np.asarray(tensor[:, i, j], dtype=np.complex128)
            style = dict(color=CASE_COLORS[case_name], linewidth=2.0, label=CASE_LABELS[case_name])
            ax_re.plot(omega_axis_ev, np.real(values), **style)
            ax_im.plot(omega_axis_ev, np.imag(values), **style)
            ax_abs.plot(omega_axis_ev, np.abs(values), **style)

        ax_re.set_title(rf"$\sigma_{{{comp}}}$ comparison", loc="left", fontsize=12)
        ax_re.set_ylabel(r"$\Re\,\sigma$")
        ax_im.set_ylabel(r"$\Im\,\sigma$")
        ax_abs.set_ylabel(r"$|\sigma|$")
        ax_abs.set_xlabel(r"$\omega_\mathrm{laser}\;(\mathrm{eV})$")

        for ax in (ax_re, ax_im, ax_abs):
            ax.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
            ax.set_xlim(float(omega_axis_ev.min()), float(omega_axis_ev.max()))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax_abs.set_ylim(bottom=0.0)
        ax_re.legend(frameon=False, fontsize=9, loc="best")

        fig.tight_layout()
        output = output_dir / f"sigma1_{comp}_comparison.png"
        fig.savefig(output, dpi=300, facecolor="white")
        plt.close(fig)
        outputs.append(output)
    return outputs


def _summarize_case_differences(
    *,
    case_tensors: dict[str, np.ndarray],
    hel_labels: tuple[str, ...],
) -> list[str]:
    lines: list[str] = []
    full = np.asarray(case_tensors["full"], dtype=np.complex128)
    full_max = float(np.max(np.abs(full)))
    for case_name, tensor in case_tensors.items():
        if case_name == "full":
            continue
        diff = np.asarray(tensor, dtype=np.complex128) - full
        max_abs_diff = float(np.max(np.abs(diff)))
        rel = max_abs_diff / max(full_max, 1.0e-30)

        best_component = ""
        best_rel = -1.0
        for i, j in np.ndindex((len(hel_labels), len(hel_labels))):
            ref = np.asarray(full[:, i, j], dtype=np.complex128)
            delta = np.asarray(diff[:, i, j], dtype=np.complex128)
            ref_scale = float(np.max(np.abs(ref)))
            if ref_scale <= 1.0e-12:
                continue
            rel_comp = float(np.max(np.abs(delta)) / ref_scale)
            if rel_comp > best_rel:
                best_rel = rel_comp
                best_component = _component_label(hel_labels, i, j)

        lines.append(
            f"[node-mask] comparacion '{case_name}' vs full: "
            f"max|delta sigma|={max_abs_diff:.6g}, rel_global={100.0*rel:.3f}%, "
            f"componente mas sensible={best_component or 'n/a'} ({100.0*max(best_rel, 0.0):.3f}%)."
        )
    return lines


def _plot_case_comparison_grid_order2(
    *,
    output_dir: Path,
    omega_axis_ev: np.ndarray,
    case_tensors: dict[str, np.ndarray],
    hel_labels: tuple[str, ...],
    selected_components: list[tuple[int, int, int]],
) -> Path | None:
    if not selected_components:
        return None

    ncomp = len(selected_components)
    ncols = min(max(int(np.ceil(np.sqrt(ncomp))), 1), 3)
    nrows = int(np.ceil(ncomp / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        squeeze=False,
        sharex=True,
    )
    panel_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for ipanel, indices in enumerate(selected_components):
        ax = axes.flat[ipanel]
        comp = _component_label_multi(hel_labels, indices)
        for case_name, tensor in case_tensors.items():
            values = np.asarray(tensor[(slice(None),) + indices], dtype=np.complex128)
            ax.plot(
                omega_axis_ev,
                np.abs(values),
                color=CASE_COLORS[case_name],
                linewidth=2.0,
                label=CASE_LABELS[case_name],
            )
        ax.set_title(rf"({panel_labels[ipanel].lower()}) $\sigma^{{(2)}}_{{{comp}}}$", loc="left", fontsize=11)
        ax.set_xlim(float(omega_axis_ev.min()), float(omega_axis_ev.max()))
        ax.set_ylim(bottom=0.0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    for ax in axes.flat[ncomp:]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\omega_\mathrm{laser}\;(\mathrm{eV})$")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(r"Helicity SHG comparison: $|\sigma^{(2)}|$ with at least one $z$", fontsize=14)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "sigma2_case_comparison_grid.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output


def _plot_case_comparison_components_order2(
    *,
    output_dir: Path,
    omega_axis_ev: np.ndarray,
    case_tensors: dict[str, np.ndarray],
    hel_labels: tuple[str, ...],
    selected_components: list[tuple[int, int, int]],
) -> list[Path]:
    outputs: list[Path] = []
    if not selected_components:
        return outputs

    output_dir.mkdir(parents=True, exist_ok=True)
    for indices in selected_components:
        comp = _component_label_multi(hel_labels, indices)
        fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.8), sharex=True, squeeze=False)
        ax_re = axes[0, 0]
        ax_im = axes[1, 0]
        ax_abs = axes[2, 0]

        for case_name, tensor in case_tensors.items():
            values = np.asarray(tensor[(slice(None),) + indices], dtype=np.complex128)
            style = dict(color=CASE_COLORS[case_name], linewidth=2.0, label=CASE_LABELS[case_name])
            ax_re.plot(omega_axis_ev, np.real(values), **style)
            ax_im.plot(omega_axis_ev, np.imag(values), **style)
            ax_abs.plot(omega_axis_ev, np.abs(values), **style)

        ax_re.set_title(rf"$\sigma^{{(2)}}_{{{comp}}}$ comparison", loc="left", fontsize=12)
        ax_re.set_ylabel(r"$\Re\,\sigma^{(2)}$")
        ax_im.set_ylabel(r"$\Im\,\sigma^{(2)}$")
        ax_abs.set_ylabel(r"$|\sigma^{(2)}|$")
        ax_abs.set_xlabel(r"$\omega_\mathrm{laser}\;(\mathrm{eV})$")

        for ax in (ax_re, ax_im, ax_abs):
            ax.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
            ax.set_xlim(float(omega_axis_ev.min()), float(omega_axis_ev.max()))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax_abs.set_ylim(bottom=0.0)
        ax_re.legend(frameon=False, fontsize=9, loc="best")

        fig.tight_layout()
        output = output_dir / f"sigma2_{comp}_comparison.png"
        fig.savefig(output, dpi=300, facecolor="white")
        plt.close(fig)
        outputs.append(output)
    return outputs


def _summarize_case_differences_order2(
    *,
    case_tensors: dict[str, np.ndarray],
    hel_labels: tuple[str, ...],
    selected_components: list[tuple[int, int, int]],
) -> list[str]:
    if not selected_components:
        return ["[node-mask] order 2: no helicity-z components survived the rotation/filter."]

    lines: list[str] = []
    full = np.asarray(case_tensors["full"], dtype=np.complex128)
    full_scale = 0.0
    for indices in selected_components:
        full_scale = max(full_scale, float(np.max(np.abs(full[(slice(None),) + indices]))))

    for case_name, tensor in case_tensors.items():
        if case_name == "full":
            continue
        max_abs_diff = 0.0
        best_component = ""
        best_rel = -1.0
        for indices in selected_components:
            ref = np.asarray(full[(slice(None),) + indices], dtype=np.complex128)
            values = np.asarray(tensor[(slice(None),) + indices], dtype=np.complex128)
            delta = values - ref
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(delta))))
            ref_scale = float(np.max(np.abs(ref)))
            if ref_scale <= 1.0e-18:
                continue
            rel_component = float(np.max(np.abs(delta)) / ref_scale)
            if rel_component > best_rel:
                best_rel = rel_component
                best_component = _component_label_multi(hel_labels, indices)

        rel_global = max_abs_diff / max(full_scale, 1.0e-30)
        lines.append(
            f"[node-mask] order 2 '{case_name}' vs full: "
            f"max|delta sigma2|={max_abs_diff:.6g}, rel_global={100.0 * rel_global:.3f}%, "
            f"componente mas sensible={best_component or 'n/a'} ({100.0 * max(best_rel, 0.0):.3f}%)."
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcula y grafica sigma^(1) y sigma^(2) del WSM en base de helicidad para tres casos: "
            "completo, sin nodos de quiralidad positiva y sin nodos de quiralidad negativa."
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Input cfg de susceptibilidad del WSM (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directorio de salida base para datos y plots.",
    )
    parser.add_argument(
        "--mask-radius",
        type=float,
        default=None,
        help="Radio absoluto de la esfera excluida alrededor de cada nodo (en unidades reciprocas del modelo).",
    )
    parser.add_argument(
        "--mask-radius-factor",
        type=float,
        default=0.35,
        help="Si no se pasa --mask-radius, usa este factor por la distancia minima entre nodos.",
    )
    parser.add_argument(
        "--max-frequencies",
        type=int,
        default=None,
        help="Limita el numero de frecuencias para una prueba rapida.",
    )
    parser.add_argument(
        "--order2-omega-chunk",
        type=int,
        default=ORDER2_OMEGA_CHUNK,
        help="Numero de frecuencias procesadas simultaneamente en sigma^(2) para acotar RAM.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    apply_paper_style()

    config = QXTIConfig.from_file(args.config)
    module, module_path = _load_model_module(config)
    if not hasattr(module, "weyl_nodes_with_chirality"):
        raise AttributeError(
            f"El modelo {module_path.name} no expone weyl_nodes_with_chirality(params)."
        )

    tagged_nodes = list(module.weyl_nodes_with_chirality(dict(config.hamiltonian.params)))
    if not tagged_nodes:
        raise ValueError("El modelo no devolvio ningun nodo de Weyl.")

    simulation = QXTISimulation(config=config)
    hamiltonian = simulation.build_hamiltonian()
    kgrid = simulation.build_kgrid(hamiltonian)
    k_points = np.asarray(kgrid.points(), dtype=np.float64)
    dim = int(hamiltonian.dimension)
    if dim < 2:
        raise ValueError("La base de helicidad solo tiene sentido para modelos 2D o 3D.")

    omega_axis = _resolve_omega_axis(config)
    if args.max_frequencies is not None:
        max_freq = max(1, int(args.max_frequencies))
        omega_axis = np.asarray(omega_axis[:max_freq], dtype=np.float64)

    node_positions = np.asarray([np.asarray(node["k"], dtype=np.float64) for node in tagged_nodes], dtype=np.float64)
    mask_radius = _resolve_mask_radius(node_positions, args)
    cart_labels = tuple(("x", "y", "z")[:dim])
    requested_orders = {int(order) for order in getattr(config.xtp, "susceptibility_orders", (1,))}
    compute_order2 = 2 in requested_orders

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[node-mask] modelo: {module_path.name} | dimension={dim} | Nk={kgrid.total_points} | "
        f"Nomega={omega_axis.size} | radio_mascara={mask_radius:.6f}",
        flush=True,
    )
    for index, node in enumerate(tagged_nodes, start=1):
        position = np.asarray(node["k"], dtype=np.float64)
        coords = ", ".join(f"{value:.6f}" for value in position[:dim])
        print(f"[node-mask] nodo {index}: chi={int(node['chirality']):+d}, k=({coords})", flush=True)

    base_weights = build_k_integration_weights(config, hamiltonian=hamiltonian, kgrid=kgrid)
    total_weight = float(np.sum(base_weights)) if base_weights.size else 1.0

    overall_start = time.perf_counter()
    case_tensors_hel: dict[str, np.ndarray] = {}
    case_tensors_hel_order2: dict[str, np.ndarray] = {}
    hel_labels_ref: tuple[str, ...] | None = None
    hel_labels_ref_order2: tuple[str, ...] | None = None
    order2_selected_components: list[tuple[int, int, int]] | None = None
    for case_name, case_label, chirality_to_remove in CASE_SPECS:
        local_mask, removed_centers, effective_radius = _build_local_node_mask(
            k_points,
            tagged_nodes=tagged_nodes,
            chirality_to_remove=chirality_to_remove,
            radius=mask_radius,
        )
        active_weight = build_k_integration_weights(
            config,
            hamiltonian=hamiltonian,
            kgrid=kgrid,
            extra_k_weight_mask=local_mask,
        )
        removed_points = int(np.count_nonzero(local_mask == 0.0))
        removed_fraction_points = removed_points / max(int(kgrid.total_points), 1)
        removed_fraction_weight = 1.0 - float(np.sum(active_weight) / total_weight) if total_weight > 0 else 0.0

        print(
            f"[node-mask] caso '{case_name}': {case_label} | centros={len(removed_centers)} | "
            f"radio_efectivo={effective_radius:.6f} | "
            f"puntos_excluidos={removed_points}/{kgrid.total_points} ({100.0*removed_fraction_points:.2f}%) | "
            f"peso_excluido={100.0*removed_fraction_weight:.2f}%",
            flush=True,
        )

        case_start = time.perf_counter()
        result = compute_linear_response_spectrum(
            config,
            omega_axis,
            progress=True,
            extra_k_weight_mask=local_mask,
        )
        sigma_cart = np.asarray(result["sigma"], dtype=np.complex128)
        sigma_hel, hel_labels = to_helicity_basis(sigma_cart, dim)
        case_tensors_hel[case_name] = np.asarray(sigma_hel, dtype=np.complex128)
        if hel_labels_ref is None:
            hel_labels_ref = tuple(hel_labels)

        case_dir = output_dir / case_name
        _save_case_outputs(
            case_dir=case_dir,
            omega_axis_au=omega_axis,
            sigma_cart=sigma_cart,
            sigma_hel=sigma_hel,
            hel_labels=hel_labels,
            cart_labels=cart_labels,
            dpi=int(config.xtp.susceptibility_plot_dpi),
        )

        metadata = {
            "case_name": case_name,
            "case_label": case_label,
            "model_file": str(module_path),
            "config_file": str(Path(args.config).resolve()),
            "mask_radius_requested": float(mask_radius),
            "mask_radius_effective": float(effective_radius),
            "removed_chirality": chirality_to_remove,
            "removed_points": removed_points,
            "removed_fraction_points": removed_fraction_points,
            "removed_fraction_weight": removed_fraction_weight,
            "runtime_seconds": float(time.perf_counter() - case_start),
            "node_chiralities": [int(node["chirality"]) for node in tagged_nodes],
            "node_positions": node_positions.tolist(),
            "removed_node_centers": np.asarray(removed_centers, dtype=np.float64).tolist(),
        }
        save_dataset_npz(
            case_dir / "data" / "mask_metadata.npz",
            {
                "local_mask": np.asarray(local_mask.reshape(kgrid.shape), dtype=np.float64),
                "base_weights": np.asarray(base_weights.reshape(kgrid.shape), dtype=np.float64),
                "active_weights": np.asarray(active_weight.reshape(kgrid.shape), dtype=np.float64),
                **metadata,
            },
            compressed=True,
        )

        if compute_order2:
            sigma2_cart = _compute_order2_sigma_masked(
                config=config,
                hamiltonian=hamiltonian,
                kgrid=kgrid,
                omega_axis=omega_axis,
                weights=active_weight,
                progress=True,
                omega_chunk_size=int(args.order2_omega_chunk),
            )
            sigma2_hel, hel_labels_order2 = _rotate_rank3_to_helicity(sigma2_cart, dim)
            case_tensors_hel_order2[case_name] = np.asarray(sigma2_hel, dtype=np.complex128)
            if hel_labels_ref_order2 is None:
                hel_labels_ref_order2 = tuple(hel_labels_order2)
            if order2_selected_components is None:
                order2_selected_components = _select_z_containing_rank3_components(
                    sigma2_hel,
                    hel_labels_order2,
                )
                selected_names = [
                    _component_label_multi(hel_labels_order2, indices)
                    for indices in order2_selected_components
                ]
                print(
                    "[node-mask] order 2 helicity components with z kept for plots: "
                    + (", ".join(selected_names) if selected_names else "none"),
                    flush=True,
                )
            _save_case_outputs_order2(
                case_dir=case_dir,
                omega_axis_au=omega_axis,
                sigma2_cart=sigma2_cart,
                sigma2_hel=sigma2_hel,
                hel_labels=hel_labels_order2,
                cart_labels=cart_labels,
                selected_components=order2_selected_components or [],
                dpi=int(config.xtp.susceptibility_plot_dpi),
            )
        print(
            f"[node-mask] caso '{case_name}' listo en {format_duration(time.perf_counter() - case_start)} -> {case_dir}",
            flush=True,
        )

    if hel_labels_ref is not None:
        comparison_dir = output_dir / "comparison" / "helicity"
        omega_axis_ev = np.asarray(omega_axis, dtype=np.float64) * AU_TO_EV
        comparison_grid = _plot_case_comparison_grid(
            output_dir=comparison_dir,
            omega_axis_ev=omega_axis_ev,
            case_tensors=case_tensors_hel,
            hel_labels=hel_labels_ref,
        )
        _plot_case_comparison_components(
            output_dir=comparison_dir / "components",
            omega_axis_ev=omega_axis_ev,
            case_tensors=case_tensors_hel,
            hel_labels=hel_labels_ref,
        )
        for line in _summarize_case_differences(case_tensors=case_tensors_hel, hel_labels=hel_labels_ref):
            print(line, flush=True)
        print(f"[node-mask] plot comparativo guardado en {comparison_grid}", flush=True)

    if compute_order2 and hel_labels_ref_order2 is not None and order2_selected_components is not None:
        comparison_dir_order2 = output_dir / "comparison" / "helicity_with_z" / "order_2"
        omega_axis_ev = np.asarray(omega_axis, dtype=np.float64) * AU_TO_EV
        comparison_grid_order2 = _plot_case_comparison_grid_order2(
            output_dir=comparison_dir_order2,
            omega_axis_ev=omega_axis_ev,
            case_tensors=case_tensors_hel_order2,
            hel_labels=hel_labels_ref_order2,
            selected_components=order2_selected_components,
        )
        _plot_case_comparison_components_order2(
            output_dir=comparison_dir_order2 / "components",
            omega_axis_ev=omega_axis_ev,
            case_tensors=case_tensors_hel_order2,
            hel_labels=hel_labels_ref_order2,
            selected_components=order2_selected_components,
        )
        for line in _summarize_case_differences_order2(
            case_tensors=case_tensors_hel_order2,
            hel_labels=hel_labels_ref_order2,
            selected_components=order2_selected_components,
        ):
            print(line, flush=True)
        if comparison_grid_order2 is not None:
            print(f"[node-mask] plot comparativo de order 2 guardado en {comparison_grid_order2}", flush=True)

    print(
        f"[node-mask] terminado: 3 casos en {format_duration(time.perf_counter() - overall_start)}. "
        f"Salida: {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
