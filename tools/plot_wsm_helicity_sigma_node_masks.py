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
from dataclasses import replace
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
TRANSVERSE_COMPONENTS = ("xy", "xz", "yz")
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


def _delta_tag(delta: float) -> str:
    value = f"{float(delta):+.6f}"
    return value.replace("+", "p").replace("-", "m").replace(".", "p")


def _resolved_output_dir(raw_output_dir: str, *, delta_override: float | None) -> Path:
    base = Path(raw_output_dir)
    if delta_override is None:
        return base
    return base / f"delta_{_delta_tag(delta_override)}"


def _config_with_delta_override(config: QXTIConfig, *, delta_override: float | None) -> QXTIConfig:
    if delta_override is None:
        return config
    params = dict(config.hamiltonian.params)
    params["Delta"] = float(delta_override)
    return replace(
        config,
        hamiltonian=replace(config.hamiltonian, params=params),
    )


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


def _cartesian_component_indices(component: str, labels: tuple[str, ...]) -> tuple[int, int]:
    key = str(component).strip().lower()
    if len(key) != 2:
        raise ValueError(f"Componente cartesiana invalida: {component!r}.")
    try:
        return labels.index(key[0]), labels.index(key[1])
    except ValueError as exc:
        raise ValueError(
            f"Componente {component!r} no disponible para etiquetas {labels}."
        ) from exc


def _sigma_quantity(values: np.ndarray, quantity: str) -> np.ndarray:
    q = str(quantity).strip().lower()
    z = np.asarray(values, dtype=np.complex128)
    if q in {"abs", "mod", "modulus", "magnitude"}:
        return np.abs(z)
    if q in {"real", "re"}:
        return np.real(z)
    if q in {"imag", "im", "imaginary"}:
        return np.imag(z)
    raise ValueError(f"Cantidad no soportada para sigma: {quantity!r}.")


def _sigma_quantity_label(quantity: str) -> str:
    q = str(quantity).strip().lower()
    if q in {"abs", "mod", "modulus", "magnitude"}:
        return r"$|\sigma^{(1)}|$"
    if q in {"real", "re"}:
        return r"$\Re\,\sigma^{(1)}$"
    if q in {"imag", "im", "imaginary"}:
        return r"$\Im\,\sigma^{(1)}$"
    raise ValueError(f"Cantidad no soportada para sigma: {quantity!r}.")


def _sigma_component_ylabel(component: str, quantity: str) -> str:
    q = str(quantity).strip().lower()
    if q in {"abs", "mod", "modulus", "magnitude"}:
        return rf"$|\sigma^{{(1)}}_{{{component}}}|$"
    if q in {"real", "re"}:
        return rf"$\Re\,\sigma^{{(1)}}_{{{component}}}$"
    if q in {"imag", "im", "imaginary"}:
        return rf"$\Im\,\sigma^{{(1)}}_{{{component}}}$"
    raise ValueError(f"Cantidad no soportada para sigma: {quantity!r}.")


def _save_figure_png_svg(fig, output_base: Path, *, dpi: int = 340) -> list[Path]:
    import matplotlib as mpl

    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_base.with_suffix(".png"), output_base.with_suffix(".svg")]
    # Preserve the square canvas requested for paper figures. Tight bounding
    # boxes are pretty for drafts but change the final aspect ratio.
    with mpl.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(outputs[0], dpi=int(dpi), facecolor="white")
        fig.savefig(outputs[1], facecolor="white")
    return outputs


def _plot_cartesian_transverse_paper(
    *,
    output_dir: Path,
    omega_axis_ev: np.ndarray,
    case_tensors: dict[str, np.ndarray],
    cart_labels: tuple[str, ...],
    components: tuple[str, ...] = TRANSVERSE_COMPONENTS,
    quantity: str = "abs",
    dpi: int = 340,
) -> list[Path]:
    """Paper-style transverse sigma plots for node-mask comparisons.

    Produces one square combined figure plus one square figure per component.
    The combined figure stacks the three transverse components vertically inside
    a square canvas; the individual figures keep each axis exactly square.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    omega_axis_ev = np.asarray(omega_axis_ev, dtype=np.float64)
    qtag = str(quantity).strip().lower()
    outputs: list[Path] = []

    # Same strong, readable palette used for the node-mask case identity.
    case_order = [case_name for case_name, _label, _chirality in CASE_SPECS if case_name in case_tensors]
    line_width = 3.8
    fill_alpha = {
        "full": 0.11,
        "no_positive_chirality": 0.18,
        "no_negative_chirality": 0.18,
    }

    fig, axes = plt.subplots(
        len(components),
        1,
        figsize=(10.8, 10.8),
        sharex=True,
        squeeze=False,
    )
    panel_letters = "abc"
    global_ymin = np.inf
    global_ymax = -np.inf
    panel_values: dict[tuple[str, str], np.ndarray] = {}
    for component in components:
        i, j = _cartesian_component_indices(component, cart_labels)
        for case_name in case_order:
            values = _sigma_quantity(case_tensors[case_name][:, i, j], quantity)
            panel_values[(component, case_name)] = values
            global_ymin = min(global_ymin, float(np.nanmin(values)))
            global_ymax = max(global_ymax, float(np.nanmax(values)))

    if str(quantity).strip().lower() in {"abs", "mod", "modulus", "magnitude"}:
        ylo, yhi = 0.0, global_ymax * 1.10 if global_ymax > 0.0 else 1.0
    else:
        span = max(abs(global_ymin), abs(global_ymax), 1.0e-30)
        ylo, yhi = -1.12 * span, 1.12 * span

    for idx, component in enumerate(components):
        ax = axes[idx, 0]
        for case_name in case_order:
            values = panel_values[(component, case_name)]
            color = CASE_COLORS[case_name]
            ax.fill_between(
                omega_axis_ev,
                0.0,
                values,
                color=color,
                alpha=fill_alpha.get(case_name, 0.16),
                lw=0,
                zorder=2,
            )
            ax.plot(
                omega_axis_ev,
                values,
                color=color,
                lw=line_width,
                label=CASE_LABELS[case_name],
                zorder=3,
            )
        ax.axhline(0.0, color="#B8C4D0", lw=1.0, zorder=1)
        ax.set_xlim(float(omega_axis_ev[0]), float(omega_axis_ev[-1]))
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel(_sigma_component_ylabel(component, quantity), fontsize=25, labelpad=13)
        ax.tick_params(axis="both", which="major", labelsize=23, length=8, width=1.35)
        ax.tick_params(axis="both", which="minor", length=4, width=1.0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.yaxis.get_offset_text().set_fontsize(20)
        ax.text(
            0.025,
            0.88,
            rf"$\mathbf{{{panel_letters[idx]}}}$",
            transform=ax.transAxes,
            fontsize=28,
            ha="left",
            va="top",
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.35)

    axes[-1, 0].set_xlabel(r"$\hbar\omega\;(\mathrm{eV})$", fontsize=27, labelpad=11)
    axes[0, 0].legend(
        frameon=False,
        fontsize=19,
        loc="upper right",
        handlelength=1.8,
        labelspacing=0.45,
    )
    fig.subplots_adjust(left=0.21, right=0.965, bottom=0.105, top=0.975, hspace=0.16)
    outputs.extend(
        _save_figure_png_svg(
            fig,
            output_dir / f"sigma1_transverse_{'_'.join(components)}_{qtag}_node_masks_paper",
            dpi=dpi,
        )
    )
    plt.close(fig)

    component_dir = output_dir / "components"
    for component in components:
        fig, ax = plt.subplots(figsize=(10.8, 10.8))
        i, j = _cartesian_component_indices(component, cart_labels)
        local_ymin = np.inf
        local_ymax = -np.inf
        local_values: dict[str, np.ndarray] = {}
        for case_name in case_order:
            values = _sigma_quantity(case_tensors[case_name][:, i, j], quantity)
            local_values[case_name] = values
            local_ymin = min(local_ymin, float(np.nanmin(values)))
            local_ymax = max(local_ymax, float(np.nanmax(values)))

        if str(quantity).strip().lower() in {"abs", "mod", "modulus", "magnitude"}:
            local_ylim = (0.0, local_ymax * 1.10 if local_ymax > 0.0 else 1.0)
        else:
            span = max(abs(local_ymin), abs(local_ymax), 1.0e-30)
            local_ylim = (-1.12 * span, 1.12 * span)

        for case_name in case_order:
            values = local_values[case_name]
            color = CASE_COLORS[case_name]
            ax.fill_between(
                omega_axis_ev,
                0.0,
                values,
                color=color,
                alpha=fill_alpha.get(case_name, 0.16),
                lw=0,
                zorder=2,
            )
            ax.plot(
                omega_axis_ev,
                values,
                color=color,
                lw=4.2,
                label=CASE_LABELS[case_name],
                zorder=3,
            )
        ax.axhline(0.0, color="#B8C4D0", lw=1.0, zorder=1)
        ax.set_xlim(float(omega_axis_ev[0]), float(omega_axis_ev[-1]))
        ax.set_ylim(*local_ylim)
        ax.set_box_aspect(1.0)
        ax.set_xlabel(r"$\hbar\omega\;(\mathrm{eV})$", fontsize=36, labelpad=13)
        ax.set_ylabel(_sigma_component_ylabel(component, quantity), fontsize=36, labelpad=15)
        ax.tick_params(axis="both", which="major", labelsize=32, length=10, width=1.45)
        ax.tick_params(axis="both", which="minor", length=5, width=1.1)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax.yaxis.get_offset_text().set_fontsize(29)
        ax.legend(
            frameon=False,
            fontsize=25,
            loc="upper right",
            handlelength=1.9,
            labelspacing=0.45,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.45)
        fig.subplots_adjust(left=0.22, right=0.96, bottom=0.16, top=0.96)
        outputs.extend(
            _save_figure_png_svg(
                fig,
                component_dir / f"sigma1_{component}_{qtag}_node_masks_paper",
                dpi=dpi,
            )
        )
        plt.close(fig)

    print(
        "[node-mask] transverse Cartesian paper plots saved: "
        + ", ".join(str(path) for path in outputs),
        flush=True,
    )
    return outputs


def _load_saved_cartesian_node_mask_cases(
    base_dir: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[str, ...]]:
    omega_axis_ev_ref: np.ndarray | None = None
    case_tensors: dict[str, np.ndarray] = {}
    for case_name, _label, _chirality in CASE_SPECS:
        dataset_path = base_dir / case_name / "data" / "sigma_node_mask_case.npz"
        if not dataset_path.exists():
            raise FileNotFoundError(f"No existe el dataset de mascara: {dataset_path}")
        data = np.load(dataset_path, allow_pickle=True)
        omega_axis_ev = np.asarray(data["omega_axis_ev"], dtype=np.float64)
        if omega_axis_ev_ref is None:
            omega_axis_ev_ref = omega_axis_ev
        elif omega_axis_ev.shape != omega_axis_ev_ref.shape or not np.allclose(omega_axis_ev, omega_axis_ev_ref):
            raise ValueError(f"El eje de frecuencia de {dataset_path} no coincide con los otros casos.")
        case_tensors[case_name] = np.asarray(data["sigma_order_1_tensor"], dtype=np.complex128)
    if omega_axis_ev_ref is None:
        raise ValueError(f"No se cargaron casos desde {base_dir}.")
    return omega_axis_ev_ref, case_tensors, ("x", "y", "z")


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
        "--delta",
        type=float,
        default=None,
        help=(
            "Sobrescribe el parametro hamiltonian.Delta solo para esta corrida. "
            "Los nodos de Weyl y las mascaras se recalculan con ese Delta."
        ),
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

    config = _config_with_delta_override(
        QXTIConfig.from_file(args.config),
        delta_override=args.delta,
    )
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

    output_dir = _resolved_output_dir(
        args.output_dir,
        delta_override=args.delta,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = []
    effective_delta = float(config.hamiltonian.params.get("Delta", 0.0))

    header = (
        f"[node-mask] modelo: {module_path.name} | dimension={dim} | Nk={kgrid.total_points} | "
        f"Nomega={omega_axis.size} | Delta={effective_delta:.6f} | radio_mascara={mask_radius:.6f}"
    )
    summary_lines.append(header)
    print(header, flush=True)
    for index, node in enumerate(tagged_nodes, start=1):
        position = np.asarray(node["k"], dtype=np.float64)
        coords = ", ".join(f"{value:.6f}" for value in position[:dim])
        line = f"[node-mask] nodo {index}: chi={int(node['chirality']):+d}, k=({coords})"
        summary_lines.append(line)
        print(line, flush=True)

    base_weights = build_k_integration_weights(config, hamiltonian=hamiltonian, kgrid=kgrid)
    total_weight = float(np.sum(base_weights)) if base_weights.size else 1.0

    overall_start = time.perf_counter()
    case_tensors_hel: dict[str, np.ndarray] = {}
    case_tensors_cart: dict[str, np.ndarray] = {}
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

        line = (
            f"[node-mask] caso '{case_name}': {case_label} | centros={len(removed_centers)} | "
            f"radio_efectivo={effective_radius:.6f} | "
            f"puntos_excluidos={removed_points}/{kgrid.total_points} ({100.0*removed_fraction_points:.2f}%) | "
            f"peso_excluido={100.0*removed_fraction_weight:.2f}%"
        )
        summary_lines.append(line)
        print(line, flush=True)

        case_start = time.perf_counter()
        result = compute_linear_response_spectrum(
            config,
            omega_axis,
            progress=True,
            extra_k_weight_mask=local_mask,
        )
        sigma_cart = np.asarray(result["sigma"], dtype=np.complex128)
        sigma_hel, hel_labels = to_helicity_basis(sigma_cart, dim)
        case_tensors_cart[case_name] = np.asarray(sigma_cart, dtype=np.complex128)
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
                summary_lines.append(
                    "[node-mask] order 2 helicity components with z kept for plots: "
                    + (", ".join(selected_names) if selected_names else "none")
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
        line = (
            f"[node-mask] caso '{case_name}' listo en "
            f"{format_duration(time.perf_counter() - case_start)} -> {case_dir}"
        )
        summary_lines.append(line)
        print(line, flush=True)

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
            summary_lines.append(line)
            print(line, flush=True)
        line = f"[node-mask] plot comparativo guardado en {comparison_grid}"
        summary_lines.append(line)
        print(line, flush=True)

        transverse_outputs = _plot_cartesian_transverse_paper(
            output_dir=output_dir / "comparison" / "cartesian_transverse",
            omega_axis_ev=omega_axis_ev,
            case_tensors=case_tensors_cart,
            cart_labels=cart_labels,
            components=TRANSVERSE_COMPONENTS,
            quantity="abs",
            dpi=int(config.xtp.susceptibility_plot_dpi),
        )
        line = (
            "[node-mask] plots transversales cartesianos guardados en "
            + ", ".join(str(path) for path in transverse_outputs)
        )
        summary_lines.append(line)
        print(line, flush=True)

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
            summary_lines.append(line)
            print(line, flush=True)
        if comparison_grid_order2 is not None:
            line = f"[node-mask] plot comparativo de order 2 guardado en {comparison_grid_order2}"
            summary_lines.append(line)
            print(line, flush=True)

    final_line = (
        f"[node-mask] terminado: 3 casos en {format_duration(time.perf_counter() - overall_start)}. "
        f"Salida: {output_dir}"
    )
    summary_lines.append(final_line)
    print(final_line, flush=True)
    summary_path = output_dir / "study_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[node-mask] resumen guardado en {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
