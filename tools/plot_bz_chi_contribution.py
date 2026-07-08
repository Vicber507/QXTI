#!/usr/bin/env python3
from __future__ import annotations

"""Mapa k-resuelto de la contribucion a chi^(2)_ijk en una seccion de la BZ.

La respuesta integrada ``chi^(2)_ijk`` es una integral sobre toda la zona de
Brillouin. Esta herramienta no reemplaza esa integral: dibuja el integrando local
en una lamina, por defecto ``kz = 0``, para identificar que regiones de la BZ
alimentan mas una componente, por ejemplo ``chi_zzz`` o ``chi_xzx``.

Ejemplo:
    python tools/plot_bz_chi_contribution.py \
        --config inputs/inputParams.wsm_orenstein.cfg \
        --component zzz --omega-ev 0.8 --plane kz --fixed 0.0
"""

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.analytics.theory_response import _resolve_distribution
from qxti.core import QXTIConfig
from qxti.core.simulation import QXTISimulation
from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
from qxti.utils.progress import format_duration

AU_TO_EV = 27.211386245988
BOHR_INV_TO_ANGSTROM_INV = 1.8897259886
DEFAULT_OUTPUT_DIR = Path("outputs") / "bz_chi_contribution"
AXES = {"x": 0, "y": 1, "z": 2}
LABELS = ("x", "y", "z")

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _component_indices(text: str) -> tuple[int, int, int]:
    clean = (
        str(text)
        .strip()
        .lower()
        .replace("chi", "")
        .replace("xi", "")
        .replace("χ", "")
        .replace("ξ", "")
        .replace("sigma", "")
        .replace("σ", "")
        .replace("^", "")
        .replace("(", "")
        .replace(")", "")
        .replace("_", "")
        .replace(",", "")
        .replace(" ", "")
    )
    if len(clean) != 3 or any(item not in AXES for item in clean):
        raise ValueError("component debe tener tres indices cartesianos, por ejemplo zzz o xzx.")
    return tuple(AXES[item] for item in clean)  # type: ignore[return-value]


def _component_label(component: tuple[int, int, int]) -> str:
    return "".join(LABELS[index] for index in component)


def _load_model_module(config: QXTIConfig):
    source = Path(str(config.hamiltonian.source_file))
    model_path = source if source.is_absolute() else PROJECT_ROOT / "models" / source
    spec = importlib.util.spec_from_file_location(f"qxti_model_{model_path.stem}", model_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_omega(args: argparse.Namespace) -> float:
    if args.omega is not None and args.omega_ev is not None:
        raise SystemExit("Usa solo uno de --omega o --omega-ev.")
    if args.omega_ev is not None:
        return float(args.omega_ev) / AU_TO_EV
    if args.omega is not None:
        return float(args.omega)
    return 0.03


def _abs_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "qxti_bz_chi",
        ["#081B33", "#0E4F6F", "#2A9D8F", "#F3E9D2", "#F4A261", "#B23A48"],
        N=256,
    )


def _signed_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "qxti_bz_chi_signed",
        ["#174A7C", "#4FA3C8", "#F7F3E8", "#E59B48", "#9E2F3D"],
        N=256,
    )


def _slice_contribution_summary(values: FloatArray, x_axis: FloatArray, y_axis: FloatArray) -> dict[str, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return {
            "raw_positive": 0.0,
            "raw_negative": 0.0,
            "raw_net": 0.0,
            "weighted_positive": 0.0,
            "weighted_negative": 0.0,
            "weighted_net": 0.0,
            "cell_area": 0.0,
            "cancellation_ratio": np.nan,
        }

    dx = float(np.mean(np.abs(np.diff(x_axis)))) if x_axis.size > 1 else 1.0
    dy = float(np.mean(np.abs(np.diff(y_axis)))) if y_axis.size > 1 else 1.0
    cell_area = dx * dy
    raw_positive = float(np.sum(finite[finite > 0.0]))
    raw_negative = float(np.sum(finite[finite < 0.0]))
    raw_net = float(np.sum(finite))
    weighted_positive = raw_positive * cell_area
    weighted_negative = raw_negative * cell_area
    weighted_net = raw_net * cell_area
    denominator = abs(weighted_net)
    cancellation_ratio = (
        (weighted_positive + abs(weighted_negative)) / denominator
        if denominator > 1.0e-300
        else np.inf
    )
    return {
        "raw_positive": raw_positive,
        "raw_negative": raw_negative,
        "raw_net": raw_net,
        "weighted_positive": weighted_positive,
        "weighted_negative": weighted_negative,
        "weighted_net": weighted_net,
        "cell_area": float(cell_area),
        "cancellation_ratio": float(cancellation_ratio),
    }


def _axis_values(bounds: tuple[float, float], n: int, *, shifted: bool) -> np.ndarray:
    lo, hi = float(bounds[0]), float(bounds[1])
    if shifted:
        step = (hi - lo) / float(n)
        return np.linspace(lo, hi, int(n), endpoint=False, dtype=np.float64) + 0.5 * step
    return np.linspace(lo, hi, int(n), dtype=np.float64)


def _batched_hamiltonian(Hf: Any, k_points: FloatArray) -> ComplexArray:
    out = np.empty((k_points.shape[0], 0, 0), dtype=np.complex128)
    matrices = [Hf(float(k[0]), float(k[1]), float(k[2])) for k in k_points]
    if not matrices:
        return out
    return np.asarray(matrices, dtype=np.complex128)


def _band_data(
    *,
    Hf: Any,
    k_points: FloatArray,
    axes_needed: tuple[int, ...],
    distribution: Any,
    mu: float,
    temperature: float,
    velocity_step: float,
) -> dict[str, Any]:
    h0 = _batched_hamiltonian(Hf, k_points)
    energies, U = np.linalg.eigh(h0)
    Udag = np.conj(np.transpose(U, (0, 2, 1)))
    vel: dict[int, ComplexArray] = {}
    for axis in sorted(set(axes_needed)):
        shift = np.zeros(3, dtype=np.float64)
        shift[axis] = float(velocity_step)
        dH = (_batched_hamiltonian(Hf, k_points + shift) - _batched_hamiltonian(Hf, k_points - shift)) / (
            2.0 * float(velocity_step)
        )
        vel[axis] = Udag @ dH @ U

    occupations = np.asarray(distribution(energies, mu, temperature), dtype=np.float64)
    eps = energies[:, :, None] - energies[:, None, :]
    fmn = occupations[:, None, :] - occupations[:, :, None]
    nb = int(energies.shape[1])
    valid = (~np.eye(nb, dtype=bool))[None] & (np.abs(eps) > 1.0e-20)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_eps = np.where(valid, 1.0 / eps, 0.0)
    dfde = (
        -occupations * (1.0 - occupations) / temperature
        if temperature > 1.0e-15
        else np.zeros_like(occupations)
    )
    return {
        "energies": energies,
        "U": U,
        "Udag": Udag,
        "vel": vel,
        "eps": eps,
        "fmn": fmn,
        "valid": valid,
        "inv_eps": inv_eps,
        "dfde": dfde,
        "nb": nb,
    }


def _rho1(data: dict[str, Any], input_axis: int, omega_complex: complex) -> ComplexArray:
    nb = int(data["nb"])
    vel = np.asarray(data["vel"][input_axis], dtype=np.complex128)
    inv_eps = np.asarray(data["inv_eps"], dtype=np.complex128)
    fmn = np.asarray(data["fmn"], dtype=np.float64)
    eps = np.asarray(data["eps"], dtype=np.float64)
    valid = np.asarray(data["valid"], dtype=bool)
    dfde = np.asarray(data["dfde"], dtype=np.float64)

    A = 1j * vel * inv_eps
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(valid, (A * fmn) / (omega_complex - eps), 0.0 + 0.0j)
    diag_src = (-1j) * dfde * np.real(np.diagonal(vel, axis1=1, axis2=2))
    rho[:, np.arange(nb), np.arange(nb)] += diag_src / omega_complex
    valid_diag = valid | np.eye(nb, dtype=bool)[None]
    return np.asarray(np.where(valid_diag, rho, 0.0 + 0.0j), dtype=np.complex128)


def _ordered_sigma2_integrand(
    *,
    current_data: dict[str, Any],
    plus_data: dict[str, Any],
    minus_data: dict[str, Any],
    component: tuple[int, int, int],
    omega: float,
    gamma: float,
    gradient_step: float,
) -> ComplexArray:
    output_axis, gradient_axis, input_axis = component
    omega1 = complex(float(omega), float(gamma))
    omega2 = complex(2.0 * float(omega), float(gamma))

    rho_curr = _rho1(current_data, input_axis, omega1)
    rho_plus = _rho1(plus_data, input_axis, omega1)
    rho_minus = _rho1(minus_data, input_axis, omega1)

    U_curr = np.asarray(current_data["U"], dtype=np.complex128)
    U_curr_dag = np.asarray(current_data["Udag"], dtype=np.complex128)
    U_plus = np.asarray(plus_data["U"], dtype=np.complex128)
    U_minus = np.asarray(minus_data["U"], dtype=np.complex128)
    W_plus = U_curr_dag @ U_plus
    W_minus = U_curr_dag @ U_minus
    W_plus_dag = np.conj(np.swapaxes(W_plus, -1, -2))
    W_minus_dag = np.conj(np.swapaxes(W_minus, -1, -2))

    transported_plus = W_plus @ rho_plus @ W_plus_dag
    transported_minus = W_minus @ rho_minus @ W_minus_dag
    dpart = (transported_plus - transported_minus) / (2.0 * float(gradient_step))

    A_gradient = (
        1j
        * np.asarray(current_data["vel"][gradient_axis], dtype=np.complex128)
        * np.asarray(current_data["inv_eps"], dtype=np.complex128)
    )
    commutator = A_gradient @ rho_curr - rho_curr @ A_gradient
    valid = np.asarray(current_data["valid"], dtype=bool)
    eps = np.asarray(current_data["eps"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_d2 = np.where(valid, 1.0 / (omega2 - eps), 0.0 + 0.0j)

    rho2 = (dpart - 1j * commutator) * inv_d2
    current_op = -np.asarray(current_data["vel"][output_axis], dtype=np.complex128)
    return np.conj(np.einsum("pmn,pnm->p", current_op, rho2, optimize=True))


def _sigma2_integrand(
    *,
    current_data: dict[str, Any],
    shifted_data_by_axis: dict[int, tuple[dict[str, Any], dict[str, Any]]],
    component: tuple[int, int, int],
    omega: float,
    gamma: float,
    gradient_step: float,
) -> ComplexArray:
    plus_data, minus_data = shifted_data_by_axis[component[1]]
    first = _ordered_sigma2_integrand(
        current_data=current_data,
        plus_data=plus_data,
        minus_data=minus_data,
        component=component,
        omega=omega,
        gamma=gamma,
        gradient_step=gradient_step,
    )
    if component[1] == component[2]:
        return first
    swapped = (component[0], component[2], component[1])
    plus_data, minus_data = shifted_data_by_axis[swapped[1]]
    second = _ordered_sigma2_integrand(
        current_data=current_data,
        plus_data=plus_data,
        minus_data=minus_data,
        component=swapped,
        omega=omega,
        gamma=gamma,
        gradient_step=gradient_step,
    )
    return 0.5 * (first + second)


def _value_to_plot(values: ComplexArray, part: str) -> FloatArray:
    key = part.strip().lower()
    if key == "abs":
        return np.abs(values)
    if key == "real":
        return np.real(values)
    if key == "imag":
        return np.imag(values)
    raise ValueError("part debe ser abs, real o imag.")


def _overlay_weyl_nodes(axis: Any, module: Any, params: dict[str, Any], plane_axis: int, fixed: float, unit_scale: float) -> None:
    if module is None or not hasattr(module, "weyl_nodes_with_chirality"):
        return
    try:
        tagged = module.weyl_nodes_with_chirality(params)
    except Exception:
        return
    plane_axes = [axis_index for axis_index in range(3) if axis_index != plane_axis]
    marker_style = {
        +1: dict(marker="o", facecolor="white", edgecolor="black"),
        -1: dict(marker="X", facecolor="black", edgecolor="white"),
    }
    for item in tagged:
        k = np.asarray(item.get("k"), dtype=float)
        if not np.isclose(k[plane_axis], fixed, atol=2.0e-3):
            continue
        chirality = int(item.get("chirality", 0))
        style = marker_style.get(chirality, dict(marker="o", facecolor="white", edgecolor="black"))
        axis.scatter(
            k[plane_axes[0]] * unit_scale,
            k[plane_axes[1]] * unit_scale,
            s=74,
            linewidth=1.2,
            zorder=6,
            label=rf"Weyl $\chi={chirality:+d}$",
            **style,
        )


def _plot_map(
    *,
    x_axis: FloatArray,
    y_axis: FloatArray,
    values: FloatArray,
    part: str,
    component_label: str,
    omega: float,
    plane_label: str,
    fixed: float,
    output_path: Path,
    unit_label: str,
    module: Any,
    params: dict[str, Any],
    plane_axis: int,
    unit_scale: float,
    log_abs: bool,
    summary: dict[str, float] | None,
    dpi: int,
) -> None:
    apply_paper_style()
    cmap = _abs_cmap() if part == "abs" else _signed_cmap()
    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij")

    if part == "abs" and log_abs:
        positive = values[np.isfinite(values) & (values > 0.0)]
        norm = LogNorm(vmin=max(float(np.min(positive)), 1.0e-30), vmax=float(np.max(positive))) if positive.size else None
    elif part in {"real", "imag"}:
        vmax = float(np.nanmax(np.abs(values))) if values.size else 0.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax) if np.isfinite(vmax) and vmax > 0.0 else None
    else:
        norm = None

    image = axis.pcolormesh(
        x_grid,
        y_grid,
        values,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    _overlay_weyl_nodes(axis, module, params, plane_axis, fixed, unit_scale)

    plane_axes = [axis_index for axis_index in range(3) if axis_index != plane_axis]
    axis.set_xlabel(rf"$k_{LABELS[plane_axes[0]]}\;({unit_label})$")
    axis.set_ylabel(rf"$k_{LABELS[plane_axes[1]]}\;({unit_label})$")
    part_tex = {"abs": r"|", "real": r"\Re\,", "imag": r"\Im\,"}[part]
    if part == "abs":
        title_quantity = rf"$|\chi^{{(2)}}_{{{component_label}}}(\mathbf{{k}})|$"
    else:
        title_quantity = rf"${part_tex}\chi^{{(2)}}_{{{component_label}}}(\mathbf{{k}})$"
    axis.set_title(
        title_quantity
        + rf", $\hbar\omega={omega * AU_TO_EV:.3f}\,\mathrm{{eV}}$, "
        + rf"${plane_label}={fixed * unit_scale:.3f}\,{unit_label}$"
    )
    if summary is not None and part in {"real", "imag"}:
        axis.text(
            0.02,
            0.98,
            "slice sum\n"
            + rf"$+={summary['weighted_positive']:.2e}$" "\n"
            + rf"$-={summary['weighted_negative']:.2e}$" "\n"
            + rf"$net={summary['weighted_net']:.2e}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.78),
        )
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(rf"{title_quantity} local integrand")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        dedup: dict[str, Any] = {}
        for handle, label in zip(handles, labels, strict=False):
            dedup.setdefault(label, handle)
        axis.legend(dedup.values(), dedup.keys(), loc="upper right", frameon=True, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mapa k-resuelto de la contribucion local a chi^(2)_ijk en una lamina de la BZ."
    )
    parser.add_argument("--config", default="inputs/inputParams.wsm_orenstein.cfg", help="Input QXTI.")
    parser.add_argument("--component", default="zzz", help="Componente chi^(2)_ijk, por ejemplo zzz o xzx.")
    parser.add_argument("--omega", type=float, default=None, help="Frecuencia laser en a.u.")
    parser.add_argument("--omega-ev", type=float, default=0.8, help="Frecuencia laser en eV. Default: 0.8.")
    parser.add_argument("--plane", choices=("kx", "ky", "kz"), default="kz", help="Plano fijo. Default: kz.")
    parser.add_argument("--fixed", type=float, default=0.0, help="Valor fijo del plano en 1/Bohr.")
    parser.add_argument(
        "--points",
        type=int,
        default=180,
        help="Puntos por eje del mapa 2D. Default par para evitar lineas degeneradas exactas.",
    )
    parser.add_argument("--normal-points", type=int, default=None, help="Define dk normal como ancho BZ / normal-points.")
    parser.add_argument("--gradient-step", type=float, default=None, help="Paso de derivada covariante normal en 1/Bohr.")
    parser.add_argument("--velocity-step", type=float, default=1.0e-4, help="Paso de derivada para velocidades en 1/Bohr.")
    parser.add_argument(
        "--part",
        choices=("abs", "real", "imag"),
        default="real",
        help="Parte graficada. Default: real, para ver que regiones suman/restan.",
    )
    parser.add_argument("--linear-abs", action="store_true", help="Usa escala lineal para |chi|.")
    parser.add_argument("--unshifted", action="store_true", help="Usa grilla no desplazada en los dos ejes del plano.")
    parser.add_argument("--units", choices=("angstrom", "au"), default="angstrom", help="Unidades del plot.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Carpeta de salida.")
    parser.add_argument("--dpi", type=int, default=320, help="Resolucion PNG.")
    args = parser.parse_args()

    if args.points <= 1:
        raise SystemExit("--points debe ser mayor que 1.")
    omega = _resolve_omega(args)
    if omega <= 0.0:
        raise SystemExit("omega debe ser positivo.")

    config = QXTIConfig.from_file(Path(args.config))
    sim = QXTISimulation(config=config)
    hamiltonian = sim.build_hamiltonian()
    if int(hamiltonian.dimension) < 3:
        raise SystemExit("Esta herramienta necesita un modelo 3D para un plano de BZ.")
    module = _load_model_module(config)
    params = dict(config.hamiltonian.params)

    component = _component_indices(args.component)
    component_label = _component_label(component)
    plane_axis = AXES[args.plane[-1]]
    plane_axes = [axis for axis in range(3) if axis != plane_axis]
    bounds = hamiltonian.reciprocal_box_bounds()
    x_internal = _axis_values(bounds[plane_axes[0]], int(args.points), shifted=not args.unshifted)
    y_internal = _axis_values(bounds[plane_axes[1]], int(args.points), shifted=not args.unshifted)
    source_name = str(config.hamiltonian.source_file).lower()
    if "wsm_orenstein" in source_name and plane_axis == AXES["z"] and np.isclose(float(args.fixed), 0.0, atol=1.0e-14):
        axes_by_index = {plane_axes[0]: x_internal, plane_axes[1]: y_internal}
        ky_axis = axes_by_index.get(AXES["y"])
        if ky_axis is not None and np.any(np.isclose(ky_axis, 0.0, atol=1.0e-14)):
            print(
                "[bz-chi] WARNING: this kz=0 grid includes ky=0, where the Orenstein model has an "
                "internal band degeneracy. The local k-resolved integrand can show a sharp gauge/"
                "band-sorting stripe there. Use an even --points value, e.g. --points 180 or 200.",
                flush=True,
            )
    x_grid, y_grid = np.meshgrid(x_internal, y_internal, indexing="ij")
    k_points = np.zeros((x_grid.size, 3), dtype=np.float64)
    k_points[:, plane_axis] = float(args.fixed)
    k_points[:, plane_axes[0]] = x_grid.reshape(-1)
    k_points[:, plane_axes[1]] = y_grid.reshape(-1)

    normal_points = int(args.normal_points) if args.normal_points is not None else int(config.kgrid.k_points[plane_axis])
    if normal_points <= 1:
        normal_points = max(int(args.points), 2)
    gradient_step = (
        float(args.gradient_step)
        if args.gradient_step is not None
        else (float(bounds[plane_axis][1]) - float(bounds[plane_axis][0])) / float(normal_points)
    )

    axes_needed = tuple(sorted(set(component)))
    dist = _resolve_distribution(config.susceptibility_solver.distribution)
    mu = float(config.susceptibility_solver.fermi_level)
    temperature = float(config.susceptibility_solver.temperature)
    gamma = 0.0 if config.susceptibility_solver.coherence_time <= 0 else 1.0 / float(config.susceptibility_solver.coherence_time)

    gradient_axes = tuple(sorted({component[1], component[2]}))
    t0 = time.perf_counter()
    print(
        f"[bz-chi] computing local chi^(2)_{component_label} on {args.plane}={args.fixed:g}, "
        f"{args.points}x{args.points}, omega={omega * AU_TO_EV:.3f} eV, "
        f"dk_grad={gradient_step:.4g} 1/Bohr",
        flush=True,
    )
    if component[1] != component[2]:
        axis_text = ", ".join(f"k{LABELS[axis]}" for axis in gradient_axes)
        print(
            f"[bz-chi] off-diagonal input indices detected: evaluating covariant derivatives along {axis_text}.",
            flush=True,
        )
    current_data = _band_data(
        Hf=hamiltonian._matrix_at,
        k_points=k_points,
        axes_needed=axes_needed,
        distribution=dist,
        mu=mu,
        temperature=temperature,
        velocity_step=float(args.velocity_step),
    )
    shifted_data_by_axis: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for gradient_axis in gradient_axes:
        shift = np.zeros(3, dtype=np.float64)
        # Each ordered chi_ijk term differentiates covariantly along its second index.
        shift[gradient_axis] = gradient_step
        plus_data = _band_data(
            Hf=hamiltonian._matrix_at,
            k_points=k_points + shift,
            axes_needed=axes_needed,
            distribution=dist,
            mu=mu,
            temperature=temperature,
            velocity_step=float(args.velocity_step),
        )
        minus_data = _band_data(
            Hf=hamiltonian._matrix_at,
            k_points=k_points - shift,
            axes_needed=axes_needed,
            distribution=dist,
            mu=mu,
            temperature=temperature,
            velocity_step=float(args.velocity_step),
        )
        shifted_data_by_axis[gradient_axis] = (plus_data, minus_data)

    sigma_integrand = _sigma2_integrand(
        current_data=current_data,
        shifted_data_by_axis=shifted_data_by_axis,
        component=component,
        omega=omega,
        gamma=gamma,
        gradient_step=gradient_step,
    )
    chi_integrand = sigma_integrand / (-1j * (2.0 * omega))
    chi_map = chi_integrand.reshape(int(args.points), int(args.points))
    value_map = _value_to_plot(chi_map, args.part)
    contribution_summary = (
        _slice_contribution_summary(value_map, x_internal, y_internal)
        if args.part in {"real", "imag"}
        else None
    )
    elapsed = time.perf_counter() - t0

    unit_scale = BOHR_INV_TO_ANGSTROM_INV if args.units == "angstrom" else 1.0
    unit_label = r"\mathrm{\AA}^{-1}" if args.units == "angstrom" else r"a_0^{-1}"
    x_plot = x_internal * unit_scale
    y_plot = y_internal * unit_scale
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"chi2_{component_label}_{args.plane}_{args.fixed:g}_{args.part}".replace("-", "m").replace(".", "p")
    data_path = output_dir / f"{stem}.npz"
    plot_path = output_dir / f"{stem}.png"
    np.savez_compressed(
        data_path,
        component=np.asarray(component, dtype=np.int16),
        component_label=component_label,
        omega=float(omega),
        omega_ev=float(omega * AU_TO_EV),
        gamma=float(gamma),
        plane=str(args.plane),
        fixed=float(args.fixed),
        gradient_step=float(gradient_step),
        gradient_axes=np.asarray(gradient_axes, dtype=np.int16),
        velocity_step=float(args.velocity_step),
        x_axis=x_internal,
        y_axis=y_internal,
        x_axis_plot=x_plot,
        y_axis_plot=y_plot,
        unit=str(args.units),
        chi_integrand=chi_map,
        sigma_integrand=sigma_integrand.reshape(int(args.points), int(args.points)),
        central_gap=current_data["energies"][:, 2].reshape(int(args.points), int(args.points))
        - current_data["energies"][:, 1].reshape(int(args.points), int(args.points)),
        plotted_values=value_map,
        signed_part=str(args.part),
        slice_cell_area=0.0 if contribution_summary is None else contribution_summary["cell_area"],
        slice_positive_sum=0.0 if contribution_summary is None else contribution_summary["weighted_positive"],
        slice_negative_sum=0.0 if contribution_summary is None else contribution_summary["weighted_negative"],
        slice_net_sum=0.0 if contribution_summary is None else contribution_summary["weighted_net"],
        slice_raw_positive_sum=0.0 if contribution_summary is None else contribution_summary["raw_positive"],
        slice_raw_negative_sum=0.0 if contribution_summary is None else contribution_summary["raw_negative"],
        slice_raw_net_sum=0.0 if contribution_summary is None else contribution_summary["raw_net"],
        slice_cancellation_ratio=np.nan if contribution_summary is None else contribution_summary["cancellation_ratio"],
        note="Local k-resolved integrand slice; not the full BZ-integrated susceptibility.",
    )
    _plot_map(
        x_axis=x_plot,
        y_axis=y_plot,
        values=value_map,
        part=str(args.part),
        component_label=component_label,
        omega=omega,
        plane_label=str(args.plane),
        fixed=float(args.fixed),
        output_path=plot_path,
        unit_label=unit_label,
        module=module,
        params=params,
        plane_axis=plane_axis,
        unit_scale=unit_scale,
        log_abs=not bool(args.linear_abs),
        summary=contribution_summary,
        dpi=int(args.dpi),
    )
    if contribution_summary is not None:
        print(
            "[bz-chi] signed slice summary "
            f"({args.part}): +={contribution_summary['weighted_positive']:.6e}, "
            f"-={contribution_summary['weighted_negative']:.6e}, "
            f"net={contribution_summary['weighted_net']:.6e}, "
            f"cancellation={contribution_summary['cancellation_ratio']:.3g}",
            flush=True,
        )
    print(
        f"[bz-chi] done in {format_duration(elapsed)}\n"
        f"[bz-chi] data: {data_path}\n"
        f"[bz-chi] plot: {plot_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
