#!/usr/bin/env python3
from __future__ import annotations

"""Mapa de fase de Haldane coloreado por Re[sigma^(2)_ijk].

Esta tool barre el espacio de fase canonico del modelo de Haldane,
``(phi0, M0/t2)``, y calcula

    sigma^(2)_ijk(2 omega; omega, omega)

para una frecuencia laser fija. Guarda un dataset con el tensor completo y,
por defecto, grafica solo la componente ``xyy`` con la misma densidad de puntos
que el mapa lineal ``sigma_xy``.

Ejemplo:
    python tools/plot_haldane_sigma2_phase_diagram.py \
        --config inputs/inputParams.haldane_topological.cfg \
        --component xyy --omega-ev 0.8
"""

import argparse
import contextlib
import io
import os
from itertools import product
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.core import QXTIConfig
from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
from qxti.utils.progress import ProgressTimer, format_duration
from tools.plot_haldane_sigma2_vs_phi import (
    AU_TO_EV,
    DEFAULT_CONFIG,
    _component_latex,
    _component_name,
    _compute_sigma2_full_gridbased,
    _prepare_config_for_phi,
)

DEFAULT_OUTPUT_DIR = Path("outputs") / "haldane_sigma2_phase_diagram"


def _phase_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "haldane_sigma2_phase",
        [
            "#163B5C",
            "#2F6C8F",
            "#8DB7B5",
            "#F3E9D2",
            "#E7B97A",
            "#C96B4B",
            "#7A1F2B",
        ],
        N=256,
    )


def _axis_with_zero(lo: float, hi: float, n: int) -> np.ndarray:
    values = np.linspace(float(lo), float(hi), max(int(n), 2), dtype=np.float64)
    if lo <= 0.0 <= hi and not np.any(np.isclose(values, 0.0, atol=1.0e-14)):
        values = np.sort(np.concatenate([values, np.array([0.0], dtype=np.float64)]))
    return values


def _symmetric_norm(values: np.ndarray) -> TwoSlopeNorm | None:
    vmax = float(np.nanmax(np.abs(values))) if np.size(values) else 0.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        return None
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _config_param(config: QXTIConfig, key: str, fallback: float) -> float:
    value = config.hamiltonian.params.get(key, fallback)
    return float(value)


def _parse_component(text: str, labels: tuple[str, ...]) -> tuple[int, int, int]:
    clean = (
        str(text)
        .strip()
        .lower()
        .replace("sigma", "")
        .replace("σ", "")
        .replace("^", "")
        .replace("(", "")
        .replace(")", "")
        .replace("_", "")
        .replace(",", "")
        .replace(" ", "")
    )
    if len(clean) != 3:
        raise ValueError("La componente debe tener tres indices, por ejemplo xyy, xxx, yxy.")
    label_to_index = {label.lower(): index for index, label in enumerate(labels)}
    try:
        return tuple(label_to_index[item] for item in clean)  # type: ignore[return-value]
    except KeyError as exc:
        allowed = "".join(labels)
        raise ValueError(f"Componente invalida '{text}'. Indices permitidos: {allowed}.") from exc


def _omega_from_args(args: argparse.Namespace) -> float:
    if args.omega is not None and args.omega_ev is not None:
        raise SystemExit("Usa solo uno de --omega o --omega-ev.")
    if args.omega_ev is not None:
        return float(args.omega_ev) / AU_TO_EV
    if args.omega is not None:
        return float(args.omega)
    return 0.05


def _point_config(
    *,
    base_config: QXTIConfig,
    phi0: float,
    m_over_t2: float,
    t2_for_axis: float,
    args: argparse.Namespace,
) -> QXTIConfig:
    return _prepare_config_for_phi(
        base_config,
        phi0=float(phi0),
        kpoints=int(args.kpoints),
        t1=args.t1,
        t2=args.t2,
        m0=float(m_over_t2) * float(t2_for_axis),
        a0=args.a0,
        gamma=args.gamma,
        coherence_time=args.coherence_time,
        temperature_au=args.temperature_au,
        fermi_level=args.fermi_level,
        distribution=args.distribution,
    )


def _compute_phase_tensor(
    *,
    base_config: QXTIConfig,
    phi_values: np.ndarray,
    m_over_t2_values: np.ndarray,
    omega: float,
    t2_for_axis: float,
    selected_component: tuple[int, int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, tuple[str, ...], list[dict[str, Any]]]:
    omega_axis = np.asarray([float(omega)], dtype=np.float64)
    sigma_maps: np.ndarray | None = None
    labels: tuple[str, ...] | None = None
    metadata: list[dict[str, Any]] = []
    timer = ProgressTimer(
        total=int(phi_values.size * m_over_t2_values.size),
        min_completed_for_eta=max(4, min(int(phi_values.size), 12)),
    )

    for im, m_over_t2 in enumerate(m_over_t2_values):
        row_start = time.perf_counter()
        for ip, phi0 in enumerate(phi_values):
            cfg = _point_config(
                base_config=base_config,
                phi0=float(phi0),
                m_over_t2=float(m_over_t2),
                t2_for_axis=t2_for_axis,
                args=args,
            )
            if args.verbose_point_logs:
                result = _compute_sigma2_full_gridbased(
                    cfg,
                    omega_axis,
                    components=None if args.all_components else (selected_component,),
                    progress=True,
                )
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = _compute_sigma2_full_gridbased(
                        cfg,
                        omega_axis,
                        components=None if args.all_components else (selected_component,),
                        progress=False,
                    )

            tensor = np.asarray(result["sigma_order_2_tensor"][0], dtype=np.complex128)
            if sigma_maps is None:
                labels = tuple(str(item) for item in result["direction_labels"])
                sigma_maps = np.zeros(
                    (m_over_t2_values.size, phi_values.size) + tensor.shape,
                    dtype=np.complex128,
                )
            sigma_maps[im, ip] = tensor
            metadata.append(result)
            timer.advance()

        if not args.quiet:
            print(
                f"[haldane-sigma2-phase] row {im + 1}/{m_over_t2_values.size} "
                f"(M0/t2={m_over_t2:+.3f}) in {format_duration(time.perf_counter() - row_start)}; "
                f"elapsed {format_duration(timer.elapsed_seconds)}, ETA {timer.eta_text()}",
                flush=True,
            )

    if sigma_maps is None or labels is None:
        raise RuntimeError("No se pudo calcular ningun punto del mapa.")
    return sigma_maps, labels, metadata


def _draw_phase_boundaries(axis: plt.Axes, phi_values: np.ndarray) -> None:
    critical = 3.0 * np.sqrt(3.0) * np.sin(phi_values)
    axis.plot(phi_values, critical, color="black", linewidth=1.1, linestyle="--", alpha=0.92)
    axis.plot(phi_values, -critical, color="black", linewidth=1.1, linestyle="--", alpha=0.92)


def _plot_component_map(
    *,
    phi_values: np.ndarray,
    m_over_t2_values: np.ndarray,
    values: np.ndarray,
    component: tuple[int, int, int],
    labels: tuple[str, ...],
    omega: float,
    output_path: Path,
    dpi: int,
) -> None:
    apply_paper_style()
    cmap = _phase_colormap()
    x_grid, y_grid = np.meshgrid(phi_values, m_over_t2_values, indexing="xy")
    norm = _symmetric_norm(values)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    image = ax.pcolormesh(x_grid, y_grid, values, shading="auto", cmap=cmap, norm=norm, alpha=0.88)
    _draw_phase_boundaries(ax, phi_values)

    ax.set_xlabel(r"$\phi_0\;(\mathrm{rad})$")
    ax.set_ylabel(r"$M_0/t_2$")
    ax.set_title(
        _component_latex(component, labels, "real")
        + rf"$,\;\hbar\omega={omega * AU_TO_EV:.3f}\,\mathrm{{eV}}$"
    )
    ax.set_xlim(float(phi_values.min()), float(phi_values.max()))
    ax.set_ylim(float(m_over_t2_values.min()), float(m_over_t2_values.max()))

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(_component_latex(component, labels, "real") + r"$\;(\mathrm{a.u.})$")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def _plot_overview(
    *,
    phi_values: np.ndarray,
    m_over_t2_values: np.ndarray,
    sigma_real: np.ndarray,
    labels: tuple[str, ...],
    omega: float,
    output_path: Path,
    dpi: int,
) -> None:
    apply_paper_style()
    dim = len(labels)
    components = list(product(range(dim), repeat=3))
    ncols = 4 if dim == 2 else 3
    nrows = int(np.ceil(len(components) / ncols))
    x_grid, y_grid = np.meshgrid(phi_values, m_over_t2_values, indexing="xy")
    cmap = _phase_colormap()
    norm = _symmetric_norm(sigma_real)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.35 * ncols, 2.85 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for axis, component in zip(axes.ravel(), components, strict=False):
        values = sigma_real[:, :, component[0], component[1], component[2]]
        image = axis.pcolormesh(x_grid, y_grid, values, shading="auto", cmap=cmap, norm=norm, alpha=0.88)
        _draw_phase_boundaries(axis, phi_values)
        axis.set_title(_component_latex(component, labels, "real"), fontsize=9)
        axis.set_xlim(float(phi_values.min()), float(phi_values.max()))
        axis.set_ylim(float(m_over_t2_values.min()), float(m_over_t2_values.max()))
        axis.tick_params(labelsize=7)

    for axis in axes.ravel()[len(components):]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel(r"$\phi_0\;(\mathrm{rad})$")
    for axis in axes[:, 0]:
        if axis.get_visible():
            axis.set_ylabel(r"$M_0/t_2$")

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.012)
        colorbar.set_label(r"$\Re\,\sigma^{(2)}_{ijk}\;(\mathrm{a.u.})$")
    fig.suptitle(rf"Haldane phase map, $\hbar\omega={omega * AU_TO_EV:.3f}\,\mathrm{{eV}}$", fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mapa de fase de Haldane para Re[sigma^(2)_ijk(2w;w,w)]."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Input base de Haldane.")
    parser.add_argument("--nphi", type=int, default=81, help="Numero de puntos en phi0.")
    parser.add_argument("--nm", type=int, default=81, help="Numero de puntos en M0/t2.")
    parser.add_argument("--kpoints", type=int, default=61, help="Puntos k por eje.")
    parser.add_argument("--phi-min", type=float, default=-np.pi, help="Limite inferior de phi0 (rad).")
    parser.add_argument("--phi-max", type=float, default=np.pi, help="Limite superior de phi0 (rad).")
    parser.add_argument("--m-over-t2-min", type=float, default=-6.0, help="Limite inferior de M0/t2.")
    parser.add_argument("--m-over-t2-max", type=float, default=6.0, help="Limite superior de M0/t2.")
    parser.add_argument("--omega", type=float, default=None, help="Frecuencia laser en a.u. Default: 0.05.")
    parser.add_argument("--omega-ev", type=float, default=None, help="Frecuencia laser en eV.")
    parser.add_argument("--gamma", type=float, default=None, help="Override gamma=1/T2 en a.u.")
    parser.add_argument("--coherence-time", type=float, default=None, help="Override T2 en a.u.")
    parser.add_argument("--temperature-au", type=float, default=None, help="Override temperatura en a.u.")
    parser.add_argument("--fermi-level", type=float, default=None, help="Override nivel de Fermi en a.u.")
    parser.add_argument("--distribution", default=None, help="Override distribucion de ocupacion.")
    parser.add_argument("--t1", type=float, default=None, help="Override t1 (a.u.).")
    parser.add_argument("--t2", type=float, default=None, help="Override t2 (a.u.).")
    parser.add_argument("--a0", type=float, default=None, help="Override a0 (a.u.).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Carpeta de salida.")
    parser.add_argument("--dataset-name", default="haldane_sigma2_phase_diagram.npz", help="Nombre del dataset.")
    parser.add_argument("--component", default="xyy", help="Componente sigma^(2)_ijk a graficar. Default: xyy.")
    parser.add_argument("--dpi", type=int, default=300, help="Resolucion PNG.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs por fila.")
    parser.add_argument("--verbose-point-logs", action="store_true", help="Muestra logs internos de cada punto.")
    parser.add_argument("--all-components", action="store_true", help="Tambien guarda overview y mapas de todas las componentes.")
    args = parser.parse_args()

    if args.nphi <= 1 or args.nm <= 1 or args.kpoints <= 1:
        raise SystemExit("nphi, nm y kpoints deben ser mayores que 1.")
    if args.gamma is not None and args.coherence_time is not None:
        raise SystemExit("Usa solo uno de --gamma o --coherence-time.")

    base_config = QXTIConfig.from_file(Path(args.config))
    t2_for_axis = float(args.t2) if args.t2 is not None else _config_param(base_config, "t2", 0.025)
    if abs(t2_for_axis) <= 1.0e-15:
        raise SystemExit("t2 no puede ser cero porque el eje vertical es M0/t2.")

    omega = _omega_from_args(args)
    if omega <= 0.0:
        raise SystemExit("omega debe ser estrictamente positivo.")

    phi_values = _axis_with_zero(args.phi_min, args.phi_max, args.nphi)
    m_over_t2_values = _axis_with_zero(args.m_over_t2_min, args.m_over_t2_max, args.nm)
    selected_component = _parse_component(args.component, ("x", "y"))
    selected_name = _component_name(selected_component, ("x", "y"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[haldane-sigma2-phase] construyendo mapa "
        f"{m_over_t2_values.size}x{phi_values.size}, k={args.kpoints}x{args.kpoints}, "
        f"hbar omega={omega * AU_TO_EV:.4f} eV",
        flush=True,
    )
    start = time.perf_counter()
    sigma_maps, labels, metadata = _compute_phase_tensor(
        base_config=base_config,
        phi_values=phi_values,
        m_over_t2_values=m_over_t2_values,
        omega=omega,
        t2_for_axis=t2_for_axis,
        selected_component=selected_component,
        args=args,
    )
    elapsed = time.perf_counter() - start
    sigma_real = np.real(sigma_maps)
    selected_component = _parse_component(args.component, labels)
    selected_name = _component_name(selected_component, labels)

    data_path = output_dir / args.dataset_name
    np.savez_compressed(
        data_path,
        phi0_axis=phi_values,
        m_over_t2_axis=m_over_t2_values,
        omega=float(omega),
        omega_ev=float(omega * AU_TO_EV),
        sigma_order_2_tensor=sigma_maps,
        sigma_order_2_real=sigma_real,
        sigma_order_2_imag=np.imag(sigma_maps),
        direction_labels=np.asarray(labels),
        component_order="sigma[m_index, phi_index, i, j, k]",
        selected_component=np.asarray(selected_component, dtype=np.int16),
        selected_component_name=str(selected_name),
        config_path=str(Path(args.config)),
        t2_for_axis=float(t2_for_axis),
        runtime_seconds=float(elapsed),
        point_runtime_seconds=np.asarray([float(item["runtime_seconds"]) for item in metadata], dtype=np.float64),
    )

    selected_path = output_dir / f"haldane_sigma2_phase_{selected_name}_real.png"
    _plot_component_map(
        phi_values=phi_values,
        m_over_t2_values=m_over_t2_values,
        values=sigma_real[
            :,
            :,
            selected_component[0],
            selected_component[1],
            selected_component[2],
        ],
        component=selected_component,
        labels=labels,
        omega=omega,
        output_path=selected_path,
        dpi=int(args.dpi),
    )

    overview_path: Path | None = None
    component_paths: list[Path] = []
    if args.all_components:
        overview_path = output_dir / "haldane_sigma2_phase_overview_real.png"
        _plot_overview(
            phi_values=phi_values,
            m_over_t2_values=m_over_t2_values,
            sigma_real=sigma_real,
            labels=labels,
            omega=omega,
            output_path=overview_path,
            dpi=int(args.dpi),
        )
        component_dir = output_dir / "components" / "real"
        for component in product(range(len(labels)), repeat=3):
            name = _component_name(component, labels)
            path = component_dir / f"haldane_sigma2_phase_{name}_real.png"
            _plot_component_map(
                phi_values=phi_values,
                m_over_t2_values=m_over_t2_values,
                values=sigma_real[:, :, component[0], component[1], component[2]],
                component=component,
                labels=labels,
                omega=omega,
                output_path=path,
                dpi=int(args.dpi),
            )
            component_paths.append(path)

    print(
        f"[haldane-sigma2-phase] listo en {format_duration(elapsed)}.\n"
        f"[haldane-sigma2-phase] data: {data_path}\n"
        f"[haldane-sigma2-phase] selected plot: {selected_path}\n"
        f"[haldane-sigma2-phase] overview: {overview_path if overview_path is not None else 'disabled'}\n"
        f"[haldane-sigma2-phase] component plots: {len(component_paths)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
