#!/usr/bin/env python3
from __future__ import annotations

"""Sweep de M0/t2 para sigma^(2)_ijk del modelo de Haldane.

Mantiene phi0 fijo, por defecto ``phi0 = pi/2``, y barre ``M0/t2`` desde
la transicion superior hasta la inferior:

    M0/t2 = +3 sqrt(3) sin(phi0)  ->  -3 sqrt(3) sin(phi0).

El grafico sigue el estilo del sweep en phi: curvas en frecuencia coloreadas
por el parametro barrido. Por defecto grafica Re[sigma^(2)_xyy].

Ejemplo:
    python tools/plot_haldane_sigma2_vs_m0.py \
        --config inputs/inputParams.haldane_topological.cfg \
        --component xyy --phi0 1.5707963267948966
"""

import argparse
import contextlib
import io
import os
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.core import QXTIConfig
from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
from qxti.utils.progress import ProgressTimer, format_duration
from tools.plot_haldane_sigma2_phase_diagram import _config_param, _parse_component
from tools.plot_haldane_sigma2_vs_phi import (
    AU_TO_EV,
    DEFAULT_CONFIG,
    _component_latex,
    _component_name,
    _compute_sigma2_full_gridbased,
    _prepare_config_for_phi,
    _value_view,
)

DEFAULT_OUTPUT_DIR = Path("outputs") / "haldane_sigma2_m0_sweep"


def _sweep_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "haldane_m0_sweep_sigma2",
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


def _omega_axis_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.omega_min_ev is not None or args.omega_max_ev is not None:
        if args.omega_min_ev is None or args.omega_max_ev is None:
            raise SystemExit("Si usas eV, pasa ambos --omega-min-ev y --omega-max-ev.")
        omega_min = float(args.omega_min_ev) / AU_TO_EV
        omega_max = float(args.omega_max_ev) / AU_TO_EV
    else:
        omega_min = float(args.omega_min)
        omega_max = float(args.omega_max)
    if omega_min <= 0.0 or omega_max <= omega_min:
        raise SystemExit("Se requiere 0 < omega-min < omega-max.")
    return np.linspace(omega_min, omega_max, int(args.nomega), dtype=np.float64)


def _transition_m_over_t2(phi0: float) -> tuple[float, float]:
    critical = 3.0 * np.sqrt(3.0) * np.sin(float(phi0))
    return float(critical), float(-critical)


def _m_over_t2_axis(args: argparse.Namespace) -> np.ndarray:
    if args.m_over_t2_start is None or args.m_over_t2_stop is None:
        start, stop = _transition_m_over_t2(float(args.phi0))
    else:
        start = float(args.m_over_t2_start)
        stop = float(args.m_over_t2_stop)
    return np.linspace(start, stop, int(args.nm), dtype=np.float64)


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


def _build_m0_sweep(
    *,
    base_config: QXTIConfig,
    m_over_t2_values: np.ndarray,
    omega_axis: np.ndarray,
    selected_component: tuple[int, int, int],
    t2_for_axis: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, tuple[str, ...], list[dict[str, Any]]]:
    sigma: np.ndarray | None = None
    labels: tuple[str, ...] | None = None
    metadata: list[dict[str, Any]] = []
    timer = ProgressTimer(total=int(m_over_t2_values.size), min_completed_for_eta=max(2, min(m_over_t2_values.size, 5)))

    for index, m_over_t2 in enumerate(m_over_t2_values):
        row_start = time.perf_counter()
        cfg = _point_config(
            base_config=base_config,
            phi0=float(args.phi0),
            m_over_t2=float(m_over_t2),
            t2_for_axis=t2_for_axis,
            args=args,
        )
        if args.verbose_point_logs:
            result = _compute_sigma2_full_gridbased(
                cfg,
                omega_axis,
                components=(selected_component,),
                progress=True,
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                result = _compute_sigma2_full_gridbased(
                    cfg,
                    omega_axis,
                    components=(selected_component,),
                    progress=False,
                )
        tensor = np.asarray(result["sigma_order_2_tensor"], dtype=np.complex128)
        if sigma is None:
            labels = tuple(str(item) for item in result["direction_labels"])
            sigma = np.zeros((m_over_t2_values.size,) + tensor.shape, dtype=np.complex128)
        sigma[index] = tensor
        metadata.append(result)
        timer.advance()
        if not args.quiet:
            print(
                f"[haldane-sigma2-m0] point {index + 1}/{m_over_t2_values.size} "
                f"(M0/t2={m_over_t2:+.5f}) in {format_duration(time.perf_counter() - row_start)}; "
                f"elapsed {format_duration(timer.elapsed_seconds)}, ETA {timer.eta_text()}",
                flush=True,
            )

    if sigma is None or labels is None:
        raise RuntimeError("No se pudo calcular ningun punto del sweep.")
    return sigma, labels, metadata


def _plot_component_sweep(
    *,
    omega_axis: np.ndarray,
    m_over_t2_values: np.ndarray,
    sigma_sweep: np.ndarray,
    component: tuple[int, int, int],
    labels: tuple[str, ...],
    phi0: float,
    value_mode: str,
    output_path: Path,
    dpi: int,
) -> None:
    apply_paper_style()
    cmap = _sweep_colormap()
    norm = Normalize(vmin=float(np.min(m_over_t2_values)), vmax=float(np.max(m_over_t2_values)))
    x_ev = omega_axis * AU_TO_EV
    values = _value_view(sigma_sweep[:, :, component[0], component[1], component[2]], value_mode)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for m_over_t2, curve in zip(m_over_t2_values, values, strict=True):
        ax.plot(x_ev, curve, color=cmap(norm(float(m_over_t2))), linewidth=1.35, alpha=0.95)
    if value_mode in {"real", "imag"}:
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9, alpha=0.75)

    upper, lower = _transition_m_over_t2(phi0)
    ax.set_xlabel(r"$\hbar\omega\;(\mathrm{eV})$")
    ax.set_ylabel(_component_latex(component, labels, value_mode) + r"$\;(\mathrm{a.u.})$")
    ax.set_title(
        _component_latex(component, labels, value_mode)
        + rf"$,\;\phi_0={phi0:.3f}\,\mathrm{{rad}},\;M_0/t_2:{upper:.3f}\to{lower:.3f}$"
    )
    ax.set_xlim(float(x_ev.min()), float(x_ev.max()))
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    colorbar.set_label(r"$M_0/t_2$")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep de M0/t2 para sigma^(2)_ijk del Haldane a phi0 fijo."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Input base de Haldane.")
    parser.add_argument("--phi0", type=float, default=0.5 * np.pi, help="Phi fijo en radianes. Default: pi/2.")
    parser.add_argument("--nm", type=int, default=81, help="Numero de valores de M0/t2.")
    parser.add_argument("--m-over-t2-start", type=float, default=None, help="Inicio del sweep. Default: +3sqrt(3)sin(phi0).")
    parser.add_argument("--m-over-t2-stop", type=float, default=None, help="Final del sweep. Default: -3sqrt(3)sin(phi0).")
    parser.add_argument("--nomega", type=int, default=160, help="Numero de frecuencias.")
    parser.add_argument("--omega-min", type=float, default=0.005, help="Frecuencia laser minima en a.u.")
    parser.add_argument("--omega-max", type=float, default=0.18, help="Frecuencia laser maxima en a.u.")
    parser.add_argument("--omega-min-ev", type=float, default=None, help="Frecuencia laser minima en eV.")
    parser.add_argument("--omega-max-ev", type=float, default=None, help="Frecuencia laser maxima en eV.")
    parser.add_argument("--kpoints", type=int, default=61, help="Puntos k por eje.")
    parser.add_argument("--component", default="xyy", help="Componente sigma^(2)_ijk. Default: xyy.")
    parser.add_argument("--value", choices=("real", "imag", "abs"), default="real", help="Cantidad que se grafica.")
    parser.add_argument("--gamma", type=float, default=None, help="Override gamma=1/T2 en a.u.")
    parser.add_argument("--coherence-time", type=float, default=None, help="Override T2 en a.u.")
    parser.add_argument("--temperature-au", type=float, default=None, help="Override temperatura en a.u.")
    parser.add_argument("--fermi-level", type=float, default=None, help="Override nivel de Fermi en a.u.")
    parser.add_argument("--distribution", default=None, help="Override distribucion de ocupacion.")
    parser.add_argument("--t1", type=float, default=None, help="Override t1 (a.u.).")
    parser.add_argument("--t2", type=float, default=None, help="Override t2 (a.u.).")
    parser.add_argument("--a0", type=float, default=None, help="Override a0 (a.u.).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Carpeta de salida.")
    parser.add_argument("--dataset-name", default="haldane_sigma2_vs_m0.npz", help="Nombre del dataset.")
    parser.add_argument("--dpi", type=int, default=300, help="Resolucion PNG.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs por punto.")
    parser.add_argument("--verbose-point-logs", action="store_true", help="Muestra logs internos de cada punto.")
    args = parser.parse_args()

    if args.nm <= 1 or args.nomega <= 1 or args.kpoints <= 1:
        raise SystemExit("nm, nomega y kpoints deben ser mayores que 1.")
    if args.gamma is not None and args.coherence_time is not None:
        raise SystemExit("Usa solo uno de --gamma o --coherence-time.")

    base_config = QXTIConfig.from_file(Path(args.config))
    t2_for_axis = float(args.t2) if args.t2 is not None else _config_param(base_config, "t2", 0.025)
    if abs(t2_for_axis) <= 1.0e-15:
        raise SystemExit("t2 no puede ser cero porque el eje del sweep es M0/t2.")

    selected_component = _parse_component(args.component, ("x", "y"))
    selected_name = _component_name(selected_component, ("x", "y"))
    m_over_t2_values = _m_over_t2_axis(args)
    omega_axis = _omega_axis_from_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[haldane-sigma2-m0] sweep "
        f"phi0={args.phi0:.6f} rad, M0/t2=[{m_over_t2_values[0]:+.5f} -> {m_over_t2_values[-1]:+.5f}], "
        f"nm={m_over_t2_values.size}, nomega={omega_axis.size}, k={args.kpoints}x{args.kpoints}",
        flush=True,
    )
    start = time.perf_counter()
    sigma_sweep, labels, metadata = _build_m0_sweep(
        base_config=base_config,
        m_over_t2_values=m_over_t2_values,
        omega_axis=omega_axis,
        selected_component=selected_component,
        t2_for_axis=t2_for_axis,
        args=args,
    )
    elapsed = time.perf_counter() - start
    selected_component = _parse_component(args.component, labels)
    selected_name = _component_name(selected_component, labels)

    data_path = output_dir / args.dataset_name
    np.savez_compressed(
        data_path,
        m_over_t2_axis=m_over_t2_values,
        m0_axis=m_over_t2_values * float(t2_for_axis),
        phi0=float(args.phi0),
        omega_axis=omega_axis,
        omega_axis_ev=omega_axis * AU_TO_EV,
        sigma_order_2_tensor=sigma_sweep,
        sigma_order_2_real=np.real(sigma_sweep),
        sigma_order_2_imag=np.imag(sigma_sweep),
        selected_component=np.asarray(selected_component, dtype=np.int16),
        selected_component_name=str(selected_name),
        direction_labels=np.asarray(labels),
        component_order="sigma[m_index, omega_index, i, j, k]",
        transition_upper=float(_transition_m_over_t2(args.phi0)[0]),
        transition_lower=float(_transition_m_over_t2(args.phi0)[1]),
        config_path=str(Path(args.config)),
        t2_for_axis=float(t2_for_axis),
        runtime_seconds=float(elapsed),
        point_runtime_seconds=np.asarray([float(item["runtime_seconds"]) for item in metadata], dtype=np.float64),
    )

    plot_path = output_dir / f"haldane_sigma2_{selected_name}_vs_m0_{args.value}.png"
    _plot_component_sweep(
        omega_axis=omega_axis,
        m_over_t2_values=m_over_t2_values,
        sigma_sweep=sigma_sweep,
        component=selected_component,
        labels=labels,
        phi0=float(args.phi0),
        value_mode=str(args.value),
        output_path=plot_path,
        dpi=int(args.dpi),
    )

    print(
        f"[haldane-sigma2-m0] listo en {format_duration(elapsed)}.\n"
        f"[haldane-sigma2-m0] data: {data_path}\n"
        f"[haldane-sigma2-m0] plot: {plot_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
