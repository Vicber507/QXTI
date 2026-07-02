#!/usr/bin/env python3
from __future__ import annotations

"""Sweep the Orenstein WSM susceptibility over several Delta values and plot it.

This tool stays OUTSIDE the QXTI core on purpose. It works with the unified
input files introduced in the repo and uses ``main.py <config> -xtp`` for any
missing datasets.

For each requested ``Delta`` it:
1. generates one TEMPORARY derived unified config,
2. optionally runs the dedicated XTP susceptibility workflow,
3. loads ``xtp_susceptibility.npz``,
4. overlays every available susceptibility component for orders 1 and 2,
   using one color per Delta and a shared colorbar,
5. removes the per-Delta intermediate config/output so only the final images
   remain in ``outputs/<base_run_name>_delta_sweep``.

Examples
--------
Reuse existing datasets only:

    python tools/plot_wsm_orenstein_susceptibility_vs_delta.py \
        --base-config inputs/inputParams.wsm_orenstein.cfg \
        --deltas 0.0 0.2 0.5 0.8 1.0

Generate any missing runs with the unified ``-xtp`` mode:

    python tools/plot_wsm_orenstein_susceptibility_vs_delta.py \
        --base-config inputs/inputParams.wsm_orenstein.cfg \
        --deltas 0.0 0.2 0.5 0.8 1.0 \
        --run-missing
"""

import argparse
import configparser
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm

from qxti.core import QXTIConfig
from qxti.data.io import load_dataset_npz
from qxti.graphics.plot_susceptibility_tensor import apply_paper_style


AU_TO_EV = 27.211386245988
AXIS_LABELS = ("x", "y", "z")
DEFAULT_DELTAS = (0.0, 0.2, 0.5, 0.8, 1.0)
DEFAULT_BASE_CONFIG = Path("inputs/inputParams.wsm_orenstein.cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay WSM Orenstein susceptibility components for several Delta "
            "values. The script understands the new unified inputs and can run "
            "missing Delta cases through `main.py <config> -xtp`."
        )
    )
    parser.add_argument(
        "--base-config",
        default=str(DEFAULT_BASE_CONFIG),
        help="Unified Orenstein input used as the template for all Delta cases.",
    )
    parser.add_argument(
        "--deltas",
        nargs="+",
        type=float,
        default=list(DEFAULT_DELTAS),
        help="Delta values to compare. Default: %(default)s",
    )
    parser.add_argument(
        "--run-missing",
        action="store_true",
        help="If a Delta dataset is missing, generate it by running `main.py <generated_cfg> -xtp`.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Recompute every requested Delta even if its dataset already exists.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Directory for the combined Delta-sweep plots. "
            "Default: outputs/<base_run_name>_delta_sweep"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=260,
        help="PNG resolution.",
    )
    return parser.parse_args()


def _sanitize_delta_label(delta: float) -> str:
    value = f"{float(delta):+.6f}"
    return value.replace("+", "p").replace("-", "m").replace(".", "p")


def _component_label(indices: Iterable[int]) -> str:
    return "".join(AXIS_LABELS[int(index)] for index in indices)


def _component_tex(indices: Iterable[int], order: int) -> str:
    return rf"\chi^{{({order})}}_{{\mathrm{{{_component_label(indices)}}}}}"


def _generated_run_name(base_run_name: str, delta: float) -> str:
    return f"{base_run_name}_delta_{_sanitize_delta_label(delta)}"


def _default_output_dir(base_config: Path) -> Path:
    run_name = QXTIConfig.from_file(base_config).run_name()
    return PROJECT_ROOT / "outputs" / f"{run_name}_delta_sweep"


def _read_parser(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read config: {path}")
    return parser


def _write_generated_config(
    *,
    base_config: Path,
    generated_dir: Path,
    delta: float,
) -> Path:
    parser = _read_parser(base_config)
    if "hamiltonian" not in parser:
        raise ValueError(f"Config {base_config} is missing [hamiltonian].")
    if "xtp" not in parser:
        parser["xtp"] = {}

    base_run_name = QXTIConfig.from_file(base_config).run_name()
    run_name = _generated_run_name(base_run_name, delta)
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_path = generated_dir / f"inputParams.{run_name}.cfg"

    parser["hamiltonian"]["Delta"] = str(float(delta))
    parser["xtp"]["susceptibility_enabled"] = "true"
    parser["xtp"]["susceptibility_orders"] = "[1,2]"

    with generated_path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return generated_path


def _dataset_path_for(config_path: Path) -> Path:
    config = QXTIConfig.from_file(config_path).with_standard_output_dirs()
    return Path(config.xtp.susceptibility_output_dir) / "data" / "xtp_susceptibility.npz"


def _run_root_for(config_path: Path) -> Path:
    config = QXTIConfig.from_file(config_path).with_standard_output_dirs()
    return Path(config.xtp.susceptibility_output_dir).parent


def _ensure_dataset(
    *,
    config_path: Path,
    dataset_path: Path,
    run_missing: bool,
    force_rerun: bool,
) -> None:
    if dataset_path.exists() and not force_rerun:
        print(f"[wsm-delta] Reusing saved dataset: {dataset_path}", flush=True)
        return

    if not run_missing and not force_rerun:
        raise FileNotFoundError(
            f"Missing dataset for {config_path.name}: {dataset_path}\n"
            "Run this tool with --run-missing, or generate the case manually with:\n"
            f"  python main.py {config_path} -xtp"
        )

    print(f"[wsm-delta] Running XTP susceptibility for {config_path.name} ...", flush=True)
    subprocess.run(
        [sys.executable, "main.py", str(config_path), "-xtp"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"QXTI finished but the expected dataset was not found: {dataset_path}"
        )


def _load_case(delta: float, dataset_path: Path) -> dict[str, object]:
    data = load_dataset_npz(dataset_path)
    omega_axis = np.asarray(data.get("laser_omega_axis", data.get("omega_axis")), dtype=np.float64)
    if omega_axis.ndim != 1 or omega_axis.size == 0:
        raise ValueError(f"Invalid frequency axis in {dataset_path}")
    case: dict[str, object] = {
        "delta": float(delta),
        "dataset_path": dataset_path,
        "omega_axis": omega_axis,
    }
    for order in (1, 2):
        tensor_key = f"chi_order_{order}_tensor"
        indices_key = f"chi_order_{order}_available_indices"
        if tensor_key in data:
            case[tensor_key] = np.asarray(data[tensor_key], dtype=np.complex128)
            case[indices_key] = np.asarray(
                data.get(indices_key, np.empty((0, order + 1), dtype=np.int64)),
                dtype=np.int64,
            )
    return case


def _validate_cases(cases: list[dict[str, object]]) -> None:
    if not cases:
        raise ValueError("No Delta cases were loaded.")

    reference_omega = np.asarray(cases[0]["omega_axis"], dtype=np.float64)
    for case in cases[1:]:
        omega_axis = np.asarray(case["omega_axis"], dtype=np.float64)
        if omega_axis.shape != reference_omega.shape or not np.allclose(omega_axis, reference_omega, rtol=0.0, atol=1.0e-14):
            raise ValueError(
                "All Delta cases must share the same laser-frequency axis. "
                f"Mismatch found in {case['dataset_path']}."
            )

    for order in (1, 2):
        tensor_key = f"chi_order_{order}_tensor"
        if tensor_key not in cases[0]:
            continue
        reference_indices = np.asarray(cases[0][f"chi_order_{order}_available_indices"], dtype=np.int64)
        for case in cases[1:]:
            if tensor_key not in case:
                raise ValueError(f"Order {order} is missing in {case['dataset_path']}.")
            indices = np.asarray(case[f"chi_order_{order}_available_indices"], dtype=np.int64)
            if indices.shape != reference_indices.shape or not np.array_equal(indices, reference_indices):
                raise ValueError(
                    f"Available susceptibility components for order {order} do not match "
                    f"between Delta cases (mismatch in {case['dataset_path']})."
                )


def _cleanup_intermediate_run(run_root: Path) -> None:
    if run_root.exists():
        shutil.rmtree(run_root)
        print(f"[wsm-delta] Removed intermediate run folder: {run_root}", flush=True)


def _plot_component_vs_delta(
    *,
    omega_ev: np.ndarray,
    deltas: np.ndarray,
    component_values: list[np.ndarray],
    order: int,
    component: tuple[int, ...],
    output_path: Path,
    cmap_name: str = "cividis",
    dpi: int = 260,
) -> Path:
    apply_paper_style()

    cmap = plt.get_cmap(cmap_name)
    delta_min = float(np.min(deltas))
    delta_max = float(np.max(deltas))
    if np.isclose(delta_min, delta_max):
        delta_min -= 0.5
        delta_max += 0.5
    norm = mcolors.Normalize(vmin=delta_min, vmax=delta_max)
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    figure, axes = plt.subplots(3, 1, figsize=(8.8, 8.6), sharex=True, constrained_layout=True)
    labels = (
        rf"$\Re\,{_component_tex(component, order)}$",
        rf"$\Im\,{_component_tex(component, order)}$",
        rf"$\left|{_component_tex(component, order)}\right|$",
    )

    for delta, values in zip(deltas, component_values, strict=True):
        color = cmap(norm(float(delta)))
        axes[0].plot(omega_ev, np.real(values), color=color, linewidth=1.7, alpha=0.95)
        axes[1].plot(omega_ev, np.imag(values), color=color, linewidth=1.7, alpha=0.95)
        axes[2].plot(omega_ev, np.abs(values), color=color, linewidth=1.7, alpha=0.95)

    for axis, ylabel in zip(axes, labels, strict=True):
        axis.axhline(0.0, color="#B8C4D0", linewidth=0.8, zorder=0)
        axis.set_xlim(float(omega_ev.min()), float(omega_ev.max()))
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axis.set_ylabel(ylabel)
    axes[2].set_ylim(bottom=0.0)
    axes[2].set_xlabel(r"$\hbar\omega_\mathrm{laser}\;(\mathrm{eV})$")

    figure.suptitle(
        rf"WSM Orenstein ${_component_tex(component, order)}(\omega)$ for varying $\Delta$",
        fontsize=14,
    )
    colorbar = figure.colorbar(scalar_mappable, ax=axes, pad=0.015)
    colorbar.set_label(r"$\Delta$ (tunes Weyl-node separation)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)
    return output_path


def _plot_all_orders(
    *,
    cases: list[dict[str, object]],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    outputs: list[Path] = []
    omega_axis = np.asarray(cases[0]["omega_axis"], dtype=np.float64)
    omega_ev = omega_axis * AU_TO_EV
    deltas = np.asarray([float(case["delta"]) for case in cases], dtype=np.float64)

    for order in (1, 2):
        tensor_key = f"chi_order_{order}_tensor"
        indices_key = f"chi_order_{order}_available_indices"
        if tensor_key not in cases[0]:
            print(f"[wsm-delta] Order {order} not present in the datasets; skipping.", flush=True)
            continue

        available_indices = np.asarray(cases[0][indices_key], dtype=np.int64)
        order_dir = output_dir / f"order_{order}"
        print(
            f"[wsm-delta] Plotting {available_indices.shape[0]} susceptibility components for order {order} ...",
            flush=True,
        )
        for row, component_indices in enumerate(available_indices, start=1):
            component = tuple(int(value) for value in component_indices.tolist())
            component_label = _component_label(component)
            values_by_delta = [
                np.asarray(case[tensor_key][(slice(None),) + component], dtype=np.complex128)
                for case in cases
            ]
            output_path = order_dir / f"chi_order_{order}_{component_label}_vs_delta.png"
            outputs.append(
                _plot_component_vs_delta(
                    omega_ev=omega_ev,
                    deltas=deltas,
                    component_values=values_by_delta,
                    order=order,
                    component=component,
                    output_path=output_path,
                    dpi=dpi,
                )
            )
            print(
                f"[wsm-delta]   component {row}/{available_indices.shape[0]} -> {output_path.name}",
                flush=True,
            )
    return outputs


def main() -> int:
    args = parse_args()
    base_config = Path(args.base_config).expanduser()
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    resolved_output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else _default_output_dir(base_config)
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    base_qxti = QXTIConfig.from_file(base_config)
    source_file = Path(base_qxti.hamiltonian.source_file).name
    if "wsm_orenstein" not in source_file:
        raise ValueError(
            "This tool is specialized for the unified Orenstein WSM input. "
            f"Expected a hamiltonian source_file like 'wsm_orenstein.py', got '{source_file}'."
        )

    deltas = [float(delta) for delta in args.deltas]
    cases: list[dict[str, object]] = []
    print(f"[wsm-delta] Base config: {base_config}", flush=True)

    with tempfile.TemporaryDirectory(prefix="qxti_wsm_delta_", dir="/private/tmp") as tmpdir:
        generated_dir = Path(tmpdir)
        for delta in deltas:
            generated_config = _write_generated_config(
                base_config=base_config,
                generated_dir=generated_dir,
                delta=delta,
            )
            dataset_path = _dataset_path_for(generated_config)
            run_root = _run_root_for(generated_config)
            _ensure_dataset(
                config_path=generated_config,
                dataset_path=dataset_path,
                run_missing=bool(args.run_missing),
                force_rerun=bool(args.force_rerun),
            )
            cases.append(_load_case(delta, dataset_path))
            _cleanup_intermediate_run(run_root)

    _validate_cases(cases)

    outputs = _plot_all_orders(cases=cases, output_dir=resolved_output_dir, dpi=int(args.dpi))
    print(
        "[wsm-delta] Done.\n"
        f"  Base config: {base_config}\n"
        f"  Output dir:  {resolved_output_dir}\n"
        f"  Figures:     {len(outputs)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
