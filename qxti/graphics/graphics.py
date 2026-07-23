from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep matplotlib/font caches in a writable place that exists on mac/win/linux
# (honours $TMPDIR, which SLURM sets per job).
import tempfile as _tempfile
_qxti_cache = os.path.join(_tempfile.gettempdir(), "qxti_cache")
os.environ.setdefault("MPLCONFIGDIR", _qxti_cache)
os.environ.setdefault("XDG_CACHE_HOME", _qxti_cache)

from qxti.core import QXTIConfig
from qxti.data import (
    ResponseData,
    load_dataset_npz,
    load_rho_orders_from_dat,
    load_rho_orders_from_npy,
)
from qxti.graphics.plot_harmonics import HarmonicGraphics, resolve_harmonic_plot_config
from qxti.graphics.plot_hamiltonian import HamiltonianGraphics
from qxti.graphics.plot_response import ResponseGraphics, resolve_response_plot_config
from qxti.graphics.plot_susceptibility_tensor import (
    SusceptibilityTensorPlotter,
    resolve_susceptibility_plot_config,
    to_helicity_basis,
)


# Default harmonic-axis cap for the full non-perturbative tddm engine, whose
# spectrum is broadband (not limited by [cmd] max_order). Shows up to harmonic 10
# (the +0.5 keeps the H10 peak fully inside the axis).
TDDM_DEFAULT_MAX_HARMONIC_ORDER = 10.5

DEFAULT_HAMILTONIAN_PLOTS = (
    "band_structure_2d",
    "band_surface_3d",
    "velocity_2d",
    "velocity_field_3d",
    "velocity_magnitude",
)


def _load_standardized_config(config_path: str | Path) -> QXTIConfig:
    """Load a config and standardize its output dirs to outputs/<model>/{cmd,xtp,
    hamiltonian} so graphics reads from the same paths main.py writes to. Guarded
    with hasattr so SimpleNamespace test mocks (which patch from_file) still work.
    """
    config = QXTIConfig.from_file(config_path)
    if hasattr(config, "with_standard_output_dirs"):
        config = config.with_standard_output_dirs()
    return config


def plot_hamiltonian_graphics_from_saved_data(
    config_path: str | Path,
) -> dict[str, Path]:
    config = _load_standardized_config(config_path)
    plot_cfg = config.hamiltonian_plots
    output_dir = Path(plot_cfg.output_dir)
    data_dir = output_dir / "data"
    requested_plots = plot_cfg.plots or DEFAULT_HAMILTONIAN_PLOTS
    outputs: dict[str, Path] = {}

    for plot_name in requested_plots:
        normalized = _normalize_plot_name(plot_name)
        dataset_path = data_dir / f"{normalized}.npz"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Missing Hamiltonian dataset '{dataset_path}'. "
                "Run `python main.py inputParams.cfg` first."
            )
        data = load_dataset_npz(dataset_path)
        print(f"[graphics] plotting Hamiltonian dataset '{dataset_path.name}'.")

        if normalized == "band_structure_2d":
            outputs[normalized] = HamiltonianGraphics.plot_band_structure_2d(
                data,
                output_dir / f"{normalized}.png",
            )
        elif normalized == "band_surface_3d":
            outputs[normalized] = HamiltonianGraphics.plot_band_surface_3d(
                data,
                output_dir / f"{normalized}.png",
                style=plot_cfg.surface_style,
            )
        elif normalized == "velocity_2d":
            outputs[normalized] = HamiltonianGraphics.plot_velocity_2d(
                data,
                output_dir / f"{normalized}.png",
            )
        elif normalized == "velocity_field_3d":
            outputs[normalized] = HamiltonianGraphics.plot_velocity_field_3d(
                data,
                output_dir / f"{normalized}.png",
                stride=plot_cfg.quiver_stride,
            )
        elif normalized == "velocity_magnitude":
            outputs[normalized] = HamiltonianGraphics.plot_velocity_magnitude(
                data,
                output_dir / f"{normalized}.png",
            )
        else:
            raise ValueError(f"Unsupported Hamiltonian plot '{plot_name}'.")

    return outputs


def plot_response_graphics_from_saved_data(
    config_path: str | Path,
    *,
    plot_config: dict[str, object] | None = None,
) -> dict[str, Path]:
    config = _load_standardized_config(config_path)
    output_dir = Path(config.cmd.output_dir)
    data_dir = output_dir / "data"
    resolved_plot_config = resolve_response_plot_config(plot_config)
    outputs: dict[str, Path] = {}

    requested_orders = _resolve_requested_orders(resolved_plot_config.get("orders"))
    population_cfg = resolved_plot_config["population"]
    coherence_cfg = resolved_plot_config["coherence"]
    population_enabled = bool(config.cmd.save_population_dataset)
    coherence_enabled = bool(config.cmd.save_coherence_dataset)

    if not population_enabled and not coherence_enabled:
        print("[graphics] response population/coherence datasets are disabled by config; skipping response graphics.")
        return {}

    population_dataset_path = data_dir / "population_kx_ky_per_band.npz"
    coherence_dataset_path = data_dir / "coherence_kx_ky_per_pair.npz"
    population_kmap_data: dict[str, object] | None = None
    coherence_kmap_data: dict[str, object] | None = None
    need_population_from_rho = population_enabled and not population_dataset_path.exists()
    need_coherence_from_rho = coherence_enabled and not coherence_dataset_path.exists()

    if population_enabled and population_dataset_path.exists():
        print("[graphics] plotting population response graphics from compact saved dataset.")
        population_kmap_data = _load_population_response_dataset(
            population_dataset_path,
            value_mode=str(population_cfg.get("value_mode", "delta")),
        )
    elif not population_enabled:
        print("[graphics] population response dataset disabled by config; skipping population graphics.")

    if coherence_enabled and coherence_dataset_path.exists():
        print("[graphics] plotting coherence response graphics from compact saved dataset.")
        coherence_kmap_data = _load_coherence_response_dataset(
            coherence_dataset_path,
            component=str(coherence_cfg["component"]),
        )
    elif not coherence_enabled:
        print("[graphics] coherence response dataset disabled by config; skipping coherence graphics.")

    if need_population_from_rho or need_coherence_from_rho:
        rho_orders, time_axis, k_points, kx_values, ky_values, kz_values = _load_response_fallback_data(
            config_path
        )
        if not rho_orders:
            raise FileNotFoundError(
                "Missing response datasets and no saved rho_order_*.npy files were found. "
                "Legacy rho_order_*.dat files are still supported if they already exist. "
                "Run `python main.py inputParams.cfg` first."
            )
        print("[graphics] plotting response graphics from saved rho_order tensors.")

        if need_population_from_rho:
            try:
                population_kmap_data = ResponseData.population_kxky_animation_data_from_saved_rho(
                    rho_orders,
                    time_axis=time_axis,
                    kx_values=kx_values,
                    ky_values=ky_values,
                    kz_values=kz_values,
                    orders=requested_orders,
                    value_mode=str(population_cfg.get("value_mode", "delta")),
                )
            except ValueError as exc:
                population_kmap_data = None
                print(f"[graphics] skipped population kx-ky data: {exc}")

        if need_coherence_from_rho:
            try:
                coherence_kmap_data = ResponseData.coherence_kxky_animation_data_from_saved_rho(
                    rho_orders,
                    time_axis=time_axis,
                    kx_values=kx_values,
                    ky_values=ky_values,
                    kz_values=kz_values,
                    orders=requested_orders,
                    component=str(coherence_cfg["component"]),
                )
            except ValueError as exc:
                coherence_kmap_data = None
                print(f"[graphics] skipped coherence kx-ky data: {exc}")

    if population_kmap_data is not None and bool(population_cfg["snapshots"]["enabled"]):
        population_snapshot_indices = ResponseGraphics.resolve_snapshot_indices(
            np.asarray(population_kmap_data["time_axis"], dtype=float),
            num_snapshots=int(population_cfg["snapshots"]["num_snapshots"]),
            snapshot_times=list(population_cfg["snapshots"]["snapshot_times"]),
            snapshot_indices=list(population_cfg["snapshots"]["snapshot_indices"]),
        )
        outputs["rho_population_snapshots"] = ResponseGraphics.plot_population_snapshots(
            population_kmap_data,
            output_dir / str(population_cfg["snapshots"]["output_file"]),
            snapshot_indices=population_snapshot_indices,
            cmap=str(population_cfg["snapshots"]["cmap"]),
            center_zero=bool(population_cfg["snapshots"].get("center_zero", False)),
            contrast_percentile=float(population_cfg["snapshots"].get("contrast_percentile", 100.0)),
        )

    if coherence_kmap_data is not None and bool(coherence_cfg["snapshots"]["enabled"]):
        coherence_snapshot_indices = ResponseGraphics.resolve_snapshot_indices(
            np.asarray(coherence_kmap_data["time_axis"], dtype=float),
            num_snapshots=int(coherence_cfg["snapshots"]["num_snapshots"]),
            snapshot_times=list(coherence_cfg["snapshots"]["snapshot_times"]),
            snapshot_indices=list(coherence_cfg["snapshots"]["snapshot_indices"]),
        )
        outputs["rho_coherence_snapshots"] = ResponseGraphics.plot_coherence_snapshots(
            coherence_kmap_data,
            output_dir / str(coherence_cfg["snapshots"]["output_file"]),
            snapshot_indices=coherence_snapshot_indices,
            cmap=str(coherence_cfg["snapshots"]["cmap"]),
            center_zero=bool(coherence_cfg["snapshots"].get("center_zero", False)),
            contrast_percentile=float(coherence_cfg["snapshots"].get("contrast_percentile", 100.0)),
        )

    if population_kmap_data is not None and bool(population_cfg["video"]["enabled"]):
        print("[graphics] static population plots completed; generating population video.")
        try:
            outputs["rho_population_kxky_video"] = ResponseGraphics.animate_population_kxky_maps(
                population_kmap_data,
                output_dir / str(population_cfg["video"]["output_file"]),
                fps=int(population_cfg["video"]["fps"]),
                duration_seconds=None if population_cfg["video"].get("duration_seconds") is None else float(population_cfg["video"]["duration_seconds"]),
                frame_stride=int(population_cfg["video"]["frame_stride"]),
                cmap=str(population_cfg["video"]["cmap"]),
                center_zero=bool(population_cfg["video"].get("center_zero", False)),
                contrast_percentile=float(population_cfg["video"].get("contrast_percentile", 100.0)),
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[graphics] skipped population video: {exc}")

    if coherence_kmap_data is not None and bool(coherence_cfg["video"]["enabled"]):
        print("[graphics] static coherence plots completed; generating coherence video.")
        try:
            outputs["rho_coherence_kxky_video"] = ResponseGraphics.animate_coherence_kxky_maps(
                coherence_kmap_data,
                output_dir / str(coherence_cfg["video"]["output_file"]),
                fps=int(coherence_cfg["video"]["fps"]),
                duration_seconds=None if coherence_cfg["video"].get("duration_seconds") is None else float(coherence_cfg["video"]["duration_seconds"]),
                frame_stride=int(coherence_cfg["video"]["frame_stride"]),
                cmap=str(coherence_cfg["video"]["cmap"]),
                center_zero=bool(coherence_cfg["video"].get("center_zero", False)),
                contrast_percentile=float(coherence_cfg["video"].get("contrast_percentile", 100.0)),
            )
        except (RuntimeError, ValueError) as exc:
            print(f"[graphics] skipped coherence video: {exc}")

    return outputs


def _load_population_response_dataset(
    dataset_path: Path,
    *,
    value_mode: str,
) -> dict[str, object]:
    data = load_dataset_npz(dataset_path)
    dataset = dict(data)
    dataset["value_mode"] = value_mode
    frames = np.asarray(dataset["population_frames"], dtype=float)
    if value_mode.strip().lower() == "delta" and dataset.get("equilibrium_population_frame") is not None:
        equilibrium = np.asarray(dataset["equilibrium_population_frame"], dtype=float)
        dataset["population_frames"] = frames - equilibrium[np.newaxis, :, :, :]
    else:
        dataset["population_frames"] = frames
    return dataset


def _load_coherence_response_dataset(
    dataset_path: Path,
    *,
    component: str,
) -> dict[str, object]:
    data = load_dataset_npz(dataset_path)
    dataset = dict(data)
    dataset["component"] = component
    frames = dataset.get("coherence_frames_complex", dataset.get("coherence_frames"))
    if frames is None:
        raise KeyError("Missing coherence frames in saved dataset.")
    values = np.asarray(frames)
    key = component.strip().lower()
    if np.iscomplexobj(values):
        if key == "magnitude":
            dataset["coherence_frames"] = np.abs(values)
        elif key == "real":
            dataset["coherence_frames"] = np.real(values)
        elif key == "imag":
            dataset["coherence_frames"] = np.imag(values)
        else:
            raise ValueError("coherence component must be one of: magnitude, real, imag.")
    else:
        dataset["coherence_frames"] = np.asarray(values, dtype=float)
    return dataset


def plot_harmonic_graphics_from_saved_data(
    config_path: str | Path,
    *,
    plot_config: dict[str, object] | None = None,
) -> dict[str, Path]:
    config = _load_standardized_config(config_path)
    output_dir = Path(config.cmd.output_dir)
    data_dir = output_dir / "data"
    resolved_plot_config = resolve_harmonic_plot_config(plot_config)
    # Show every harmonic that was actually computed by bumping the display cap.
    #   * pfddm / ptddm (perturbative): the harmonic order IS the highest computed
    #     order, so cap at [cmd] max_order (+0.5). The 3.5 default would otherwise
    #     hide orders 4+ from a 7-order run.
    #   * tddm (full non-perturbative): the solve produces a BROADBAND spectrum (all
    #     harmonics up to Nyquist); max_order there only sizes the dataset schema,
    #     NOT the harmonic content. Capping at max_order would hide the real H4+
    #     peaks, so default the tddm display to harmonic 10 (raise, never lower).
    _method = str(getattr(config.cmd, "response_method", "ptddm"))
    _max_h = float(config.cmd.max_order) + 0.5
    if _method in ("tddm", "all"):
        _max_h = max(_max_h, TDDM_DEFAULT_MAX_HARMONIC_ORDER)
    for _sec in resolved_plot_config.values():
        if isinstance(_sec, dict) and _sec.get("max_harmonic_order") is not None:
            _sec["max_harmonic_order"] = max(float(_sec["max_harmonic_order"]), _max_h)
    outputs: dict[str, Path] = {}

    if not bool(config.cmd.save_xtp_dataset):
        dataset_path = data_dir / "current_spectrum.npz"
        if not dataset_path.exists():
            print("[graphics] XTP dataset disabled by config and no saved current_spectrum.npz was found; skipping harmonic graphics.")
            return outputs

    dataset_name = None
    for section_name in (
        "field_current_time",
        "current_total_spectrum",
        "current_components_spectrum",
        "current_inter_intra_spectrum",
        "current_circular_spectrum",
        "current_overview_spectrum",
    ):
        section_cfg = resolved_plot_config[section_name]
        if bool(section_cfg["enabled"]):
            dataset_name = str(section_cfg["dataset_file"])
            break
    if dataset_name is None:
        return outputs

    dataset_path = data_dir / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing harmonic dataset '{dataset_path}'. "
            "Run `python main.py inputParams.cfg` first."
        )

    data = load_dataset_npz(dataset_path)
    data = _augment_legacy_harmonic_dataset(data)
    print(f"[graphics] plotting harmonic dataset '{dataset_path.name}'.")
    reference_omega = _harmonic_reference_omega(config)

    field_current_cfg = resolved_plot_config["field_current_time"]
    if bool(field_current_cfg["enabled"]):
        outputs["field_current_time"] = HarmonicGraphics.plot_field_current_time_comparison(
            np.asarray(data["time_axis"], dtype=float),
            np.asarray(data["electric_field_time"], dtype=float),
            np.asarray(data["current_time"], dtype=float),
            output_dir / str(field_current_cfg["output_file"]),
            directions=tuple(str(direction) for direction in field_current_cfg["directions"]),
            include_total=bool(field_current_cfg["include_total"]),
            combine_planar=bool(field_current_cfg.get("combine_planar", False)),
        )

    total_cfg = resolved_plot_config["current_total_spectrum"]
    if bool(total_cfg["enabled"]):
        outputs["current_total_spectrum"] = HarmonicGraphics.plot_total_current_spectrum(
            np.asarray(data["omega_axis"], dtype=float),
            np.asarray(data["current_total_magnitude"], dtype=float),
            output_dir / str(total_cfg["output_file"]),
            orders=tuple(int(order) for order in data.get("orders", ())),
            positive_only=bool(total_cfg["positive_only"]),
            omega_min=None if total_cfg["omega_min"] is None else float(total_cfg["omega_min"]),
            omega_max=None if total_cfg["omega_max"] is None else float(total_cfg["omega_max"]),
            fundamental_omega=reference_omega,
            use_harmonic_order=bool(total_cfg.get("use_harmonic_order", False)),
            max_harmonic_order=None if total_cfg.get("max_harmonic_order") is None else float(total_cfg["max_harmonic_order"]),
            log_scale=bool(total_cfg["log_scale"]),
        )

    components_cfg = resolved_plot_config["current_components_spectrum"]
    if bool(components_cfg["enabled"]):
        outputs["current_components_spectrum"] = HarmonicGraphics.plot_current_magnitude_spectrum(
            np.asarray(data["omega_axis"], dtype=float),
            np.asarray(data["current_spectrum"], dtype=np.complex128),
            output_dir / str(components_cfg["output_file"]),
            orders=tuple(int(order) for order in data.get("orders", ())),
            directions=tuple(str(direction) for direction in components_cfg["directions"]),
            positive_only=bool(components_cfg["positive_only"]),
            omega_min=None if components_cfg["omega_min"] is None else float(components_cfg["omega_min"]),
            omega_max=None if components_cfg["omega_max"] is None else float(components_cfg["omega_max"]),
            fundamental_omega=reference_omega,
            use_harmonic_order=bool(components_cfg.get("use_harmonic_order", False)),
            max_harmonic_order=None if components_cfg.get("max_harmonic_order") is None else float(components_cfg["max_harmonic_order"]),
            log_scale=bool(components_cfg["log_scale"]),
        )

    inter_intra_cfg = resolved_plot_config["current_inter_intra_spectrum"]
    if bool(inter_intra_cfg["enabled"]) and bool(data.get("current_decomposition_available", False)):
        outputs["current_inter_intra_spectrum"] = HarmonicGraphics.plot_inter_intra_current_spectrum(
            np.asarray(data["omega_axis"], dtype=float),
            np.asarray(data["current_total_magnitude_intraband"], dtype=float),
            np.asarray(data["current_total_magnitude_interband"], dtype=float),
            output_dir / str(inter_intra_cfg["output_file"]),
            orders=tuple(int(order) for order in data.get("orders", ())),
            positive_only=bool(inter_intra_cfg["positive_only"]),
            omega_min=None if inter_intra_cfg["omega_min"] is None else float(inter_intra_cfg["omega_min"]),
            omega_max=None if inter_intra_cfg["omega_max"] is None else float(inter_intra_cfg["omega_max"]),
            fundamental_omega=reference_omega,
            use_harmonic_order=bool(inter_intra_cfg.get("use_harmonic_order", False)),
            max_harmonic_order=None if inter_intra_cfg.get("max_harmonic_order") is None else float(inter_intra_cfg["max_harmonic_order"]),
            log_scale=bool(inter_intra_cfg["log_scale"]),
        )

    circular_cfg = resolved_plot_config["current_circular_spectrum"]
    if bool(circular_cfg["enabled"]):
        outputs["current_circular_spectrum"] = HarmonicGraphics.plot_circular_current_spectrum(
            np.asarray(data["omega_axis"], dtype=float),
            np.asarray(data["current_spectrum"], dtype=np.complex128),
            output_dir / str(circular_cfg["output_file"]),
            orders=tuple(int(order) for order in data.get("orders", ())),
            positive_only=bool(circular_cfg["positive_only"]),
            omega_min=None if circular_cfg["omega_min"] is None else float(circular_cfg["omega_min"]),
            omega_max=None if circular_cfg["omega_max"] is None else float(circular_cfg["omega_max"]),
            fundamental_omega=reference_omega,
            use_harmonic_order=bool(circular_cfg.get("use_harmonic_order", False)),
            max_harmonic_order=None if circular_cfg.get("max_harmonic_order") is None else float(circular_cfg["max_harmonic_order"]),
            log_scale=bool(circular_cfg["log_scale"]),
        )

    overview_cfg = resolved_plot_config["current_overview_spectrum"]
    if bool(overview_cfg["enabled"]):
        outputs["current_overview_spectrum"] = HarmonicGraphics.plot_current_overview_spectrum(
            np.asarray(data["omega_axis"], dtype=float),
            np.asarray(data["current_spectrum"], dtype=np.complex128),
            np.asarray(data["current_total_magnitude"], dtype=float),
            output_dir / str(overview_cfg["output_file"]),
            orders=tuple(int(order) for order in data.get("orders", ())),
            directions=tuple(str(direction) for direction in overview_cfg.get("directions", ("x", "y", "z"))),
            intraband_magnitude=(
                np.asarray(data["current_total_magnitude_intraband"], dtype=float)
                if bool(data.get("current_decomposition_available", False))
                else None
            ),
            interband_magnitude=(
                np.asarray(data["current_total_magnitude_interband"], dtype=float)
                if bool(data.get("current_decomposition_available", False))
                else None
            ),
            positive_only=bool(overview_cfg["positive_only"]),
            omega_min=None if overview_cfg["omega_min"] is None else float(overview_cfg["omega_min"]),
            omega_max=None if overview_cfg["omega_max"] is None else float(overview_cfg["omega_max"]),
            fundamental_omega=reference_omega,
            use_harmonic_order=bool(overview_cfg.get("use_harmonic_order", False)),
            max_harmonic_order=None if overview_cfg.get("max_harmonic_order") is None else float(overview_cfg["max_harmonic_order"]),
            log_scale=bool(overview_cfg["log_scale"]),
        )

    return outputs


def _emit_tensor_plots(
    *,
    tensor: np.ndarray,
    omega_axis: np.ndarray,
    tensor_name: str,
    direction_labels: tuple[str, ...],
    available_components: list[tuple[int, ...]] | None,
    dpi: int,
    include_ev_axis: bool,
    x_label: str,
    argument_label: str,
    base_dir: Path,
    key_prefix: str,
    do_overview: bool,
    do_grid: bool,
    do_components: bool,
) -> dict[str, Path]:
    """Emit overview/grid (in ``base_dir/overview``) and individual component
    plots (in ``base_dir/components``) for one response tensor.

    Used for both the cartesian and helicity bases, and for chi and sigma.
    """
    overview_dir = base_dir / "overview"
    components_dir = base_dir / "components"
    plotter = SusceptibilityTensorPlotter(
        x_axis=omega_axis,
        tensor=tensor,
        output_dir=base_dir,
        x_label=x_label,
        argument_label=argument_label,
        tensor_name=tensor_name,
        direction_labels=direction_labels,
        available_components=available_components or None,
        dpi=dpi,
        include_ev_axis=include_ev_axis,
    )
    out: dict[str, Path] = {}
    if do_overview:
        out[f"{key_prefix}_overview"] = plotter.plot_overview(
            output_path=overview_dir / f"{tensor_name}_overview.png"
        )
    if do_grid:
        out[f"{key_prefix}_grid"] = plotter.plot_grid(
            output_path=overview_dir / f"{tensor_name}_grid.png"
        )
    if do_components:
        modulus_dir = components_dir / "modulus"
        for component in plotter.component_indices():
            label = plotter._component_label(component)
            # Sanitize the filename (helicity labels use +/- which are kept in
            # the LaTeX titles but mapped to p/m in file names).
            file_label = label.replace("+", "p").replace("-", "m")
            out[f"{key_prefix}_{label}"] = plotter.plot_component(
                component, output_path=components_dir / f"{tensor_name}_{file_label}.png"
            )
            # Extra modulus-only plot per component (y-axis from 0, paper style).
            out[f"{key_prefix}_{label}_modulus"] = plotter.plot_component_modulus(
                component, output_path=modulus_dir / f"{tensor_name}_{file_label}_modulus.png"
            )
    return out


def plot_susceptibility_graphics_from_saved_data(
    config_path: str | Path,
    *,
    plot_config: dict[str, object] | None = None,
) -> dict[str, Path]:
    config = _load_standardized_config(config_path)
    output_dir = _susceptibility_plot_output_dir(config)
    data_dir = Path(config.xtp.susceptibility_output_dir) / "data"
    resolved_plot_config = _resolve_susceptibility_plot_config_from_xtp(config, plot_config)
    dataset_path = _resolve_susceptibility_dataset_path(
        data_dir=data_dir,
        dataset_file=str(resolved_plot_config["dataset_file"]),
    )
    outputs: dict[str, Path] = {}

    # A susceptibility config is identified by susceptibility_enabled (the same
    # flag used to run the sweep). Calling graphics on such a config generates
    # its plots; on a non-susceptibility config it is silently skipped. No
    # separate plot flag is needed in the input.
    if not bool(config.xtp.susceptibility_enabled):
        print("[graphics] not a susceptibility config (susceptibility_enabled=false); skipping susceptibility graphics.")
        return outputs
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing susceptibility dataset '{dataset_path}'. "
            f"Run `python main.py {Path(config_path).name}` first."
        )

    data = load_dataset_npz(dataset_path)
    print(f"[graphics] plotting susceptibility dataset '{dataset_path.name}'.")

    saved_orders = tuple(int(order) for order in data.get("orders", ()))
    requested_orders = _resolve_requested_orders(resolved_plot_config.get("orders"))
    if requested_orders is None:
        selected_orders = saved_orders
    else:
        selected_orders = tuple(order for order in saved_orders if order in requested_orders)
    if not selected_orders:
        print("[graphics] no susceptibility orders matched the requested configuration; skipping susceptibility graphics.")
        return outputs

    omega_axis_full = np.asarray(
        data.get("laser_omega_axis", data.get("omega_axis", np.empty(0, dtype=float))),
        dtype=float,
    )
    mask = _select_frequency_mask(
        omega_axis_full,
        positive_only=bool(resolved_plot_config["positive_only"]),
        omega_min=None if resolved_plot_config["omega_min"] is None else float(resolved_plot_config["omega_min"]),
        omega_max=None if resolved_plot_config["omega_max"] is None else float(resolved_plot_config["omega_max"]),
    )
    direction_labels = tuple(str(label) for label in data.get("direction_labels", ("x", "y", "z")))
    dpi = int(resolved_plot_config["dpi"])

    dimension = len(direction_labels)
    # Plot the laser frequency axis in electron-volts (primary axis). The
    # dataset stores omega in atomic units, so convert here. The redundant
    # secondary eV axis is therefore disabled.
    AU_TO_EV = 27.211386245988
    include_ev_axis = False
    x_label = r"$\omega_\mathrm{laser}\;(\mathrm{eV})$"
    argument_label = r"\omega_\mathrm{laser}"

    overview_cfg = resolved_plot_config["overview"]
    grid_cfg = resolved_plot_config["grid"]
    components_cfg = resolved_plot_config["components"]
    do_overview = bool(overview_cfg["enabled"])
    do_grid = bool(grid_cfg["enabled"])
    do_components = bool(components_cfg["enabled"])
    conductivity_cfg = resolved_plot_config.get("conductivity", {})
    conductivity_on = isinstance(conductivity_cfg, dict) and bool(conductivity_cfg.get("enabled", False))

    omega_axis = omega_axis_full[mask] * AU_TO_EV  # atomic units -> eV for plotting

    def _components_from_indices(key: str, fallback: np.ndarray | None) -> list[tuple[int, ...]]:
        arr = np.asarray(data.get(key, fallback if fallback is not None else np.empty((0,), dtype=int)), dtype=int)
        return [
            tuple(int(index) for index in row)
            for row in np.atleast_2d(arr)
            if arr.size > 0
        ]

    for order in selected_orders:
        tensor_key = f"chi_order_{order}_tensor"
        if tensor_key not in data:
            print(f"[graphics] missing susceptibility tensor for order {order}; skipping.")
            continue

        order_dir = output_dir / f"order_{order}"
        cartesian_dir = order_dir / "cartesian"
        helicity_dir = order_dir / "helicity"

        # ---- susceptibility chi ----
        chi_cart = np.asarray(data[tensor_key], dtype=np.complex128)[mask]
        chi_components = _components_from_indices(
            f"chi_order_{order}_available_indices", np.empty((0, order + 1), dtype=int)
        )
        outputs.update(
            {
                f"susceptibility_order_{order}_{k}": v
                for k, v in _emit_tensor_plots(
                    tensor=chi_cart, omega_axis=omega_axis, tensor_name=f"chi{order}",
                    direction_labels=direction_labels, available_components=chi_components,
                    dpi=dpi, include_ev_axis=include_ev_axis, x_label=x_label,
                    argument_label=argument_label, base_dir=cartesian_dir, key_prefix="cartesian",
                    do_overview=do_overview, do_grid=do_grid, do_components=do_components,
                ).items()
            }
        )
        # Helicity (circular) basis: only well-defined for the rank-2 linear tensor.
        if order == 1 and dimension >= 2:
            chi_hel, hel_labels = to_helicity_basis(chi_cart, dimension)
            outputs.update(
                {
                    f"susceptibility_order_{order}_{k}": v
                    for k, v in _emit_tensor_plots(
                        tensor=chi_hel, omega_axis=omega_axis, tensor_name=f"chi{order}",
                        direction_labels=hel_labels, available_components=None,
                        dpi=dpi, include_ev_axis=include_ev_axis, x_label=x_label,
                        argument_label=argument_label, base_dir=helicity_dir, key_prefix="helicity",
                        do_overview=do_overview, do_grid=do_grid, do_components=do_components,
                    ).items()
                }
            )

        # ---- conductivity sigma ----
        sigma_key = f"sigma_order_{order}_tensor"
        if conductivity_on and sigma_key in data:
            sigma_cart = np.asarray(data[sigma_key], dtype=np.complex128)[mask]
            sigma_components = _components_from_indices(
                f"sigma_order_{order}_available_indices", np.empty((0, order + 1), dtype=int)
            )
            cond_overview = conductivity_cfg.get("overview", {})
            cond_grid = conductivity_cfg.get("grid", {})
            cond_components = conductivity_cfg.get("components", {})
            sig_do_overview = isinstance(cond_overview, dict) and bool(cond_overview.get("enabled", False))
            sig_do_grid = isinstance(cond_grid, dict) and bool(cond_grid.get("enabled", False))
            sig_do_components = isinstance(cond_components, dict) and bool(cond_components.get("enabled", False))
            outputs.update(
                {
                    f"conductivity_order_{order}_{k}": v
                    for k, v in _emit_tensor_plots(
                        tensor=sigma_cart, omega_axis=omega_axis, tensor_name=f"sigma{order}",
                        direction_labels=direction_labels, available_components=sigma_components,
                        dpi=dpi, include_ev_axis=include_ev_axis, x_label=x_label,
                        argument_label=argument_label, base_dir=cartesian_dir, key_prefix="cartesian",
                        do_overview=sig_do_overview, do_grid=sig_do_grid, do_components=sig_do_components,
                    ).items()
                }
            )
            if order == 1 and dimension >= 2:
                sigma_hel, hel_labels = to_helicity_basis(sigma_cart, dimension)
                outputs.update(
                    {
                        f"conductivity_order_{order}_{k}": v
                        for k, v in _emit_tensor_plots(
                            tensor=sigma_hel, omega_axis=omega_axis, tensor_name=f"sigma{order}",
                            direction_labels=hel_labels, available_components=None,
                            dpi=dpi, include_ev_axis=include_ev_axis, x_label=x_label,
                            argument_label=argument_label, base_dir=helicity_dir, key_prefix="helicity",
                            do_overview=sig_do_overview, do_grid=sig_do_grid, do_components=sig_do_components,
                        ).items()
                    }
                )
        elif conductivity_on and sigma_key not in data:
            print(
                f"[graphics] missing saved conductivity tensor for order {order}; "
                "rerun the susceptibility workflow to generate J/E conductivity plots."
            )

    return outputs


def _susceptibility_plot_output_dir(config: QXTIConfig) -> Path:
    raw = config.xtp.susceptibility_plot_output_dir.strip()
    if raw:
        return Path(raw)
    return Path(config.xtp.susceptibility_output_dir) / "xtp_susceptibility"


def _resolve_susceptibility_dataset_path(*, data_dir: Path, dataset_file: str) -> Path:
    requested_path = data_dir / dataset_file
    if requested_path.exists():
        return requested_path
    if dataset_file == "xtp_susceptibility.npz":
        legacy_path = data_dir / "susceptibility_scan.npz"
        if legacy_path.exists():
            return legacy_path
    return requested_path


def _resolve_susceptibility_plot_config_from_xtp(
    config: QXTIConfig,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    xtp = config.xtp
    base = {
        "dataset_file": xtp.susceptibility_plot_dataset_file,
        "orders": "all" if xtp.susceptibility_plot_orders is None else xtp.susceptibility_plot_orders,
        "positive_only": bool(xtp.susceptibility_plot_positive_only),
        "omega_min": xtp.susceptibility_plot_omega_min,
        "omega_max": xtp.susceptibility_plot_omega_max,
        "dpi": int(xtp.susceptibility_plot_dpi),
        "include_ev_axis": bool(xtp.susceptibility_plot_ev_axis),
        "overview": {
            "enabled": bool(xtp.susceptibility_plot_overview_enabled),
            "output_file_template": "chi{order}_overview.png",
        },
        "grid": {
            "enabled": bool(xtp.susceptibility_plot_grid_enabled),
            "output_file_template": "chi{order}_grid.png",
        },
        "components": {
            "enabled": bool(xtp.susceptibility_plot_components_enabled),
            "output_file_template": "chi{order}_{label}.png",
        },
        "conductivity": {
            "enabled": bool(xtp.susceptibility_plot_conductivity_enabled),
            "overview": {
                "enabled": bool(xtp.susceptibility_plot_overview_enabled),
                "output_file_template": "sigma{order}_overview.png",
            },
            "grid": {
                "enabled": bool(xtp.susceptibility_plot_grid_enabled),
                "output_file_template": "sigma{order}_grid.png",
            },
            "components": {
                "enabled": bool(xtp.susceptibility_plot_components_enabled),
                "output_file_template": "sigma{order}_{label}.png",
            },
        },
    }
    resolved = resolve_susceptibility_plot_config()
    _deep_update_local(resolved, base)
    if overrides:
        _deep_update_local(resolved, overrides)
    return resolved

def _deep_update_local(target: dict[str, object], updates: dict[str, object]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update_local(target[key], value)  # type: ignore[index]
        else:
            target[key] = value


def plot_ldos_graphics_from_saved_data(
    config_path: str | Path,
) -> dict[str, Path]:
    """Generate the density-of-states figures from a saved ``ldos.npz`` dataset.

    Produces the total DOS (with cumulative N(E)), the orbital-projected PDOS
    (when present) and the momentum-resolved spectral map (when present). Reads
    from the standardized ``outputs/<model>/ldos/data/ldos.npz`` path.
    """
    from qxti.graphics.plot_dos import (
        plot_dos_projected,
        plot_dos_surface_bulk,
        plot_dos_total,
        plot_finite_ldos_map,
        plot_finite_spectrum,
        plot_spectral_map,
        plot_spectral_plane,
    )

    config = _load_standardized_config(config_path)
    lcfg = config.ldos
    output_dir = Path(lcfg.output_dir)
    dataset_path = output_dir / "data" / "ldos.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing density-of-states dataset '{dataset_path}'. "
            f"Run `python main.py {Path(config_path).name} -ldos` first."
        )

    data = load_dataset_npz(dataset_path)
    if str(data.get("method", "")) == "surface" and "surface_compare_layers" not in data:
        data = dict(data)
        normal = str(data.get("surface_normal", getattr(lcfg, "surface_normal", "auto"))).strip().lower()
        explicit_layers = int(getattr(lcfg, "surface_compare_layers", 0) or 0)
        if explicit_layers > 0:
            compare_layers = explicit_layers
        elif normal == "x":
            compare_layers = max(int(getattr(lcfg, "finite_nx", 1)), 1)
        elif normal == "y":
            compare_layers = max(int(getattr(lcfg, "finite_ny", 1)), 1)
        else:
            compare_layers = max(
                int(getattr(lcfg, "finite_nx", 1)),
                int(getattr(lcfg, "finite_ny", 1)),
                1,
            )
        data["surface_compare_layers"] = compare_layers
    print(f"[graphics] plotting density-of-states dataset '{dataset_path.name}'.")
    dpi = int(getattr(lcfg, "plot_dpi", 300))
    outputs: dict[str, Path] = {}

    outputs["dos_total"] = plot_dos_total(data, output_dir / "dos_total.png", dpi=dpi)
    if str(data.get("method", "")) == "surface":
        legacy_surface_bulk = plot_dos_surface_bulk(data, output_dir / "dos_surface_bulk.png", dpi=dpi)
        if legacy_surface_bulk is not None:
            outputs["dos_surface_bulk"] = legacy_surface_bulk
    projected = plot_dos_projected(data, output_dir / "dos_projected.png", dpi=dpi)
    if projected is not None:
        outputs["dos_projected"] = projected
        if str(data.get("method", "")) == "surface":
            legacy_projected_bulk = plot_dos_projected(
                data,
                output_dir / "dos_projected_bulk.png",
                dpi=dpi,
            )
            if legacy_projected_bulk is not None:
                outputs["dos_projected_bulk"] = legacy_projected_bulk
    surface_projected = plot_dos_projected(
        data,
        output_dir / "dos_projected_surface.png",
        dpi=dpi,
        pdos_key="surface_pdos",
        total_key="surface_dos",
        total_label=r"surface total $g_\mathrm{surf}(E)$",
        x_label_override=r"$g_{\alpha}^{\mathrm{surf}}(E)\;(\mathrm{states}/\mathrm{eV})$",
    )
    if surface_projected is not None:
        outputs["dos_projected_surface"] = surface_projected
    spectral = plot_spectral_map(data, output_dir / "spectral_map.png", dpi=dpi)
    if spectral is not None:
        outputs["spectral_map"] = spectral
    plane = plot_spectral_plane(data, output_dir / "spectral_plane.png", dpi=dpi)
    if plane is not None:
        outputs["spectral_plane"] = plane
    finite_spectrum = plot_finite_spectrum(data, output_dir / "finite_spectrum.png", dpi=dpi)
    if finite_spectrum is not None:
        outputs["finite_spectrum"] = finite_spectrum
    finite_ldos_map = plot_finite_ldos_map(data, output_dir / "finite_ldos_map.png", dpi=dpi)
    if finite_ldos_map is not None:
        outputs["finite_ldos_map"] = finite_ldos_map
    return outputs


def plot_all_graphics_from_saved_data(
    config_path: str | Path,
) -> dict[str, Path]:
    """Generate every graphics family that has saved data for this config.

    Families whose datasets are missing are skipped (not fatal), so a
    susceptibility-only config produces only its susceptibility plots, an
    HHG-only config produces only its harmonic/response plots, an LDOS-only
    config produces only its DOS plots, and so on.
    """
    outputs: dict[str, Path] = {}
    families = (
        ("hamiltonian", plot_hamiltonian_graphics_from_saved_data),
        ("harmonics", plot_harmonic_graphics_from_saved_data),
        ("susceptibility", plot_susceptibility_graphics_from_saved_data),
        ("ldos", plot_ldos_graphics_from_saved_data),
        ("response", plot_response_graphics_from_saved_data),
    )
    for name, plotter in families:
        try:
            outputs.update(plotter(config_path))
        except FileNotFoundError as exc:
            print(f"[graphics] skipping {name} graphics (no saved data): {exc}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate QXTI graphics from saved datasets without recalculating physics."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="inputParams.cfg",
        help="Path to the configuration file. Defaults to inputParams.cfg.",
    )
    parser.add_argument(
        "--family",
        choices=("all", "hamiltonian", "response", "harmonics", "susceptibility", "ldos"),
        default="all",
        help="Choose which graphics family to generate from saved data.",
    )
    # Convenience flags so the SAME flag used to run a calculation in main.py also
    # selects its plots here (e.g. `graphics.py -ldos <config>` == `--family ldos`).
    shortcuts = parser.add_mutually_exclusive_group()
    shortcuts.add_argument("-cmd", "--cmd", "-hhg", "--hhg", dest="shortcut",
                           action="store_const", const="harmonics",
                           help="Shortcut for --family harmonics (the -cmd calculation's plots). "
                           "-hhg is a deprecated alias.")
    shortcuts.add_argument("-xtp", "--xtp", dest="shortcut", action="store_const", const="susceptibility",
                           help="Shortcut for --family susceptibility (the -xtp calculation's plots).")
    shortcuts.add_argument("-ldos", "--ldos", dest="shortcut", action="store_const", const="ldos",
                           help="Shortcut for --family ldos (the -ldos calculation's plots).")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    # A -hhg/-xtp/-ldos shortcut, if given, overrides --family.
    family = args.shortcut if getattr(args, "shortcut", None) else args.family

    if family == "all":
        outputs = plot_all_graphics_from_saved_data(args.config)
    elif family == "hamiltonian":
        outputs = plot_hamiltonian_graphics_from_saved_data(args.config)
    elif family == "harmonics":
        outputs = plot_harmonic_graphics_from_saved_data(args.config)
    elif family == "susceptibility":
        outputs = plot_susceptibility_graphics_from_saved_data(args.config)
    elif family == "ldos":
        outputs = plot_ldos_graphics_from_saved_data(args.config)
    else:
        outputs = plot_response_graphics_from_saved_data(args.config)

    print(f"Generated {len(outputs)} graphics from saved data in {args.config}:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


def _normalize_plot_name(plot_name: str) -> str:
    key = plot_name.strip().lower()
    aliases = {
        "bands_2d": "band_structure_2d",
        "bandas_2d": "band_structure_2d",
        "band_structure_2d": "band_structure_2d",
        "bands_3d": "band_surface_3d",
        "bandas_3d": "band_surface_3d",
        "band_surface_3d": "band_surface_3d",
        "velocities_2d": "velocity_2d",
        "velocidades_2d": "velocity_2d",
        "velocity_2d": "velocity_2d",
        "velocities_3d": "velocity_field_3d",
        "velocidades_3d": "velocity_field_3d",
        "velocity_field_3d": "velocity_field_3d",
        "velocity_magnitude": "velocity_magnitude",
        "modulo_velocidad": "velocity_magnitude",
        "modulo_de_velocidad": "velocity_magnitude",
    }
    return aliases.get(key, key)


def _harmonic_reference_omega(config: QXTIConfig) -> float:
    if config.laser.pulses:
        omegas = [float(pulse.get("omega", pulse.get("w0", 0.0))) for pulse in config.laser.pulses]
        positive_omegas = [omega for omega in omegas if omega > 0.0]
        if positive_omegas:
            return float(min(positive_omegas))
    return float(config.laser.omega)


def _augment_legacy_harmonic_dataset(data: dict[str, object]) -> dict[str, object]:
    augmented = dict(data)
    if "current_total_magnitude" not in augmented and "current_spectrum" in augmented:
        spectrum = np.asarray(augmented["current_spectrum"], dtype=np.complex128)
        augmented["current_total_magnitude"] = np.sqrt(np.sum(np.abs(spectrum) ** 2, axis=1))
    if "current_decomposition_available" not in augmented:
        augmented["current_decomposition_available"] = False
    return augmented


def _load_response_fallback_data(
    config_path: str | Path,
) -> tuple[dict[int, object], object, object, object, object, object]:
    config = _load_standardized_config(config_path)
    output_dir = Path(config.cmd.output_dir)
    from qxti.core import QXTISimulation

    simulation = QXTISimulation.from_file(config_path)
    hamiltonian = simulation.build_hamiltonian()
    cmd = simulation.build_cmd(hamiltonian)

    rho_orders = load_rho_orders_from_npy(output_dir, nt=cmd.timegrid.Nt)
    if rho_orders:
        return (
            rho_orders,
            np.asarray(cmd.timegrid.generate(), dtype=float),
            np.asarray(cmd.kgrid.points(), dtype=float),
            np.asarray(cmd.kgrid.kx_values, dtype=float),
            np.asarray(cmd.kgrid.ky_values, dtype=float),
            np.asarray(cmd.kgrid.kz_values, dtype=float),
        )

    rho_orders_dat, k_points, time_axis = load_rho_orders_from_dat(output_dir)
    if rho_orders_dat:
        kx_values = np.unique(k_points[:, 0])
        ky_values = np.unique(k_points[:, 1])
        kz_values = np.unique(k_points[:, 2])
        return (
            rho_orders_dat,
            time_axis,
            k_points,
            kx_values,
            ky_values,
            kz_values,
        )

    empty = []
    return ({}, empty, empty, empty, empty, empty)


def _resolve_requested_orders(config_value: object) -> tuple[int, ...] | None:
    if config_value is None:
        return None
    if isinstance(config_value, str) and config_value.strip().lower() in {"", "all", "none"}:
        return None
    if isinstance(config_value, int):
        return (int(config_value),)
    if isinstance(config_value, (list, tuple)):
        return tuple(int(item) for item in config_value)
    raise ValueError("plot config 'orders' must be 'all' or a list/tuple of ints.")


def _select_frequency_mask(
    omega_axis: np.ndarray,
    *,
    positive_only: bool,
    omega_min: float | None,
    omega_max: float | None,
) -> np.ndarray:
    omega = np.asarray(omega_axis, dtype=float)
    mask = np.ones_like(omega, dtype=bool)
    if positive_only:
        mask &= omega >= 0.0
    if omega_min is not None:
        mask &= omega >= float(omega_min)
    if omega_max is not None:
        mask &= omega <= float(omega_max)
    if not np.any(mask):
        raise ValueError("The selected susceptibility frequency window is empty.")
    return mask


if __name__ == "__main__":
    raise SystemExit(main())
