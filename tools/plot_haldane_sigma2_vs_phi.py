#!/usr/bin/env python3
from __future__ import annotations

"""Barrido en phi0 de la conductividad no lineal sigma^(2)_ijk del Haldane.

La tool calcula el tensor completo simetrizado

    sigma^(2)_ijk(2 omega; omega, omega)

para varios valores de ``phi0`` y dibuja cada componente como curvas en
frecuencia coloreadas por phi0. Usa el mismo motor de respuesta analitica de
QXTI que el barrido de susceptibilidad, extendido aqui para reconstruir todos
los pares de indices de entrada (j,k), no solo los componentes diagonales j=j.

Ejemplo:
    python tools/plot_haldane_sigma2_vs_phi.py \
        --config inputs/inputParams.haldane_topological.cfg \
        --nphi 21 --kpoints 61 --nomega 160
"""

import argparse
import copy
import os
from pathlib import Path
import sys
import time
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray

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

from qxti.analytics.theory_response import build_k_integration_weights, _resolve_distribution
from qxti.core import QXTIConfig
from qxti.core.simulation import QXTISimulation
from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
from qxti.utils.progress import ProgressTimer, format_duration

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

DEFAULT_OUTPUT_DIR = Path("outputs") / "haldane_sigma2_phi_sweep"
DEFAULT_CONFIG = Path("inputs") / "inputParams.haldane_topological.cfg"
AU_TO_EV = 27.211386245988


def _phi_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "haldane_phi_sweep_sigma2",
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


def _phi_axis(phi_min: float, phi_max: float, nphi: int) -> np.ndarray:
    values = np.linspace(float(phi_min), float(phi_max), max(int(nphi), 2), dtype=np.float64)
    if phi_min <= 0.0 <= phi_max and not np.any(np.isclose(values, 0.0, atol=1e-14)):
        values = np.sort(np.concatenate([values, np.array([0.0], dtype=np.float64)]))
    return values


def _value_view(values: np.ndarray, mode: str) -> np.ndarray:
    key = mode.strip().lower()
    if key == "real":
        return np.real(values)
    if key == "imag":
        return np.imag(values)
    if key in {"abs", "modulus", "magnitude"}:
        return np.abs(values)
    raise ValueError("value mode must be one of: real, imag, abs.")


def _value_prefix(mode: str) -> str:
    key = mode.strip().lower()
    if key == "real":
        return r"\Re"
    if key == "imag":
        return r"\Im"
    return r"|"


def _component_latex(component: tuple[int, int, int], labels: tuple[str, ...], mode: str) -> str:
    i, j, k = component
    indices = f"{labels[i]}{labels[j]}{labels[k]}"
    key = mode.strip().lower()
    if key == "abs":
        return rf"$|\sigma^{{(2)}}_{{{indices}}}|$"
    return rf"${_value_prefix(key)}\,\sigma^{{(2)}}_{{{indices}}}$"


def _component_name(component: tuple[int, int, int], labels: tuple[str, ...]) -> str:
    i, j, k = component
    return f"{labels[i]}{labels[j]}{labels[k]}"


def _load_base_config(path: Path) -> QXTIConfig:
    if not path.exists():
        raise FileNotFoundError(f"No existe el input base: {path}")
    return QXTIConfig.from_file(path)


def _prepare_config_for_phi(
    base_config: QXTIConfig,
    *,
    phi0: float,
    kpoints: int | None,
    t1: float | None,
    t2: float | None,
    m0: float | None,
    a0: float | None,
    gamma: float | None,
    coherence_time: float | None,
    temperature_au: float | None,
    fermi_level: float | None,
    distribution: str | None,
) -> QXTIConfig:
    cfg = copy.deepcopy(base_config)
    cfg.hamiltonian.source_file = "haldane.py"
    cfg.hamiltonian.params["phi0"] = float(phi0)
    if t1 is not None:
        cfg.hamiltonian.params["t1"] = float(t1)
    if t2 is not None:
        cfg.hamiltonian.params["t2"] = float(t2)
    if m0 is not None:
        cfg.hamiltonian.params["M0"] = float(m0)
    if a0 is not None:
        cfg.hamiltonian.params["a0"] = float(a0)
    if kpoints is not None:
        n = max(int(kpoints), 3)
        cfg.kgrid.dimension = 2
        cfg.kgrid.k_points = (n, n)
        cfg.kgrid.points_per_axis = None
        cfg.kgrid.shifted = True
    if gamma is not None:
        cfg.susceptibility_solver.coherence_time = float("inf") if gamma <= 0.0 else 1.0 / float(gamma)
    if coherence_time is not None:
        cfg.susceptibility_solver.coherence_time = float(coherence_time)
    if temperature_au is not None:
        cfg.susceptibility_solver.temperature = float(temperature_au)
    if fermi_level is not None:
        cfg.susceptibility_solver.fermi_level = float(fermi_level)
    if distribution:
        cfg.susceptibility_solver.distribution = str(distribution)
    cfg.susceptibility_solver.max_order = max(int(cfg.susceptibility_solver.max_order), 2)
    return cfg


def _compute_sigma2_full_gridbased(
    config: QXTIConfig,
    omega_axis: np.ndarray,
    *,
    components: tuple[tuple[int, int, int], ...] | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Full sigma^(2)_ijk with equal input frequencies.

    The implementation mirrors ``qxti.analytics.theory_response._order2_gridbased``
    but keeps the first-order input direction and the covariant-gradient
    direction independent. The returned tensor is symmetrized in the two input
    indices because the sweep is degenerate: omega_j = omega_k = omega.
    """
    t0 = time.perf_counter()
    sim = QXTISimulation(config=config)
    hamiltonian = sim.build_hamiltonian()
    kgrid = sim.build_kgrid(hamiltonian)
    dim = int(hamiltonian.dimension)
    nb = int(hamiltonian.basis_size)
    if dim != 2:
        raise ValueError("Esta tool esta pensada para Haldane 2D; dimension debe ser 2.")
    if components is None:
        requested_components = tuple(product(range(dim), repeat=3))
    else:
        requested_components = tuple(dict.fromkeys(tuple(int(x) for x in component) for component in components))
        for component in requested_components:
            if len(component) != 3 or any(index < 0 or index >= dim for index in component):
                raise ValueError(f"Componente sigma^(2) invalida para dimension {dim}: {component}")
    needed_ordered_components: set[tuple[int, int, int]] = set()
    for output_axis, first_input_axis, second_input_axis in requested_components:
        needed_ordered_components.add((output_axis, first_input_axis, second_input_axis))
        needed_ordered_components.add((output_axis, second_input_axis, first_input_axis))
    gradient_axes = sorted({component[1] for component in needed_ordered_components})
    input_axes_by_gradient = {
        gradient_axis: sorted(
            {component[2] for component in needed_ordered_components if component[1] == gradient_axis}
        )
        for gradient_axis in gradient_axes
    }
    output_axes_by_pair = {
        (gradient_axis, input_axis): sorted(
            {
                component[0]
                for component in needed_ordered_components
                if component[1] == gradient_axis and component[2] == input_axis
            }
        )
        for gradient_axis in gradient_axes
        for input_axis in input_axes_by_gradient[gradient_axis]
    }
    shape = tuple(int(kgrid.shape[a]) for a in range(3))
    k_points = np.asarray(kgrid.points(), dtype=np.float64)
    nk = int(k_points.shape[0])
    omega_axis = np.asarray(omega_axis, dtype=np.float64)
    nw = int(omega_axis.size)
    Hf = hamiltonian._matrix_at
    bounds = hamiltonian.reciprocal_box_bounds()
    dks = [(float(bounds[a][1]) - float(bounds[a][0])) / float(shape[a]) for a in range(dim)]
    weights = build_k_integration_weights(config, hamiltonian=hamiltonian, kgrid=kgrid)

    ccfg = config.susceptibility_solver
    distribution = _resolve_distribution(ccfg.distribution)
    mu = float(ccfg.fermi_level)
    temperature = float(ccfg.temperature)
    gamma = 0.0 if ccfg.coherence_time <= 0 else 1.0 / float(ccfg.coherence_time)
    dk = float(getattr(hamiltonian, "dk_derivative", 1.0e-5) or 1.0e-5)

    if progress:
        component_text = "full tensor" if components is None else ", ".join(
            "".join(("x", "y", "z")[index] for index in component) for component in requested_components
        )
        print(
            f"[haldane-sigma2-phi] order 2 {component_text}: diagonalizing {nk} k-points "
            f"for {nw} frequencies.",
            flush=True,
        )

    def _Hbatch(kc: FloatArray) -> ComplexArray:
        out = np.empty((kc.shape[0], nb, nb), dtype=np.complex128)
        for index, k in enumerate(kc):
            out[index] = Hf(float(k[0]), float(k[1]), float(k[2]))
        return out

    k_chunk = max(512, min(8192, nk))
    energies = np.empty((nk, nb), dtype=np.float64)
    U = np.empty((nk, nb, nb), dtype=np.complex128)
    for start in range(0, nk, k_chunk):
        stop = min(start + k_chunk, nk)
        evals_chunk, U_chunk = np.linalg.eigh(_Hbatch(k_points[start:stop]))
        energies[start:stop] = evals_chunk
        U[start:stop] = U_chunk

    vel = [np.empty((nk, nb, nb), dtype=np.complex128) for _ in range(dim)]
    for axis in range(dim):
        if progress:
            print(f"[haldane-sigma2-phi] velocity operator {('x','y')[axis]} ...", flush=True)
        shift = np.zeros(3, dtype=np.float64)
        shift[axis] = dk
        for start in range(0, nk, k_chunk):
            stop = min(start + k_chunk, nk)
            kc = k_points[start:stop]
            dH = (_Hbatch(kc + shift) - _Hbatch(kc - shift)) / (2.0 * dk)
            U_chunk = U[start:stop]
            Udag_chunk = np.conj(np.transpose(U_chunk, (0, 2, 1)))
            vel[axis][start:stop] = Udag_chunk @ dH @ U_chunk

    occupations = np.asarray(distribution(energies, mu, temperature), dtype=np.float64)
    eps = energies[:, :, None] - energies[:, None, :]
    fmn = occupations[:, None, :] - occupations[:, :, None]
    offdiag = ~np.eye(nb, dtype=bool)
    valid = offdiag[None] & (np.abs(eps) > 1.0e-20)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_eps = np.where(valid, 1.0 / eps, 0.0)
    dfde = (
        -occupations * (1.0 - occupations) / temperature
        if temperature > 1.0e-15
        else np.zeros_like(occupations)
    )

    U_mesh = U.reshape(*shape, nb, nb)
    vel_mesh = [item.reshape(*shape, nb, nb) for item in vel]
    eps_mesh = eps.reshape(*shape, nb, nb)
    fmn_mesh = fmn.reshape(*shape, nb, nb)
    valid_mesh = valid.reshape(*shape, nb, nb)
    inv_eps_mesh = inv_eps.reshape(*shape, nb, nb)
    dfde_mesh = dfde.reshape(*shape, nb)
    weights_mesh = np.asarray(weights, dtype=np.float64).reshape(*shape)

    max_slice_points = max(int(nk // max(shape[axis], 1)) for axis in range(dim))
    bytes_per_frequency = max_slice_points * nb * nb * 16 * 10
    omega_chunk = max(1, min(nw, 16, int(1.5e8 / max(bytes_per_frequency, 1))))
    sigma_ordered = np.full((nw, dim, dim, dim), np.nan + 1j * np.nan, dtype=np.complex128)
    for component in needed_ordered_components:
        sigma_ordered[:, component[0], component[1], component[2]] = 0.0 + 0.0j
    total_component_slices = sum(
        int(shape[gradient_axis]) * len(input_axes_by_gradient[gradient_axis])
        for gradient_axis in gradient_axes
    )
    timer = ProgressTimer(total=max(1, total_component_slices))

    def _take_matrix(mesh_array: np.ndarray, axis: int, idx: int) -> ComplexArray:
        return np.asarray(np.take(mesh_array, int(idx) % shape[axis], axis=axis), dtype=np.complex128).reshape(-1, nb, nb)

    def _take_float_matrix(mesh_array: np.ndarray, axis: int, idx: int) -> FloatArray:
        return np.asarray(np.take(mesh_array, int(idx) % shape[axis], axis=axis), dtype=np.float64).reshape(-1, nb, nb)

    def _take_bool_matrix(mesh_array: np.ndarray, axis: int, idx: int) -> NDArray[np.bool_]:
        return np.asarray(np.take(mesh_array, int(idx) % shape[axis], axis=axis), dtype=bool).reshape(-1, nb, nb)

    def _take_vector(mesh_array: np.ndarray, axis: int, idx: int) -> ComplexArray:
        return np.asarray(np.take(mesh_array, int(idx) % shape[axis], axis=axis), dtype=np.complex128).reshape(-1, nb)

    def _take_weights(axis: int, idx: int) -> FloatArray:
        return np.asarray(np.take(weights_mesh, int(idx) % shape[axis], axis=axis), dtype=np.float64).reshape(-1)

    def _rho1_flat(
        *,
        input_axis: int,
        gradient_axis: int,
        slice_index: int,
        omega_chunk_values: ComplexArray,
    ) -> tuple[ComplexArray, ComplexArray, ComplexArray, FloatArray, NDArray[np.bool_], FloatArray]:
        vel_l = _take_matrix(vel_mesh[input_axis], gradient_axis, slice_index)
        inv_eps_l = _take_matrix(inv_eps_mesh, gradient_axis, slice_index)
        fmn_l = _take_float_matrix(fmn_mesh, gradient_axis, slice_index)
        valid_l = _take_bool_matrix(valid_mesh, gradient_axis, slice_index)
        eps_l = _take_float_matrix(eps_mesh, gradient_axis, slice_index)
        dfde_l = np.asarray(
            np.take(dfde_mesh, slice_index % shape[gradient_axis], axis=gradient_axis),
            dtype=np.float64,
        ).reshape(-1, nb)
        diag_src = (-1j) * dfde_l * np.real(np.diagonal(vel_l, axis1=1, axis2=2))
        A_l = 1j * vel_l * inv_eps_l
        drive = A_l * fmn_l
        denom = omega_chunk_values[None, :, None, None] - eps_l[:, None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            rho1 = np.where(valid_l[:, None, :, :], drive[:, None, :, :] / denom, 0.0 + 0.0j)
        rho1[:, :, np.arange(nb), np.arange(nb)] += diag_src[:, None, :] / omega_chunk_values[None, :, None]
        return (
            np.asarray(rho1, dtype=np.complex128),
            np.asarray(A_l, dtype=np.complex128),
            vel_l,
            eps_l,
            valid_l,
            inv_eps_l,
        )

    for gradient_axis in gradient_axes:
        n_slices = int(shape[gradient_axis])
        if progress:
            print(
                f"[haldane-sigma2-phi] covariant gradient along {('x','y')[gradient_axis]} "
                f"({n_slices} slices, omega chunks of {omega_chunk}).",
                flush=True,
            )
        for wstart in range(0, nw, omega_chunk):
            wstop = min(wstart + omega_chunk, nw)
            omega_chunk_values = np.asarray(omega_axis[wstart:wstop] + 1j * gamma, dtype=np.complex128)
            omega2_chunk_values = np.asarray(2.0 * omega_axis[wstart:wstop] + 1j * gamma, dtype=np.complex128)

            for slice_index in range(n_slices):
                U_curr = _take_matrix(U_mesh, gradient_axis, slice_index)
                U_plus = _take_matrix(U_mesh, gradient_axis, slice_index + 1)
                U_minus = _take_matrix(U_mesh, gradient_axis, slice_index - 1)
                U_curr_dag = np.conj(np.swapaxes(U_curr, -1, -2))
                W_plus = U_curr_dag @ U_plus
                W_minus = U_curr_dag @ U_minus
                W_plus_dag = np.conj(np.swapaxes(W_plus, -1, -2))
                W_minus_dag = np.conj(np.swapaxes(W_minus, -1, -2))
                weights_curr = _take_weights(gradient_axis, slice_index)
                eps_curr = _take_float_matrix(eps_mesh, gradient_axis, slice_index)
                valid_curr = _take_bool_matrix(valid_mesh, gradient_axis, slice_index)
                inv_d2 = np.zeros((weights_curr.size, wstop - wstart, nb, nb), dtype=np.complex128)
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv_d2 = np.where(
                        valid_curr[:, None, :, :],
                        1.0 / (omega2_chunk_values[None, :, None, None] - eps_curr[:, None, :, :]),
                        0.0 + 0.0j,
                    )
                A_gradient = 1j * _take_matrix(vel_mesh[gradient_axis], gradient_axis, slice_index) * _take_matrix(
                    inv_eps_mesh,
                    gradient_axis,
                    slice_index,
                )

                for input_axis in input_axes_by_gradient[gradient_axis]:
                    rho1_curr, _, _, _, _, _ = _rho1_flat(
                        input_axis=input_axis,
                        gradient_axis=gradient_axis,
                        slice_index=slice_index,
                        omega_chunk_values=omega_chunk_values,
                    )
                    rho1_plus, _, _, _, _, _ = _rho1_flat(
                        input_axis=input_axis,
                        gradient_axis=gradient_axis,
                        slice_index=slice_index + 1,
                        omega_chunk_values=omega_chunk_values,
                    )
                    rho1_minus, _, _, _, _, _ = _rho1_flat(
                        input_axis=input_axis,
                        gradient_axis=gradient_axis,
                        slice_index=slice_index - 1,
                        omega_chunk_values=omega_chunk_values,
                    )
                    transported_plus = W_plus[:, None, :, :] @ rho1_plus @ W_plus_dag[:, None, :, :]
                    transported_minus = W_minus[:, None, :, :] @ rho1_minus @ W_minus_dag[:, None, :, :]
                    dpart = (transported_plus - transported_minus) / (2.0 * dks[gradient_axis])
                    commutator = (
                        A_gradient[:, None, :, :] @ rho1_curr
                        - rho1_curr @ A_gradient[:, None, :, :]
                    )
                    rho2 = (dpart - 1j * commutator) * inv_d2
                    for output_axis in output_axes_by_pair[(gradient_axis, input_axis)]:
                        current_op = -_take_matrix(vel_mesh[output_axis], gradient_axis, slice_index)
                        sigma_ordered[wstart:wstop, output_axis, gradient_axis, input_axis] += np.conj(
                            np.einsum("pmn,pwnm,p->w", current_op, rho2, weights_curr, optimize=True)
                        )
                    timer.advance()
                    if progress and timer.completed % max(1, timer.total // 10) == 0:
                        print(
                            f"[haldane-sigma2-phi]   progress {timer.completed}/{timer.total} "
                            f"({100 * timer.completed // timer.total}%), ETA {timer.eta_text()}",
                            flush=True,
                        )

    sigma_sym = 0.5 * (sigma_ordered + np.swapaxes(sigma_ordered, -1, -2))
    available = np.asarray(requested_components, dtype=np.int16)
    return {
        "omega_axis": omega_axis,
        "sigma_order_2_tensor": np.asarray(sigma_sym, dtype=np.complex128),
        "sigma_order_2_ordered_tensor": np.asarray(sigma_ordered, dtype=np.complex128),
        "sigma_order_2_available_indices": available,
        "direction_labels": tuple(("x", "y", "z")[:dim]),
        "dimension": dim,
        "gamma": gamma,
        "runtime_seconds": time.perf_counter() - t0,
    }


def _build_phi_sweep(
    *,
    base_config: QXTIConfig,
    phi_values: np.ndarray,
    omega_axis: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, tuple[str, ...], list[dict[str, Any]]]:
    config0 = _prepare_config_for_phi(
        base_config,
        phi0=float(phi_values[0]),
        kpoints=args.kpoints,
        t1=args.t1,
        t2=args.t2,
        m0=args.m0,
        a0=args.a0,
        gamma=args.gamma,
        coherence_time=args.coherence_time,
        temperature_au=args.temperature_au,
        fermi_level=args.fermi_level,
        distribution=args.distribution,
    )
    first = _compute_sigma2_full_gridbased(config0, omega_axis, progress=not args.quiet)
    direction_labels = tuple(str(x) for x in first["direction_labels"])
    tensor_shape = (phi_values.size,) + np.asarray(first["sigma_order_2_tensor"]).shape
    sigma = np.zeros(tensor_shape, dtype=np.complex128)
    sigma[0] = np.asarray(first["sigma_order_2_tensor"], dtype=np.complex128)
    metadata = [first]

    timer = ProgressTimer(total=int(phi_values.size), min_completed_for_eta=max(2, min(phi_values.size, 5)))
    timer.advance()
    print(
        f"[haldane-sigma2-phi] phi 1/{phi_values.size} "
        f"(phi0={phi_values[0]:+.4f} rad) done in {format_duration(first['runtime_seconds'])}; "
        f"elapsed {format_duration(timer.elapsed_seconds)}, ETA {timer.eta_text()}",
        flush=True,
    )

    for ip in range(1, phi_values.size):
        row_start = time.perf_counter()
        cfg = _prepare_config_for_phi(
            base_config,
            phi0=float(phi_values[ip]),
            kpoints=args.kpoints,
            t1=args.t1,
            t2=args.t2,
            m0=args.m0,
            a0=args.a0,
            gamma=args.gamma,
            coherence_time=args.coherence_time,
            temperature_au=args.temperature_au,
            fermi_level=args.fermi_level,
            distribution=args.distribution,
        )
        result = _compute_sigma2_full_gridbased(cfg, omega_axis, progress=not args.quiet)
        sigma[ip] = np.asarray(result["sigma_order_2_tensor"], dtype=np.complex128)
        metadata.append(result)
        timer.advance()
        print(
            f"[haldane-sigma2-phi] phi {ip + 1}/{phi_values.size} "
            f"(phi0={phi_values[ip]:+.4f} rad) in {format_duration(time.perf_counter() - row_start)}; "
            f"elapsed {format_duration(timer.elapsed_seconds)}, ETA {timer.eta_text()}",
            flush=True,
        )
    return sigma, direction_labels, metadata


def _plot_component(
    *,
    omega_axis: np.ndarray,
    phi_values: np.ndarray,
    sigma_phi: np.ndarray,
    component: tuple[int, int, int],
    direction_labels: tuple[str, ...],
    output_path: Path,
    value_mode: str,
    dpi: int,
) -> None:
    apply_paper_style()
    cmap = _phi_colormap()
    norm = Normalize(vmin=float(phi_values.min()), vmax=float(phi_values.max()))
    x_ev = omega_axis * AU_TO_EV
    values = _value_view(sigma_phi[:, :, component[0], component[1], component[2]], value_mode)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for phi0, curve in zip(phi_values, values, strict=True):
        ax.plot(x_ev, curve, color=cmap(norm(float(phi0))), linewidth=1.35, alpha=0.95)
    if value_mode in {"real", "imag"}:
        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9, alpha=0.75)
    ax.set_xlabel(r"$\hbar\omega\;(\mathrm{eV})$")
    ax.set_ylabel(_component_latex(component, direction_labels, value_mode) + r"$\;(\mathrm{a.u.})$")
    ax.set_title(_component_latex(component, direction_labels, value_mode) + r" vs $\phi_0$")
    ax.set_xlim(float(x_ev.min()), float(x_ev.max()))
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    colorbar.set_label(r"$\phi_0\;(\mathrm{rad})$")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def _plot_overview(
    *,
    omega_axis: np.ndarray,
    phi_values: np.ndarray,
    sigma_phi: np.ndarray,
    direction_labels: tuple[str, ...],
    output_path: Path,
    value_mode: str,
    dpi: int,
) -> None:
    apply_paper_style()
    dim = len(direction_labels)
    components = list(product(range(dim), repeat=3))
    cmap = _phi_colormap()
    norm = Normalize(vmin=float(phi_values.min()), vmax=float(phi_values.max()))
    x_ev = omega_axis * AU_TO_EV
    ncols = 4 if dim == 2 else 3
    nrows = int(np.ceil(len(components) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.35 * ncols, 2.35 * nrows), squeeze=False)
    for axis, component in zip(axes.ravel(), components, strict=False):
        values = _value_view(sigma_phi[:, :, component[0], component[1], component[2]], value_mode)
        for phi0, curve in zip(phi_values, values, strict=True):
            axis.plot(x_ev, curve, color=cmap(norm(float(phi0))), linewidth=0.8, alpha=0.85)
        if value_mode in {"real", "imag"}:
            axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.6, alpha=0.7)
        axis.set_title(_component_latex(component, direction_labels, value_mode), fontsize=9)
        axis.set_xlim(float(x_ev.min()), float(x_ev.max()))
        axis.tick_params(labelsize=7)
    for axis in axes.ravel()[len(components):]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel(r"$\hbar\omega\;(\mathrm{eV})$")
    for axis in axes[:, 0]:
        if axis.get_visible():
            axis.set_ylabel(r"$\sigma^{(2)}\;(\mathrm{a.u.})$")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), pad=0.012)
    colorbar.set_label(r"$\phi_0\;(\mathrm{rad})$")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Barrido en phi0 de sigma^(2)_ijk del modelo de Haldane."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Input base de Haldane.")
    parser.add_argument("--nphi", type=int, default=21, help="Numero de valores de phi0.")
    parser.add_argument("--phi-min", type=float, default=-0.5 * np.pi, help="Phi minimo (rad).")
    parser.add_argument("--phi-max", type=float, default=0.5 * np.pi, help="Phi maximo (rad).")
    parser.add_argument("--nomega", type=int, default=160, help="Numero de frecuencias.")
    parser.add_argument("--omega-min", type=float, default=0.005, help="Frecuencia laser minima (a.u.).")
    parser.add_argument("--omega-max", type=float, default=0.18, help="Frecuencia laser maxima (a.u.).")
    parser.add_argument("--kpoints", type=int, default=None, help="Override: puntos k por eje.")
    parser.add_argument("--t1", type=float, default=None, help="Override t1 (a.u.).")
    parser.add_argument("--t2", type=float, default=None, help="Override t2 (a.u.).")
    parser.add_argument("--m0", type=float, default=None, help="Override M0 (a.u.).")
    parser.add_argument("--a0", type=float, default=None, help="Override a0 (a.u.).")
    parser.add_argument("--gamma", type=float, default=None, help="Override de ensanchamiento gamma=1/T2 (a.u.).")
    parser.add_argument("--coherence-time", type=float, default=None, help="Override T2 (a.u.).")
    parser.add_argument("--temperature-au", type=float, default=None, help="Override temperatura (a.u.).")
    parser.add_argument("--fermi-level", type=float, default=None, help="Override nivel de Fermi (a.u.).")
    parser.add_argument("--distribution", default=None, help="Override distribucion: valence_occupation, fermi_dirac, etc.")
    parser.add_argument("--value", choices=("real", "imag", "abs"), default="real", help="Cantidad que se grafica.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Carpeta de salida.")
    parser.add_argument("--dataset-name", default="haldane_sigma2_vs_phi.npz", help="Nombre del .npz.")
    parser.add_argument("--dpi", type=int, default=300, help="Resolucion PNG.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs internos por phi.")
    parser.add_argument("--no-component-plots", action="store_true", help="Solo guarda overview y dataset.")
    args = parser.parse_args()

    if args.nphi <= 1 or args.nomega <= 1:
        raise SystemExit("nphi y nomega deben ser mayores que 1.")
    if args.omega_min <= 0.0 or args.omega_max <= args.omega_min:
        raise SystemExit("Se requiere 0 < omega-min < omega-max.")
    if args.gamma is not None and args.coherence_time is not None:
        raise SystemExit("Usa solo uno de --gamma o --coherence-time.")

    base_config = _load_base_config(Path(args.config))
    phi_values = _phi_axis(args.phi_min, args.phi_max, args.nphi)
    omega_axis = np.linspace(args.omega_min, args.omega_max, args.nomega, dtype=np.float64)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[haldane-sigma2-phi] sweep "
        f"nphi={phi_values.size}, nomega={omega_axis.size}, "
        f"phi=[{phi_values.min():+.4f}, {phi_values.max():+.4f}] rad",
        flush=True,
    )
    sigma_phi, direction_labels, metadata = _build_phi_sweep(
        base_config=base_config,
        phi_values=phi_values,
        omega_axis=omega_axis,
        args=args,
    )

    data_path = output_dir / args.dataset_name
    np.savez_compressed(
        data_path,
        phi0_axis=phi_values,
        omega_axis=omega_axis,
        omega_axis_ev=omega_axis * AU_TO_EV,
        sigma_order_2_tensor=sigma_phi,
        sigma_order_2_real=np.real(sigma_phi),
        sigma_order_2_imag=np.imag(sigma_phi),
        direction_labels=np.asarray(direction_labels),
        value_mode=str(args.value),
        config_path=str(Path(args.config)),
        component_order="sigma_phi[iphi, iw, i, j, k]",
        runtime_seconds=np.asarray([float(item["runtime_seconds"]) for item in metadata], dtype=np.float64),
        gamma=np.asarray([float(item["gamma"]) for item in metadata], dtype=np.float64),
    )

    overview_path = output_dir / f"haldane_sigma2_vs_phi_overview_{args.value}.png"
    _plot_overview(
        omega_axis=omega_axis,
        phi_values=phi_values,
        sigma_phi=sigma_phi,
        direction_labels=direction_labels,
        output_path=overview_path,
        value_mode=args.value,
        dpi=int(args.dpi),
    )

    component_paths: list[Path] = []
    if not args.no_component_plots:
        component_dir = output_dir / "components" / args.value
        for component in product(range(len(direction_labels)), repeat=3):
            name = _component_name(component, direction_labels)
            path = component_dir / f"haldane_sigma2_{name}_vs_phi_{args.value}.png"
            _plot_component(
                omega_axis=omega_axis,
                phi_values=phi_values,
                sigma_phi=sigma_phi,
                component=component,
                direction_labels=direction_labels,
                output_path=path,
                value_mode=args.value,
                dpi=int(args.dpi),
            )
            component_paths.append(path)

    print(
        f"[haldane-sigma2-phi] data: {data_path}\n"
        f"[haldane-sigma2-phi] overview: {overview_path}\n"
        f"[haldane-sigma2-phi] component plots: {len(component_paths)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
