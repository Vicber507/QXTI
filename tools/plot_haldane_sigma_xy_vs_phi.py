#!/usr/bin/env python3
from __future__ import annotations

"""Plot de Re[sigma_xy^(1)(omega)] del modelo de Haldane para varios phi0.

Genera un barrido en ``phi0`` entre ``-pi/2`` y ``pi/2`` (incluyendo 0) y
dibuja las curvas de la parte real de ``sigma_xy`` con un colorbar continuo en
``phi0``. El calculo usa la misma formulacion lineal de Kubo empleada en la
tool del mapa de fase, pero aqui produce el espectro completo en frecuencia.

Ejemplo:
    python tools/plot_haldane_sigma_xy_vs_phi.py \
        --nphi 21 --kpoints 101 --omega-max 0.12 --nomega 300
"""

import argparse
import importlib.util
import os
from pathlib import Path
import sys

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

from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
from qxti.utils.progress import ProgressTimer, format_duration

DEFAULT_OUTPUT_DIR = Path("outputs") / "haldane_sigma_xy_phi_sweep"
AU_TO_EV = 27.211386245988


def _load_haldane_module():
    module_path = PROJECT_ROOT / "models" / "haldane.py"
    spec = importlib.util.spec_from_file_location("qxti_haldane_model_sigma_phi", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el modelo de Haldane desde {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fermi_occupations(energies: np.ndarray, mu: float, temperature_au: float) -> np.ndarray:
    if temperature_au <= 1.0e-15:
        return np.where(energies <= mu + 1.0e-14, 1.0, 0.0).astype(np.float64)
    return np.asarray(1.0 / (np.exp((energies - mu) / temperature_au) + 1.0), dtype=np.float64)


def _dfde(occupations: np.ndarray, temperature_au: float) -> np.ndarray:
    if temperature_au <= 1.0e-15:
        return np.zeros_like(occupations, dtype=np.float64)
    return np.asarray(-occupations * (1.0 - occupations) / temperature_au, dtype=np.float64)


def _build_precomputed_k_cache(
    *,
    a0: float,
    nkx: int,
    nky: int,
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> dict[str, np.ndarray | float]:
    kx_max = np.pi / (np.sqrt(3.0) * a0)
    ky_max = 2.0 * np.pi / (3.0 * a0)
    kx_values = np.linspace(-kx_max, kx_max, nkx, endpoint=False, dtype=np.float64) + (kx_max / nkx)
    ky_values = np.linspace(-ky_max, ky_max, nky, endpoint=False, dtype=np.float64) + (ky_max / nky)

    mesh_kx, mesh_ky = np.meshgrid(kx_values, ky_values, indexing="ij")
    k_flat = np.column_stack([mesh_kx.reshape(-1), mesh_ky.reshape(-1)])

    angles_a = np.asarray(k_flat @ vec_a.T, dtype=np.float64)
    angles_b = np.asarray(k_flat @ vec_b.T, dtype=np.float64)
    cos_a = np.cos(angles_a)
    sin_a = np.sin(angles_a)
    cos_b = np.cos(angles_b)
    sin_b = np.sin(angles_b)

    ax = np.asarray(vec_a[:, 0], dtype=np.float64)
    ay = np.asarray(vec_a[:, 1], dtype=np.float64)
    bx = np.asarray(vec_b[:, 0], dtype=np.float64)
    by = np.asarray(vec_b[:, 1], dtype=np.float64)

    width_x = 2.0 * kx_max
    width_y = 2.0 * ky_max
    bz_area = width_x * width_y
    weight_per_k = bz_area / float(nkx * nky)

    return {
        "kx_values": kx_values,
        "ky_values": ky_values,
        "cos_a_sum": np.asarray(np.sum(cos_a, axis=1), dtype=np.float64),
        "sin_a_sum": np.asarray(np.sum(sin_a, axis=1), dtype=np.float64),
        "cos_b_sum": np.asarray(np.sum(cos_b, axis=1), dtype=np.float64),
        "sin_b_sum": np.asarray(np.sum(sin_b, axis=1), dtype=np.float64),
        "sin_a_ax": np.asarray(sin_a @ ax, dtype=np.float64),
        "sin_a_ay": np.asarray(sin_a @ ay, dtype=np.float64),
        "cos_a_ax": np.asarray(cos_a @ ax, dtype=np.float64),
        "cos_a_ay": np.asarray(cos_a @ ay, dtype=np.float64),
        "sin_b_bx": np.asarray(sin_b @ bx, dtype=np.float64),
        "sin_b_by": np.asarray(sin_b @ by, dtype=np.float64),
        "cos_b_bx": np.asarray(cos_b @ bx, dtype=np.float64),
        "cos_b_by": np.asarray(cos_b @ by, dtype=np.float64),
        "bz_area": float(bz_area),
        "weight_per_k": float(weight_per_k),
    }


def _sigma_xy_spectrum_haldane(
    *,
    t1: float,
    t2: float,
    phi0: float,
    m0: float,
    omega_axis: np.ndarray,
    gamma: float,
    mu: float,
    temperature_au: float,
    spin_deg: int,
    cache: dict[str, np.ndarray | float],
) -> np.ndarray:
    cos_phi = float(np.cos(phi0))
    sin_phi = float(np.sin(phi0))

    cos_a_sum = np.asarray(cache["cos_a_sum"], dtype=np.float64)
    sin_a_sum = np.asarray(cache["sin_a_sum"], dtype=np.float64)
    cos_b_sum = np.asarray(cache["cos_b_sum"], dtype=np.float64)
    sin_b_sum = np.asarray(cache["sin_b_sum"], dtype=np.float64)
    sin_a_ax = np.asarray(cache["sin_a_ax"], dtype=np.float64)
    sin_a_ay = np.asarray(cache["sin_a_ay"], dtype=np.float64)
    cos_a_ax = np.asarray(cache["cos_a_ax"], dtype=np.float64)
    cos_a_ay = np.asarray(cache["cos_a_ay"], dtype=np.float64)
    sin_b_bx = np.asarray(cache["sin_b_bx"], dtype=np.float64)
    sin_b_by = np.asarray(cache["sin_b_by"], dtype=np.float64)
    cos_b_bx = np.asarray(cache["cos_b_bx"], dtype=np.float64)
    cos_b_by = np.asarray(cache["cos_b_by"], dtype=np.float64)
    weight_per_k = float(cache["weight_per_k"])
    bz_area = float(cache["bz_area"])

    b0 = 2.0 * t2 * cos_phi * cos_b_sum
    b1 = t1 * cos_a_sum
    b2 = t1 * sin_a_sum
    b3 = m0 - 2.0 * t2 * sin_phi * sin_b_sum

    db0_dkx = -2.0 * t2 * cos_phi * sin_b_bx
    db0_dky = -2.0 * t2 * cos_phi * sin_b_by
    db1_dkx = -t1 * sin_a_ax
    db1_dky = -t1 * sin_a_ay
    db2_dkx = t1 * cos_a_ax
    db2_dky = t1 * cos_a_ay
    db3_dkx = -2.0 * t2 * sin_phi * cos_b_bx
    db3_dky = -2.0 * t2 * sin_phi * cos_b_by

    nk = b0.shape[0]
    h = np.empty((nk, 2, 2), dtype=np.complex128)
    vx = np.empty((nk, 2, 2), dtype=np.complex128)
    vy = np.empty((nk, 2, 2), dtype=np.complex128)

    h[:, 0, 0] = b0 + b3
    h[:, 1, 1] = b0 - b3
    h[:, 0, 1] = b1 - 1j * b2
    h[:, 1, 0] = b1 + 1j * b2

    vx[:, 0, 0] = db0_dkx + db3_dkx
    vx[:, 1, 1] = db0_dkx - db3_dkx
    vx[:, 0, 1] = db1_dkx - 1j * db2_dkx
    vx[:, 1, 0] = db1_dkx + 1j * db2_dkx

    vy[:, 0, 0] = db0_dky + db3_dky
    vy[:, 1, 1] = db0_dky - db3_dky
    vy[:, 0, 1] = db1_dky - 1j * db2_dky
    vy[:, 1, 0] = db1_dky + 1j * db2_dky

    energies, unitary = np.linalg.eigh(h)
    unitary_dag = np.conj(np.transpose(unitary, (0, 2, 1)))
    vx_band = unitary_dag @ vx @ unitary
    vy_band = unitary_dag @ vy @ unitary

    occupations = _fermi_occupations(energies, mu=mu, temperature_au=temperature_au)
    dfde = _dfde(occupations, temperature_au=temperature_au)

    omega_complex = np.asarray(omega_axis, dtype=np.float64) + 1j * float(gamma)
    eps_mn = energies[:, :, None] - energies[:, None, :]
    f_nm = occupations[:, None, :] - occupations[:, :, None]
    valid = (~np.eye(2, dtype=bool))[None, :, :] & (np.abs(eps_mn) > 1.0e-20)

    with np.errstate(divide="ignore", invalid="ignore"):
        interband_prefactor = np.where(valid, f_nm / eps_mn, 0.0)
        interband_numer = (
            np.transpose(vx_band, (0, 2, 1)) * vy_band * interband_prefactor
        )

    diagonal_vx = np.diagonal(vx_band, axis1=1, axis2=2)
    diagonal_vy = np.diagonal(vy_band, axis1=1, axis2=2).real
    intraband_sum = np.sum(diagonal_vx * (dfde * diagonal_vy), axis=1)

    sigma_xy = np.zeros(omega_complex.size, dtype=np.complex128)
    for iw, omega in enumerate(omega_complex):
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = omega - eps_mn
            interband = np.where(valid, interband_numer / denom, 0.0 + 0.0j)
        intraband = (-1j / omega) * intraband_sum
        s_k = np.sum(interband, axis=(1, 2)) + intraband
        sigma_xy[iw] = -1j * float(spin_deg) * np.sum((weight_per_k / bz_area) * s_k)

    return np.asarray(sigma_xy, dtype=np.complex128)


def _build_sigma_phi_sweep(
    *,
    t1: float,
    t2: float,
    m0: float,
    a0: float,
    omega_axis: np.ndarray,
    gamma: float,
    mu: float,
    temperature_au: float,
    spin_deg: int,
    phi_values: np.ndarray,
    nkx: int,
    nky: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    haldane_module = _load_haldane_module()
    vec_a = np.asarray(haldane_module._nn_vectors(a0), dtype=np.float64)
    vec_b = np.asarray(haldane_module._nnn_vectors(a0), dtype=np.float64)
    cache = _build_precomputed_k_cache(a0=a0, nkx=nkx, nky=nky, vec_a=vec_a, vec_b=vec_b)

    sigma_curves = np.zeros((phi_values.size, omega_axis.size), dtype=np.complex128)
    timer = ProgressTimer(total=int(phi_values.size), min_completed_for_eta=max(4, min(phi_values.size, 8)))
    for ip, phi0 in enumerate(phi_values):
        row_start = timer.elapsed_seconds
        sigma_curves[ip] = _sigma_xy_spectrum_haldane(
            t1=t1,
            t2=t2,
            phi0=float(phi0),
            m0=m0,
            omega_axis=omega_axis,
            gamma=gamma,
            mu=mu,
            temperature_au=temperature_au,
            spin_deg=spin_deg,
            cache=cache,
        )
        timer.advance()
        print(
            f"[haldane-sigma-phi] phi {ip + 1}/{phi_values.size} "
            f"(phi0={phi0:+.4f} rad) in {format_duration(timer.elapsed_seconds - row_start)}; "
            f"elapsed {format_duration(timer.elapsed_seconds)}, ETA {timer.eta_text()}",
            flush=True,
        )

    return sigma_curves, np.asarray(cache["kx_values"]), np.asarray(cache["ky_values"])


def _phi_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "haldane_phi_sweep",
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


def _plot_sigma_phi_sweep(
    *,
    omega_axis: np.ndarray,
    phi_values: np.ndarray,
    sigma_curves: np.ndarray,
    output_path: Path,
) -> None:
    apply_paper_style()

    cmap = _phi_colormap()
    norm = Normalize(vmin=float(phi_values.min()), vmax=float(phi_values.max()))
    x_ev = np.asarray(omega_axis, dtype=np.float64) * AU_TO_EV
    values = np.real(sigma_curves)

    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    for phi0, curve in zip(phi_values, values, strict=True):
        axis.plot(
            x_ev,
            curve,
            color=cmap(norm(float(phi0))),
            linewidth=1.5,
            alpha=0.95,
        )

    axis.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8)
    axis.set_xlabel(r"$\hbar\omega\;(\mathrm{eV})$")
    axis.set_ylabel(r"$\Re\,\sigma_{xy}^{(1)}(\omega)\;(\mathrm{a.u.})$")
    axis.set_title(r"Haldane: $\Re\,\sigma_{xy}^{(1)}(\omega)$ barrido en $\phi_0$")
    axis.set_xlim(float(x_ev.min()), float(x_ev.max()))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = figure.colorbar(sm, ax=axis, pad=0.02)
    colorbar.set_label(r"$\phi_0\;(\mathrm{rad})$")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot de Re[sigma_xy(omega)] del Haldane con curvas coloreadas por phi0."
    )
    parser.add_argument("--nphi", type=int, default=21, help="Numero de valores de phi0.")
    parser.add_argument("--phi-min", type=float, default=-0.5 * np.pi, help="Limite inferior de phi0 (rad).")
    parser.add_argument("--phi-max", type=float, default=0.5 * np.pi, help="Limite superior de phi0 (rad).")
    parser.add_argument("--nomega", type=int, default=300, help="Numero de frecuencias del espectro.")
    parser.add_argument("--omega-min", type=float, default=1.0e-3, help="Frecuencia minima (a.u.).")
    parser.add_argument("--omega-max", type=float, default=0.12, help="Frecuencia maxima (a.u.).")
    parser.add_argument("--kpoints", type=int, default=101, help="Numero de puntos por eje del grid k desplazado.")
    parser.add_argument("--gamma", type=float, default=5.0e-4, help="Ensanchamiento/decoherencia efectiva (a.u.).")
    parser.add_argument("--mu", type=float, default=0.0, help="Nivel de Fermi (a.u.).")
    parser.add_argument("--temperature-au", type=float, default=0.0, help="Temperatura en unidades atomicas.")
    parser.add_argument("--spin-deg", type=int, default=1, help="Degeneracion de spin del modelo de Haldane.")
    parser.add_argument("--t1", type=float, default=0.075, help="Hopping NN t1 (a.u.).")
    parser.add_argument("--t2", type=float, default=0.025, help="Hopping NNN t2 (a.u.).")
    parser.add_argument("--m0", type=float, default=0.0, help="Masa de subred M0 (a.u.).")
    parser.add_argument("--a0", type=float, default=1.0 / 0.529177, help="Constante de red a0 (a.u.).")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "haldane_sigma_xy_vs_phi.png"),
        help="Ruta del PNG de salida.",
    )
    parser.add_argument(
        "--data-output",
        default=str(DEFAULT_OUTPUT_DIR / "haldane_sigma_xy_vs_phi.npz"),
        help="Ruta del dataset .npz con las curvas y ejes.",
    )
    args = parser.parse_args()

    if args.nphi <= 1 or args.nomega <= 1 or args.kpoints <= 1:
        raise SystemExit("nphi, nomega y kpoints deben ser mayores que 1.")
    if args.omega_min <= 0.0 or args.omega_max <= args.omega_min:
        raise SystemExit("Se requiere 0 < omega-min < omega-max.")
    if args.gamma < 0.0:
        raise SystemExit("gamma no puede ser negativo.")

    phi_values = np.linspace(args.phi_min, args.phi_max, args.nphi, dtype=np.float64)
    omega_axis = np.linspace(args.omega_min, args.omega_max, args.nomega, dtype=np.float64)

    print(
        "[haldane-sigma-phi] construyendo barrido "
        f"nphi={args.nphi}, nomega={args.nomega}, grid k {args.kpoints}x{args.kpoints}, "
        f"phi in [{args.phi_min:.4f}, {args.phi_max:.4f}]",
        flush=True,
    )
    timer = ProgressTimer(total=1)
    sigma_curves, kx_values, ky_values = _build_sigma_phi_sweep(
        t1=float(args.t1),
        t2=float(args.t2),
        m0=float(args.m0),
        a0=float(args.a0),
        omega_axis=omega_axis,
        gamma=float(args.gamma),
        mu=float(args.mu),
        temperature_au=float(args.temperature_au),
        spin_deg=int(args.spin_deg),
        phi_values=phi_values,
        nkx=int(args.kpoints),
        nky=int(args.kpoints),
    )
    elapsed = timer.elapsed_seconds

    output_path = Path(args.output)
    data_output_path = Path(args.data_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        data_output_path,
        phi0_axis=phi_values,
        omega_axis=omega_axis,
        omega_axis_ev=omega_axis * AU_TO_EV,
        sigma_xy_curves=sigma_curves,
        sigma_xy_real=np.real(sigma_curves),
        sigma_xy_imag=np.imag(sigma_curves),
        gamma=float(args.gamma),
        t1=float(args.t1),
        t2=float(args.t2),
        m0=float(args.m0),
        a0=float(args.a0),
        mu=float(args.mu),
        temperature_au=float(args.temperature_au),
        spin_deg=int(args.spin_deg),
        kx_axis=kx_values,
        ky_axis=ky_values,
    )

    _plot_sigma_phi_sweep(
        omega_axis=omega_axis,
        phi_values=phi_values,
        sigma_curves=sigma_curves,
        output_path=output_path,
    )

    print(
        f"[haldane-sigma-phi] listo en {format_duration(elapsed)}.\n"
        f"[haldane-sigma-phi] plot: {output_path}\n"
        f"[haldane-sigma-phi] data: {data_output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
