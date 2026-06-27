#!/usr/bin/env python3
from __future__ import annotations

"""Mapa de fase de Haldane coloreado por Re[sigma_xy(omega -> 0)].

Usa la formula de Kubo lineal de primer orden sobre un grid k desplazado
(Monkhorst-Pack) y dibuja el mapa sobre el espacio de fase canonico
(phi0, M0/t2). El estilo grafico reutiliza la estetica paper-like del
plotter de susceptibilidad/conductividad de QXTI.

Ejemplo:
    python tools/plot_haldane_sigma_xy_phase_diagram.py \
        --nphi 81 --nm 81 --kpoints 61 \
        --omega-dc 1.0e-4 --gamma 5.0e-4
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.graphics.plot_susceptibility_tensor import apply_paper_style
from qxti.utils.progress import ProgressTimer, format_duration

DEFAULT_OUTPUT_DIR = Path("outputs") / "haldane_sigma_xy_phase_diagram"


def _load_haldane_module():
    module_path = PROJECT_ROOT / "models" / "haldane.py"
    spec = importlib.util.spec_from_file_location("qxti_haldane_model_phase_map", module_path)
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


def _sigma_xy_low_frequency_haldane(
    *,
    t1: float,
    t2: float,
    phi0: float,
    m_over_t2: float,
    omega_dc: float,
    gamma: float,
    mu: float,
    temperature_au: float,
    spin_deg: int,
    cache: dict[str, np.ndarray | float],
) -> complex:
    m0 = float(m_over_t2) * float(t2)
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

    omega_complex = complex(float(omega_dc), float(gamma))
    eps_mn = energies[:, :, None] - energies[:, None, :]
    f_nm = occupations[:, None, :] - occupations[:, :, None]
    valid = (~np.eye(2, dtype=bool))[None, :, :] & (np.abs(eps_mn) > 1.0e-20)

    with np.errstate(divide="ignore", invalid="ignore"):
        interband = np.where(
            valid,
            np.transpose(vx_band, (0, 2, 1)) * vy_band * f_nm / eps_mn / (omega_complex - eps_mn),
            0.0 + 0.0j,
        )

    diagonal_vx = np.diagonal(vx_band, axis1=1, axis2=2)
    diagonal_vy = np.diagonal(vy_band, axis1=1, axis2=2).real
    intraband = (-1j / omega_complex) * np.sum(diagonal_vx * (dfde * diagonal_vy), axis=1)
    s_k = np.sum(interband, axis=(1, 2)) + intraband

    sigma_xy = -1j * float(spin_deg) * np.sum((weight_per_k / bz_area) * s_k)
    return complex(sigma_xy)


def _build_phase_map(
    *,
    t1: float,
    t2: float,
    a0: float,
    omega_dc: float,
    gamma: float,
    mu: float,
    temperature_au: float,
    spin_deg: int,
    phi_values: np.ndarray,
    m_over_t2_values: np.ndarray,
    nkx: int,
    nky: int,
) -> tuple[np.ndarray, np.ndarray]:
    haldane_module = _load_haldane_module()
    vec_a = np.asarray(haldane_module._nn_vectors(a0), dtype=np.float64)
    vec_b = np.asarray(haldane_module._nnn_vectors(a0), dtype=np.float64)
    cache = _build_precomputed_k_cache(a0=a0, nkx=nkx, nky=nky, vec_a=vec_a, vec_b=vec_b)

    sigma_map = np.zeros((m_over_t2_values.size, phi_values.size), dtype=np.complex128)
    timer = ProgressTimer(total=int(m_over_t2_values.size * phi_values.size), min_completed_for_eta=max(4, phi_values.size))

    for im, m_over_t2 in enumerate(m_over_t2_values):
        row_start = timer.elapsed_seconds
        for ip, phi0 in enumerate(phi_values):
            sigma_map[im, ip] = _sigma_xy_low_frequency_haldane(
                t1=t1,
                t2=t2,
                phi0=float(phi0),
                m_over_t2=float(m_over_t2),
                omega_dc=omega_dc,
                gamma=gamma,
                mu=mu,
                temperature_au=temperature_au,
                spin_deg=spin_deg,
                cache=cache,
            )
            timer.advance()
        print(
            f"[haldane-phase] row {im + 1}/{m_over_t2_values.size} "
            f"(M0/t2={m_over_t2:+.3f}) done in {format_duration(timer.elapsed_seconds - row_start)}; "
            f"elapsed {format_duration(timer.elapsed_seconds)}, ETA {timer.eta_text()}",
            flush=True,
        )

    return sigma_map, np.asarray(cache["kx_values"]), np.asarray(cache["ky_values"])


def _plot_phase_map(
    *,
    phi_values: np.ndarray,
    m_over_t2_values: np.ndarray,
    sigma_map: np.ndarray,
    output_path: Path,
    omega_dc: float,
    gamma: float,
    nkx: int,
    nky: int,
    value_part: str,
) -> None:
    apply_paper_style()

    phase_cmap = LinearSegmentedColormap.from_list(
        "haldane_phase",
        [
            "#163B5C",  # azul profundo
            "#2F6C8F",  # azul petroleo
            "#8DB7B5",  # salvia fria
            "#F3E9D2",  # marfil calido
            "#E7B97A",  # arena dorada
            "#C96B4B",  # terracota
            "#7A1F2B",  # vino oscuro
        ],
        N=256,
    )

    if value_part == "real":
        values = np.real(sigma_map)
        colorbar_label = rf"$\Re\,\sigma_{{xy}}^{{(1)}}(\omega_{{\mathrm{{dc}}}})$"
    elif value_part == "imag":
        values = np.imag(sigma_map)
        colorbar_label = rf"$\Im\,\sigma_{{xy}}^{{(1)}}(\omega_{{\mathrm{{dc}}}})$"
    else:
        values = np.abs(sigma_map)
        colorbar_label = rf"$|\sigma_{{xy}}^{{(1)}}(\omega_{{\mathrm{{dc}}}})|$"

    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    x_grid, y_grid = np.meshgrid(phi_values, m_over_t2_values, indexing="xy")

    if value_part in {"real", "imag"} and np.nanmin(values) < 0.0 < np.nanmax(values):
        norm = TwoSlopeNorm(vmin=float(np.nanmin(values)), vcenter=0.0, vmax=float(np.nanmax(values)))
    else:
        norm = None

    image = axis.pcolormesh(
        x_grid,
        y_grid,
        values,
        shading="auto",
        cmap=phase_cmap,
        norm=norm,
        alpha=0.84,
    )

    critical = 3.0 * np.sqrt(3.0) * np.sin(phi_values)
    axis.plot(phi_values, critical, color="black", linewidth=1.35, linestyle="--", alpha=0.95)
    axis.plot(phi_values, -critical, color="black", linewidth=1.35, linestyle="--", alpha=0.95)

    axis.set_xlabel(r"$\phi_0\;(\mathrm{rad})$")
    axis.set_ylabel(r"$M_0 / t_2$")
    axis.set_title(rf"Haldane: {colorbar_label} near $\omega \to 0$")

    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(colorbar_label + r"$\;(\mathrm{a.u.})$")

    text_x = float(phi_values.min()) + 0.03 * float(phi_values.max() - phi_values.min())
    text_y = float(m_over_t2_values.max()) - 0.10 * float(m_over_t2_values.max() - m_over_t2_values.min())
    axis.text(text_x, text_y, "C=+1", fontsize=11.5, color="black")
    axis.text(text_x, -text_y, "C=-1", fontsize=11.5, color="black")

    axis.set_xlim(float(phi_values.min()), float(phi_values.max()))
    axis.set_ylim(float(m_over_t2_values.min()), float(m_over_t2_values.max()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot de mapa de fase del modelo de Haldane usando sigma_xy(omega~0)."
    )
    parser.add_argument("--nphi", type=int, default=81, help="Numero de puntos en phi0.")
    parser.add_argument("--nm", type=int, default=81, help="Numero de puntos en M0/t2.")
    parser.add_argument("--kpoints", type=int, default=61, help="Numero de puntos por eje del grid k desplazado.")
    parser.add_argument("--phi-min", type=float, default=-np.pi, help="Limite inferior de phi0 (rad).")
    parser.add_argument("--phi-max", type=float, default=np.pi, help="Limite superior de phi0 (rad).")
    parser.add_argument("--m-over-t2-min", type=float, default=-6.0, help="Limite inferior de M0/t2.")
    parser.add_argument("--m-over-t2-max", type=float, default=6.0, help="Limite superior de M0/t2.")
    parser.add_argument("--omega-dc", type=float, default=1.0e-4, help="Frecuencia pequena que aproxima omega->0 (a.u.).")
    parser.add_argument("--gamma", type=float, default=5.0e-4, help="Ensanchamiento/decoherencia efectiva (a.u.).")
    parser.add_argument("--mu", type=float, default=0.0, help="Nivel de Fermi (a.u.).")
    parser.add_argument("--temperature-au", type=float, default=0.0, help="Temperatura en unidades atomicas.")
    parser.add_argument("--spin-deg", type=int, default=1, help="Degeneracion de spin del modelo de Haldane.")
    parser.add_argument("--t1", type=float, default=0.075, help="Hopping NN t1 (a.u.).")
    parser.add_argument("--t2", type=float, default=0.025, help="Hopping NNN t2 (a.u.).")
    parser.add_argument("--a0", type=float, default=1.0 / 0.529177, help="Constante de red a0 (a.u.).")
    parser.add_argument("--part", choices=("real", "imag", "abs"), default="real", help="Parte del tensor a colorear.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "haldane_sigma_xy_dc.png"),
        help="Ruta del PNG de salida.",
    )
    parser.add_argument(
        "--data-output",
        default=str(DEFAULT_OUTPUT_DIR / "haldane_sigma_xy_dc.npz"),
        help="Ruta del dataset .npz con el mapa y ejes.",
    )
    args = parser.parse_args()

    if args.nphi <= 1 or args.nm <= 1 or args.kpoints <= 1:
        raise SystemExit("nphi, nm y kpoints deben ser mayores que 1.")
    if args.omega_dc <= 0.0:
        raise SystemExit("omega-dc debe ser estrictamente positivo.")
    if args.gamma < 0.0:
        raise SystemExit("gamma no puede ser negativo.")
    if args.t2 == 0.0:
        raise SystemExit("t2 no puede ser cero si el eje vertical es M0/t2.")

    phi_values = np.linspace(args.phi_min, args.phi_max, args.nphi, dtype=np.float64)
    m_over_t2_values = np.linspace(args.m_over_t2_min, args.m_over_t2_max, args.nm, dtype=np.float64)

    print(
        "[haldane-phase] construyendo mapa "
        f"{args.nm}x{args.nphi} con grid k {args.kpoints}x{args.kpoints}, "
        f"omega_dc={args.omega_dc:.2e}, gamma={args.gamma:.2e}",
        flush=True,
    )
    start = ProgressTimer(total=1)
    sigma_map, kx_values, ky_values = _build_phase_map(
        t1=float(args.t1),
        t2=float(args.t2),
        a0=float(args.a0),
        omega_dc=float(args.omega_dc),
        gamma=float(args.gamma),
        mu=float(args.mu),
        temperature_au=float(args.temperature_au),
        spin_deg=int(args.spin_deg),
        phi_values=phi_values,
        m_over_t2_values=m_over_t2_values,
        nkx=int(args.kpoints),
        nky=int(args.kpoints),
    )
    elapsed = start.elapsed_seconds

    output_path = Path(args.output)
    data_output_path = Path(args.data_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        data_output_path,
        phi0_axis=phi_values,
        m_over_t2_axis=m_over_t2_values,
        sigma_xy_map=sigma_map,
        sigma_xy_real=np.real(sigma_map),
        sigma_xy_imag=np.imag(sigma_map),
        omega_dc=float(args.omega_dc),
        gamma=float(args.gamma),
        t1=float(args.t1),
        t2=float(args.t2),
        a0=float(args.a0),
        mu=float(args.mu),
        temperature_au=float(args.temperature_au),
        spin_deg=int(args.spin_deg),
        kx_axis=kx_values,
        ky_axis=ky_values,
    )

    _plot_phase_map(
        phi_values=phi_values,
        m_over_t2_values=m_over_t2_values,
        sigma_map=sigma_map,
        output_path=output_path,
        omega_dc=float(args.omega_dc),
        gamma=float(args.gamma),
        nkx=int(args.kpoints),
        nky=int(args.kpoints),
        value_part=str(args.part),
    )

    print(
        f"[haldane-phase] listo en {format_duration(elapsed)}.\n"
        f"[haldane-phase] plot: {output_path}\n"
        f"[haldane-phase] data: {data_output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
