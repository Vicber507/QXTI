from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.analytics.hipolito2018 import analytical_sigma1_fast
from qxti.analytics.rho_analytic import compare_rho_vs_qxti, extract_frequency_component, rho_order_s, sigma1_kubo
from qxti.grids import TimeGrid


def toy_hamiltonian(kx: float, ky: float, kz: float) -> np.ndarray:
    del kz
    mass = 0.4
    velocity = 1.2
    return np.array(
        [
            [mass, velocity * (kx - 1j * ky)],
            [velocity * (kx + 1j * ky), -mass],
        ],
        dtype=np.complex128,
    )


def test_rho_order_s_scales_with_field_amplitude() -> None:
    kx, ky, kz = 0.13, -0.09, 0.0
    omega = 0.42
    gamma = 1.0e-3
    mu = 0.0
    t_au = 0.0
    scale = 0.37
    e_field = np.array([2.0e-4, 0.0, 0.0], dtype=np.complex128)

    reference = rho_order_s(
        toy_hamiltonian,
        kx,
        ky,
        kz,
        e_field,
        omega,
        gamma,
        mu,
        t_au,
        max_order=3,
    )
    scaled = rho_order_s(
        toy_hamiltonian,
        kx,
        ky,
        kz,
        scale * e_field,
        omega,
        gamma,
        mu,
        t_au,
        max_order=3,
    )

    for order in (1, 2, 3):
        np.testing.assert_allclose(
            scaled[order],
            (scale ** order) * reference[order],
            rtol=5.0e-6,
            atol=5.0e-11,
        )


def test_analytical_sigma1_fast_matches_sigma1_kubo() -> None:
    omega_axis = np.array([0.25, 0.60], dtype=np.float64)
    kwargs = dict(
        gamma=1.0e-3,
        mu=0.0,
        spin_deg=1,
        bz_bounds=(-np.pi / 2.0, np.pi / 2.0),
        dimension=2,
    )

    sigma_reference = sigma1_kubo(
        toy_hamiltonian,
        (9, 9),
        omega_axis,
        T_K=0.0,
        verbose=False,
        **kwargs,
    )
    sigma_fast = analytical_sigma1_fast(
        toy_hamiltonian,
        (9, 9),
        omega_axis,
        T=0.0,
        **kwargs,
    )

    np.testing.assert_allclose(sigma_fast, sigma_reference, atol=1.0e-10, rtol=1.0e-10)


def test_extract_frequency_component_matches_qxti_fft_convention() -> None:
    timegrid = TimeGrid(0.0, 12.0, 256, fft_window="hann", zero_padding=True, padding_factor=2)
    rng = np.random.default_rng(1234)
    signal_t = rng.normal(size=(len(timegrid), 2, 2)) + 1j * rng.normal(size=(len(timegrid), 2, 2))

    window = np.asarray(timegrid.apply_window(np.ones(len(timegrid), dtype=float)), dtype=np.float64)
    weighted = signal_t * window[:, np.newaxis, np.newaxis]
    manual_spectrum = timegrid.dt * np.fft.fft(
        weighted,
        n=len(timegrid) * timegrid.padding_factor,
        axis=0,
    )
    omega_axis = np.asarray(timegrid.frequency_axis(), dtype=np.float64)
    target_omega = float(omega_axis[17])

    sampled_omega, sampled_component = extract_frequency_component(
        signal_t,
        target_omega=target_omega,
        timegrid=timegrid,
    )

    index = int(np.argmin(np.abs(omega_axis - target_omega)))
    np.testing.assert_allclose(sampled_omega, omega_axis[index], atol=1.0e-14)
    np.testing.assert_allclose(sampled_component, manual_spectrum[index], atol=1.0e-12, rtol=1.0e-12)


def test_compare_rho_vs_qxti_normalizes_by_field_spectrum(tmp_path: Path) -> None:
    nt = 128
    timegrid = TimeGrid(0.0, 40.0, nt, fft_window="none", zero_padding=False, padding_factor=2)
    omega_axis = np.asarray(timegrid.frequency_axis(), dtype=np.float64)
    omega_index = 7
    order = 2
    omega = float(omega_axis[omega_index])
    target_index = order * omega_index
    k_point = np.array([[0.17, -0.08, 0.0]], dtype=np.float64)
    e_plus = 3.0e-4 + 8.0e-5j
    e_field_analytic = np.array([2.5e-4 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    gamma = 1.0e-3
    mu = 0.0
    t_au = 0.0

    rho_analytic = rho_order_s(
        toy_hamiltonian,
        float(k_point[0, 0]),
        float(k_point[0, 1]),
        float(k_point[0, 2]),
        e_field_analytic,
        omega,
        gamma,
        mu,
        t_au,
        max_order=order,
    )[order]

    rho_spectrum = np.zeros((nt, 2, 2), dtype=np.complex128)
    rho_spectrum[target_index] = rho_analytic * (e_plus / e_field_analytic[0]) ** order
    rho_t = np.fft.ifft(rho_spectrum, axis=0) / timegrid.dt

    field_spectrum = np.zeros((nt, 3), dtype=np.complex128)
    field_spectrum[omega_index, 0] = e_plus
    field_t = np.fft.ifft(field_spectrum, axis=0) / timegrid.dt

    rho_tensor = rho_t[np.newaxis, :, :, :]
    rho_path = tmp_path / "rho_order_2.npy"
    np.save(rho_path, rho_tensor)

    comparison = compare_rho_vs_qxti(
        toy_hamiltonian,
        str(rho_path),
        k_point,
        timegrid.generate(),
        e_field_analytic,
        omega,
        gamma=gamma,
        mu=mu,
        T_K=0.0,
        order=order,
        k_indices=[0],
        timegrid=timegrid,
        electric_field_time=field_t,
        normalize_by_field=True,
        verbose=False,
    )

    assert len(comparison["error_rel"]) == 1
    assert comparison["error_rel"][0] < 1.0e-10
    np.testing.assert_allclose(
        comparison["rho_numeric_normalized"][0],
        comparison["rho_analytic_normalized"][0],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
