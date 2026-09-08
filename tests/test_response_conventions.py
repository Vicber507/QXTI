"""Physical sign / phase / helicity conventions of the response engines.

These are CONVENTION tests, independent of any engine-vs-engine agreement in
magnitude: they check things that physics fixes unambiguously.

* Linear conductivity: ``Re sigma_xx(w) > 0`` above the gap (passivity) and the
  mesh order 1 equals a Kubo formula written from scratch (electron, ``H' = +E.r``,
  ``j = -v``, e^{-i w t}) -- ratio +1, not -1.
* The time-domain perturbative path returns a REAL current with no ad-hoc per-order
  phase, and agrees with the closed form in COMPLEX amplitude at every s*w0.
* ``helicity_components`` labels R = positive helicity = the rotation sense of a
  drive with ``ellip > 0`` (on numpy +w FFT bins), for a tilted frame too; and
  ``to_helicity_basis`` labels the FFT-convention tensors the same way.
* pfddm and tddm agree in PHASE (not just magnitude) at H1 for a circular drive and
  both put the H1 current in the same helicity channel as the field.
"""
from __future__ import annotations

from dataclasses import replace as R
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qxti.analytics.mesh_response import (  # noqa: E402
    harmonic_currents, precompute_band_data, time_domain_currents, uniform_mp_grid,
)
from qxti.graphics.plot_harmonics import helicity_components  # noqa: E402
from qxti.graphics.plot_susceptibility_tensor import to_helicity_basis  # noqa: E402
from qxti.physics.laser import Laser  # noqa: E402

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_BOUNDS = ((-np.pi, np.pi), (-np.pi, np.pi), (0.0, 0.0))


def _qwz(M: float):
    """Qi-Wu-Zhang two-band model: gapped for M != 0, +-2 (Chern for |M| < 2)."""
    def H(kx, ky, kz):
        return np.sin(kx) * _SX + np.sin(ky) * _SY + (M + np.cos(kx) + np.cos(ky)) * _SZ
    return H


def _kubo_from_scratch(H, kpts, w, omega, gamma, mu=0.0):
    """sigma_ab = sum_k w sum_{n!=m} (-v_a)_nm rho1_mn / E_b, with the physical
    rho1_mn = i E_b (D_k rho0)_mn / (w + i g - (E_m - E_n)),
    (D_k rho0)_mn = v_b,mn (f_m - f_n)/(E_m - E_n).   (T = 0, no Drude term)"""
    dk = 1e-5
    s = np.zeros((2, 2), dtype=complex)
    for kp, wk in zip(kpts, w):
        e, U = np.linalg.eigh(H(*kp))
        f = (e <= mu).astype(float)
        v = [U.conj().T @ ((H(kp[0] + dk, kp[1], 0) - H(kp[0] - dk, kp[1], 0)) / (2 * dk)) @ U,
             U.conj().T @ ((H(kp[0], kp[1] + dk, 0) - H(kp[0], kp[1] - dk, 0)) / (2 * dk)) @ U]
        for a in range(2):
            for b in range(2):
                for n in range(2):
                    for m in range(2):
                        if n == m:
                            continue
                        rho1 = 1j * v[b][m, n] * (f[m] - f[n]) / (e[m] - e[n]) \
                            / (omega + 1j * gamma - (e[m] - e[n]))
                        s[a, b] += wk * (-v[a][n, m]) * rho1
    return s


@pytest.mark.parametrize("M", [3.0, -1.0])
def test_linear_conductivity_has_positive_absorption_and_matches_kubo(M):
    H = _qwz(M)
    N = 24
    kpts, w = uniform_mp_grid(_BOUNDS, (N, N, 1))
    band = precompute_band_data(H, kpts, (N, N, 1), _BOUNDS, mu=0.0, T_au=0.0, dimension=2)
    omega, gamma = 2.6, 0.05                       # above the gap (2.0)
    J = harmonic_currents(band, w, np.array([1, 0, 0], dtype=complex), omega, 1, gamma=gamma)[1]
    sigma_mesh = -J[:2]                            # J = -sum_k w Tr[v rho]  (electron)
    sigma_ref = _kubo_from_scratch(H, kpts, w, omega, gamma)
    assert sigma_mesh[0].real > 0.0, "Re sigma_xx must be positive (absorption)"
    np.testing.assert_allclose(sigma_mesh[0], sigma_ref[0, 0], rtol=1e-8)
    np.testing.assert_allclose(sigma_mesh[1], sigma_ref[1, 0], rtol=1e-8)


def test_time_domain_current_is_real_and_matches_closed_form_in_phase():
    H = _qwz(3.0)
    N = 16
    kpts, w = uniform_mp_grid(_BOUNDS, (N, N, 1))
    band = precompute_band_data(H, kpts, (N, N, 1), _BOUNDS, mu=0.0, T_au=0.0, dimension=2)
    dt, Nt = 0.05, 8000
    t = np.arange(Nt) * dt
    T = Nt * dt
    omega = 2.0 * np.pi * 64 / T                   # exactly on the FFT grid
    env = np.sin(np.pi * t / T) ** 2
    E0 = 2e-2
    E_t = np.zeros((Nt, 3))
    E_t[:, 0] = E0 * env * np.cos(omega * t)       # circular drive: (x + i y) e^{-i w t}
    E_t[:, 1] = E0 * env * np.sin(omega * t)
    td = time_domain_currents(band, w, E_t[:, :2], dt, 3, gamma=0.02, gamma_pop=0.02)
    scale = max(np.abs(td["J_t"][s].real).max() for s in (1, 2, 3))   # order 1 sets the scale
    for s in (1, 2, 3):                        # (order 2 is ~0 here: QWZ has inversion symmetry)
        Jt = td["J_t"][s]
        assert np.abs(Jt.imag).max() < 1e-10 * scale, f"order {s}: J(t) is not real"
    # closed form with E(t) = E_cw e^{-i w t} + c.c.  ->  E_cw = (E0/2)(1, i)
    Jcw = harmonic_currents(band, w, np.array([E0 / 2, 1j * E0 / 2, 0]), omega, 3,
                            gamma=0.02, gamma_pop=0.02)
    freq = 2.0 * np.pi * np.fft.fftfreq(Nt, d=dt)
    for s, tol in ((1, 1e-6), (3, 2e-2)):
        b = int(np.argmin(np.abs(freq + s * omega)))       # e^{-i s w t} sits in the -s w bin
        for a in range(2):
            amp_td = td["J_omega"][s][b, a] / Nt
            amp_cw = Jcw[s][a] * np.mean(env ** s)
            ratio = amp_td / amp_cw
            assert abs(ratio - 1.0) < tol, f"H{s} comp {a}: complex ratio td/cw = {ratio}"


@pytest.mark.parametrize("ellip", [1.0, -1.0])
def test_helicity_components_label_follows_the_drive(ellip):
    d = np.array([1.0, 1.0, 2.0]); d /= np.linalg.norm(d)      # (112) tilted propagation
    L = Laser(omega=0.2, E0=1.0, ellip=ellip, ncycles=8.0, envname="gauss",
              thetaz=float(np.arccos(d[2])), phiz=float(np.arctan2(d[1], d[0])), phix=0.3)
    t = np.linspace(-400.0, 400.0, 8000)
    E = np.array([L.electric_field(x) for x in t])            # (Nt, 3) real
    S = np.fft.fft(E, axis=0)
    freq = 2.0 * np.pi * np.fft.fftfreq(t.size, d=t[1] - t[0])
    i = int(np.argmin(np.abs(freq - L.omega)))                # POSITIVE-frequency bin
    JR, JL, Jlong = helicity_components(S, L.rotation_matrix())
    assert abs(Jlong[i]) < 1e-6 * abs(S[i]).max()
    if ellip > 0:
        assert abs(JR[i]) > 1e3 * abs(JL[i]), "ellip>0 (helicity>0) must be labelled R"
    else:
        assert abs(JL[i]) > 1e3 * abs(JR[i]), "ellip<0 (helicity<0) must be labelled L"
    # power closure
    np.testing.assert_allclose(abs(JR[i]) ** 2 + abs(JL[i]) ** 2 + abs(Jlong[i]) ** 2,
                               (abs(S[i]) ** 2).sum(), rtol=1e-12)


def test_helicity_basis_labels_fft_convention_tensor_physically():
    # physical (e^{-i w t}) tensor with a known helicity structure: sigma_++ = 2, sigma_-- = 1
    ep = np.array([1, 1j, 0]) / np.sqrt(2)
    em = np.array([1, -1j, 0]) / np.sqrt(2)
    ez = np.array([0, 0, 1.0])
    sigma_phys = 2.0 * np.outer(ep, ep.conj()) + 1.0 * np.outer(em, em.conj()) + 0.5 * np.outer(ez, ez)
    sigma_fft = np.conj(sigma_phys)[None]                      # what QXTI stores (J(w)/E(w) on +w bins)
    hel, labels = to_helicity_basis(sigma_fft, 3)
    assert labels == ("+", "-", "z")
    np.testing.assert_allclose(hel[0, 0, 0], np.conj(2.0), atol=1e-12)   # "+" = positive helicity
    np.testing.assert_allclose(hel[0, 1, 1], np.conj(1.0), atol=1e-12)
    np.testing.assert_allclose(hel[0, 0, 1], 0.0, atol=1e-12)


def test_pfddm_and_tddm_agree_in_phase_and_helicity_for_a_circular_drive():
    from qxti.core import QXTIConfig
    from qxti.analytics.theory_response import compute_hhg_spectrum
    from qxti.analytics.tddm import compute_hhg_spectrum_tddm

    cfg = QXTIConfig.from_file(str(PROJECT_ROOT / "inputs" / "inputParams.haldane_trivial.cfg"))
    cfg = R(cfg, kgrid=R(cfg.kgrid, k_points=(12, 12)))
    cfg = R(cfg, timegrid=R(cfg.timegrid, dt=0.5, zero_padding=False))
    cfg = R(cfg, laser=R(cfg.laser, ellip=1.0, E0=7e-4))
    cfg = R(cfg, cmd=R(cfg.cmd, max_order=2))
    w0 = float(cfg.laser.omega)

    def peaks(ds):
        w = np.asarray(ds["omega_axis"]); S = np.asarray(ds["current_spectrum"])
        E_t = np.asarray(ds["electric_field_time"]); J_t = np.asarray(ds["current_time"])
        t = np.asarray(ds["time_axis"]); dt = t[1] - t[0]
        i1 = int(np.argmin(np.abs(w - w0)))
        ER, EL, _ = helicity_components(np.fft.fft(E_t, axis=0), None)
        JR, JL, _ = helicity_components(S, None)
        absorbed = float(np.sum(J_t * E_t) * dt)
        return S[i1, :2], (abs(ER[i1]), abs(EL[i1])), (abs(JR[i1]), abs(JL[i1])), absorbed

    Jp, Ep, JRLp, Wp = peaks(compute_hhg_spectrum(cfg, max_order=2, progress=False)["dataset"])
    Jt, Et, JRLt, Wt = peaks(compute_hhg_spectrum_tddm(cfg, max_order=2, progress=False)["dataset"])
    assert Ep[0] > 1e3 * Ep[1] and Et[0] > 1e3 * Et[1]          # field: ellip>0 -> R
    # Haldane is C3-symmetric, so H1 is purely co-rotating; the small opposite-helicity
    # leakage (~0.2-0.4%) is the square 12x12 k-grid breaking C3, not physics.
    assert JRLp[0] > 50 * JRLp[1], "pfddm H1 must rotate with the field (R)"
    assert JRLt[0] > 50 * JRLt[1], "tddm H1 must rotate with the field (R)"
    assert Wp > 0.0 and Wt > 0.0, "absorbed energy must be positive in both engines"
    for a in range(2):
        ratio = Jp[a] / Jt[a]
        assert abs(abs(ratio) - 1.0) < 0.05, f"|H1| pfddm/tddm = {abs(ratio)}"
        assert abs(np.degrees(np.angle(ratio))) < 5.0, f"H1 phase pfddm vs tddm = {np.degrees(np.angle(ratio))} deg"
