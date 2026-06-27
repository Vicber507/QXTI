"""Analytical (theory) linear response spectrum, comparable to the CMD simulation.

Computes sigma^(1)(omega) and chi^(1)(omega) directly in the frequency domain
(Kubo / Hipolito-Taghizadeh-Pedersen 2018 Eq. A2) using QXTI's OWN gauge-consistent
band operators (band_gauge_frame.current / connection / omega_matrix) and the SAME
Brillouin-zone integration weights as the XTP simulation. This guarantees the
normalization matches the time-domain result, so the two are directly comparable.

The whole spectrum is produced in a single pass over the k-grid (no laser-frequency
sweep, no time propagation), so it is typically orders of magnitude faster than the
simulation.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qxti.core.config import QXTIConfig
from qxti.core.simulation import QXTISimulation
from qxti.response.distributions import (
    bose_einstein,
    fermi_dirac,
    full_occupation,
    maxwell_boltzmann,
    valence_occupation,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_DISTRIBUTIONS = {
    "fermi_dirac": fermi_dirac,
    "valence_occupation": valence_occupation,
    "full_occupation": full_occupation,
    "maxwell_boltzmann": maxwell_boltzmann,
    "bose_einstein": bose_einstein,
}


def _resolve_distribution(name: str):
    key = (name or "fermi_dirac").strip().lower()
    aliases = {"valence": "valence_occupation", "fd": "fermi_dirac",
               "mb": "maxwell_boltzmann", "be": "bose_einstein"}
    key = aliases.get(key, key)
    return _DISTRIBUTIONS.get(key, fermi_dirac)


def compute_linear_response_spectrum(
    config: QXTIConfig,
    omega_axis: FloatArray,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    """Return analytical sigma^(1)(omega) and chi^(1)(omega) on ``omega_axis``.

    sigma^(1)_ab(omega) = sum_k w_k * sum_{m!=n} J_a,mn(k) * A_b,nm(k)
                          * (f_m - f_n) / (omega + i*gamma - (e_n - e_m))

    where J_a = current operator (= -v) and A_b = Berry connection, both in QXTI's
    smoothed band gauge, w_k are the XTP BZ weights, and gamma = 1/T2.

    chi^(1)_ab(omega) = sigma^(1)_ab(omega) / (-i * omega)   (P = J / (-i omega)).

    Returns a dict with ``omega_axis``, ``sigma`` (nw, dim, dim), ``chi``,
    ``runtime_seconds``, and metadata.
    """
    t_start = time.perf_counter()

    sim = QXTISimulation(config=config)
    hamiltonian = sim.build_hamiltonian()
    kgrid = sim.build_kgrid(hamiltonian)

    dim = hamiltonian.dimension
    nb = hamiltonian.basis_size
    k_points = kgrid.points()
    nk = kgrid.total_points

    # Same BZ integration weights as the simulation (XTP), built from the same
    # static quadrature rule so the normalization matches exactly.
    from qxti.response.xtp import XTP
    bounds = hamiltonian.reciprocal_box_bounds()
    periodic = bool(getattr(kgrid, "shifted", False))
    axis_values = (kgrid.kx_values, kgrid.ky_values, kgrid.kz_values)
    axis_weights = []
    for axis in range(3):
        active = axis < dim
        axis_weights.append(
            XTP._axis_integration_weights(
                np.asarray(axis_values[axis], dtype=np.float64),
                bounds[axis] if active else (0.0, 0.0),
                active=active,
                periodic=periodic,
            )
        )
    weights = (
        axis_weights[0][:, None, None]
        * axis_weights[1][None, :, None]
        * axis_weights[2][None, None, :]
    ).reshape(-1).astype(np.float64)  # (nk,)

    ccfg = config.susceptibility_solver
    distribution = _resolve_distribution(ccfg.distribution)
    mu = float(ccfg.fermi_level)
    temperature = float(ccfg.temperature)
    gamma = 0.0 if ccfg.coherence_time <= 0 else 1.0 / float(ccfg.coherence_time)

    omega_axis = np.asarray(omega_axis, dtype=np.float64)
    nw = omega_axis.size
    sigma = np.zeros((nw, dim, dim), dtype=np.complex128)

    # BATCHED over k-points in chunks: one vectorized LAPACK eigh per chunk
    # (shared between energies AND the velocity transform -- removing the
    # redundant per-k diagonalization), plus a vectorized BZ sum. Memory stays
    # bounded by the chunk size, so even millions of k-points run fast and
    # without OOM. The BZ-summed sigma is gauge-invariant, so the raw eigenvector
    # gauge gives the same result as the smoothed one.
    from qxti.utils.progress import ProgressTimer
    timer = ProgressTimer(total=nk)
    offdiag_mask = ~np.eye(nb, dtype=bool)
    Hf = hamiltonian._matrix_at
    dk = 1e-4
    # Chunk size: keep the (C, nw, nb, nb) work array near ~100 MB.
    chunk = max(256, min(20000, int(1.0e8 / max(nw * nb * nb * 16, 1))))

    def _Hbatch(kc):  # (C,3) -> (C,nb,nb)
        return np.array([Hf(float(k[0]), float(k[1]), float(k[2])) for k in kc],
                        dtype=np.complex128)

    omega_c = omega_axis[:, None, None] + 1j * gamma  # (nw,1,1)
    for start in range(0, nk, chunk):
        stop = min(start + chunk, nk)
        kc = k_points[start:stop]                       # (C,3)
        C = kc.shape[0]
        H0 = _Hbatch(kc)
        energies, U = np.linalg.eigh(H0)                # (C,nb), (C,nb,nb)
        Udag = np.conj(np.transpose(U, (0, 2, 1)))
        # Velocity in band basis per active axis via batched central differences.
        vel = []
        for axis in range(dim):
            shift = np.zeros(3); shift[axis] = dk
            dH = (_Hbatch(kc + shift) - _Hbatch(kc - shift)) / (2 * dk)
            vel.append(Udag @ dH @ U)                   # (C,nb,nb) band basis
        f = np.asarray(distribution(energies, mu, temperature), dtype=np.float64)  # (C,nb)
        w_c = weights[start:stop]                        # (C,)

        eps = energies[:, :, None] - energies[:, None, :]     # (C,n,m)
        fmn = f[:, None, :] - f[:, :, None]                   # (C,n,m) = f_m - f_n
        valid = offdiag_mask[None] & (np.abs(eps) > 1e-20)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_eps = np.where(valid, 1.0 / eps, 0.0)         # (C,n,m)
            denom = omega_c[None] - eps[:, None, :, :]        # (C,nw,n,m)
            inv_denom = np.where(valid[:, None], 1.0 / denom, 0.0)

        for a in range(dim):
            Ja = -vel[a]                                      # (C,n,m)
            for b in range(dim):
                Ab = 1j * vel[b] * inv_eps                    # A_b,nm
                # num[c,n,m] = J_a,mn * A_b,nm * (f_m - f_n) * w_k
                num = (np.transpose(Ja, (0, 2, 1)) * Ab) * fmn * w_c[:, None, None]
                sigma[:, a, b] += np.einsum("cnm,cwnm->w", num, inv_denom, optimize=True)

        timer.completed = stop
        if progress:
            print(
                f"[theory] k-points {stop}/{nk} ({100 * stop // nk}%), "
                f"elapsed {timer.elapsed_seconds:.1f}s, ETA {timer.eta_text()}",
                flush=True,
            )

    if progress:
        print(f"[theory] done: {nk}/{nk} k-points in {timer.elapsed_seconds:.1f}s")

    # chi = sigma / (-i omega); guard omega=0
    safe_omega = np.where(np.abs(omega_axis) > 1e-30, omega_axis, np.nan)
    chi = sigma / (-1j * safe_omega)[:, None, None]

    # Match QXTI's FFT sign convention. QXTI extracts sigma = J(omega)/E(omega)
    # with numpy's forward FFT (e^{-i omega t}), so its spectrum equals the
    # complex conjugate of the physical retarded response computed above
    # (sigma_fft(omega) = sigma_phys(-omega) = sigma_phys(omega)* for a real signal).
    sigma = np.conj(sigma)
    chi = np.conj(chi)

    runtime = time.perf_counter() - t_start
    return {
        "omega_axis": omega_axis,
        "sigma": sigma,
        "chi": chi,
        "runtime_seconds": runtime,
        "dimension": dim,
        "n_kpoints": nk,
        "gamma": gamma,
        "method": "theory",
    }


def compute_hhg_spectrum(
    config: QXTIConfig,
    *,
    max_order: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Perturbative (theory) HHG current spectrum J(omega), comparable to the CMD run.

    Uses the actual laser pulse E(t) of the config:

      - Order 1 (linear): J^(1)(omega) = sigma^(1)(omega) * E(omega)  -- full spectrum.
      - Orders s>=2: harmonic peak J^(s) at s*omega0 from the analytic rho^(s)(k, s*omega0)
        (closed forms of Hipolito+2018 A1b/A1c), BZ-summed and gauge-consistent.

    The linear part is essentially free (single k-grid pass); the harmonic peaks add
    one analytic rho recursion per k-point per order. This is far cheaper than the
    full time-domain solve, which integrates the density matrix over all time steps.

    Returns dict with ``omega_axis``, ``J_total`` (nw, dim), ``J_order`` {s: (nw,dim)},
    ``harmonic_peaks`` {s: complex per direction}, ``runtime_seconds``.
    """
    t_start = time.perf_counter()
    sim = QXTISimulation(config=config)
    hamiltonian = sim.build_hamiltonian()
    kgrid = sim.build_kgrid(hamiltonian)
    laser_system = sim.build_laser_system()
    timegrid = sim.build_timegrid(laser_system)
    dim = hamiltonian.dimension
    nk = kgrid.total_points

    ccfg = config.cmd
    if max_order is None:
        max_order = int(ccfg.max_order)
    omega0 = float(config.laser.omega)
    gamma = 0.0 if ccfg.coherence_time <= 0 else 1.0 / float(ccfg.coherence_time)
    mu = float(ccfg.fermi_level)
    temperature = float(ccfg.temperature)
    distribution = _resolve_distribution(ccfg.distribution)

    # Time/frequency axes matching the simulation FFT grid.
    Nt = int(timegrid.Nt)
    dt = (float(timegrid.tf) - float(timegrid.t0)) / max(Nt - 1, 1)
    t_axis = float(timegrid.t0) + np.arange(Nt) * dt
    E_t = np.array([laser_system.electric_field(t) for t in t_axis])  # (Nt, 3)
    E_w = np.fft.fft(E_t, axis=0) * dt                                 # (Nt, 3)
    freq = np.fft.fftfreq(Nt, d=dt) * 2.0 * np.pi

    # --- Order 1: full linear current J^(1)(omega) = sigma^(1)(omega) E(omega) ---
    if progress:
        print(
            f"[theory-HHG] order 1: linear sigma^(1)(omega) over {nk} k-points "
            "(smooth -> coarse omega grid, interpolated to the FFT axis).",
            flush=True,
        )
    # sigma(omega) is smooth, so a coarse omega axis is enough; it is interpolated
    # onto the full FFT grid below. Keep it small so the k-pass stays fast.
    omega_sigma = np.linspace(max(freq.min(), -3 * max_order * omega0 - 1.0),
                              3 * max_order * omega0 + 1.0, 200)
    lin = compute_linear_response_spectrum(config, omega_sigma, progress=progress)
    sigma1 = lin["sigma"]  # (n_omega_sigma, dim, dim)
    # Interpolate sigma(omega) onto the FFT freq grid (per component), then J=sigma*E
    J_total = np.zeros((Nt, dim), dtype=np.complex128)
    J_order: dict[int, np.ndarray] = {}
    J1 = np.zeros((Nt, dim), dtype=np.complex128)
    for a in range(dim):
        for b in range(dim):
            sig_re = np.interp(freq, omega_sigma, sigma1[:, a, b].real)
            sig_im = np.interp(freq, omega_sigma, sigma1[:, a, b].imag)
            J1[:, a] += (sig_re + 1j * sig_im) * E_w[:, b]
    J_order[1] = J1
    J_total += J1

    # Linear current in time (exact): J^(1)(t) = IFFT[ sigma^(1)(omega) E(omega) ].
    J1_t = np.fft.ifft(J1, axis=0).real  # (Nt, dim)

    # Pulse envelope and CW field amplitude from the analytic signal of E(t).
    analytic = np.zeros((Nt, dim), dtype=np.complex128)
    pos = freq > 0
    for d in range(dim):
        spec = np.fft.fft(E_t[:, d])
        spec_a = np.zeros_like(spec)
        spec_a[pos] = 2.0 * spec[pos]
        spec_a[np.abs(freq) < 1e-30] = spec[np.abs(freq) < 1e-30]
        analytic[:, d] = np.fft.ifft(spec_a)
    env_t = np.abs(analytic).max(axis=1)
    env_peak = float(env_t.max()) if env_t.max() > 0 else 1.0
    env_norm = env_t / env_peak                       # (Nt,) normalized to 1
    i_peak = int(np.argmax(env_norm))
    # +omega0 complex amplitude per direction (half the analytic amplitude at peak).
    E_cw = 0.5 * analytic[i_peak, :dim]

    # --- Orders >=2: harmonic peaks via analytic rho^(s)(k, s*omega0) ---
    harmonic_peaks: dict[int, np.ndarray] = {}
    J_harm_t = np.zeros((Nt, dim), dtype=np.float64)
    if max_order >= 2:
        from qxti.analytics.rho_analytic import rho_order_s, _velocity_band
        bounds = hamiltonian.reciprocal_box_bounds()
        from qxti.response.xtp import XTP
        periodic = bool(getattr(kgrid, "shifted", False))
        axv = (kgrid.kx_values, kgrid.ky_values, kgrid.kz_values)
        aw = [XTP._axis_integration_weights(np.asarray(axv[ax], float),
              bounds[ax] if ax < dim else (0.0, 0.0), active=ax < dim, periodic=periodic)
              for ax in range(3)]
        weights = (aw[0][:, None, None] * aw[1][None, :, None] * aw[2][None, None, :]).reshape(-1)
        k_points = kgrid.points()
        T_au = temperature
        E_field = np.asarray(list(E_cw) + [0.0] * (3 - dim), dtype=complex)

        from qxti.utils.progress import ProgressTimer
        for s in range(2, max_order + 1):
            timer = ProgressTimer(total=nk)
            hb = max(1, nk // 10)
            Js = np.zeros(dim, dtype=np.complex128)
            for ik in range(nk):
                if progress and ik % hb == 0:
                    print(
                        f"[theory-HHG] order {s}: k-point {ik+1}/{nk} "
                        f"({100*(ik+1)//nk}%), elapsed {timer.elapsed_seconds:.1f}s, "
                        f"ETA {timer.eta_text()}",
                        flush=True,
                    )
                kx, ky, kz = (float(k_points[ik, 0]), float(k_points[ik, 1]), float(k_points[ik, 2]))
                vel = _velocity_band(hamiltonian._matrix_at, kx, ky, kz)
                try:
                    rhos = rho_order_s(hamiltonian._matrix_at, kx, ky, kz, E_field,
                                       omega0, gamma, mu, T_au, max_order=s)
                except Exception:
                    timer.advance(); continue
                rho_s = rhos.get(s)
                if rho_s is None:
                    timer.advance(); continue
                for a in range(dim):
                    Js[a] += weights[ik] * np.trace((-vel[a]) @ rho_s)
                timer.advance()
            harmonic_peaks[s] = Js
            # Time-domain harmonic, modulated by the pulse envelope^s at s*omega0:
            #   J^(s)(t) = 2 Re[ J^(s)_+ * env_norm(t)^s * e^{-i s omega0 t} ]
            phase = np.exp(-1j * s * omega0 * t_axis)
            for a in range(dim):
                J_harm_t[:, a] += 2.0 * np.real(Js[a] * (env_norm ** s) * phase)
            if progress:
                print(f"[theory-HHG] order {s} harmonic done in {timer.elapsed_seconds:.1f}s", flush=True)

    # --- Assemble a current_spectrum.npz-compatible dataset (for graphics) ---
    current_time = np.zeros((Nt, 3), dtype=np.float64)
    current_time[:, :dim] = J1_t + J_harm_t
    field_time = np.zeros((Nt, 3), dtype=np.float64)
    field_time[:, :dim] = E_t[:, :dim].real
    current_spectrum = np.fft.fft(current_time, axis=0)
    current_magnitude = np.abs(current_spectrum)
    current_total_magnitude = np.sqrt(np.sum(current_magnitude ** 2, axis=1))
    zeros3 = np.zeros((Nt, 3), dtype=np.float64)
    zeros3c = np.zeros((Nt, 3), dtype=np.complex128)
    dataset = {
        "omega_axis": freq,
        "current_spectrum": current_spectrum,
        "current_magnitude": current_magnitude,
        "current_total_magnitude": current_total_magnitude,
        "current_time": current_time,
        "current_time_total": current_time,
        "current_spectrum_total": current_spectrum,
        "polarization_time": zeros3,
        "equilibrium_current_time": zeros3,
        "equilibrium_polarization_time": zeros3,
        "time_axis": t_axis,
        "electric_field_time": field_time,
        # intra/inter split is not separated in theory mode: report all as interband.
        "current_time_intraband": zeros3,
        "current_time_interband": current_time,
        "current_spectrum_intraband": zeros3c,
        "current_spectrum_interband": current_spectrum,
        "current_total_magnitude_intraband": np.zeros(Nt),
        "current_total_magnitude_interband": current_total_magnitude,
        "orders": np.asarray(tuple(range(1, max_order + 1))),
    }

    runtime = time.perf_counter() - t_start
    return {
        "omega_axis": freq,
        "J_total": current_spectrum[:, :dim],
        "J_order": J_order,
        "harmonic_peaks": harmonic_peaks,
        "omega0": omega0,
        "max_order": max_order,
        "runtime_seconds": runtime,
        "dimension": dim,
        "method": "theory",
        "dataset": dataset,
    }


def _susc_available_indices(order: int, dimension: int) -> list[tuple[int, ...]]:
    """Tensor components a single-direction probe can reconstruct (matches the sweep)."""
    if order == 1:
        return [(i, j) for i in range(dimension) for j in range(dimension)]
    return [(i,) + (j,) * order for i in range(dimension) for j in range(dimension)]


def compute_susceptibility_spectrum(
    config: QXTIConfig,
    omega_axis: FloatArray,
    orders: tuple[int, ...],
    *,
    progress: bool = True,
) -> dict[str, Any]:
    """Analytical susceptibility/conductivity tensors for all requested orders.

    Produces a dataset in the SAME format as the time-domain sweep
    (``xtp_susceptibility.npz``), so the standard susceptibility graphics work
    transparently. Order 1 uses the fast streaming Kubo pass; orders s>=2 use the
    closed-form rho^(s)(k, s*omega) (Hipolito A1b/A1c) summed over the BZ, for the
    diagonal-input components chi^(s)_{i j..j}.
    """
    t_start = time.perf_counter()
    sim = QXTISimulation(config=config)
    hamiltonian = sim.build_hamiltonian()
    kgrid = sim.build_kgrid(hamiltonian)
    dim = hamiltonian.dimension
    k_points = kgrid.points()
    nk = kgrid.total_points
    omega_axis = np.asarray(omega_axis, dtype=np.float64)
    nw = omega_axis.size
    orders = tuple(sorted({int(o) for o in orders}))

    direction_labels = tuple(("x", "y", "z")[:dim])
    dataset: dict[str, Any] = {
        "scan_type": "laser_frequency_sweep",
        "orders": np.asarray(orders),
        "dimension": int(dim),
        "direction_labels": list(direction_labels),
        "laser_omega_axis": omega_axis,
        "output_frequency_rule": "omega_out = order * omega_laser",
        "engine": "theory",
    }

    # --- Order 1: fast streaming Kubo (full tensor) ---
    if 1 in orders:
        if progress:
            print("[theory-susc] order 1: streaming Kubo sigma^(1)(omega) ...", flush=True)
        lin = compute_linear_response_spectrum(config, omega_axis, progress=progress)
        dataset["sigma_order_1_tensor"] = lin["sigma"]              # (nw,dim,dim)
        dataset["chi_order_1_tensor"] = lin["chi"]
        idx1 = np.asarray(_susc_available_indices(1, dim), dtype=np.int16)
        dataset["sigma_order_1_available_indices"] = idx1
        dataset["chi_order_1_available_indices"] = idx1

    # --- Orders >=2 ---
    higher = [s for s in orders if s >= 2]
    ccfg = config.susceptibility_solver
    from qxti.response.xtp import XTP
    periodic = bool(getattr(kgrid, "shifted", False))
    bounds = hamiltonian.reciprocal_box_bounds()
    axv = (kgrid.kx_values, kgrid.ky_values, kgrid.kz_values)
    aw = [XTP._axis_integration_weights(np.asarray(axv[ax], float),
          bounds[ax] if ax < dim else (0.0, 0.0), active=ax < dim, periodic=periodic)
          for ax in range(3)]
    weights = (aw[0][:, None, None] * aw[1][None, :, None] * aw[2][None, None, :]).reshape(-1)

    # Order 2: fast grid-based mesh covariant gradient.
    if 2 in higher:
        sig2, idx2 = _order2_gridbased(hamiltonian, kgrid, omega_axis, weights, ccfg, progress=progress)
        omega_b = (2 * omega_axis)[(slice(None),) + (None,) * 3]
        dataset["sigma_order_2_tensor"] = sig2
        dataset["chi_order_2_tensor"] = sig2 / (-1j * omega_b)
        dataset["sigma_order_2_available_indices"] = idx2
        dataset["chi_order_2_available_indices"] = idx2
        higher = [s for s in higher if s != 2]

    # Orders >=3: closed-form rho^(s)(k, s*omega) per input direction (per-point).
    if higher:
        from qxti.analytics.rho_analytic import rho_order_s, _band_frame, _velocity_band
        from qxti.utils.progress import ProgressTimer
        distribution = _resolve_distribution(ccfg.distribution)
        mu = float(ccfg.fermi_level)
        T_au = float(ccfg.temperature)
        gamma = 0.0 if ccfg.coherence_time <= 0 else 1.0 / float(ccfg.coherence_time)

        max_s = max(higher)
        # sigma_s[s] has shape (nw,)+(dim,)*(s+1); only (i, j..j) filled, rest NaN.
        sig_tensors = {s: np.full((nw,) + (dim,) * (s + 1), np.nan + 1j * np.nan, np.complex128)
                       for s in higher}
        total_steps = nw * dim * nk

        # --- Up-front cost estimate (orders >=2 use the expensive covariant
        # gradient per (freq, dir, k); calibrate on a few k-points and warn). ---
        import time as _time
        kx0, ky0, kz0 = (float(k_points[0, 0]), float(k_points[0, 1]), float(k_points[0, 2]))
        E_cal = np.zeros(3, dtype=complex); E_cal[0] = 1.0
        n_cal = min(20, nk)
        t_cal0 = _time.perf_counter()
        for ic in range(n_cal):
            kc = k_points[ic % nk]
            try:
                rho_order_s(hamiltonian._matrix_at, float(kc[0]), float(kc[1]), float(kc[2]),
                            E_cal, float(omega_axis[0]), gamma, mu, T_au, max_order=max_s)
            except Exception:
                pass
        per_call = (_time.perf_counter() - t_cal0) / max(n_cal, 1)
        est_seconds = per_call * total_steps
        if progress:
            from qxti.utils.progress import format_duration as _fmt
            print(
                f"[theory-susc] orders {higher}: {nw} freq x {dim} dirs x {nk} k-points "
                f"= {total_steps:,} rho^(s) evaluations.\n"
                f"[theory-susc] calibrated ~{per_call*1e3:.1f} ms/eval -> estimated "
                f"~{_fmt(est_seconds)} total.",
                flush=True,
            )
            if est_seconds > 600:
                print(
                    "[theory-susc] WARNING: this is a LONG order>=2 run. The closed-form "
                    "rho^(s) uses the covariant k-gradient (neighbour diagonalizations) per "
                    "frequency/direction, so cost ~ Nk x Nfreq x Ndim. Order>=2 converges with "
                    f"a MODEST grid: try k_points ~ [12,12,12] (here {nk:,} points). Reduce the "
                    "grid or susceptibility_num_frequencies to cut the time. (Order 1 is cheap "
                    "and already done.) Ctrl-C to stop and adjust.",
                    flush=True,
                )

        timer = ProgressTimer(total=total_steps)
        step = 0
        k_hb = max(1, nk // 10)
        for iw, omega in enumerate(omega_axis):
            for j in range(dim):                       # input direction
                E_field = np.zeros(3, dtype=complex); E_field[j] = 1.0
                Js = {s: np.zeros(dim, dtype=np.complex128) for s in higher}
                if progress:
                    print(f"[theory-susc] freq {iw+1}/{nw} (omega={float(omega):.4f}), "
                          f"dir {('x','y','z')[j]}: {step:,}/{total_steps:,} "
                          f"({100*step//total_steps}%), ETA {timer.eta_text()}", flush=True)
                for ik in range(nk):
                    if progress and ik and ik % k_hb == 0:
                        print(f"[theory-susc]   ...k {ik}/{nk} (overall {100*step//total_steps}%, "
                              f"ETA {timer.eta_text()})", flush=True)
                    kx, ky, kz = (float(k_points[ik, 0]), float(k_points[ik, 1]), float(k_points[ik, 2]))
                    vel = _velocity_band(hamiltonian._matrix_at, kx, ky, kz)
                    try:
                        rhos = rho_order_s(hamiltonian._matrix_at, kx, ky, kz, E_field,
                                           float(omega), gamma, mu, T_au, max_order=max_s)
                    except Exception:
                        step += 1; timer.advance(); continue
                    for s in higher:
                        rho_s = rhos.get(s)
                        if rho_s is None:
                            continue
                        for i in range(dim):
                            Js[s][i] += weights[ik] * np.trace((-vel[i]) @ rho_s)
                    step += 1; timer.advance()
                # chi^(s)_{i j..j} = sigma / (-i * s*omega); E_j = 1 so sigma = J.
                for s in higher:
                    out_omega = s * float(omega)
                    for i in range(dim):
                        comp = (iw, i) + (j,) * s
                        sig_tensors[s][comp] = np.conj(Js[s][i])
        for s in higher:
            # broadcast omega over the (s+1) trailing spatial axes of the tensor
            omega_b = (s * omega_axis)[(slice(None),) + (None,) * (s + 1)]
            chi_t = sig_tensors[s] / (-1j * omega_b)
            dataset[f"sigma_order_{s}_tensor"] = sig_tensors[s]
            dataset[f"chi_order_{s}_tensor"] = chi_t
            idx = np.asarray(_susc_available_indices(s, dim), dtype=np.int16)
            dataset[f"sigma_order_{s}_available_indices"] = idx
            dataset[f"chi_order_{s}_available_indices"] = idx

    return {"dataset": dataset, "runtime_seconds": time.perf_counter() - t_start,
            "orders": orders, "method": "theory"}


def _order2_gridbased(hamiltonian, kgrid, omega_axis, weights, ccfg, *, progress=True):
    """Fast grid-based chi^(2)/sigma^(2)(omega) via the mesh covariant gradient.

    Precomputes band data over the whole k-mesh ONCE (batched eigh), builds
    rho^(1)(k, omega) vectorized over all frequencies, then takes the covariant
    k-gradient by Wilson-link finite differences across mesh neighbours
    (np.roll), and assembles rho^(2)(k, 2omega). This replaces the per-(freq,dir,k)
    rho recursion -- typically 50-100x faster -- while preserving crystal symmetry.

    Returns (sigma2_tensor[nw, dim, dim, dim], available_indices).
    """
    import time as _time
    from qxti.analytics.rho_analytic import _band_frame  # noqa
    dim = hamiltonian.dimension
    nb = hamiltonian.basis_size
    shape = tuple(kgrid.shape[a] for a in range(3))      # (Nx,Ny,Nz)
    k_points = kgrid.points()                            # (nk,3), 'ij' order
    nk = k_points.shape[0]
    omega_axis = np.asarray(omega_axis, dtype=np.float64)
    nw = omega_axis.size
    Hf = hamiltonian._matrix_at
    bounds = hamiltonian.reciprocal_box_bounds()
    dks = [ (float(bounds[a][1]) - float(bounds[a][0])) / shape[a] for a in range(dim) ]
    distribution = _resolve_distribution(ccfg.distribution)
    mu = float(ccfg.fermi_level); T_au = float(ccfg.temperature)
    gamma = 0.0 if ccfg.coherence_time <= 0 else 1.0 / float(ccfg.coherence_time)
    dk = 1e-4

    t0 = _time.perf_counter()
    if progress:
        print(f"[theory-susc] order 2 (grid-based): diagonalizing {nk} k-points (batched) ...", flush=True)

    # --- batched band data over the whole grid ---
    def _Hbatch(kc):
        return np.array([Hf(float(k[0]), float(k[1]), float(k[2])) for k in kc], dtype=np.complex128)
    H0 = _Hbatch(k_points)
    energies, U = np.linalg.eigh(H0)                     # (nk,nb),(nk,nb,nb)
    Udag = np.conj(np.transpose(U, (0, 2, 1)))
    vel = []
    for axis in range(dim):
        sh = np.zeros(3); sh[axis] = dk
        dH = (_Hbatch(k_points + sh) - _Hbatch(k_points - sh)) / (2 * dk)
        vel.append(Udag @ dH @ U)                        # (nk,nb,nb) band basis
    f = np.asarray(distribution(energies, mu, T_au), dtype=np.float64)  # (nk,nb)
    eps = energies[:, :, None] - energies[:, None, :]    # (nk,n,m)=e_n-e_m
    fmn = f[:, None, :] - f[:, :, None]                  # (nk,n,m)=f_m-f_n
    offdiag = ~np.eye(nb, dtype=bool)
    valid = offdiag[None] & (np.abs(eps) > 1e-20)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_eps = np.where(valid, 1.0 / eps, 0.0)
        dfde = (-f * (1.0 - f) / T_au) if T_au > 1e-15 else np.zeros_like(f)
    A = [1j * vel[a] * inv_eps for a in range(dim)]      # Berry connection (nk,n,m)

    U_mesh = U.reshape(*shape, nb, nb)
    Udag_mesh = Udag.reshape(*shape, nb, nb)
    ow1 = omega_axis[:, None, None] + 1j * gamma         # (nw,1,1)
    ow2 = 2.0 * omega_axis[:, None, None] + 1j * gamma
    inv_d2 = np.where(valid[:, None],
                      1.0 / (ow2[None] - eps[:, None, :, :]), 0.0)  # (nk,nw,n,m)
    sig2 = np.full((nw, dim, dim, dim), np.nan + 1j * np.nan, dtype=np.complex128)

    for j in range(dim):                                  # input direction
        if progress:
            print(f"[theory-susc] order 2 (grid-based): input dir {('x','y','z')[j]} "
                  f"({j+1}/{dim}) ...", flush=True)
        # rho^(1)(k,omega) driven by E_j=1: interband + intraband
        rho1 = np.zeros((nk, nw, nb, nb), dtype=np.complex128)
        rho1 += (A[j][:, None, :, :] * fmn[:, None, :, :]) / (ow1[None] - eps[:, None, :, :])
        # intraband diagonal: rho1_nn = (-i * dfde_n * v_j,nn) / omega_bar
        diag_src = (-1j) * dfde * np.real(np.diagonal(vel[j], axis1=1, axis2=2))  # (nk,nb)
        if np.any(diag_src):
            di = np.arange(nb)
            rho1[:, :, di, di] += diag_src[:, None, :] / ow1[None, :, 0, :]
        rho1 = np.where(valid[:, None] | np.eye(nb, dtype=bool)[None, None], rho1, 0.0)

        # covariant gradient D_j rho1 = d_kj rho1 - i[A_j, rho1], mesh Wilson FD
        rho1_mesh = rho1.reshape(*shape, nw, nb, nb)
        Up = np.roll(U_mesh, -1, axis=j); Um = np.roll(U_mesh, +1, axis=j)
        Wp = Udag_mesh @ Up; Wm = Udag_mesh @ Um         # (*shape,nb,nb)
        Wp_d = np.conj(np.swapaxes(Wp, -1, -2)); Wm_d = np.conj(np.swapaxes(Wm, -1, -2))
        rp = np.roll(rho1_mesh, -1, axis=j); rm = np.roll(rho1_mesh, +1, axis=j)
        tp = Wp[..., None, :, :] @ rp @ Wp_d[..., None, :, :]
        tm = Wm[..., None, :, :] @ rm @ Wm_d[..., None, :, :]
        dpart = (tp - tm) / (2.0 * dks[j])
        Aj = A[j].reshape(*shape, nb, nb)[..., None, :, :]
        comm = Aj @ rho1_mesh - rho1_mesh @ Aj
        Dj_rho1 = (dpart - 1j * comm).reshape(nk, nw, nb, nb)

        # rho^(2)(k,2omega) = E_j * D_j rho1 / (2omega_bar - eps), E_j = 1
        rho2 = Dj_rho1 * inv_d2                           # (nk,nw,nb,nb)
        # J^(2)_i = sum_k w_k Tr[(-v_i) rho2]
        for i in range(dim):
            Ji = -vel[i]                                  # (nk,nb,nb)
            # Tr[Ji rho2] over (n,m): sum_nm Ji[k,m,n] rho2[k,w,n,m]
            tr = np.einsum("kmn,kwnm->wk", Ji, rho2, optimize=True)  # (nw,nk)
            sig2[:, i, j, j] = np.conj(tr @ weights)      # sum over k with BZ weights
    if progress:
        print(f"[theory-susc] order 2 (grid-based) done in {_time.perf_counter()-t0:.1f}s", flush=True)
    idx = np.asarray(_susc_available_indices(2, dim), dtype=np.int16)
    return sig2, idx
