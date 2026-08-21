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


def _resolve_extra_k_weight_mask(
    extra_k_weight_mask: FloatArray | None,
    *,
    kgrid,
) -> FloatArray | None:
    if extra_k_weight_mask is None:
        return None

    mask = np.asarray(extra_k_weight_mask, dtype=np.float64)
    if mask.shape == tuple(kgrid.shape):
        resolved = mask.reshape(-1)
    elif mask.ndim == 1 and mask.size == int(kgrid.total_points):
        resolved = mask.reshape(-1)
    else:
        raise ValueError(
            "extra_k_weight_mask must have shape kgrid.shape or length kgrid.total_points."
        )
    if not np.all(np.isfinite(resolved)):
        raise ValueError("extra_k_weight_mask must contain only finite values.")
    return np.asarray(resolved, dtype=np.float64)


def _bz_mask_reference_radius(bounds: tuple[tuple[float, float], ...], dimension: int) -> float:
    active_bounds = bounds[:dimension]
    return float(min(max(abs(lower), abs(upper)) for lower, upper in active_bounds))


def _bz_mask_radius(config: QXTIConfig, bounds: tuple[tuple[float, float], ...], dimension: int) -> float:
    return 0.01 * float(config.xtp.bz_mask_radius_percent) * _bz_mask_reference_radius(bounds, dimension)


def _bz_mask_sigma(config: QXTIConfig, bounds: tuple[tuple[float, float], ...], dimension: int) -> float:
    if config.xtp.bz_mask_sigma is not None:
        return float(max(config.xtp.bz_mask_sigma, 1.0e-15))
    radius = _bz_mask_radius(config, bounds, dimension)
    sigma_percent = (
        100.0
        if config.xtp.bz_mask_sigma_percent_legacy is None
        else float(config.xtp.bz_mask_sigma_percent_legacy)
    )
    return float(max(0.01 * sigma_percent * radius, 1.0e-15))


def _bz_radial_mask_weights(
    config: QXTIConfig,
    *,
    kgrid,
    bounds: tuple[tuple[float, float], ...],
    dimension: int,
) -> FloatArray:
    if not bool(config.xtp.bz_mask_enabled):
        return np.ones(int(kgrid.total_points), dtype=np.float64)

    mesh = kgrid.mesh(indexing="ij")
    active_coordinates = [np.asarray(mesh[axis], dtype=np.float64) for axis in range(dimension)]
    radial_distance = np.sqrt(np.sum([coordinate**2 for coordinate in active_coordinates], axis=0))

    radius = _bz_mask_radius(config, bounds, dimension)
    sigma = _bz_mask_sigma(config, bounds, dimension)
    weights = np.exp(-0.5 * (radial_distance / sigma) ** 2)
    weights = np.where(radial_distance <= radius, weights, 0.0)
    return np.asarray(weights.reshape(-1), dtype=np.float64)


def _mesh_parallel_settings(config: QXTIConfig) -> tuple[int | None, float]:
    """(n_workers, reserve_gb) for the streaming mesh engine, from ``[cmd]``.

    ``n_workers`` <= 0 -> None (engine uses all performance cores).  ``reserve_gb``
    defaults to 1 GB (RAM always kept free) if the config predates the field.
    """
    ccfg = getattr(config, "cmd", None)
    nw = int(getattr(ccfg, "n_workers", 0) or 0)
    reserve = float(getattr(ccfg, "reserve_gb", 1.0))
    return (None if nw <= 0 else nw), reserve


def _model_h_batch(hamiltonian: Any):
    """Return a vectorized ``kpts->(nk,nb,nb)`` builder if the model exposes
    ``H_batch`` (big speedup on large grids); else None (per-k fallback)."""
    module = getattr(hamiltonian, "_module", None)
    fn = getattr(module, "H_batch", None)
    if callable(fn):
        params = dict(getattr(hamiltonian, "params", {}) or {})
        return lambda kp: fn(kp, params)
    return None


def build_k_integration_weights(
    config: QXTIConfig,
    *,
    hamiltonian: Any,
    kgrid,
    extra_k_weight_mask: FloatArray | None = None,
) -> FloatArray:
    """Return theory-side BZ integration weights matching XTP.

    The result includes:
    - the same axis quadrature rule used by XTP,
    - the optional radial BZ mask configured in ``[xtp]``,
    - and an optional extra multiplicative mask in k-space.
    """
    from qxti.response.xtp import XTP

    dim = int(hamiltonian.dimension)
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
    ).reshape(-1).astype(np.float64)
    weights *= _bz_radial_mask_weights(config, kgrid=kgrid, bounds=bounds, dimension=dim)

    resolved_extra_mask = _resolve_extra_k_weight_mask(extra_k_weight_mask, kgrid=kgrid)
    if resolved_extra_mask is not None:
        weights *= resolved_extra_mask
    return np.asarray(weights, dtype=np.float64)


def compute_linear_response_spectrum(
    config: QXTIConfig,
    omega_axis: FloatArray,
    *,
    progress: bool = True,
    extra_k_weight_mask: FloatArray | None = None,
) -> dict[str, Any]:
    """Return analytical sigma^(1)(omega) and chi^(1)(omega) on ``omega_axis``.

    sigma^(1)_ab(omega) = sum_k w_k * sum_{m!=n} J_a,mn(k) * A_b,nm(k)
                          * (f_m - f_n) / (omega + i*gamma - (e_n - e_m))

    where J_a = current operator (= -v) and A_b = Berry connection, both in QXTI's
    smoothed band gauge, w_k are the XTP BZ weights, and gamma = 1/T2.

    chi^(1)_ab(omega) = sigma^(1)_ab(omega) / (-i * omega)   (P = J / (-i omega)).

    Parameters
    ----------
    extra_k_weight_mask:
        Optional multiplicative k-space mask/weight. It must have either shape
        ``kgrid.shape`` or length ``kgrid.total_points``. This is applied on top
        of the standard XTP quadrature and the optional global radial BZ mask.

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

    # Same BZ integration weights as the simulation (XTP), including the
    # optional radial BZ mask and any extra node/local mask supplied here.
    weights = build_k_integration_weights(
        config,
        hamiltonian=hamiltonian,
        kgrid=kgrid,
        extra_k_weight_mask=extra_k_weight_mask,
    )

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
    from qxti.utils.progress import LiveProgress
    live = LiveProgress("theory", nk) if progress else None
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

        if live is not None:
            live.update(stop)

    if live is not None:
        live.update(nk, force=True)
        live.close()

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
        "k_weight_mask_applied": extra_k_weight_mask is not None,
    }


def order2_tensor_at_omega(hamiltonian, kgrid, omega, weights, ccfg):
    """Full second-order conductivity tensor sigma^(2)_{ijk}(2*omega) at ONE omega.

    Grid-based covariant-gradient BZ integral (the CONVERGENT scheme, vectorized
    with ``np.roll`` between mesh neighbours), extended to all off-diagonal input
    components. Returns sigma[i, j, k] (dim^3, complex), symmetric in (j, k).

    The current at the second harmonic for a field E (per Cartesian direction) is
    ``J_i(2 omega) = sum_{jk} sigma[i,j,k] E_j E_k``.
    """
    dim = int(hamiltonian.dimension)
    nb = int(hamiltonian.basis_size)
    shape = tuple(int(kgrid.shape[a]) for a in range(3))
    k_points = np.asarray(kgrid.points(), dtype=np.float64)
    nk = k_points.shape[0]
    Hf = hamiltonian._matrix_at
    bounds = hamiltonian.reciprocal_box_bounds()
    dks = [(float(bounds[a][1]) - float(bounds[a][0])) / shape[a] for a in range(dim)]
    distribution = _resolve_distribution(ccfg.distribution)
    mu = float(ccfg.fermi_level)
    T = float(ccfg.temperature)
    gamma = 0.0 if ccfg.coherence_time <= 0 else 1.0 / float(ccfg.coherence_time)
    dk = 1.0e-4

    def _Hbatch(kc):
        return np.array([Hf(float(k[0]), float(k[1]), float(k[2])) for k in kc], dtype=np.complex128)

    energies, U = np.linalg.eigh(_Hbatch(k_points))
    Udag = np.conj(np.transpose(U, (0, 2, 1)))
    vel = []
    for a in range(dim):
        sh = np.zeros(3); sh[a] = dk
        vel.append(Udag @ ((_Hbatch(k_points + sh) - _Hbatch(k_points - sh)) / (2 * dk)) @ U)
    f = np.asarray(distribution(energies, mu, T), dtype=np.float64)
    eps = energies[:, :, None] - energies[:, None, :]
    fmn = f[:, None, :] - f[:, :, None]
    offdiag = ~np.eye(nb, dtype=bool)
    valid = offdiag[None] & (np.abs(eps) > 1e-20)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_eps = np.where(valid, 1.0 / eps, 0.0)
        dfde = (-f * (1.0 - f) / T) if T > 1e-15 else np.zeros_like(f)
    A = [1j * vel[a] * inv_eps for a in range(dim)]
    ow1 = omega + 1j * gamma
    ow2 = 2.0 * omega + 1j * gamma
    diagidx = np.arange(nb)
    valid_diag = valid | np.eye(nb, dtype=bool)[None]

    rho1 = []
    for c in range(dim):
        r = np.where(valid, (A[c] * fmn) / (ow1 - eps), 0.0 + 0.0j)
        diag_src = (-1j) * dfde * np.real(np.diagonal(vel[c], axis1=1, axis2=2))
        r[:, diagidx, diagidx] += diag_src / ow1
        rho1.append(np.where(valid_diag, r, 0.0))

    U_mesh = U.reshape(*shape, nb, nb)
    Udag_mesh = Udag.reshape(*shape, nb, nb)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_d2 = np.where(valid, 1.0 / (ow2 - eps), 0.0 + 0.0j)
    w = np.asarray(weights, dtype=np.float64).reshape(nk)

    Tt = np.zeros((dim, dim, dim), dtype=np.complex128)
    for b in range(dim):
        Up = np.roll(U_mesh, -1, axis=b); Um = np.roll(U_mesh, +1, axis=b)
        Wp = (Udag_mesh @ Up).reshape(nk, nb, nb); Wm = (Udag_mesh @ Um).reshape(nk, nb, nb)
        Wp_d = np.conj(np.swapaxes(Wp, -1, -2)); Wm_d = np.conj(np.swapaxes(Wm, -1, -2))
        for c in range(dim):
            rc_mesh = rho1[c].reshape(*shape, nb, nb)
            rp = np.roll(rc_mesh, -1, axis=b).reshape(nk, nb, nb)
            rm = np.roll(rc_mesh, +1, axis=b).reshape(nk, nb, nb)
            # dpart is the Wilson-TRANSPORTED mesh gradient = the FULL covariant
            # derivative D_k rho^(1) = grad rho - i[A, rho] in one shot, so no
            # separate commutator is subtracted (adding it double-counts the Berry
            # connection and cancels the intraband/population channel).
            dpart = (Wp @ rp @ Wp_d - Wm @ rm @ Wm_d) / (2.0 * dks[b])
            rho2 = dpart * inv_d2
            for i in range(dim):
                tr = np.einsum("kmn,knm->k", -vel[i], rho2, optimize=True)
                Tt[i, b, c] = np.conj(np.sum(tr * w))
    return 0.5 * (Tt + np.transpose(Tt, (0, 2, 1)))


def _hhg_multilaser_result(config, hamiltonian, kgrid, dim, nk, max_order, gamma,
                           mu, temperature, distribution, Nt, dt, t_axis, E_t, freq,
                           progress, extra_k_weight_mask, t_start):
    """Multi-laser HHG current spectrum via the TIME-DOMAIN perturbative engine.

    When more than one laser/frequency is present the single-carrier closed form
    (harmonic peaks at s*omega0) is ill-defined.  Here every order is integrated in
    time with the FULL field E(t) (all lasers, over the union time window that spans
    from the first pulse onset to the last pulse end), so every mixing frequency
    (omega_i +/- omega_j +/- ...) appears automatically in the FFT of the current.
    Reduces to the closed form for a single laser (validated in the study).
    """
    from qxti.analytics.mesh_response import time_domain_currents_meshed, default_worker_count
    from qxti.utils.progress import format_duration, LiveProgress

    weights = build_k_integration_weights(config, hamiltonian=hamiltonian, kgrid=kgrid,
                                          extra_k_weight_mask=extra_k_weight_mask)
    k_points = kgrid.points()
    shape = tuple(int(kgrid.shape[a]) for a in range(3))
    bounds = hamiltonian.reciprocal_box_bounds()
    n_workers, reserve_gb = _mesh_parallel_settings(config)
    h_batch = _model_h_batch(hamiltonian)
    if progress:
        print(f"[theory-HHG] MULTI-LASER ({int(getattr(kgrid,'total_points',nk))} k-points): "
              f"memory-safe STREAMING time-domain recursion over the full E(t) "
              f"({Nt} time steps, window [{t_axis[0]:.1f}, {t_axis[-1]:.1f}] a.u.), "
              f">={reserve_gb:g} GB RAM kept free. Live progress with ETA below.", flush=True)
    _t_band = time.perf_counter()
    _blive: dict[str, LiveProgress] = {}   # lazy: #blocks is known only in the callback

    def _block_cb(done, total):
        if not progress:
            return
        lv = _blive.get("p")
        if lv is None:
            lv = _blive["p"] = LiveProgress("theory-HHG", max(1, total))
        lv.update(done, message="multi-laser mesh")

    # Stream the k-mesh in halo-padded blocks so the (nk, nb, nb, Nt) frequency
    # tensors never exist for the whole grid (that all-at-once allocation is what
    # OOM'd multi-laser at large nk -> the reason such runs needed ptddm).
    td = time_domain_currents_meshed(
        hamiltonian._matrix_at, k_points, shape, bounds, weights, E_t[:, :3], dt, max_order,
        gamma=gamma, gamma_pop=gamma, mu=mu, T_au=temperature, dimension=dim,
        distribution=distribution, n_workers=n_workers, reserve_gb=reserve_gb,
        h_batch=h_batch, return_intraband=True, progress_cb=_block_cb)
    if progress:
        lv = _blive.get("p")
        if lv is not None:
            lv.update(lv.total, force=True)
            lv.close()
        print(f"[theory-HHG] multi-laser mesh recursion done: elapsed "
              f"{format_duration(time.perf_counter() - _t_band)}.", flush=True)

    # Per-order physical current in time (real) and its spectrum; plus the REAL
    # intra/inter split (intraband = Σ_n v_nn ρ_nn diagonal, inter = total − intra).
    J_order: dict[int, np.ndarray] = {}
    current_dim = np.zeros((Nt, dim), dtype=np.float64)
    intra_dim = np.zeros((Nt, dim), dtype=np.float64)
    for s in range(1, max_order + 1):
        Js_t = td["J_t"][s][:, :dim].real
        current_dim += Js_t
        intra_dim += td["J_t_intra"][s][:, :dim].real
        J_order[s] = np.fft.fft(Js_t, axis=0)

    current_time = np.zeros((Nt, 3), dtype=np.float64)
    current_time[:, :dim] = current_dim
    current_time_intra = np.zeros((Nt, 3), dtype=np.float64)
    current_time_intra[:, :dim] = intra_dim
    current_time_inter = current_time - current_time_intra
    field_time = np.zeros((Nt, 3), dtype=np.float64)
    field_time[:, :dim] = E_t[:, :dim].real
    current_spectrum = np.fft.fft(current_time, axis=0)
    spectrum_intra = np.fft.fft(current_time_intra, axis=0)
    spectrum_inter = current_spectrum - spectrum_intra
    current_magnitude = np.abs(current_spectrum)
    current_total_magnitude = np.sqrt(np.sum(current_magnitude ** 2, axis=1))
    zeros3 = np.zeros((Nt, 3), dtype=np.float64)
    dataset = {
        "omega_axis": freq, "current_spectrum": current_spectrum,
        "current_magnitude": current_magnitude,
        "current_total_magnitude": current_total_magnitude,
        "current_time": current_time, "current_time_total": current_time,
        "current_spectrum_total": current_spectrum, "polarization_time": zeros3,
        "equilibrium_current_time": zeros3, "equilibrium_polarization_time": zeros3,
        "time_axis": t_axis, "electric_field_time": field_time,
        "current_decomposition_available": True,
        "current_time_intraband": current_time_intra, "current_time_interband": current_time_inter,
        "current_spectrum_intraband": spectrum_intra, "current_spectrum_interband": spectrum_inter,
        "current_total_magnitude_intraband": np.sqrt(np.sum(np.abs(spectrum_intra) ** 2, axis=1)),
        "current_total_magnitude_interband": np.sqrt(np.sum(np.abs(spectrum_inter) ** 2, axis=1)),
        "orders": np.asarray(tuple(range(1, max_order + 1))),
    }
    if progress:
        print("[theory-HHG] multi-laser time-domain done.", flush=True)
    return {
        "omega_axis": freq, "J_total": current_spectrum[:, :dim], "J_order": J_order,
        "harmonic_peaks": {}, "omega0": float(config.laser.omega), "max_order": max_order,
        "runtime_seconds": time.perf_counter() - t_start, "dimension": dim,
        "method": "theory-multilaser", "dataset": dataset,
    }


def compute_hhg_spectrum(
    config: QXTIConfig,
    *,
    max_order: int | None = None,
    progress: bool = True,
    extra_k_weight_mask: FloatArray | None = None,
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
    freq = np.fft.fftfreq(Nt, d=dt) * 2.0 * np.pi

    # Multi-laser / multi-frequency drive: the single-carrier harmonic peaks are
    # ill-defined, so integrate the perturbative response in the TIME domain over
    # the full multi-pulse field E(t).  Single laser keeps the fast closed form.
    if int(laser_system.number_of_lasers()) > 1:
        return _hhg_multilaser_result(
            config, hamiltonian, kgrid, dim, nk, max_order, gamma, mu, temperature,
            distribution, Nt, dt, t_axis, E_t, freq, progress, extra_k_weight_mask, t_start)

    # --- Order 1: from the SAME mesh recursion as orders >=2 (single, consistent
    #     path). J^(1) = sum_k w_k Tr[-v rho^(1)] is the FULL linear current --
    #     interband AND intraband (Drude) -- and is dressed with env^1 below, exactly
    #     like the higher harmonics. This is the physically complete order 1; the old
    #     separate sigma^(1)(omega) pass was interband-only and re-diagonalized the grid.
    J_order: dict[int, np.ndarray] = {}
    J1_t = np.zeros((Nt, dim), dtype=np.float64)   # order 1 lives in J_harm_t below

    # Pulse envelope and CW field amplitude from the analytic signal of E(t).
    analytic = np.zeros((Nt, dim), dtype=np.complex128)
    pos = freq > 0
    for d in range(dim):
        spec = np.fft.fft(E_t[:, d])
        spec_a = np.zeros_like(spec)
        spec_a[pos] = 2.0 * spec[pos]
        # DROP the DC bin: the pulse envelope is the modulation of the CARRIER only.
        # Keeping DC would make |analytic| = A(t) + E_dc·cos(φ) — a ripple at the
        # carrier frequency — which beats against exp(-i s ω0 t) in the reconstruction
        # and injects SPURIOUS even harmonics (2ω0, 4ω0, ...).  See docs/INTEGRATORS.md.
        spec_a[np.abs(freq) < 1e-30] = 0.0
        analytic[:, d] = np.fft.ifft(spec_a)
    env_t = np.abs(analytic).max(axis=1)
    env_peak = float(env_t.max()) if env_t.max() > 0 else 1.0
    env_norm = env_t / env_peak                       # (Nt,) normalized to 1
    i_peak = int(np.argmax(env_norm))
    # +omega0 complex amplitude per direction (half the analytic amplitude at peak).
    E_cw = 0.5 * analytic[i_peak, :dim]

    # --- Orders 1..max: harmonic peaks via analytic rho^(s)(k, s*omega0) ---
    harmonic_peaks: dict[int, np.ndarray] = {}
    J_harm_t = np.zeros((Nt, dim), dtype=np.float64)
    J_harm_t_intra = np.zeros((Nt, dim), dtype=np.float64)   # diagonal (Drude/Bloch) part
    if max_order >= 1:
        # Single VECTORIZED mesh recursion for ALL orders (1..max) -- replaces the old
        # per-k rho_order_s loop, the double-counting order-2 tensor path, AND the
        # separate interband-only sigma^(1) pass for order 1.
        from qxti.analytics.mesh_response import harmonic_currents_meshed, default_worker_count
        weights = build_k_integration_weights(config, hamiltonian=hamiltonian, kgrid=kgrid,
                                              extra_k_weight_mask=extra_k_weight_mask)
        k_points = kgrid.points()
        shape = tuple(int(kgrid.shape[a]) for a in range(3))
        bounds = hamiltonian.reciprocal_box_bounds()
        E_field = np.asarray(list(E_cw) + [0.0] * (3 - dim), dtype=complex)
        from qxti.utils.progress import LiveProgress, format_duration
        # Memory-safe + multi-core: n_workers/reserve from [cmd]; use the model's
        # vectorized H_batch when it exposes one (big speedup on large grids).
        n_workers, reserve_gb = _mesh_parallel_settings(config)
        grad_stencil = int(getattr(ccfg, "gradient_stencil", 2) or 2)
        h_batch = _model_h_batch(hamiltonian)
        # Describe the model so the mesh can run each k-block in a separate PROCESS
        # (its vectorized per-block numpy is GIL-bound -> a ThreadPool stalls at ~1
        # core for any nb).  Without a source path we omit it -> mesh keeps threads.
        _src = str(getattr(hamiltonian, "_module_path", "") or "")
        model_spec = None
        if _src:
            model_spec = {
                "source_file": _src,
                "function_name": str(getattr(hamiltonian, "function_name", "H")),
                "params": dict(getattr(hamiltonian, "params", {}) or {}),
                "h_batch_name": "H_batch",
                "dist_name": str(getattr(ccfg, "distribution", "") or ""),
            }
        if progress:
            print(f"[theory-HHG] orders 1..{max_order}: memory-safe streaming mesh recursion "
                  f"({nk} k-points, up to {n_workers or default_worker_count()} cores, "
                  f">={reserve_gb:g} GB RAM kept free). Live progress with ETA below.", flush=True)
        _t_band = time.perf_counter()
        _blive: dict[str, LiveProgress] = {}   # created lazily: total (#blocks) is known only in the callback

        def _block_cb(done, total):
            if not progress:
                return
            live = _blive.get("p")
            if live is None:
                live = _blive["p"] = LiveProgress("theory-HHG", max(1, total))
            live.update(done, message="mesh recursion")

        # return_intraband: the mesh also returns the diagonal (Drude/Bloch) part
        # J_intra = Σ_k w Σ_n v_nn ρ_nn, so we can build the REAL intra/inter split
        # (interband = total − intra), instead of faking it (was: all -> interband).
        J_all, J_intra_all = harmonic_currents_meshed(
            hamiltonian._matrix_at, k_points, shape, bounds, weights, E_field, omega0, max_order,
            gamma=gamma, gamma_pop=gamma, mu=mu, T_au=temperature, dimension=dim,
            distribution=distribution, n_workers=n_workers, reserve_gb=reserve_gb,
            h_batch=h_batch, return_intraband=True, progress_cb=_block_cb,
            grad_stencil=grad_stencil, model_spec=model_spec)
        if progress:
            _live = _blive.get("p")
            if _live is not None:
                _live.update(_live.total, force=True)
                _live.close()
            print(f"[theory-HHG] mesh recursion done: elapsed "
                  f"{format_duration(time.perf_counter() - _t_band)}.", flush=True)
        for s in range(1, max_order + 1):
            Js = -np.asarray(J_all[s][:dim], dtype=np.complex128)         # sum_k w Tr[-v rho^(s)]
            Js_intra = -np.asarray(J_intra_all[s][:dim], dtype=np.complex128)  # diagonal (intraband)
            harmonic_peaks[s] = Js
            if s == 1:
                J_order[1] = Js                                     # order 1 (inter + intraband)
            # Time-domain harmonic, modulated by the pulse envelope^s at s*omega0:
            phase = np.exp(-1j * s * omega0 * t_axis)
            for a in range(dim):
                J_harm_t[:, a] += 2.0 * np.real(Js[a] * (env_norm ** s) * phase)
                J_harm_t_intra[:, a] += 2.0 * np.real(Js_intra[a] * (env_norm ** s) * phase)
        if progress:
            print("[theory-HHG] orders 1..max done.", flush=True)

    # --- Assemble a current_spectrum.npz-compatible dataset (for graphics) ---
    current_time = np.zeros((Nt, 3), dtype=np.float64)
    current_time[:, :dim] = J1_t + J_harm_t
    field_time = np.zeros((Nt, 3), dtype=np.float64)
    field_time[:, :dim] = E_t[:, :dim].real
    current_spectrum = np.fft.fft(current_time, axis=0)
    current_magnitude = np.abs(current_spectrum)
    current_total_magnitude = np.sqrt(np.sum(current_magnitude ** 2, axis=1))
    # REAL intra/inter split (interband = total − intraband).
    current_time_intra = np.zeros((Nt, 3), dtype=np.float64)
    current_time_intra[:, :dim] = J_harm_t_intra
    current_time_inter = current_time - current_time_intra
    spectrum_intra = np.fft.fft(current_time_intra, axis=0)
    spectrum_inter = current_spectrum - spectrum_intra
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
        # REAL intra/inter split: intraband = Σ_n v_nn ρ_nn (diagonal), inter = total − intra.
        "current_decomposition_available": True,
        "current_time_intraband": current_time_intra,
        "current_time_interband": current_time_inter,
        "current_spectrum_intraband": spectrum_intra,
        "current_spectrum_interband": spectrum_inter,
        "current_total_magnitude_intraband": np.sqrt(np.sum(np.abs(spectrum_intra) ** 2, axis=1)),
        "current_total_magnitude_interband": np.sqrt(np.sum(np.abs(spectrum_inter) ** 2, axis=1)),
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


def _mesh_susceptibility(hamiltonian, kgrid, omega_axis, weights, ccfg, orders_ge2, *, progress=True):
    """sigma^(s)_{i, j..j}(s*omega) tensors for orders >= 2 via the VECTORIZED mesh recursion.

    Single fast path that replaces BOTH the old per-k ``rho_order_s`` loop (orders >= 3)
    and the ``_order2_gridbased`` order-2 path (which double-counted the Berry
    commutator).  Band data (batched eigh + velocities) is computed ONCE and reused
    across the whole (frequency x drive-direction) sweep.  Returns the diagonal-input
    components only (matches ``_susc_available_indices``); the SAME convention as the
    old orders>=3 branch:  sigma = conj( sum_k w_k Tr[(-v_i) rho^(s)] ).
    """
    from concurrent.futures import ThreadPoolExecutor
    from qxti.analytics.mesh_response import (precompute_band_data, harmonic_currents,
                                              default_worker_count, _LIVE_ARRAYS_PER_PLANE)
    from qxti.utils import memory as _mem
    dim = int(hamiltonian.dimension)
    shape = tuple(int(kgrid.shape[a]) for a in range(3))
    kpts = np.asarray(kgrid.points(), dtype=np.float64)
    bounds = hamiltonian.reciprocal_box_bounds()
    mu = float(ccfg.fermi_level)
    T_au = float(ccfg.temperature)
    distribution = _resolve_distribution(ccfg.distribution)
    gamma = 0.0 if ccfg.coherence_time <= 0 else 1.0 / float(ccfg.coherence_time)
    orders_ge2 = tuple(sorted(orders_ge2))
    max_s = max(orders_ge2)
    omega_axis = np.asarray(omega_axis, dtype=np.float64)
    nw = omega_axis.size
    n_workers = int(getattr(ccfg, "n_workers", 0) or 0) or default_worker_count()
    reserve_gb = float(getattr(ccfg, "reserve_gb", 1.0))
    grad_stencil = int(getattr(ccfg, "gradient_stencil", 2) or 2)

    # RAM guard: the shared band data + one recursion per thread must fit.  eigh is
    # done ONCE and reused across (freq x dir), so it never exceeds the full mesh.
    nb = int(hamiltonian.basis_size)
    band_bytes = kpts.shape[0] * nb * nb * 16.0 * _LIVE_ARRAYS_PER_PLANE
    _mem.ensure_headroom(band_bytes, reserve_gb=reserve_gb, label="theory-susc band")

    if progress:
        print(f"[theory-susc] orders {list(orders_ge2)}: vectorized mesh recursion "
              f"({kpts.shape[0]} k-points, batched eigh once), {nw} frequencies x "
              f"{dim} directions on up to {n_workers} cores. Live progress with ETA below.",
              flush=True)

    from qxti.utils.progress import LiveProgress
    # Pad the mesh so the covariant gradient reads real out-of-box neighbours at the
    # BZ box edge (no periodic wrap); halo = (max_order-1)*(stencil reach).
    band = precompute_band_data(hamiltonian._matrix_at, kpts, shape, bounds,
                                mu=mu, T_au=T_au, dimension=dim,
                                distribution=distribution,
                                halo=(max_s - 1) * (grad_stencil // 2))
    sig = {s: np.full((nw,) + (dim,) * (s + 1), np.nan + 1j * np.nan, np.complex128)
           for s in orders_ge2}

    def _one(iw_j):
        iw, j = iw_j
        E_field = np.zeros(3, dtype=np.complex128); E_field[j] = 1.0
        J = harmonic_currents(band, weights, E_field, float(omega_axis[iw]), max_s,
                              gamma=gamma, gamma_pop=gamma, grad_stencil=grad_stencil)
        return iw, j, J

    tasks = [(iw, j) for iw in range(nw) for j in range(dim)]
    live = LiveProgress("theory-susc", len(tasks)) if progress else None
    _done = [0]                                   # single-threaded: _store runs on the main thread

    def _store(iw, j, J):
        for s in orders_ge2:
            for i in range(dim):
                sig[s][(iw, i) + (j,) * s] = np.conj(-J[s][i])
        _done[0] += 1
        if live is not None:
            live.update(_done[0], message="freq x dir")

    if n_workers <= 1:
        for t in tasks:
            _store(*_one(t))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for iw, j, J in ex.map(_one, tasks):
                _store(iw, j, J)
    if live is not None:
        live.update(len(tasks), force=True)
        live.close()
    return sig


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

    # Multi-laser drive: a single-omega susceptibility sweep cannot represent the
    # mixing of several frequencies.  Fall back to the TIME-DOMAIN perturbative
    # engine on the actual multi-pulse field E(t) and return its current response
    # (all mixing products), sharing the exact machinery of the HHG multi-laser
    # path.  A single laser keeps the fast closed-form chi^(s)(s*omega) sweep.
    laser_system = sim.build_laser_system()
    if int(laser_system.number_of_lasers()) > 1:
        ccfg = config.susceptibility_solver
        max_order = max(orders) if orders else 2
        gamma = 0.0 if ccfg.coherence_time <= 0 else 1.0 / float(ccfg.coherence_time)
        mu = float(ccfg.fermi_level)
        temperature = float(ccfg.temperature)
        distribution = _resolve_distribution(ccfg.distribution)
        timegrid = sim.build_timegrid(laser_system)
        Nt = int(timegrid.Nt)
        dt = (float(timegrid.tf) - float(timegrid.t0)) / max(Nt - 1, 1)
        t_axis = float(timegrid.t0) + np.arange(Nt) * dt
        E_t = np.array([laser_system.electric_field(t) for t in t_axis])
        freq = np.fft.fftfreq(Nt, d=dt) * 2.0 * np.pi
        if progress:
            print("[theory-susc] MULTI-LASER: single-omega susceptibility is undefined; "
                  "returning the time-domain multi-frequency current response.", flush=True)
        res = _hhg_multilaser_result(
            config, hamiltonian, kgrid, dim, nk, max_order, gamma, mu, temperature,
            distribution, Nt, dt, t_axis, E_t, freq, progress, None, t_start)
        res["dataset"]["scan_type"] = "multilaser_time_domain"
        res["dataset"]["engine"] = "theory-multilaser"
        return {"dataset": res["dataset"], "runtime_seconds": res["runtime_seconds"],
                "method": "theory-multilaser", "orders": orders}

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
    weights = build_k_integration_weights(config, hamiltonian=hamiltonian, kgrid=kgrid)

    # Orders >= 2: single VECTORIZED mesh recursion (replaces the old per-k
    # rho_order_s loop AND the double-counting order-2 grid path).
    if higher:
        sig = _mesh_susceptibility(hamiltonian, kgrid, omega_axis, weights,
                                   ccfg, tuple(higher), progress=progress)
        for s in higher:
            omega_b = (s * omega_axis)[(slice(None),) + (None,) * (s + 1)]
            dataset[f"sigma_order_{s}_tensor"] = sig[s]
            dataset[f"chi_order_{s}_tensor"] = sig[s] / (-1j * omega_b)
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

    # --- batched band data over the whole grid (memory-safe precompute) ---
    def _Hbatch(kc):
        out = np.empty((kc.shape[0], nb, nb), dtype=np.complex128)
        for index, k in enumerate(kc):
            out[index] = Hf(float(k[0]), float(k[1]), float(k[2]))
        return out

    k_chunk = max(512, min(8192, nk))
    energies = np.empty((nk, nb), dtype=np.float64)
    U = np.empty((nk, nb, nb), dtype=np.complex128)
    for start in range(0, nk, k_chunk):
        stop = min(start + k_chunk, nk)
        H0 = _Hbatch(k_points[start:stop])
        evals_chunk, U_chunk = np.linalg.eigh(H0)
        energies[start:stop] = evals_chunk
        U[start:stop] = U_chunk
        if progress and (stop == nk or stop % max(k_chunk, nk // 10 or 1) == 0):
            print(
                f"[theory-susc] order 2 (grid-based): eig {stop}/{nk} "
                f"({100 * stop // nk}%)",
                flush=True,
            )

    vel = [np.empty((nk, nb, nb), dtype=np.complex128) for _ in range(dim)]
    for axis in range(dim):
        if progress:
            print(
                f"[theory-susc] order 2 (grid-based): velocity operator along "
                f"{('x', 'y', 'z')[axis]} ...",
                flush=True,
            )
        sh = np.zeros(3, dtype=np.float64)
        sh[axis] = dk
        for start in range(0, nk, k_chunk):
            stop = min(start + k_chunk, nk)
            kc = k_points[start:stop]
            dH = (_Hbatch(kc + sh) - _Hbatch(kc - sh)) / (2 * dk)
            U_chunk = U[start:stop]
            Udag_chunk = np.conj(np.transpose(U_chunk, (0, 2, 1)))
            vel[axis][start:stop] = Udag_chunk @ dH @ U_chunk
    f = np.asarray(distribution(energies, mu, T_au), dtype=np.float64)  # (nk,nb)
    eps = energies[:, :, None] - energies[:, None, :]    # (nk,n,m)=e_n-e_m
    fmn = f[:, None, :] - f[:, :, None]                  # (nk,n,m)=f_m-f_n
    offdiag = ~np.eye(nb, dtype=bool)
    valid = offdiag[None] & (np.abs(eps) > 1e-20)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_eps = np.where(valid, 1.0 / eps, 0.0)
        dfde = (-f * (1.0 - f) / T_au) if T_au > 1e-15 else np.zeros_like(f)

    U_mesh = U.reshape(*shape, nb, nb)
    vel_mesh = [item.reshape(*shape, nb, nb) for item in vel]
    eps_mesh = eps.reshape(*shape, nb, nb)
    fmn_mesh = fmn.reshape(*shape, nb, nb)
    valid_mesh = valid.reshape(*shape, nb, nb)
    inv_eps_mesh = inv_eps.reshape(*shape, nb, nb)
    dfde_mesh = dfde.reshape(*shape, nb)
    weights_mesh = np.asarray(weights, dtype=np.float64).reshape(*shape)

    max_slice_points = max(int(nk // max(shape[axis], 1)) for axis in range(dim))
    bytes_per_frequency = max_slice_points * nb * nb * 16 * 8
    omega_chunk = max(1, min(nw, 16, int(1.5e8 / max(bytes_per_frequency, 1))))
    if progress:
        print(
            f"[theory-susc] order 2 (grid-based): using k-chunk={k_chunk} and "
            f"omega-chunk={omega_chunk} to keep RAM bounded.",
            flush=True,
        )

    def _take_matrix(mesh_array, axis: int, idx: int) -> ComplexArray:
        return np.asarray(np.take(mesh_array, int(idx) % shape[axis], axis=axis), dtype=np.complex128).reshape(-1, nb, nb)

    def _take_vector(mesh_array, axis: int, idx: int) -> ComplexArray:
        return np.asarray(np.take(mesh_array, int(idx) % shape[axis], axis=axis), dtype=np.complex128).reshape(-1, nb)

    def _take_weights(axis: int, idx: int) -> FloatArray:
        return np.asarray(np.take(weights_mesh, int(idx) % shape[axis], axis=axis), dtype=np.float64).reshape(-1)

    def _rho1_flat(
        *,
        vel_j_flat: ComplexArray,
        inv_eps_flat: ComplexArray,
        fmn_flat: FloatArray,
        valid_flat: NDArray[np.bool_],
        eps_flat: FloatArray,
        diag_src_flat: ComplexArray,
        ow1_flat: ComplexArray,
    ) -> tuple[ComplexArray, ComplexArray]:
        Aj_flat = 1j * vel_j_flat * inv_eps_flat
        drive = Aj_flat * fmn_flat
        denom1 = ow1_flat[None, :, None, None] - eps_flat[:, None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            rho1 = np.where(
                valid_flat[:, None, :, :],
                drive[:, None, :, :] / denom1,
                0.0 + 0.0j,
            )
        rho1[:, :, np.arange(nb), np.arange(nb)] += diag_src_flat[:, None, :] / ow1_flat[None, :, None]
        return np.asarray(rho1, dtype=np.complex128), np.asarray(Aj_flat, dtype=np.complex128)

    sig2 = np.full((nw, dim, dim, dim), np.nan + 1j * np.nan, dtype=np.complex128)

    for j in range(dim):                                  # input direction
        if progress:
            print(
                f"[theory-susc] order 2 (grid-based): input dir {('x','y','z')[j]} "
                f"({j+1}/{dim}) ...",
                flush=True,
            )
        sig2[:, :, j, j] = 0.0 + 0.0j
        diag_src_mesh = np.asarray(
            (-1j) * dfde_mesh * np.real(np.diagonal(vel_mesh[j], axis1=-2, axis2=-1)),
            dtype=np.complex128,
        )
        n_slices = int(shape[j])
        n_omega_chunks = (nw + omega_chunk - 1) // omega_chunk

        for omega_chunk_index, wstart in enumerate(range(0, nw, omega_chunk), start=1):
            wstop = min(wstart + omega_chunk, nw)
            omega_chunk_values = np.asarray(omega_axis[wstart:wstop], dtype=np.float64)
            ow1_flat = np.asarray(omega_chunk_values + 1j * gamma, dtype=np.complex128)
            ow2_flat = np.asarray(2.0 * omega_chunk_values + 1j * gamma, dtype=np.complex128)
            if progress:
                print(
                    f"[theory-susc] order 2 (grid-based):   omega chunk "
                    f"{omega_chunk_index}/{n_omega_chunks} "
                    f"({wstart + 1}-{wstop}/{nw}) for input {('x','y','z')[j]}",
                    flush=True,
                )

            for slice_index in range(n_slices):
                U_curr = _take_matrix(U_mesh, j, slice_index)
                U_plus = _take_matrix(U_mesh, j, slice_index + 1)
                U_minus = _take_matrix(U_mesh, j, slice_index - 1)
                U_curr_dag = np.conj(np.swapaxes(U_curr, -1, -2))
                W_plus = U_curr_dag @ U_plus
                W_minus = U_curr_dag @ U_minus
                W_plus_dag = np.conj(np.swapaxes(W_plus, -1, -2))
                W_minus_dag = np.conj(np.swapaxes(W_minus, -1, -2))

                vel_curr = _take_matrix(vel_mesh[j], j, slice_index)
                inv_eps_curr = _take_matrix(inv_eps_mesh, j, slice_index)
                fmn_curr = np.asarray(np.take(fmn_mesh, slice_index % shape[j], axis=j), dtype=np.float64).reshape(-1, nb, nb)
                valid_curr = np.asarray(np.take(valid_mesh, slice_index % shape[j], axis=j), dtype=bool).reshape(-1, nb, nb)
                eps_curr = np.asarray(np.take(eps_mesh, slice_index % shape[j], axis=j), dtype=np.float64).reshape(-1, nb, nb)
                diag_curr = _take_vector(diag_src_mesh, j, slice_index)
                rho1_curr, Aj_curr = _rho1_flat(
                    vel_j_flat=vel_curr,
                    inv_eps_flat=inv_eps_curr,
                    fmn_flat=fmn_curr,
                    valid_flat=valid_curr,
                    eps_flat=eps_curr,
                    diag_src_flat=diag_curr,
                    ow1_flat=ow1_flat,
                )

                vel_plus = _take_matrix(vel_mesh[j], j, slice_index + 1)
                inv_eps_plus = _take_matrix(inv_eps_mesh, j, slice_index + 1)
                fmn_plus = np.asarray(np.take(fmn_mesh, (slice_index + 1) % shape[j], axis=j), dtype=np.float64).reshape(-1, nb, nb)
                valid_plus = np.asarray(np.take(valid_mesh, (slice_index + 1) % shape[j], axis=j), dtype=bool).reshape(-1, nb, nb)
                eps_plus = np.asarray(np.take(eps_mesh, (slice_index + 1) % shape[j], axis=j), dtype=np.float64).reshape(-1, nb, nb)
                diag_plus = _take_vector(diag_src_mesh, j, slice_index + 1)
                rho1_plus, _ = _rho1_flat(
                    vel_j_flat=vel_plus,
                    inv_eps_flat=inv_eps_plus,
                    fmn_flat=fmn_plus,
                    valid_flat=valid_plus,
                    eps_flat=eps_plus,
                    diag_src_flat=diag_plus,
                    ow1_flat=ow1_flat,
                )

                vel_minus = _take_matrix(vel_mesh[j], j, slice_index - 1)
                inv_eps_minus = _take_matrix(inv_eps_mesh, j, slice_index - 1)
                fmn_minus = np.asarray(np.take(fmn_mesh, (slice_index - 1) % shape[j], axis=j), dtype=np.float64).reshape(-1, nb, nb)
                valid_minus = np.asarray(np.take(valid_mesh, (slice_index - 1) % shape[j], axis=j), dtype=bool).reshape(-1, nb, nb)
                eps_minus = np.asarray(np.take(eps_mesh, (slice_index - 1) % shape[j], axis=j), dtype=np.float64).reshape(-1, nb, nb)
                diag_minus = _take_vector(diag_src_mesh, j, slice_index - 1)
                rho1_minus, _ = _rho1_flat(
                    vel_j_flat=vel_minus,
                    inv_eps_flat=inv_eps_minus,
                    fmn_flat=fmn_minus,
                    valid_flat=valid_minus,
                    eps_flat=eps_minus,
                    diag_src_flat=diag_minus,
                    ow1_flat=ow1_flat,
                )

                transported_plus = W_plus[:, None, :, :] @ rho1_plus @ W_plus_dag[:, None, :, :]
                transported_minus = W_minus[:, None, :, :] @ rho1_minus @ W_minus_dag[:, None, :, :]
                # dpart is the Wilson-TRANSPORTED mesh gradient = the FULL covariant
                # derivative D_k rho^(1) in one shot; no separate commutator (that
                # double-counts the Berry connection).
                dpart = (transported_plus - transported_minus) / (2.0 * dks[j])
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv_d2_curr = np.where(
                        valid_curr[:, None, :, :],
                        1.0 / (ow2_flat[None, :, None, None] - eps_curr[:, None, :, :]),
                        0.0 + 0.0j,
                    )
                rho2 = dpart * inv_d2_curr
                weights_curr = _take_weights(j, slice_index)

                for i in range(dim):
                    Ji_curr = -_take_matrix(vel_mesh[i], j, slice_index)
                    sig2[wstart:wstop, i, j, j] += np.conj(
                        np.einsum("pmn,pwnm,p->w", Ji_curr, rho2, weights_curr, optimize=True)
                    )
    if progress:
        print(f"[theory-susc] order 2 (grid-based) done in {_time.perf_counter()-t0:.1f}s", flush=True)
    idx = np.asarray(_susc_available_indices(2, dim), dtype=np.int16)
    return sig2, idx
