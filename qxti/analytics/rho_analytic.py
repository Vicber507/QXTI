"""Analytical perturbative density matrix — Hipolito, Taghizadeh & Pedersen (2018).

Implements the frequency-domain perturbative solution of the length-gauge
Liouville equation (Eqs. A1a–A1c of arXiv:1802.01430v3) for any tight-binding
Hamiltonian H(kx, ky, kz).

Scope / what is and is NOT implemented
---------------------------------------
A1a  ρ^(1)(ω)    — COMPLETE for any field direction and any ω.
A1b  ρ^(2)(2ω)   — SHG component only (ω₁=ω₂=ω, output at 2ω).
                    Optical rectification (output at 0) is NOT included.
A1c  ρ^(3)(3ω)   — THG component only (ω₁=ω₂=ω₃=ω, output at 3ω).
                    Optical Kerr effect (output at ω) is NOT included.
A1…  ρ^(s)(s·ω)  — the recursion is now generalized to ARBITRARY order s.
                    Each driven order keeps ONLY the top harmonic s·ω (the
                    "all-plus" channel ω+ω+…+ω), i.e. the χ^(s)(s·ω) component.
                    Lower-harmonic mixing channels (e.g. ρ^(4) at 0, 2ω) are
                    NOT included — they are not needed for χ^(s)(s·ω).

This is sufficient to compare with QXTI's susceptibility sweep, which measures
χ^(n) at the n-th harmonic frequency n·ω.

Recursion (same structure as QXTI's CMD, just in frequency instead of time):
    ρ^(0)          = f(ε_k)         equilibrium
    ρ^(1)(ω)       = E·D_kρ^(0) / (ω − ω_mn + iγ)
    ρ^(2)(2ω)      = E·D_kρ^(1)(ω) / (2ω − ω_mn + iγ)
    ρ^(3)(3ω)      = E·D_kρ^(2)(2ω) / (3ω − ω_mn + iγ)
    ρ^(s)(s·ω)     = E·D_kρ^(s-1)((s-1)ω) / (s·ω − ω_mn + iγ)   [general s]

Practical validity of the high-order recursion
-----------------------------------------------
The closed form is formally exact to all orders in the perturbative (weak-field)
regime, but each extra order applies ONE more numerical covariant k-gradient to
the previous order.  Those finite-difference gradients are nested, so BZ-grid /
finite-difference noise and the powers of the resonant denominators grow with s.
In practice orders s ≤ 3–4 are robust with the default steps; s = 5 needs a fine
grid + small dk_grad + a sensible γ; s ≳ 6–7 is exploratory (use analytic
derivatives or a much finer grid).  See ``rho_order_s`` for the cost scaling.

D_k ρ = ∂_k ρ − i[A, ρ]  (covariant gradient; ∂_k via Wilson-link-aligned FD).

All quantities in atomic units (ℏ = e = 1).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

_KB_AU = 3.1668114e-6    # Boltzmann constant in Hartree / K
_AU_TO_EV = 27.211386245988


# ─── low-level primitives ────────────────────────────────────────────────────

def _window_values(size: int, fft_window: str) -> FloatArray:
    name = (fft_window or "none").strip().lower()
    if name == "hann":
        window = np.hanning(size)
    elif name == "hamming":
        window = np.hamming(size)
    elif name == "blackman":
        window = np.blackman(size)
    else:
        window = np.ones(size, dtype=float)
    return np.asarray(window, dtype=np.float64)


def _frequency_axis_from_dt(nt: int, dt: float, *, zero_padding: bool, padding_factor: int) -> FloatArray:
    nfft = nt * max(1, int(padding_factor)) if zero_padding else nt
    omega = 2.0 * np.pi * np.fft.fftfreq(nfft, d=float(dt))
    return np.asarray(omega, dtype=np.float64)


def _nearest_frequency_index(
    omega_axis: FloatArray,
    omega: float,
    *,
    prefer_positive: bool = True,
) -> int:
    axis = np.asarray(omega_axis, dtype=np.float64)
    if prefer_positive and omega >= 0.0:
        positive = np.flatnonzero(axis >= 0.0)
        if positive.size:
            local = int(np.argmin(np.abs(axis[positive] - float(omega))))
            return int(positive[local])
    return int(np.argmin(np.abs(axis - float(omega))))


def extract_frequency_component(
    signal_t: ComplexArray,
    *,
    target_omega: float,
    t_axis: FloatArray | None = None,
    timegrid: object | None = None,
    fft_window: str = "hann",
    zero_padding: bool = False,
    padding_factor: int = 2,
    prefer_positive: bool = True,
) -> tuple[float, ComplexArray]:
    """Return one FFT component using the same convention as QXTI/XTP.

    Parameters
    ----------
    signal_t:
        Time-domain array with time on axis 0.
    target_omega:
        Angular frequency to sample in atomic units.
    t_axis / timegrid:
        Either the raw time axis or a QXTI ``TimeGrid``-like object providing
        ``dt``, ``zero_padding``, ``padding_factor``, ``frequency_axis()``, and
        ``apply_window()``.  When ``timegrid`` is provided it takes precedence.
    """
    values = np.asarray(signal_t, dtype=np.complex128)
    if values.ndim < 1:
        raise ValueError("signal_t must have at least one dimension.")
    nt = int(values.shape[0])
    if nt == 0:
        raise ValueError("signal_t must contain at least one time sample.")

    if timegrid is not None:
        window = np.asarray(timegrid.apply_window(np.ones(nt, dtype=float)), dtype=np.float64)
        dt = float(timegrid.dt)
        nfft = nt * int(timegrid.padding_factor) if bool(timegrid.zero_padding) else nt
        omega_axis = np.asarray(timegrid.frequency_axis(), dtype=np.float64)
    else:
        if t_axis is None:
            raise ValueError("t_axis is required when timegrid is not provided.")
        time_values = np.asarray(t_axis, dtype=np.float64)
        if time_values.ndim != 1 or time_values.size != nt:
            raise ValueError("t_axis must be 1D and match the length of signal_t.")
        dt = float(time_values[1] - time_values[0]) if nt > 1 else 1.0
        window = _window_values(nt, fft_window)
        nfft = nt * max(1, int(padding_factor)) if zero_padding else nt
        omega_axis = _frequency_axis_from_dt(
            nt,
            dt,
            zero_padding=zero_padding,
            padding_factor=padding_factor,
        )

    reshape = (nt,) + (1,) * (values.ndim - 1)
    weighted = values * window.reshape(reshape)
    spectrum = dt * np.fft.fft(weighted, n=nfft, axis=0)
    index = _nearest_frequency_index(
        np.asarray(omega_axis, dtype=np.float64),
        float(target_omega),
        prefer_positive=prefer_positive,
    )
    return float(omega_axis[index]), np.asarray(spectrum[index], dtype=np.complex128)

def _fermi(E: FloatArray, mu: float, T_au: float) -> FloatArray:
    if T_au < 1e-15:
        return np.where(E <= mu + 1e-14, 1.0, 0.0).astype(float)
    return 1.0 / (np.exp((E - mu) / T_au) + 1.0)


def _occupation(E: FloatArray, mu: float, T_au: float, distribution=None) -> FloatArray:
    """Band occupation f(E).  ``distribution(E, mu, T) -> f`` honors the CONFIGURED
    filling (e.g. valence_occupation, which fills by band index to match antelope);
    ``None`` falls back to the energy Fermi step so existing callers/tests are
    unchanged.  Same convention as ``mesh_response.precompute_band_data``."""
    if distribution is None:
        return _fermi(E, mu, T_au)
    return np.asarray(distribution(E, mu, T_au), dtype=np.float64)


def _dfde(f: FloatArray, T_au: float) -> FloatArray:
    """∂f/∂ε for Fermi–Dirac."""
    if T_au < 1e-15:
        return np.zeros_like(f)
    return -f * (1.0 - f) / T_au


def _band_frame(H_func: Callable, kx: float, ky: float, kz: float
                ) -> tuple[FloatArray, ComplexArray]:
    """Eigenvalues and (column) eigenvectors at one k-point."""
    H = np.asarray(H_func(kx, ky, kz), dtype=np.complex128)
    return np.linalg.eigh(H)


def _velocity_band(H_func: Callable, kx: float, ky: float, kz: float,
                   dk: float = 1e-4) -> list[ComplexArray]:
    """Velocity matrices [vx, vy, vz] in the band basis at (kx,ky,kz).

    Uses central finite differences: v^α = ∂H/∂k_α in the band basis.
    """
    _, U = _band_frame(H_func, kx, ky, kz)
    def to_band(dH): return U.conj().T @ dH @ U
    def dH(ax, ay, az):
        Hp = np.asarray(H_func(kx+ax, ky+ay, kz+az), dtype=np.complex128)
        Hm = np.asarray(H_func(kx-ax, ky-ay, kz-az), dtype=np.complex128)
        return (Hp - Hm) / (2 * dk)
    return [to_band(dH(dk,0,0)), to_band(dH(0,dk,0)), to_band(dH(0,0,dk))]


def _berry_offdiag(vel: list[ComplexArray], evals: FloatArray) -> list[ComplexArray]:
    """Off-diagonal Berry connection A^α_mn = i·v^α_mn / ε_mn  (m≠n); diagonal=0.

    In the band basis A^α_mn = i⟨m|∂_k_α|n⟩ = i·v^α_mn / ε_mn for m≠n.
    """
    nb = len(evals)
    result = []
    for v in vel:
        a = np.zeros((nb, nb), dtype=np.complex128)
        for m in range(nb):
            for n in range(nb):
                if m != n:
                    eps_mn = evals[m] - evals[n]
                    if abs(eps_mn) > 1e-20:
                        a[m, n] = 1j * v[m, n] / eps_mn
        result.append(a)
    return result


# ─── covariant gradient via Wilson-link-aligned finite differences ────────────

def _rho1_local(H_func: Callable, kx: float, ky: float, kz: float,
                E_field: FloatArray, ow1: complex,
                mu: float, T_au: float, dk_vel: float,
                distribution=None) -> ComplexArray:
    """ρ^(1)(k, ω) in the LOCAL band frame of k.  (No Wilson-link rotation applied.)

    This is the building block called by _drho_dk_numerical for the gradient.
    """
    evals, _ = _band_frame(H_func, kx, ky, kz)
    vel = _velocity_band(H_func, kx, ky, kz, dk_vel)
    A = _berry_offdiag(vel, evals)
    f = _occupation(evals, mu, T_au, distribution)
    dfde = _dfde(f, T_au)
    nb = len(evals)

    rho1 = np.zeros((nb, nb), dtype=np.complex128)
    for alpha, (v_a, A_a) in enumerate(zip(vel, A)):
        Ea = E_field[alpha]
        if abs(Ea) < 1e-40:
            continue
        # Interband (Eq. A1a, off-diagonal)
        for m in range(nb):
            for n in range(nb):
                if m == n:
                    continue
                omega_mn = evals[m] - evals[n]
                f_nm = f[n] - f[m]
                rho1[m, n] += Ea * A_a[m, n] * f_nm / (ow1 - omega_mn)
        # Intraband (Eq. A1a, diagonal)
        for n in range(nb):
            rho1[n, n] += Ea * (-1j * dfde[n] * v_a[n, n].real) / ow1

    return rho1


def _rho2_local(H_func: Callable, kx: float, ky: float, kz: float,
                U_ref: ComplexArray,
                E_field: FloatArray, ow1: complex, ow2: complex,
                mu: float, T_au: float, dk_grad: float, dk_vel: float) -> ComplexArray:
    """ρ^(2)(k, 2ω) in the LOCAL band frame of k.  (No Wilson-link rotation.)

    Legacy helper — superseded by the generic ``_rho_local_order``; kept for any
    external importer.  Uses the same one-shot covariant derivative (Wilson-link FD
    already includes −i[A, ρ]; no separate commutator).
    """
    evals, U = _band_frame(H_func, kx, ky, kz)
    nb = len(evals)

    # D_k ρ^(1) via Wilson-link-aligned FD — this IS the covariant derivative
    # ∂ρ^(1)/∂k − i[A, ρ^(1)] in one shot (do not add the commutator again).
    def rho1_local_func(kx2, ky2, kz2):
        return _rho1_local(H_func, kx2, ky2, kz2, E_field, ow1, mu, T_au, dk_vel)

    Dk_rho1 = _drho_dk_numerical(H_func, kx, ky, kz, U, rho1_local_func, dk_grad)

    rho2 = np.zeros((nb, nb), dtype=np.complex128)
    for alpha in range(3):
        Ea = E_field[alpha]
        if abs(Ea) < 1e-40:
            continue
        src = Dk_rho1[alpha]
        for m in range(nb):
            for n in range(nb):
                omega_mn = evals[m] - evals[n]
                rho2[m, n] += Ea * src[m, n] / (ow2 - omega_mn)

    return rho2


def _drho_dk_numerical(H_func: Callable,
                       kx: float, ky: float, kz: float,
                       U_ref: ComplexArray,
                       rho_local_func: Callable,
                       dq: float) -> list[ComplexArray]:
    """∂ρ/∂k_α by central FD, with Wilson-link alignment of neighbor frames.

    ``rho_local_func(kx', ky', kz')`` must return ρ in the LOCAL band frame of
    (kx', ky', kz'), i.e. expressed in the eigenvectors U(k').  This function
    then rotates each neighbor into the reference frame U_ref via the Wilson link
    W = U_ref† U(k'), giving a gauge-covariant finite difference.
    """
    def aligned(dx, dy, dz):
        _, U_nb = _band_frame(H_func, kx+dx, ky+dy, kz+dz)
        W = U_ref.conj().T @ U_nb          # Wilson link to reference frame
        r_local = rho_local_func(kx+dx, ky+dy, kz+dz)
        return W @ r_local @ W.conj().T    # rotate to reference frame

    r_px = aligned(dq, 0, 0);  r_mx = aligned(-dq, 0, 0)
    r_py = aligned(0, dq, 0);  r_my = aligned(0, -dq, 0)
    r_pz = aligned(0, 0, dq);  r_mz = aligned(0, 0, -dq)

    return [(r_px - r_mx) / (2*dq),
            (r_py - r_my) / (2*dq),
            (r_pz - r_mz) / (2*dq)]


# ─── generic-order recursive building block ──────────────────────────────────

def _rho_local_order(H_func: Callable, kx: float, ky: float, kz: float,
                     E: ComplexArray, omega: float, s: int,
                     gamma: float, mu: float, T_au: float,
                     dk_grad: float, dk_vel: float,
                     distribution=None) -> ComplexArray:
    """ρ^(s)(k, s·ω) in the LOCAL band frame at (kx,ky,kz), for any order s ≥ 0.

    Recursive building block for the arbitrary-order recursion.  It is the value
    used to evaluate ρ^(s-1) at NEIGHBOUR k-points inside the covariant
    k-gradient, so it must return ρ expressed in the eigenvector frame U(k) of
    its OWN k-point (no Wilson-link pre-rotation); ``_drho_dk_numerical`` rotates
    each neighbour into the reference frame.

    For s = 1 and s = 2 this reproduces ``_rho1_local`` / ``_rho2_local``
    bit-for-bit; it simply extends the same construction to any s.
    """
    if s <= 0:
        evals, _ = _band_frame(H_func, kx, ky, kz)
        f = _occupation(evals, mu, T_au, distribution)
        return np.diag(f.astype(np.complex128))

    ow_s = complex(s * omega + 1j * gamma)
    if s == 1:
        return _rho1_local(H_func, kx, ky, kz, E, ow_s, mu, T_au, dk_vel, distribution)

    evals, U = _band_frame(H_func, kx, ky, kz)
    nb = len(evals)

    def prev_local(kx2, ky2, kz2):
        return _rho_local_order(H_func, kx2, ky2, kz2, E, omega,
                                s - 1, gamma, mu, T_au, dk_grad, dk_vel, distribution)

    # _drho_dk_numerical parallel-transports each neighbour (Wilson links) before
    # differencing, so it already returns the FULL covariant derivative
    # D_k ρ = ∂_k ρ − i[A, ρ] in one shot — exactly the one-shot quantity CMD uses
    # (see cmd._covariant_gradient_for_k_index, which returns it WITHOUT a separate
    # commutator).  Do NOT subtract −i[A, ρ] again: that double-counts the Berry
    # connection and spuriously cancels the intraband (population) channel.
    Dk_prev = _drho_dk_numerical(H_func, kx, ky, kz, U, prev_local, dk_grad)

    rho_s = np.zeros((nb, nb), dtype=np.complex128)
    for alpha in range(3):
        if abs(E[alpha]) < 1e-40:
            continue
        for m in range(nb):
            for n in range(nb):
                rho_s[m, n] += E[alpha] * Dk_prev[alpha][m, n] / (ow_s - (evals[m] - evals[n]))
    return rho_s


# ─── main public function ─────────────────────────────────────────────────────

def rho_order_s(H_func: Callable, kx: float, ky: float, kz: float,
                E_field: FloatArray, omega: float,
                gamma: float, mu: float, T_au: float,
                max_order: int = 3,
                dk_grad: float = 1e-3,
                dk_vel: float = 1e-4,
                distribution=None) -> dict[int, ComplexArray]:
    """Analytical ρ^(0..max_order)(k, s·ω) for one k-point.

    Implements Eqs. A1a–A1c of Hipolito+2018, generalized to any Hamiltonian AND
    to any perturbative order (the same recursion is applied s times).

    Scope:
      s=1: ρ^(1)(ω)    — complete (Eq. A1a)
      s=2: ρ^(2)(2ω)   — SHG component only (ω+ω→2ω)
      s=3: ρ^(3)(3ω)   — THG component only (ω+ω+ω→3ω)
      s≥4: ρ^(s)(s·ω)  — top-harmonic component only (ω+…+ω→s·ω); this is the
                          χ^(s)(s·ω) channel and the natural continuation of A1b/A1c.

    Cost: order s makes ~7^(s-1) evaluations of the order-1 kernel per k-point
    (7 = one on-site + 6 gradient neighbours, nested once per order), each a small
    diagonalization.  s≤4 is cheap; s=5–6 is heavy; s=7 is expensive but works.

    Parameters
    ----------
    H_func   : H(kx,ky,kz,[params]) → (nb,nb) hermitian matrix in a.u.
    E_field  : [Ex, Ey, Ez] — complex amplitude of the e^{−iωt} component, a.u.
    omega    : laser frequency ω in a.u.
    gamma    : dephasing rate (1/T2) in a.u.  — adds as Im part: ω̄ = ω + iγ
    mu       : chemical potential in a.u.
    T_au     : temperature in a.u.  (K × kB = K × 3.167e-6 Ha/K)
    max_order: 1, 2, or 3
    dk_grad  : step for the k-gradient finite difference (a.u.)
    dk_vel   : step for the velocity-operator finite difference (a.u.)

    Returns
    -------
    {0: ρ^(0), 1: ρ^(1), 2: ρ^(2), ...}  each (nb, nb) complex128
    """
    evals, U = _band_frame(H_func, kx, ky, kz)
    vel = _velocity_band(H_func, kx, ky, kz, dk_vel)
    A = _berry_offdiag(vel, evals)
    f = _occupation(evals, mu, T_au, distribution)
    dfde = _dfde(f, T_au)
    nb = len(evals)
    E = np.asarray(E_field, dtype=np.complex128)

    rhos: dict[int, ComplexArray] = {}

    # ── ρ^(0) ────────────────────────────────────────────────────────────────
    rhos[0] = np.diag(f.astype(np.complex128))
    if max_order < 1:
        return rhos

    # ── ρ^(1)(ω) — Eq. A1a ───────────────────────────────────────────────────
    # Source = D_k ρ^(0):
    #   Diagonal (intraband):   (∂f_n/∂k_α) = (df/dε)·v^α_nn
    #   Off-diagonal (interband): -i[A^α, ρ^(0)]_mn = A^α_mn·(f_n−f_m)  m≠n
    # → ρ^(1)_mn = Σ_α E_α·source_mn^α / (ω̄ − ω_mn)
    ow1 = complex(omega + 1j * gamma)
    rhos[1] = _rho1_local(H_func, kx, ky, kz, E, ow1, mu, T_au, dk_vel, distribution)
    if max_order < 2:
        return rhos

    # ── ρ^(s)(s·ω), s ≥ 2 — Eqs. A1b/A1c generalized to any order ────────────
    # Each driven order is built from the previous one by the SAME recursion:
    #     ρ^(s)_mn(s·ω) = Σ_α E_α · [D_k ρ^(s-1)((s-1)·ω)]_mn / (s·ω̄ − ω_mn)
    #     D_k ρ = ∂_k ρ − i[A, ρ]   (covariant k-derivative)
    # The covariant derivative is obtained in ONE shot from the Wilson-link
    # (parallel-transport) finite difference — it already contains the −i[A, ρ]
    # Berry term, so it must NOT be added a second time (that double-count cancels
    # the intraband/population channel).  This matches CMD's covariant-gradient path.
    # ρ^(s-1) at neighbour k-points is supplied by the recursive _rho_local_order.
    # s=2 reproduces Eq. A1b (SHG) and s=3 reproduces Eq. A1c (THG); s≥4 is the
    # natural continuation (ρ^(4) at 4ω, ρ^(5) at 5ω, …).
    for s in range(2, max_order + 1):
        ow_s = complex(s * omega + 1j * gamma)

        def prev_local(kx2, ky2, kz2, _s=s):
            return _rho_local_order(H_func, kx2, ky2, kz2, E, omega,
                                    _s - 1, gamma, mu, T_au, dk_grad, dk_vel, distribution)

        Dk_prev = _drho_dk_numerical(H_func, kx, ky, kz, U, prev_local, dk_grad)

        rho_s = np.zeros((nb, nb), dtype=np.complex128)
        for alpha in range(3):
            if abs(E[alpha]) < 1e-40:
                continue
            for m in range(nb):
                for n in range(nb):
                    rho_s[m, n] += E[alpha] * Dk_prev[alpha][m, n] / (ow_s - (evals[m] - evals[n]))
        rhos[s] = rho_s

    return rhos


# ─── BZ-integrated conductivity ───────────────────────────────────────────────

def sigma1_kubo(H_func: Callable,
                kpoints: tuple[int, ...],
                omega_axis: FloatArray,
                *,
                gamma: float = 1e-3,
                mu: float = 0.0,
                T_K: float = 10.0,
                spin_deg: int = 2,
                bz_bounds: tuple[float, float] = (-np.pi, np.pi),
                dimension: int = 2,
                dk_vel: float = 1e-4,
                verbose: bool = True) -> ComplexArray:
    """Linear conductivity tensor σ^(1)_φα(ω) via the direct Kubo formula (Eq. A2).

    Implements EXACTLY Eq. A2 of Hipolito+2018 in the band basis:

        σ^(1)_φα(ω) = (ig/V_BZ) Σ_k { Σ_{m≠n} v^φ_nm v^α_mn f_nm / ε_mn / (ω̄−ε_mn)
                                       + Σ_n (−i/ω̄) v^φ_nn ∂f_n/∂k_α }

    where v^φ_nm ≡ ⟨n|∂H/∂k_φ|m⟩ in the band basis.

    Returns (nw, 3, 3) complex128 tensor σ_φα at each ω in omega_axis.
    """
    T_au = T_K * _KB_AU
    lo, hi = float(bz_bounds[0]), float(bz_bounds[1])
    Ns = tuple(kpoints) + (1,) * (3 - len(kpoints))
    nkx, nky, nkz = Ns
    dk_bz = hi - lo
    V_BZ = dk_bz ** dimension

    kx_arr = lo + (np.arange(nkx) + 0.5) * dk_bz / nkx
    ky_arr = lo + (np.arange(nky) + 0.5) * dk_bz / nky if dimension >= 2 else np.array([0.0])
    kz_arr = lo + (np.arange(nkz) + 0.5) * dk_bz / nkz if dimension >= 3 else np.array([0.0])
    nk = nkx * len(ky_arr) * len(kz_arr)
    w_k = V_BZ / nk

    nw = len(omega_axis)
    sigma = np.zeros((nw, 3, 3), dtype=np.complex128)
    active = list(range(dimension))

    for ik, (kx, ky, kz) in enumerate(
            (float(kx), float(ky), float(kz))
            for kx in kx_arr for ky in ky_arr for kz in kz_arr):
        if verbose and ik % max(1, nk // 10) == 0:
            print(f"  k {ik+1}/{nk}", end="\r", flush=True)
        try:
            evals, _ = _band_frame(H_func, kx, ky, kz)
            vel = _velocity_band(H_func, kx, ky, kz, dk_vel)
        except Exception:
            continue

        nb = len(evals)
        f = _fermi(evals, mu, T_au)
        dfde = _dfde(f, T_au)

        for iw, omega in enumerate(omega_axis):
            ow = complex(omega + 1j * gamma)

            for phi in active:
                for alpha in active:
                    s_k = 0.0 + 0.0j

                    # Interband: v^φ_nm · v^α_mn · f_nm/ε_mn / (ω̄−ε_mn)
                    # In band basis: v^φ_nm = vel[phi][n,m]  (⟨n|v^φ|m⟩)
                    for m in range(nb):
                        for n in range(nb):
                            if m == n:
                                continue
                            eps_mn = evals[m] - evals[n]
                            if abs(eps_mn) < 1e-20:
                                continue
                            f_nm = f[n] - f[m]
                            v_phi_nm = vel[phi][n, m]   # ⟨n|v^φ|m⟩
                            v_alpha_mn = vel[alpha][m, n]  # ⟨m|v^α|n⟩
                            s_k += v_phi_nm * v_alpha_mn * f_nm / eps_mn / (ow - eps_mn)

                    # Intraband (Drude): (−i/ω̄) · v^φ_nn · ∂f_n/∂k_α
                    for n in range(nb):
                        df_dk_alpha = dfde[n] * vel[alpha][n, n].real
                        s_k += (-1j / ow) * vel[phi][n, n] * df_dk_alpha

                    # Sign: paper uses j = -g·e·v/Ω with e>0; in a.u. e=1, giving
                    # the factor ig/Ω in Eq. A2.  The minus sign from j=-v is
                    # already encoded in the f_nm vs. f_mn convention in s_k.
                    sigma[iw, phi, alpha] += -1j * spin_deg * w_k / V_BZ * s_k

    if verbose:
        print()
    return sigma


def sigma_analytic(H_func: Callable,
                   kpoints: tuple[int, ...],
                   omega_axis: FloatArray,
                   E_field: FloatArray,
                   *,
                   gamma: float = 1e-3,
                   mu: float = 0.0,
                   T_K: float = 10.0,
                   spin_deg: int = 2,
                   bz_bounds: tuple[float, float] = (-np.pi, np.pi),
                   dimension: int = 2,
                   max_order: int = 1,
                   dk_grad: float = 5e-3,
                   dk_vel: float = 1e-4,
                   verbose: bool = True) -> dict[int, ComplexArray]:
    """BZ-integrated σ^(s)(s·ω) for orders 1..max_order via the ρ recursion.

    For order 1, prefer ``sigma1_kubo()`` which directly implements Eq. A2
    without factor-of-i ambiguity.

    Returns {order: array(nw, 3)} — σ^(s)_φ driven by E_field at each ω.
    """
    if max_order == 1:
        # Delegate to the unambiguous Kubo formula for s=1.
        sig = sigma1_kubo(H_func, kpoints, omega_axis,
                          gamma=gamma, mu=mu, T_K=T_K, spin_deg=spin_deg,
                          bz_bounds=bz_bounds, dimension=dimension,
                          dk_vel=dk_vel, verbose=verbose)
        # Return only the column driven by E_field direction.
        alpha = int(np.argmax(np.abs(np.asarray(E_field))))
        return {1: sig[:, :, alpha]}

    T_au = T_K * _KB_AU
    lo, hi = float(bz_bounds[0]), float(bz_bounds[1])
    Ns = tuple(kpoints) + (1,) * (3 - len(kpoints))
    nkx, nky, nkz = Ns
    dk_bz = hi - lo
    V_BZ = dk_bz ** dimension

    kx_arr = lo + (np.arange(nkx) + 0.5) * dk_bz / nkx
    ky_arr = lo + (np.arange(nky) + 0.5) * dk_bz / nky if dimension >= 2 else np.array([0.0])
    kz_arr = lo + (np.arange(nkz) + 0.5) * dk_bz / nkz if dimension >= 3 else np.array([0.0])
    nk = nkx * len(ky_arr) * len(kz_arr)
    w_k = V_BZ / nk

    E = np.asarray(E_field, dtype=np.complex128)
    E_norm = float(np.max(np.abs(E)))
    active = list(range(dimension))

    nw = len(omega_axis)
    sigma = {s: np.zeros((nw, 3), dtype=np.complex128) for s in range(1, max_order + 1)}

    for ik, (kx, ky, kz) in enumerate(
            (float(kx), float(ky), float(kz))
            for kx in kx_arr for ky in ky_arr for kz in kz_arr):
        if verbose and ik % max(1, nk // 10) == 0:
            print(f"  k {ik+1}/{nk}", end="\r", flush=True)
        try:
            vel = _velocity_band(H_func, kx, ky, kz, dk_vel)
        except Exception:
            continue

        for iw, omega in enumerate(omega_axis):
            try:
                rhos = rho_order_s(H_func, kx, ky, kz, E, omega,
                                   gamma, mu, T_au,
                                   max_order=max_order,
                                   dk_grad=dk_grad, dk_vel=dk_vel)
            except Exception:
                continue

            for s in range(2, max_order + 1):  # s=1 handled by Kubo above
                rho_s = rhos.get(s)
                if rho_s is None:
                    continue
                # J^(s)_φ = Σ_k w_k/V_BZ · Tr[v^φ · ρ^(s)] / E_norm^s
                # The correct prefactor comes from comparing with the Kubo formula.
                # Factor of 1j comes from A = iv/ε in the Berry connection,
                # which appears once per field interaction (see module docstring).
                # For s=2 (SHG): one extra factor of 1j from D_k ρ^(1).
                for phi in active:
                    tr = np.trace(vel[phi] @ rho_s)
                    sigma[s][iw, phi] += (1j ** s) * spin_deg * w_k / V_BZ * tr / E_norm**s

    if verbose:
        print()

    # Fill s=1 via Kubo.
    if 1 in range(1, max_order + 1):
        s1 = sigma1_kubo(H_func, kpoints, omega_axis,
                         gamma=gamma, mu=mu, T_K=T_K, spin_deg=spin_deg,
                         bz_bounds=bz_bounds, dimension=dimension,
                         dk_vel=dk_vel, verbose=False)
        alpha = int(np.argmax(np.abs(E)))
        sigma[1] = s1[:, :, alpha]

    return sigma


# ─── compare with QXTI's saved rho ───────────────────────────────────────────

def compare_rho_vs_qxti(H_func: Callable,
                        rho_npy_path: str,
                        k_points: np.ndarray,
                        t_axis: np.ndarray,
                        E_field: FloatArray,
                        omega: float,
                        *,
                        gamma: float = 1e-3,
                        mu: float = 0.0,
                        T_K: float = 10.0,
                        order: int = 1,
                        k_indices: list[int] | None = None,
                        timegrid: object | None = None,
                        fft_window: str = "hann",
                        zero_padding: bool = False,
                        padding_factor: int = 2,
                        electric_field_time: np.ndarray | None = None,
                        normalize_by_field: bool = False,
                        prefer_positive: bool = True,
                        verbose: bool = True) -> dict:
    """Element-wise comparison of QXTI numerical ρ^(s)(k,t) vs. analytic ρ^(s)(k,ω).

    QXTI stores ρ^(s)(k,t) with shape (Nk, Nt, Nb, Nb).
    We FFT each k-slice using the same window / zero-padding convention as
    QXTI/XTP and extract the component at ``s·ω``.

    When ``normalize_by_field`` is ``True`` and ``electric_field_time`` is
    provided, the function additionally compares the response coefficients
    ``ρ^(s)(sω) / E(ω)^s``.  This is the safer diagnostic for finite pulses,
    because the raw FFT amplitude of ``ρ`` scales with the pulse spectrum and
    duration.  The normalization assumes a single dominant driving axis.

    Returns dict with keys 'rho_numeric', 'rho_analytic', 'error_rel' per k-point.
    """
    T_au = T_K * _KB_AU
    E = np.asarray(E_field, dtype=np.complex128)

    rho_data = np.load(rho_npy_path, mmap_mode="r")
    if rho_data.dtype.kind != "c":
        # float16_complex packing — expand
        try:
            from qxti.utils.io_utils import expand_rho_tensor_time_axis
            rho_data = expand_rho_tensor_time_axis(rho_data)
        except Exception:
            rho_data = rho_data.view(np.float16).astype(np.float32).view(np.complex64).astype(np.complex128)

    Nk, Nt, Nb, _ = rho_data.shape

    if k_indices is None:
        k_indices = list(np.linspace(0, Nk - 1, min(6, Nk), dtype=int))

    field_component = None
    analytic_component = None
    if normalize_by_field:
        if electric_field_time is None:
            raise ValueError("electric_field_time is required when normalize_by_field=True.")
        dominant_axis = int(np.argmax(np.abs(E))) if np.any(np.abs(E) > 0.0) else 0
        sampled_field_omega, field_component = extract_frequency_component(
            np.asarray(electric_field_time, dtype=np.complex128)[:, dominant_axis],
            target_omega=float(omega),
            t_axis=np.asarray(t_axis, dtype=np.float64),
            timegrid=timegrid,
            fft_window=fft_window,
            zero_padding=zero_padding,
            padding_factor=padding_factor,
            prefer_positive=prefer_positive,
        )
        analytic_component = complex(E[dominant_axis])
        if abs(field_component) <= 1.0e-30:
            raise ValueError(
                f"The sampled field spectrum at omega={sampled_field_omega:.6g} is too small for normalization."
            )
        if abs(analytic_component) <= 1.0e-30:
            raise ValueError("E_field must contain a non-zero dominant driving component for normalization.")

    out = {
        "k_indices": k_indices,
        "rho_numeric": [],
        "rho_analytic": [],
        "error_rel": [],
        "sampled_omega": [],
    }
    if normalize_by_field:
        out["rho_numeric_normalized"] = []
        out["rho_analytic_normalized"] = []
        out["field_spectrum_component"] = complex(field_component)
        out["analytic_field_component"] = complex(analytic_component)

    for ik in k_indices:
        kx, ky, kz = float(k_points[ik, 0]), float(k_points[ik, 1]), float(k_points[ik, 2])

        # Numeric: FFT at s·ω
        rho_t = np.asarray(rho_data[ik], dtype=np.complex128)
        sampled_omega, rho_num = extract_frequency_component(
            rho_t,
            target_omega=float(order * omega),
            t_axis=np.asarray(t_axis, dtype=np.float64),
            timegrid=timegrid,
            fft_window=fft_window,
            zero_padding=zero_padding,
            padding_factor=padding_factor,
            prefer_positive=prefer_positive,
        )

        # Analytic
        rhos = rho_order_s(H_func, kx, ky, kz, E, omega, gamma, mu, T_au,
                           max_order=order)
        rho_ana = rhos.get(order, np.zeros((Nb, Nb), dtype=np.complex128))

        if normalize_by_field:
            rho_num_cmp = rho_num / (field_component ** order)
            rho_ana_cmp = rho_ana / (analytic_component ** order)
        else:
            rho_num_cmp = rho_num
            rho_ana_cmp = rho_ana

        err = (
            np.linalg.norm(rho_num_cmp - rho_ana_cmp)
            / max(np.linalg.norm(rho_ana_cmp), 1e-30)
        )
        out["rho_numeric"].append(rho_num)
        out["rho_analytic"].append(rho_ana)
        out["error_rel"].append(float(err))
        out["sampled_omega"].append(float(sampled_omega))
        if normalize_by_field:
            out["rho_numeric_normalized"].append(rho_num_cmp)
            out["rho_analytic_normalized"].append(rho_ana_cmp)

        if verbose:
            nrm_num = np.linalg.norm(rho_num_cmp)
            nrm_ana = np.linalg.norm(rho_ana_cmp)
            suffix = " (normalized by E(ω)^s)" if normalize_by_field else ""
            print(
                f"  k[{ik:4d}] ({kx:+.3f},{ky:+.3f},{kz:+.3f})  "
                f"omega={sampled_omega:+.6f}  "
                f"||ρ_num||={nrm_num:.3e}  ||ρ_ana||={nrm_ana:.3e}  "
                f"err_rel={err:.3e}{suffix}"
            )

    return out
