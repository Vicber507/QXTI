"""Density of states (DOS / LDOS) engine for QXTI.

This is the third calculation family, next to the HHG/time-domain response
([cmd], ``-hhg``) and the conductivity/susceptibility tensors ([xtp], ``-xtp``).
It reuses QXTI's own Hamiltonian and k-grid, so the DOS is computed for exactly
the same model and Brillouin-zone discretization as the response calculations.

Physics
-------
For a Bloch Hamiltonian H(k) of size ``nb = basis_size`` discretized on the
[kgrid] mesh, the bulk density of states is

    g(E) = sum_k w_k sum_n  L_eta( E - E_n(k) ),

where ``E_n(k)`` are the eigenvalues of H(k), ``w_k`` are the Brillouin-zone
quadrature weights normalized to ``sum_k w_k = 1`` (the same axis quadrature the
XTP/theory engines use), and ``L_eta`` is a normalized broadening kernel of
width ``eta`` (Lorentzian or Gaussian). With this normalization

    \\int g(E) dE = nb        (states per unit cell = number of bands),

which is checked at the end and reported.

The Lorentzian choice is not arbitrary: it equals the spectral trace of the
single-particle Green function,

    L_eta(E - E_n) summed over n  =  -(1/pi) Im Tr [ (E + i eta) I - H(k) ]^{-1},

so the eigenvalue method below is exactly the (broadened) Green-function DOS,
just evaluated through a cheap Hermitian diagonalization instead of a matrix
inverse at every energy.

Quantities returned
-------------------
* ``dos``        total DOS g(E).
* ``pdos``       orbital/sublattice-projected PDOS g_alpha(E) (if requested),
                 with ``sum_alpha g_alpha = g`` because ``sum_alpha |U_{alpha n}|^2 = 1``.
* ``cumulative`` N(E) = \\int_{-inf}^{E} g, i.e. the band filling vs energy.
* ``spectral``   optional momentum-resolved spectral function A(s, E) along a
                 straight k-path (the broadened band structure).

Everything is produced in chunked passes over the k-grid so memory stays bounded
even for million-point 3-D grids.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qxti.core.config import QXTIConfig
from qxti.core.simulation import QXTISimulation
from qxti.utils.progress import ProgressTimer

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
_AU_TO_EV = 27.211386245988


def _dos_k_weights(hamiltonian: Any, kgrid: Any) -> FloatArray:
    """Return BZ quadrature weights (NOT normalized) for the DOS sum.

    Uses the SAME per-axis quadrature rule as the XTP/theory engines (uniform
    midpoint for shifted/periodic grids, Simpson/trapezoid for edge-inclusive
    grids), but without the laser-only radial BZ mask: the DOS is a property of
    the band structure, not of any probe field.
    """
    from qxti.response.xtp import XTP

    dim = int(hamiltonian.dimension)
    bounds = hamiltonian.reciprocal_box_bounds()
    periodic = bool(getattr(kgrid, "shifted", False))
    axis_values = (kgrid.kx_values, kgrid.ky_values, kgrid.kz_values)

    axis_weights = []
    for axis in range(3):
        active = axis < dim
        axis_bounds = bounds[axis] if active and axis < len(bounds) else (0.0, 0.0)
        axis_weights.append(
            XTP._axis_integration_weights(
                np.asarray(axis_values[axis], dtype=np.float64),
                axis_bounds,
                active=active,
                periodic=periodic,
            )
        )
    weights = (
        axis_weights[0][:, None, None]
        * axis_weights[1][None, :, None]
        * axis_weights[2][None, None, :]
    )
    return np.asarray(weights.reshape(-1), dtype=np.float64)


def _broadening_kernel(diff: FloatArray, eta: float, kind: str) -> FloatArray:
    """Normalized line-shape L_eta(diff) with unit area in energy.

    ``diff`` is ``E_grid - E_level``. Both kernels satisfy ``\\int L dE = 1``.
    """
    if kind == "gaussian":
        sigma = max(float(eta), 1e-300)
        norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        return norm * np.exp(-0.5 * (diff / sigma) ** 2)
    # Lorentzian (default) = Green-function spectral trace broadening.
    eta = max(float(eta), 1e-300)
    return (eta / np.pi) / (diff * diff + eta * eta)


def _batched_eigvalsh(Hf, k_points: FloatArray, start: int, stop: int) -> FloatArray:
    kc = k_points[start:stop]
    H0 = np.array([Hf(float(k[0]), float(k[1]), float(k[2])) for k in kc], dtype=np.complex128)
    return np.linalg.eigvalsh(H0)


def _resolve_energy_window(
    config: QXTIConfig,
    Hf,
    k_points: FloatArray,
    nk: int,
    chunk: int,
    *,
    progress: bool,
) -> tuple[float, float]:
    """Return (e_min, e_max). Auto-detected from band extrema when not set."""
    lcfg = config.ldos
    if lcfg.e_min is not None and lcfg.e_max is not None:
        return float(lcfg.e_min), float(lcfg.e_max)

    if progress:
        print("[ldos] scanning band extrema for the energy window ...", flush=True)
    emin = np.inf
    emax = -np.inf
    for start in range(0, nk, chunk):
        stop = min(start + chunk, nk)
        ev = _batched_eigvalsh(Hf, k_points, start, stop)
        emin = min(emin, float(ev.min()))
        emax = max(emax, float(ev.max()))
    span = max(emax - emin, 1e-12)
    pad = 0.05 * span
    e_min = float(lcfg.e_min) if lcfg.e_min is not None else emin - pad
    e_max = float(lcfg.e_max) if lcfg.e_max is not None else emax + pad
    return e_min, e_max


def _compute_bulk_dos_components(
    *,
    Hf,
    k_points: FloatArray,
    weights: FloatArray,
    energies: FloatArray,
    eta: float,
    broadening: str,
    basis_size: int,
    projected: bool,
    progress: bool,
    progress_label: str,
) -> tuple[FloatArray, FloatArray | None]:
    """Compute bulk DOS/PDOS on a fixed energy grid.

    Used by both the periodic eigenvalue LDOS mode and the surface mode, where
    we want to show the true bulk DOS together with the surface observables.
    """
    nk = int(k_points.shape[0])
    num_e = int(energies.size)
    chunk = max(32, min(20000, int(5.0e6 / max(basis_size * num_e, 1))))

    wsum = float(weights.sum())
    if wsum <= 0.0:
        raise ValueError("Total BZ integration weight is non-positive; check [kgrid].")
    wnorm = np.asarray(weights / wsum, dtype=np.float64)

    dos = np.zeros(num_e, dtype=np.float64)
    pdos = np.zeros((basis_size, num_e), dtype=np.float64) if projected else None

    timer = ProgressTimer(total=nk)
    for start in range(0, nk, chunk):
        stop = min(start + chunk, nk)
        kc = k_points[start:stop]
        H0 = np.array([Hf(float(k[0]), float(k[1]), float(k[2])) for k in kc], dtype=np.complex128)
        w_c = wnorm[start:stop]

        if projected:
            ev, U = np.linalg.eigh(H0)
        else:
            ev = np.linalg.eigvalsh(H0)

        diff = energies[None, None, :] - ev[:, :, None]
        kernel = _broadening_kernel(diff, eta, broadening)
        dos += np.einsum("c,cne->e", w_c, kernel, optimize=True)

        if projected and pdos is not None:
            P = (U.conj() * U).real
            pdos += np.einsum("c,can,cne->ae", w_c, P, kernel, optimize=True)

        timer.completed = stop
        if progress:
            print(
                f"{progress_label} k-points {stop}/{nk} ({100 * stop // max(nk, 1)}%), "
                f"elapsed {timer.elapsed_seconds:.1f}s, ETA {timer.eta_text()}",
                flush=True,
            )

    return np.asarray(dos, dtype=np.float64), None if pdos is None else np.asarray(pdos, dtype=np.float64)


def compute_dos_spectrum(config: QXTIConfig, *, progress: bool = True) -> dict[str, Any]:
    """Compute the density of states (and optional spectral function) for a model.

    Returns a dict with ``runtime_seconds``, a ``dataset`` ready to be saved with
    :func:`qxti.data.save_dataset_npz` and read back by the graphics layer, plus
    a few summary scalars (the integral check, the band filling at the Fermi
    level, etc.).
    """
    lcfg = config.ldos
    method = (lcfg.method or "eigenvalues").strip().lower()
    if method == "surface":
        return compute_surface_spectrum(config, progress=progress)
    if method == "finite":
        return compute_finite_spectrum(config, progress=progress)
    if method not in {"eigenvalues"}:
        raise ValueError(
            f"ldos.method must be 'eigenvalues', 'surface', or 'finite' (got {method!r})."
        )

    t_start = time.perf_counter()

    sim = QXTISimulation(config=config)
    hamiltonian = sim.build_hamiltonian()
    kgrid = sim.build_kgrid(hamiltonian)

    broadening = (lcfg.broadening or "lorentzian").strip().lower()
    if broadening not in {"lorentzian", "gaussian"}:
        raise ValueError(f"ldos.broadening must be 'lorentzian' or 'gaussian' (got {broadening!r}).")

    nb = int(hamiltonian.basis_size)
    k_points = kgrid.points()
    nk = int(kgrid.total_points)
    Hf = hamiltonian._matrix_at

    num_e = max(int(lcfg.num_energies), 2)
    # Keep the (chunk, nb, num_e) broadening work array near ~50 MB.
    chunk = max(32, min(20000, int(5.0e6 / max(nb * num_e, 1))))

    e_min, e_max = _resolve_energy_window(config, Hf, k_points, nk, chunk, progress=progress)
    if not (e_max > e_min):
        e_max = e_min + 1.0
    energies = np.linspace(e_min, e_max, num_e)
    spacing = (e_max - e_min) / (num_e - 1)

    eta = float(lcfg.eta)
    if eta <= 0.0:
        # Auto: a few energy-grid steps -> a smooth, sampling-robust line shape.
        eta = 3.0 * spacing
    eta_auto = lcfg.eta <= 0.0

    weights = _dos_k_weights(hamiltonian, kgrid)
    projected = bool(lcfg.projected)
    dos, pdos = _compute_bulk_dos_components(
        Hf=Hf,
        k_points=k_points,
        weights=weights,
        energies=energies,
        eta=eta,
        broadening=broadening,
        basis_size=nb,
        projected=projected,
        progress=progress,
        progress_label="[ldos]",
    )

    # Cumulative state count N(E) = \int_{-inf}^{E} g(E') dE'  (states per cell).
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (dos[1:] + dos[:-1]) * np.diff(energies))))
    integral = float(cumulative[-1])

    # Optional momentum-resolved spectral function A(s, E) along a k-path.
    spectral = path_s = path_k = None
    path_ticks: list[float] = []
    path_tick_labels: list[str] = []
    if bool(lcfg.spectral_enabled):
        path_k, path_s, path_ticks, path_tick_labels = _build_spectral_path(hamiltonian, lcfg)
        spectral = _compute_spectral_map(Hf, path_k, energies, eta, broadening, progress=progress)

    # Optional constant-energy spectral map A(kx, ky; E0) (Fermi-surface cut).
    plane = plane_kx = plane_ky = None
    plane_energy = None
    if bool(lcfg.spectral_plane_enabled):
        plane_energy = (
            float(lcfg.spectral_plane_energy)
            if lcfg.spectral_plane_energy is not None
            else float(lcfg.fermi_level)
        )
        plane_kx, plane_ky, plane = _compute_spectral_plane(
            hamiltonian, Hf, lcfg, plane_energy, eta, broadening, progress=progress
        )

    runtime = time.perf_counter() - t_start
    if progress:
        print(
            f"[ldos] done: {nk}/{nk} k-points in {runtime:.1f}s. "
            f"\\int g dE = {integral:.4f} (expected {nb} bands)."
        )

    fermi = float(lcfg.fermi_level)
    filling = float(np.interp(fermi, energies, cumulative))

    dataset: dict[str, Any] = {
        "kind": "ldos",
        "method": method,
        "broadening": broadening,
        "eta": float(eta),
        "eta_auto": bool(eta_auto),
        "energies": np.asarray(energies, dtype=np.float64),
        "dos": np.asarray(dos, dtype=np.float64),
        "cumulative": np.asarray(cumulative, dtype=np.float64),
        "projected": bool(projected),
        "basis_size": int(nb),
        "dimension": int(hamiltonian.dimension),
        "total_k_points": int(nk),
        "fermi_level": fermi,
        "filling_at_fermi": filling,
        "integral": integral,
        "model_name": str(getattr(hamiltonian, "model_name", "") or config.run_name()),
        "au_to_ev": _AU_TO_EV,
    }
    if projected and pdos is not None:
        dataset["pdos"] = np.asarray(pdos, dtype=np.float64)
        dataset["orbital_labels"] = _orbital_labels(hamiltonian, nb)
    if spectral is not None:
        dataset["spectral"] = np.asarray(spectral, dtype=np.float64)            # (num_k, num_e)
        dataset["spectral_path_s"] = np.asarray(path_s, dtype=np.float64)       # (num_k,)
        dataset["spectral_path_k"] = np.asarray(path_k, dtype=np.float64)       # (num_k,3)
        if path_ticks:
            dataset["spectral_path_ticks"] = np.asarray(path_ticks, dtype=np.float64)
            dataset["spectral_path_tick_labels"] = list(path_tick_labels)
    if plane is not None:
        dataset["spectral_plane"] = np.asarray(plane, dtype=np.float64)         # (nky, nkx)
        dataset["spectral_plane_kx"] = np.asarray(plane_kx, dtype=np.float64)   # (nkx,)
        dataset["spectral_plane_ky"] = np.asarray(plane_ky, dtype=np.float64)   # (nky,)
        dataset["spectral_plane_energy"] = float(plane_energy)
    # Plot-style metadata so the graphics layer reproduces the example exactly.
    dataset["spectral_cmap"] = str(lcfg.spectral_cmap)
    dataset["spectral_log_scale"] = bool(lcfg.spectral_log_scale)
    dataset["spectral_log_vmin"] = lcfg.spectral_log_vmin
    dataset["spectral_log_vmax"] = lcfg.spectral_log_vmax
    dataset["spectral_alpha_min"] = float(lcfg.spectral_alpha_min)
    dataset["spectral_alpha_max"] = float(lcfg.spectral_alpha_max)

    return {"dataset": dataset, "runtime_seconds": runtime, "integral": integral, "expected": nb}


def _orbital_labels(hamiltonian: Any, nb: int) -> list[str]:
    labels = getattr(hamiltonian, "orbital_labels", None)
    if labels is not None and len(labels) == nb:
        return [str(x) for x in labels]
    return [f"orb {i}" for i in range(nb)]


def _pad_vertex(vec: tuple[float, ...]) -> np.ndarray:
    arr = np.zeros(3, dtype=np.float64)
    n = min(3, len(vec))
    arr[:n] = np.asarray(vec, dtype=np.float64)[:n]
    return arr


def _build_single_axis_vertices(lcfg: Any, dim: int) -> list[np.ndarray] | None:
    axis_name = str(getattr(lcfg, "spectral_axis", "")).strip().lower()
    if not axis_name:
        return None

    axis_index = {"kx": 0, "ky": 1, "kz": 2}.get(axis_name)
    if axis_index is None:
        raise ValueError("ldos.spectral_axis must be one of: kx, ky, kz.")

    start = getattr(lcfg, "spectral_axis_start", None)
    end = getattr(lcfg, "spectral_axis_end", None)
    if start is None or end is None:
        raise ValueError(
            "ldos.spectral_axis_start and ldos.spectral_axis_end are required when ldos.spectral_axis is set."
        )

    if axis_index >= int(dim):
        raise ValueError(
            f"ldos.spectral_axis = {axis_name!r} is not compatible with a {dim}D Hamiltonian."
        )

    origin = _pad_vertex(tuple(getattr(lcfg, "spectral_axis_origin", ())))
    p0 = origin.copy()
    p1 = origin.copy()
    p0[axis_index] = float(start)
    p1[axis_index] = float(end)
    return [p0, p1]


def _build_spectral_path(
    hamiltonian: Any, lcfg: Any
) -> tuple[FloatArray, FloatArray, list[float], list[str]]:
    """Return (path_k (num_k,3), arclength s (num_k,), vertex ticks, vertex labels).

    Path source, in order of precedence:
      1. ``spectral_path`` -- a list of high-symmetry vertices (multi-segment),
         optionally labelled by ``spectral_path_labels``;
      2. ``spectral_axis`` + ``spectral_axis_start/end`` -- a simple single-axis segment;
      3. ``spectral_k0`` / ``spectral_k1`` -- a single straight segment;
      4. default -- a straight line across the BZ box along kx through the origin.

    Note the path is NOT clipped to the integration box: H(k) is periodic, so a
    path may run through the true hexagonal-BZ corners (e.g. the Dirac points K),
    which lie outside the rectangular quadrature box.
    """
    dim = int(hamiltonian.dimension)
    bounds = hamiltonian.reciprocal_box_bounds()

    if lcfg.spectral_path and len(lcfg.spectral_path) >= 2:
        vertices = [_pad_vertex(tuple(v)) for v in lcfg.spectral_path]
        labels = list(lcfg.spectral_path_labels) if lcfg.spectral_path_labels else []
    elif str(getattr(lcfg, "spectral_axis", "")).strip():
        vertices = _build_single_axis_vertices(lcfg, dim) or []
        labels = []
    elif lcfg.spectral_k0 and lcfg.spectral_k1:
        vertices = [_pad_vertex(tuple(lcfg.spectral_k0)), _pad_vertex(tuple(lcfg.spectral_k1))]
        labels = []
    else:
        lo, hi = bounds[0]
        vertices = [np.array([lo, 0.0, 0.0]), np.array([hi, 0.0, 0.0])]
        labels = []

    # Distribute the total point budget across segments proportional to length.
    num_k = max(int(lcfg.spectral_num_k), len(vertices))
    seg_lengths = [float(np.linalg.norm(vertices[i + 1] - vertices[i])) for i in range(len(vertices) - 1)]
    total_len = sum(seg_lengths) or 1.0

    path_pts: list[np.ndarray] = []
    ticks: list[float] = [0.0]
    s_acc = 0.0
    for i, seg_len in enumerate(seg_lengths):
        n_seg = max(2, int(round(num_k * seg_len / total_len)))
        # Drop the first point of every segment after the first to avoid duplicates.
        t = np.linspace(0.0, 1.0, n_seg)
        pts = (1.0 - t)[:, None] * vertices[i][None, :] + t[:, None] * vertices[i + 1][None, :]
        if i > 0:
            pts = pts[1:]
        path_pts.append(pts)
        s_acc += seg_len
        ticks.append(s_acc)

    path_k = np.concatenate(path_pts, axis=0)
    seg = np.linalg.norm(np.diff(path_k, axis=0), axis=1)
    path_s = np.concatenate(([0.0], np.cumsum(seg)))
    # Arclength at each vertex: the END index of each assembled segment.
    ticks = [0.0]
    cum = 0
    for pts in path_pts:
        cum += len(pts)
        ticks.append(float(path_s[cum - 1]))
    # ticks has len(segments)+1 entries == number of vertices.
    tick_labels = [_fmt_label(l) for l in labels] if labels and len(labels) == len(ticks) else []
    return (
        np.asarray(path_k, dtype=np.float64),
        np.asarray(path_s, dtype=np.float64),
        ticks,
        tick_labels,
    )


def _fmt_label(label: str) -> str:
    """Wrap a high-symmetry label in mathtext (e.g. '\\Gamma' -> '$\\Gamma$')."""
    s = str(label).strip()
    if not s:
        return ""
    return s if s.startswith("$") else f"${s}$"


def _compute_spectral_map(
    Hf,
    path_k: FloatArray,
    energies: FloatArray,
    eta: float,
    broadening: str,
    *,
    progress: bool,
) -> FloatArray:
    """A(s, E) = sum_n L_eta(E - E_n(k(s))): the broadened band structure."""
    num_k = path_k.shape[0]
    num_e = energies.size
    spectral = np.zeros((num_k, num_e), dtype=np.float64)
    chunk = max(16, min(4000, int(5.0e6 / max(num_e, 1))))
    if progress:
        print(f"[ldos] spectral function on a {num_k}-point k-path ...", flush=True)
    for start in range(0, num_k, chunk):
        stop = min(start + chunk, num_k)
        kc = path_k[start:stop]
        H0 = np.array([Hf(float(k[0]), float(k[1]), float(k[2])) for k in kc], dtype=np.complex128)
        ev = np.linalg.eigvalsh(H0)                              # (C,nb)
        diff = energies[None, None, :] - ev[:, :, None]          # (C,nb,num_e)
        spectral[start:stop] = _broadening_kernel(diff, eta, broadening).sum(axis=1)
    return spectral


# =============================================================================
# Finite 2-D plaque mode: open boundaries in both lattice directions.
# =============================================================================
def _supports_finite_plaque(hamiltonian: Any) -> bool:
    source = str(getattr(hamiltonian, "source_file", "")).strip().lower()
    stem = source.rsplit("/", 1)[-1].removesuffix(".py")
    model_name = str(getattr(hamiltonian, "model_name", "")).strip().lower()
    return stem == "haldane" or "haldane" in model_name


def _build_finite_haldane_plaque(
    *,
    t1: float,
    t2: float,
    m0: float,
    phi0: float,
    a0: float,
    nx: int,
    ny: int,
) -> tuple[FloatArray, NDArray[np.int64], NDArray[np.bool_], ComplexArray, NDArray[np.bool_]]:
    """Return positions, cell-index metadata, edge mask and H for a finite plaque.

    The plaque is rectangular in the two Bravais directions of the honeycomb
    lattice and open along both. It matches the Bloch convention used by
    ``models/haldane.py``: the A-sublattice next-nearest hoppings carry
    ``+phi0`` and the B-sublattice ones carry ``-phi0``.
    """
    nx = max(int(nx), 1)
    ny = max(int(ny), 1)
    n_cells = nx * ny
    n_sites = 2 * n_cells

    a1 = np.array([np.sqrt(3.0) * 0.5 * a0, 1.5 * a0], dtype=np.float64)
    a2 = np.array([-np.sqrt(3.0) * 0.5 * a0, 1.5 * a0], dtype=np.float64)
    delta_ab = np.array([0.0, a0], dtype=np.float64)

    positions = np.zeros((n_sites, 2), dtype=np.float64)
    cell_indices = np.zeros((n_sites, 2), dtype=np.int64)
    sublattice = np.zeros(n_sites, dtype=bool)

    def idx(i: int, j: int, is_b: bool) -> int:
        return 2 * (j * nx + i) + int(is_b)

    for j in range(ny):
        for i in range(nx):
            r0 = i * a1 + j * a2
            ia = idx(i, j, False)
            ib = idx(i, j, True)
            positions[ia] = r0
            positions[ib] = r0 + delta_ab
            cell_indices[ia] = (i, j)
            cell_indices[ib] = (i, j)
            sublattice[ib] = True

    h = np.zeros((n_sites, n_sites), dtype=np.complex128)
    for j in range(ny):
        for i in range(nx):
            ia = idx(i, j, False)
            ib = idx(i, j, True)
            h[ia, ia] += m0
            h[ib, ib] += -m0

    # Nearest neighbors: A(i,j) -> B(i,j), B(i-1,j), B(i,j-1).
    nn_offsets = ((0, 0), (-1, 0), (0, -1))
    for j in range(ny):
        for i in range(nx):
            ia = idx(i, j, False)
            for di, dj in nn_offsets:
                ii = i + di
                jj = j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    ib = idx(ii, jj, True)
                    h[ia, ib] += t1
                    h[ib, ia] += t1

    # Positive-oriented next-nearest neighbors matching the Bloch b_i vectors:
    #   -a1, a2, a1-a2  ->  cell offsets (-1,0), (0,1), (1,-1).
    nnn_offsets = ((-1, 0), (0, 1), (1, -1))
    amp_a = t2 * np.exp(1j * phi0)
    amp_b = t2 * np.exp(-1j * phi0)
    for j in range(ny):
        for i in range(nx):
            ia = idx(i, j, False)
            ib = idx(i, j, True)
            for di, dj in nnn_offsets:
                ii = i + di
                jj = j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    ja = idx(ii, jj, False)
                    jb = idx(ii, jj, True)
                    h[ia, ja] += amp_a
                    h[ja, ia] += np.conj(amp_a)
                    h[ib, jb] += amp_b
                    h[jb, ib] += np.conj(amp_b)

    edge_mask = (
        (cell_indices[:, 0] == 0)
        | (cell_indices[:, 0] == nx - 1)
        | (cell_indices[:, 1] == 0)
        | (cell_indices[:, 1] == ny - 1)
    )
    positions -= positions.mean(axis=0, keepdims=True)
    return positions, cell_indices, sublattice, h, edge_mask


def compute_finite_spectrum(config: QXTIConfig, *, progress: bool = True) -> dict[str, Any]:
    """Finite 2-D plaque LDOS for open boundaries in both lattice directions."""
    t_start = time.perf_counter()
    sim = QXTISimulation(config=config)
    hamiltonian = sim.build_hamiltonian()
    lcfg = config.ldos

    if int(hamiltonian.dimension) != 2:
        raise ValueError("ldos.method = 'finite' currently requires a 2-D Hamiltonian.")
    if not _supports_finite_plaque(hamiltonian):
        raise ValueError(
            "ldos.method = 'finite' is currently implemented for the Haldane honeycomb model."
        )

    broadening = (lcfg.broadening or "lorentzian").strip().lower()
    if broadening not in {"lorentzian", "gaussian"}:
        raise ValueError(f"ldos.broadening must be 'lorentzian' or 'gaussian' (got {broadening!r}).")

    nx = max(int(lcfg.finite_nx), 1)
    ny = max(int(lcfg.finite_ny), 1)
    edge_cells = max(int(lcfg.finite_edge_cells), 1)
    params = dict(getattr(hamiltonian, "params", {}))
    try:
        t1 = float(params["t1"])
        t2 = float(params["t2"])
        m0 = float(params["M0"])
        phi0 = float(params["phi0"])
        a0 = float(params["a0"])
    except KeyError as exc:
        raise ValueError(
            "The finite Haldane plaque requires Hamiltonian parameters t1, t2, M0, phi0 and a0."
        ) from exc

    if progress:
        print(
            f"[ldos-finite] building finite Haldane plaque on a {nx}x{ny} cell grid "
            f"({2 * nx * ny} sites, open in both lattice directions).",
            flush=True,
        )
        if bool(lcfg.spectral_enabled) or bool(lcfg.spectral_plane_enabled):
            print(
                "[ldos-finite] ignoring spectral k-space plots: a finite plaque has no good crystal momentum. "
                "This mode produces DOS(E), finite-state spectrum, and LDOS(r,E).",
                flush=True,
            )

    positions, cell_indices, sublattice, h, outer_edge_mask = _build_finite_haldane_plaque(
        t1=t1,
        t2=t2,
        m0=m0,
        phi0=phi0,
        a0=a0,
        nx=nx,
        ny=ny,
    )

    if edge_cells > 1:
        edge_mask = (
            (cell_indices[:, 0] < edge_cells)
            | (cell_indices[:, 0] >= nx - edge_cells)
            | (cell_indices[:, 1] < edge_cells)
            | (cell_indices[:, 1] >= ny - edge_cells)
        )
    else:
        edge_mask = outer_edge_mask
    interior_mask = ~edge_mask

    if progress:
        h_gib = h.nbytes / (1024.0 ** 3)
        print(
            f"[ldos-finite] dense real-space Hamiltonian assembled ({h.shape[0]}x{h.shape[1]}, "
            f"{h_gib:.2f} GiB in memory). Diagonalizing ...",
            flush=True,
        )
    eigenvalues, eigenvectors = np.linalg.eigh(h)

    if lcfg.e_min is not None and lcfg.e_max is not None:
        e_min = float(lcfg.e_min)
        e_max = float(lcfg.e_max)
    else:
        emin = float(np.min(eigenvalues))
        emax = float(np.max(eigenvalues))
        span = max(emax - emin, 1.0e-12)
        pad = 0.05 * span
        e_min = float(lcfg.e_min) if lcfg.e_min is not None else emin - pad
        e_max = float(lcfg.e_max) if lcfg.e_max is not None else emax + pad
    if not (e_max > e_min):
        e_max = e_min + 1.0

    num_e = max(int(lcfg.num_energies), 2)
    energies = np.linspace(e_min, e_max, num_e)
    spacing = (e_max - e_min) / (num_e - 1)
    eta = float(lcfg.eta)
    if eta <= 0.0:
        eta = 3.0 * spacing
    eta_auto = lcfg.eta <= 0.0

    if progress:
        print(
            f"[ldos-finite] broadening the discrete spectrum on {num_e} energies "
            f"with {broadening} eta={eta:.4g} a.u. ...",
            flush=True,
        )
    diff = energies[None, :] - eigenvalues[:, None]
    kernel = _broadening_kernel(diff, eta, broadening)
    dos = kernel.sum(axis=0)

    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (dos[1:] + dos[:-1]) * np.diff(energies))))
    integral = float(cumulative[-1])
    fermi = float(lcfg.fermi_level)
    filling = float(np.interp(fermi, energies, cumulative))

    probabilities = np.abs(eigenvectors) ** 2
    edge_weight = probabilities[edge_mask].sum(axis=0).real
    ldos_energy = float(lcfg.finite_ldos_energy) if lcfg.finite_ldos_energy is not None else fermi
    ldos_kernel = _broadening_kernel(ldos_energy - eigenvalues, eta, broadening)
    site_ldos = np.einsum("sn,n->s", probabilities, ldos_kernel, optimize=True)
    edge_fraction = float(site_ldos[edge_mask].sum() / max(site_ldos.sum(), 1.0e-300))

    projected = bool(lcfg.projected)
    pdos = None
    orbital_labels = None
    if projected:
        prob_a = probabilities[~sublattice].sum(axis=0).real
        prob_b = probabilities[sublattice].sum(axis=0).real
        pdos = np.vstack(
            [
                np.einsum("n,ne->e", prob_a, kernel, optimize=True),
                np.einsum("n,ne->e", prob_b, kernel, optimize=True),
            ]
        )
        orbital_labels = ["A", "B"]

    runtime = time.perf_counter() - t_start
    if progress:
        print(
            f"[ldos-finite] done in {runtime:.1f}s. "
            f"Integrated DOS = {integral:.4f} states; edge fraction at E={ldos_energy:.4g} a.u. is {edge_fraction:.3f}.",
            flush=True,
        )

    dataset: dict[str, Any] = {
        "kind": "ldos",
        "method": "finite",
        "broadening": broadening,
        "eta": float(eta),
        "eta_auto": bool(eta_auto),
        "energies": np.asarray(energies, dtype=np.float64),
        "dos": np.asarray(dos, dtype=np.float64),
        "cumulative": np.asarray(cumulative, dtype=np.float64),
        "projected": bool(projected),
        "basis_size": int(h.shape[0]),
        "dimension": 2,
        "fermi_level": fermi,
        "filling_at_fermi": filling,
        "integral": integral,
        "model_name": str(getattr(hamiltonian, "model_name", "") or config.run_name()),
        "au_to_ev": _AU_TO_EV,
        "finite_num_cells": [int(nx), int(ny)],
        "finite_num_sites": int(h.shape[0]),
        "finite_edge_cells": int(edge_cells),
        "finite_ldos_energy": float(ldos_energy),
        "finite_positions": np.asarray(positions, dtype=np.float64),
        "finite_cell_indices": np.asarray(cell_indices, dtype=np.int64),
        "finite_boundary_mask": np.asarray(edge_mask, dtype=bool),
        "finite_interior_mask": np.asarray(interior_mask, dtype=bool),
        "finite_sublattice": np.asarray(sublattice, dtype=bool),
        "finite_site_ldos": np.asarray(site_ldos, dtype=np.float64),
        "finite_eigenvalues": np.asarray(eigenvalues, dtype=np.float64),
        "finite_edge_weight": np.asarray(edge_weight, dtype=np.float64),
        "finite_edge_fraction": float(edge_fraction),
    }
    if projected and pdos is not None and orbital_labels is not None:
        dataset["pdos"] = np.asarray(pdos, dtype=np.float64)
        dataset["orbital_labels"] = list(orbital_labels)

    return {
        "dataset": dataset,
        "runtime_seconds": runtime,
        "integral": integral,
        "expected": int(h.shape[0]),
    }


# =============================================================================
# Surface / edge mode: semi-infinite Lopez-Sancho surface Green function.
# =============================================================================
def _resolve_normal_axis(surface_normal: str, dim: int) -> int:
    """Return the axis index (0,1,2) made semi-infinite."""
    key = (surface_normal or "auto").strip().lower()
    if key in {"x", "0"}:
        return 0
    if key in {"y", "1"}:
        return 1
    if key in {"z", "2"}:
        return 2
    # auto: the last periodic axis (z in 3D, y in 2D, x in 1D).
    return max(0, dim - 1)


def _surface_compare_layers(lcfg: Any, normal: int) -> int:
    """Effective thickness for plotting surface contribution vs bulk.

    The Lopez-Sancho surface Green function returns a local surface LDOS per
    boundary layer. To compare it as a contribution to a finite ribbon/slab DOS,
    the plotting layer scales it by this effective number of layers.
    """
    explicit = int(getattr(lcfg, "surface_compare_layers", 0) or 0)
    if explicit > 0:
        return explicit
    if int(normal) == 0:
        return max(int(getattr(lcfg, "finite_nx", 1)), 1)
    if int(normal) == 1:
        return max(int(getattr(lcfg, "finite_ny", 1)), 1)
    # There is no finite_nz key today; use the larger finite in-plane knob as a
    # conservative default for 3D slab-style comparisons.
    return max(int(getattr(lcfg, "finite_nx", 1)), int(getattr(lcfg, "finite_ny", 1)), 1)


def _principal_layer_blocks(
    Hf, kpar: np.ndarray, normal: int, c_n: float, n_kn: int, h_batch=None
) -> tuple[ComplexArray, ComplexArray]:
    """Fourier-transform H along the normal into principal-layer blocks.

    ``H(k_n) = H00 + H01 e^{i k_n c} + H01^dag e^{-i k_n c}`` (nearest-layer
    approximation), so

        H00 = <H(k_n)>_{k_n},   H01 = <H(k_n) e^{-i k_n c}>_{k_n}.

    ``kpar`` is a full 3-vector whose normal component is overwritten by the
    sampling; only its in-plane components matter.  ``h_batch`` (if given) builds
    all ``n_kn`` matrices in one vectorized call -> large speedup (the per-k
    Python loop over the model H dominates the surface cost).
    """
    kn = np.linspace(-np.pi / c_n, np.pi / c_n, int(n_kn), endpoint=False)
    kpts = np.repeat(np.array(kpar, dtype=np.float64)[None, :], kn.size, axis=0)
    kpts[:, normal] = kn
    if h_batch is not None:
        Hs = np.asarray(h_batch(kpts), dtype=np.complex128)
    else:
        Hs = np.array([np.asarray(Hf(float(k[0]), float(k[1]), float(k[2])), dtype=np.complex128)
                       for k in kpts])
    H00 = Hs.mean(axis=0)
    H01 = (Hs * np.exp(-1j * kn * c_n)[:, None, None]).mean(axis=0)
    H00 = 0.5 * (H00 + H00.conj().T)
    return H00, H01


def _sancho_surface_green(
    z: np.ndarray, H00: ComplexArray, H01: ComplexArray, max_iter: int, tol: float,
    side: str = "bottom",
) -> ComplexArray:
    """Surface Green function G_s(z) via Lopez-Sancho decimation, batched over z.

    The decimation yields BOTH semi-infinite terminations at once: ``eps_s``
    accumulates ``alpha g beta`` (the "bottom" surface) and ``eps_s_top``
    accumulates ``beta g alpha`` (the opposite "top" surface). Their surface
    states are complementary -- e.g. a Weyl slab's Fermi arcs route one way on
    the bottom face and the complementary way on the top face.

    ``side`` selects "bottom", "top", or "both" (sum of the two Green functions).
    Returns ``G_s`` of shape ``(z.size, nb, nb)``.
    """
    nz = z.size
    nb = H00.shape[0]
    eye = np.eye(nb, dtype=np.complex128)
    eps_s = np.broadcast_to(H00, (nz, nb, nb)).copy()       # bottom surface
    eps_s_top = eps_s.copy()                                # top surface
    eps = eps_s.copy()                                      # bulk
    alpha = np.broadcast_to(H01, (nz, nb, nb)).copy()
    beta = np.broadcast_to(H01.conj().T, (nz, nb, nb)).copy()
    zI = z[:, None, None] * eye

    for _ in range(int(max_iter)):
        g = np.linalg.inv(zI - eps)
        agb = alpha @ g @ beta
        bga = beta @ g @ alpha
        eps_s = eps_s + agb
        eps_s_top = eps_s_top + bga
        eps = eps + agb + bga
        alpha = alpha @ g @ alpha
        beta = beta @ g @ beta
        if max(
            float(np.linalg.norm(alpha, axis=(1, 2)).max()),
            float(np.linalg.norm(beta, axis=(1, 2)).max()),
        ) < float(tol):
            break

    side = (side or "bottom").strip().lower()
    if side == "top":
        return np.linalg.inv(zI - eps_s_top)
    if side == "both":
        return np.linalg.inv(zI - eps_s) + np.linalg.inv(zI - eps_s_top)
    return np.linalg.inv(zI - eps_s)


def _surface_spectral_vs_energy(
    Hf, kpar: np.ndarray, normal: int, c_n: float, energies: FloatArray, eta: float,
    max_iter: int, tol: float, n_kn: int, side: str = "bottom", h_batch=None,
) -> FloatArray:
    """A_surf(E) = -Im Tr G_s(k_par, E+i*eta)/pi for all energies at one k_par."""
    H00, H01 = _principal_layer_blocks(Hf, kpar, normal, c_n, n_kn, h_batch=h_batch)
    z = energies + 1j * eta
    g = _sancho_surface_green(z, H00, H01, max_iter, tol, side=side)
    return -np.imag(np.einsum("zii->z", g)) / np.pi


def _surface_observables_vs_energy(
    Hf,
    kpar: np.ndarray,
    normal: int,
    c_n: float,
    energies: FloatArray,
    eta: float,
    max_iter: int,
    tol: float,
    n_kn: int,
    side: str = "bottom",
    h_batch=None,
) -> tuple[FloatArray, FloatArray]:
    """Return total and orbital-resolved surface spectral weight vs energy."""
    H00, H01 = _principal_layer_blocks(Hf, kpar, normal, c_n, n_kn, h_batch=h_batch)
    z = energies + 1j * eta
    g = _sancho_surface_green(z, H00, H01, max_iter, tol, side=side)
    total = -np.imag(np.einsum("zii->z", g)) / np.pi
    orbital = -np.imag(np.diagonal(g, axis1=1, axis2=2).T) / np.pi
    return np.asarray(total, dtype=np.float64), np.asarray(orbital, dtype=np.float64)


def compute_surface_spectrum(config: QXTIConfig, *, progress: bool = True) -> dict[str, Any]:
    """Semi-infinite surface/edge spectral function via Lopez-Sancho decimation.

    Produces the surface spectral map A_surf(k_par, E) along a path (the headline
    plot that shows surface/edge states in the projected bulk gap), an optional
    constant-energy map A_surf(kx, ky; E0), and the k_par-integrated surface LDOS
    g_surf(E). All use ``-Im Tr G_surf/pi``.
    """
    t_start = time.perf_counter()
    sim = QXTISimulation(config=config)
    hamiltonian = sim.build_hamiltonian()
    kgrid = sim.build_kgrid(hamiltonian)
    lcfg = config.ldos
    Hf = hamiltonian._matrix_at
    nb = int(hamiltonian.basis_size)
    dim = int(hamiltonian.dimension)
    bounds = hamiltonian.reciprocal_box_bounds()

    normal = _resolve_normal_axis(lcfg.surface_normal, dim)
    if normal >= len(bounds):
        normal = len(bounds) - 1
    c_n = float(np.pi / max(abs(bounds[normal][1]), 1e-12))
    max_iter = int(lcfg.surface_max_iter)
    tol = float(lcfg.surface_tol)
    n_kn = int(lcfg.surface_kn_points)
    side = str(getattr(lcfg, "surface_side", "bottom")).strip().lower() or "bottom"
    if side not in {"bottom", "top", "both"}:
        raise ValueError("ldos.surface_side must be one of: bottom, top, both.")
    # Surface momenta are independent -> parallelize the Lopez-Sancho decimation
    # across cores (each k does its own small linear algebra; embarrassingly par).
    from concurrent.futures import ThreadPoolExecutor
    from qxti.analytics.mesh_response import default_worker_count
    from qxti.analytics.theory_response import _model_h_batch
    n_workers = int(getattr(lcfg, "n_workers", 0) or 0) or default_worker_count()
    # Vectorized H builder (if the model exposes H_batch): the per-k layer FT is
    # the surface bottleneck -> building all n_kn layers at once is a big speedup.
    h_batch = _model_h_batch(hamiltonian)

    num_e = max(int(lcfg.num_energies), 2)
    chunk = max(32, min(20000, int(5.0e6 / max(nb * num_e, 1))))
    e_min, e_max = _resolve_energy_window(config, Hf, kgrid.points(), int(kgrid.total_points), chunk, progress=progress)
    if not (e_max > e_min):
        e_max = e_min + 1.0
    energies = np.linspace(e_min, e_max, num_e)
    spacing = (e_max - e_min) / (num_e - 1)
    eta = float(lcfg.eta) if lcfg.eta > 0.0 else 3.0 * spacing
    # The bulk DOS is a broadened eigenvalue sum on a finite k-grid. If the
    # broadening is narrower than the energy-grid step, the Lorentzians fall
    # between grid points -> spiky DOS AND a sum-rule deficit. The surface Green
    # function keeps the sharp user eta (needed for crisp surface states); only
    # the bulk DOS is broadened to at least a few energy-grid steps.
    bulk_eta = max(eta, 3.0 * spacing)

    axis_names = "xyz"
    if progress:
        print(
            f"[ldos-surface] semi-infinite along {axis_names[normal]} (layer spacing "
            f"c={c_n:.4f} a.u.), Lopez-Sancho decimation. Surface momentum = the "
            f"other {max(dim - 1, 1)} direction(s).",
            flush=True,
        )

    bulk_dos, bulk_pdos = _compute_bulk_dos_components(
        Hf=Hf,
        k_points=kgrid.points(),
        weights=_dos_k_weights(hamiltonian, kgrid),
        energies=energies,
        eta=bulk_eta,
        broadening="lorentzian",
        basis_size=nb,
        projected=bool(lcfg.projected),
        progress=progress,
        progress_label="[ldos-surface bulk]",
    )
    bulk_cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (bulk_dos[1:] + bulk_dos[:-1]) * np.diff(energies)))
    )
    bulk_integral = float(bulk_cumulative[-1])
    bulk_fermi = float(lcfg.fermi_level)
    bulk_filling = float(np.interp(bulk_fermi, energies, bulk_cumulative))

    # --- Surface spectral map A_surf(k_par, E) along the path ---
    spectral = spectral_bottom = spectral_top = path_s = path_k = None
    path_ticks: list[float] = []
    path_tick_labels: list[str] = []
    if bool(lcfg.spectral_enabled):
        path_k, path_s, path_ticks, path_tick_labels = _build_spectral_path(hamiltonian, lcfg)
        num_k = path_k.shape[0]
        spectral = np.zeros((num_k, num_e), dtype=np.float64)
        if side == "both":
            spectral_bottom = np.zeros((num_k, num_e), dtype=np.float64)
            spectral_top = np.zeros((num_k, num_e), dtype=np.float64)
        timer = ProgressTimer(total=num_k)

        def _path_row(i):
            if side == "both":
                b = _surface_spectral_vs_energy(Hf, path_k[i], normal, c_n, energies, eta,
                                                max_iter, tol, n_kn, side="bottom", h_batch=h_batch)
                t = _surface_spectral_vs_energy(Hf, path_k[i], normal, c_n, energies, eta,
                                                max_iter, tol, n_kn, side="top", h_batch=h_batch)
                return i, b, t
            return i, _surface_spectral_vs_energy(Hf, path_k[i], normal, c_n, energies, eta,
                                                  max_iter, tol, n_kn, side=side, h_batch=h_batch), None

        def _emit(done):
            timer.completed = done
            if progress and (done % max(1, num_k // 20) == 0 or done == num_k):
                print(f"[ldos-surface] path {done}/{num_k} ({n_workers} cores), "
                      f"ETA {timer.eta_text()}", flush=True)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for n, (i, a, b) in enumerate(ex.map(_path_row, range(num_k)), 1):
                if side == "both":
                    spectral_bottom[i] = a; spectral_top[i] = b; spectral[i] = a + b
                else:
                    spectral[i] = a
                _emit(n)

    # --- Constant-energy surface map A_surf(kx, ky; E0) ---
    plane = plane_bottom = plane_top = plane_kx = plane_ky = None
    plane_energy = None
    if bool(lcfg.spectral_plane_enabled):
        plane_energy = float(lcfg.spectral_plane_energy) if lcfg.spectral_plane_energy is not None else float(lcfg.fermi_level)
        bx = bounds[0] if len(bounds) > 0 else (-np.pi, np.pi)
        by = bounds[1] if len(bounds) > 1 else (-np.pi, np.pi)
        kx_vals = np.linspace(
            lcfg.spectral_plane_kx_min if lcfg.spectral_plane_kx_min is not None else bx[0],
            lcfg.spectral_plane_kx_max if lcfg.spectral_plane_kx_max is not None else bx[1],
            max(int(lcfg.spectral_plane_nkx), 2),
        )
        ky_vals = np.linspace(
            lcfg.spectral_plane_ky_min if lcfg.spectral_plane_ky_min is not None else by[0],
            lcfg.spectral_plane_ky_max if lcfg.spectral_plane_ky_max is not None else by[1],
            max(int(lcfg.spectral_plane_nky), 2),
        )
        z0 = np.array([plane_energy + 1j * eta])
        plane = np.zeros((ky_vals.size, kx_vals.size), dtype=np.float64)
        if side == "both":
            plane_bottom = np.zeros((ky_vals.size, kx_vals.size), dtype=np.float64)
            plane_top = np.zeros((ky_vals.size, kx_vals.size), dtype=np.float64)
        if progress:
            print(f"[ldos-surface] constant-energy map E0={plane_energy:.4g} a.u. "
                  f"({ky_vals.size}x{kx_vals.size}, {n_workers} cores) ...", flush=True)

        def _plane_row(iy):
            ky = ky_vals[iy]
            rb = np.zeros(kx_vals.size); rt = np.zeros(kx_vals.size); rtot = np.zeros(kx_vals.size)
            for ix, kx in enumerate(kx_vals):
                kpar = np.array([kx, ky, float(lcfg.spectral_plane_kz)], dtype=np.float64)
                H00, H01 = _principal_layer_blocks(Hf, kpar, normal, c_n, n_kn, h_batch=h_batch)
                if side == "both":
                    gb = _sancho_surface_green(z0, H00, H01, max_iter, tol, side="bottom")
                    gt = _sancho_surface_green(z0, H00, H01, max_iter, tol, side="top")
                    rb[ix] = -np.imag(np.trace(gb[0])) / np.pi
                    rt[ix] = -np.imag(np.trace(gt[0])) / np.pi
                    rtot[ix] = rb[ix] + rt[ix]
                else:
                    g = _sancho_surface_green(z0, H00, H01, max_iter, tol, side=side)
                    rtot[ix] = -np.imag(np.trace(g[0])) / np.pi
            return iy, rb, rt, rtot

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for iy, rb, rt, rtot in ex.map(_plane_row, range(ky_vals.size)):
                plane[iy] = rtot
                if side == "both":
                    plane_bottom[iy] = rb; plane_top[iy] = rt
        plane_kx, plane_ky = kx_vals, ky_vals

    # --- Surface LDOS g_surf(E): integrate A_surf over the in-plane grid ---
    surface_dos = np.zeros(num_e, dtype=np.float64)
    surface_pdos = np.zeros((nb, num_e), dtype=np.float64) if bool(lcfg.projected) else None
    dos_bottom = np.zeros(num_e, dtype=np.float64) if side == "both" else None
    dos_top = np.zeros(num_e, dtype=np.float64) if side == "both" else None
    pdos_bottom = np.zeros((nb, num_e), dtype=np.float64) if side == "both" and bool(lcfg.projected) else None
    pdos_top = np.zeros((nb, num_e), dtype=np.float64) if side == "both" and bool(lcfg.projected) else None
    if bool(lcfg.surface_ldos_enabled) or bool(lcfg.projected):
        inplane = [a for a in range(3) if a != normal]
        axis_vals = (kgrid.kx_values, kgrid.ky_values, kgrid.kz_values)
        a0 = np.asarray(axis_vals[inplane[0]], dtype=np.float64) if inplane else np.array([0.0])
        a1 = np.asarray(axis_vals[inplane[1]], dtype=np.float64) if len(inplane) > 1 else np.array([0.0])
        total = a0.size * a1.size
        timer = ProgressTimer(total=total)
        done = 0
        for v0 in a0:
            for v1 in a1:
                kpar = np.zeros(3, dtype=np.float64)
                if inplane:
                    kpar[inplane[0]] = v0
                if len(inplane) > 1:
                    kpar[inplane[1]] = v1
                if side == "both":
                    total_bottom, orbital_bottom = _surface_observables_vs_energy(
                        Hf, kpar, normal, c_n, energies, eta, max_iter, tol, n_kn, side="bottom",
                        h_batch=h_batch
                    )
                    total_top, orbital_top = _surface_observables_vs_energy(
                        Hf, kpar, normal, c_n, energies, eta, max_iter, tol, n_kn, side="top",
                        h_batch=h_batch
                    )
                    if bool(lcfg.surface_ldos_enabled):
                        dos_bottom += total_bottom
                        dos_top += total_top
                        surface_dos += total_bottom + total_top
                    if surface_pdos is not None and pdos_bottom is not None and pdos_top is not None:
                        pdos_bottom += orbital_bottom
                        pdos_top += orbital_top
                        surface_pdos += orbital_bottom + orbital_top
                else:
                    total_side, orbital_side = _surface_observables_vs_energy(
                        Hf, kpar, normal, c_n, energies, eta, max_iter, tol, n_kn, side=side,
                        h_batch=h_batch
                    )
                    if bool(lcfg.surface_ldos_enabled):
                        surface_dos += total_side
                    if surface_pdos is not None:
                        surface_pdos += orbital_side
                done += 1
            timer.completed = done
            if progress:
                print(f"[ldos-surface] surface LDOS {done}/{total}, ETA {timer.eta_text()}", flush=True)
        norm = float(total)
        if bool(lcfg.surface_ldos_enabled):
            surface_dos /= norm
            if dos_bottom is not None and dos_top is not None:
                dos_bottom /= norm
                dos_top /= norm
        if surface_pdos is not None:
            surface_pdos /= norm
            if pdos_bottom is not None and pdos_top is not None:
                pdos_bottom /= norm
                pdos_top /= norm

    if surface_pdos is not None and not bool(lcfg.surface_ldos_enabled):
        surface_dos = np.sum(surface_pdos, axis=0, dtype=np.float64)

    runtime = time.perf_counter() - t_start
    if progress:
        print(
            f"[ldos-surface] done in {runtime:.1f}s. "
            f"\\int g_bulk dE = {bulk_integral:.4f} (expected {nb} bands).",
            flush=True,
        )

    dataset: dict[str, Any] = {
        "kind": "ldos",
        "method": "surface",
        "surface_normal": axis_names[normal],
        "surface_side": side,
        "surface_compare_layers": int(_surface_compare_layers(lcfg, normal)),
        "broadening": "lorentzian",  # surface Green eta is intrinsically Lorentzian
        "eta": float(eta),
        "energies": np.asarray(energies, dtype=np.float64),
        "dos": np.asarray(bulk_dos, dtype=np.float64),
        "surface_dos": np.asarray(surface_dos, dtype=np.float64),
        "projected": bool(bulk_pdos is not None),
        "basis_size": int(nb),
        "dimension": dim,
        "fermi_level": float(lcfg.fermi_level),
        "cumulative": np.asarray(bulk_cumulative, dtype=np.float64),
        "integral": bulk_integral,
        "filling_at_fermi": bulk_filling,
        "model_name": str(getattr(hamiltonian, "model_name", "") or config.run_name()),
        "au_to_ev": _AU_TO_EV,
        # plot style metadata (shared with bulk mode)
        "spectral_cmap": str(lcfg.spectral_cmap),
        "spectral_log_scale": bool(lcfg.spectral_log_scale),
        "spectral_log_vmin": lcfg.spectral_log_vmin,
        "spectral_log_vmax": lcfg.spectral_log_vmax,
        "spectral_alpha_min": float(lcfg.spectral_alpha_min),
        "spectral_alpha_max": float(lcfg.spectral_alpha_max),
    }
    if bulk_pdos is not None:
        dataset["pdos"] = np.asarray(bulk_pdos, dtype=np.float64)
        dataset["orbital_labels"] = _orbital_labels(hamiltonian, nb)
    if surface_pdos is not None:
        dataset["surface_pdos"] = np.asarray(surface_pdos, dtype=np.float64)
    if dos_bottom is not None and dos_top is not None:
        dataset["surface_bottom_dos"] = np.asarray(dos_bottom, dtype=np.float64)
        dataset["surface_top_dos"] = np.asarray(dos_top, dtype=np.float64)
    if pdos_bottom is not None and pdos_top is not None:
        dataset["surface_bottom_pdos"] = np.asarray(pdos_bottom, dtype=np.float64)
        dataset["surface_top_pdos"] = np.asarray(pdos_top, dtype=np.float64)
    if spectral is not None:
        dataset["spectral"] = np.asarray(spectral, dtype=np.float64)
        if spectral_bottom is not None and spectral_top is not None:
            dataset["spectral_bottom"] = np.asarray(spectral_bottom, dtype=np.float64)
            dataset["spectral_top"] = np.asarray(spectral_top, dtype=np.float64)
        dataset["spectral_path_s"] = np.asarray(path_s, dtype=np.float64)
        dataset["spectral_path_k"] = np.asarray(path_k, dtype=np.float64)
        if path_ticks:
            dataset["spectral_path_ticks"] = np.asarray(path_ticks, dtype=np.float64)
            dataset["spectral_path_tick_labels"] = list(path_tick_labels)
    if plane is not None:
        dataset["spectral_plane"] = np.asarray(plane, dtype=np.float64)
        if plane_bottom is not None and plane_top is not None:
            dataset["spectral_plane_bottom"] = np.asarray(plane_bottom, dtype=np.float64)
            dataset["spectral_plane_top"] = np.asarray(plane_top, dtype=np.float64)
        dataset["spectral_plane_kx"] = np.asarray(plane_kx, dtype=np.float64)
        dataset["spectral_plane_ky"] = np.asarray(plane_ky, dtype=np.float64)
        dataset["spectral_plane_energy"] = float(plane_energy)

    return {"dataset": dataset, "runtime_seconds": runtime, "integral": float("nan"), "expected": nb}


def _compute_spectral_plane(
    hamiltonian: Any,
    Hf,
    lcfg: Any,
    energy0: float,
    eta: float,
    broadening: str,
    *,
    progress: bool,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Constant-energy spectral map A(kx, ky; E0) on a (nky, nkx) grid at fixed kz.

    ``A(kx, ky; E0) = sum_n L_eta(E0 - E_n(kx, ky, kz))`` -- a Fermi-surface /
    constant-energy cut (the kx-ky display mode from the original example).
    """
    bounds = hamiltonian.reciprocal_box_bounds()
    bx = bounds[0] if len(bounds) > 0 else (-np.pi, np.pi)
    by = bounds[1] if len(bounds) > 1 else (-np.pi, np.pi)

    kx_min = lcfg.spectral_plane_kx_min if lcfg.spectral_plane_kx_min is not None else bx[0]
    kx_max = lcfg.spectral_plane_kx_max if lcfg.spectral_plane_kx_max is not None else bx[1]
    ky_min = lcfg.spectral_plane_ky_min if lcfg.spectral_plane_ky_min is not None else by[0]
    ky_max = lcfg.spectral_plane_ky_max if lcfg.spectral_plane_ky_max is not None else by[1]
    kz = float(lcfg.spectral_plane_kz)

    nkx = max(int(lcfg.spectral_plane_nkx), 2)
    nky = max(int(lcfg.spectral_plane_nky), 2)
    kx_vals = np.linspace(float(kx_min), float(kx_max), nkx)
    ky_vals = np.linspace(float(ky_min), float(ky_max), nky)

    plane = np.zeros((nky, nkx), dtype=np.float64)
    if progress:
        print(
            f"[ldos] constant-energy map A(kx,ky; E0={energy0:.4g} a.u.) "
            f"on a {nky}x{nkx} grid ...",
            flush=True,
        )
    for iy, ky in enumerate(ky_vals):
        H0 = np.array([Hf(float(kx), float(ky), kz) for kx in kx_vals], dtype=np.complex128)
        ev = np.linalg.eigvalsh(H0)                              # (nkx,nb)
        diff = energy0 - ev                                      # (nkx,nb)
        plane[iy] = _broadening_kernel(diff, eta, broadening).sum(axis=1)
    return np.asarray(kx_vals, dtype=np.float64), np.asarray(ky_vals, dtype=np.float64), plane
