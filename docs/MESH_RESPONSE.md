# Mesh-vectorized perturbative response (`qxti/analytics/mesh_response.py`)

Fast, drop-in equivalent of the per-k analytic recursion
(`rho_analytic.rho_order_s`) for the perturbative harmonic currents
`J^(s)(s·ω) = Σ_k w_k Tr[v ρ^(s)]`.  Same physics, **21×–684× faster** (growing
with order), validated to **machine precision** against the old path.

> **This is now the single production path for orders ≥ 2.**  Both
> `theory_response.compute_hhg_spectrum` and `compute_susceptibility_spectrum`
> build the harmonics through this module (order 1 stays the streaming Kubo
> pass).  The old per-k `rho_order_s` loop and the `_order2_gridbased` order-2
> branch are no longer called in production; `rho_order_s`/`perk_harmonic_currents`
> remain only as the slow per-k **reference for the tests**.

## Why it is faster

The per-k recursion solves the length-gauge A1 recursion *independently at every
k-point*.  Because each order takes a **nested** finite-difference covariant
k-gradient (order `s` needs the gradient of `ρ^(s-1)`, which itself needs the
gradient of `ρ^(s-2)`, …), the cost is

```
per-k:   O(Nk × 7^(s-1))   diagonalizations         (7 = on-site + 6 FD neighbours, nested)
```

so order 4 at 120³ was ~12 h.  The mesh path solves the **same** recursion
vectorized over the whole Brillouin-zone mesh:

* one batched `eigh` over all k-points,
* `ρ^(1)(k, ω)` built on the full mesh at once,
* **one** Wilson-link covariant *mesh* gradient (`np.roll` between grid
  neighbours) per order to get `ρ^(2), ρ^(3), …`

```
mesh:    O(orders × Nk)     diagonalizations
```

so the whole 120³ order-4 sweep runs in seconds.

## The physics is identical

Both paths solve the same equation

```
ρ^(s)_mn(s·ω) = Σ_α E_α [D_k ρ^(s-1)]_mn / (s·ω + iγ − ω_mn),   D_k ρ = ∂_k ρ − i[A, ρ]
```

The covariant derivative `D_k ρ` is obtained **in one shot** from the
Wilson-transported finite difference (it already contains the `−i[A, ρ]` Berry
term), so **no separate commutator is added** — adding it double-counts the
connection and cancels the intraband/population channel (same convention as
`rho_order_s` and CMD's `_covariant_gradient_for_k_index`).

### Occupation: honor the configured `distribution`

`ρ^(1)` (and therefore every higher order) depends on the **band occupation**
`f`.  QXTI supports two fillings via `distribution` in the config:

- `valence_occupation` — fills by **band index** (band 0 occupied, band 1 empty,
  …), the fixed-valence convention that matches antelope's `UniformValence`;
- `fermi` — the **energy** Fermi step / Fermi–Dirac at the configured `μ, T`.

For a Weyl/Dirac semimetal these differ **a lot near the nodes** (up to ~0.5 per
state, because the band *index* stops tracking the energy ordering where bands
cross) — enough to change `σ^(2)` by a factor of several.  The mesh therefore
**honors the configured distribution** (`precompute_band_data(..., distribution=…)`,
wired from `_resolve_distribution(ccfg.distribution)` in both production paths),
so orders ≥ 2 use the **same filling as order 1 and as the configured engine**.

`distribution=None` defaults to the energy Fermi step — that is exactly what the
per-k `rho_order_s` reference uses, so the machine-precision tests below still
hold.  (Historically `rho_order_s` *always* used the energy step and silently
ignored `distribution`; routing the config through the mesh fixes that latent
inconsistency, and makes the QXTI-vs-antelope comparison — which is run with
`valence_occupation` — self-consistent across all orders.)

### Discretization: the one subtlety

`rho_order_s` differentiates with a fixed step `dk_grad` (evaluating H at
`k ± dk_grad`); the mesh differentiates across the k-**grid** neighbours (spacing
set by the grid).  These are two discretizations of the same continuous `∂_k`:

- On a **periodic lattice** with `dk_grad` = grid spacing, they are the **same
  finite-difference tree** — one vectorized, one looped — so they agree to
  **machine precision** (see validation below).
- With a different `dk_grad`, or a non-periodic (continuum) model whose grid
  wraps at the box edge, they differ only by the discretization and **converge
  to the same continuum value** as the grid refines.

## Validation

All numbers below use the periodic 4-band TaAs Weyl model (`natphys`, 3D), with
`dk_grad` = grid spacing so the two paths use the identical discretization.
Reproduce with `tests/test_mesh_response.py`, or run the full comparative +
convergence study (any config, honoring its `distribution`) with

```
python tools/compare_mesh_vs_perk.py --config inputs/inputParams.wsm.cfg
```

which sweeps the k-grid, times mesh vs per-k, checks |J^(s)| agreement, and
writes `outputs/mesh_vs_perk_comparison.png` (observable vs Nk, time vs Nk,
speedup, rel-error).  On the wsm WSM config (`valence_occupation`) the two paths
agree to **1e-13–1e-12** at every grid and the mesh is **65–100× faster by
Nk≈1.7k** (and reaches Nk≈1.8×10⁵ in ~5 s, where the per-k path would take hours);
the WSM harmonics converge only for fine grids (N≳32) because of the near-node
resonances — which is exactly why the fast path is needed.

### 1. Correctness — machine precision

`mesh_harmonic_currents` vs `perk_harmonic_currents` (relative error of the
BZ-summed current `J^(s)`):

| model                     | order 1 | order 2 | order 3 |
|---------------------------|---------|---------|---------|
| NatPhys 4-band (periodic) | 1.7e-15 | 9.1e-14 | 4.4e-14 |
| Orenstein 4-band (QXTI)   | 3.6e-16 | 1.4e-13 | 7.5e-13 |
| toy 2-band (continuum)*   | 1.4e-16 |  ~0     | 6.5e-5  |

\* the toy Dirac model is **not** periodic in the box, so `np.roll` wraps at the
edge while the per-k step evaluates the continuum H there — the residual `6.5e-5`
is purely that edge effect; for a lattice model it vanishes.

### 2. Speed — grows with order

NatPhys 4-band, 8³, z-drive:

| order | mesh (s) | per-k (s) | speedup | error   |
|-------|----------|-----------|---------|---------|
| 2     | 0.014    | 0.29      | **21×** | 1.4e-13 |
| 3     | 0.014    | 1.66      | **116×**| 1.4e-13 |
| 4     | 0.015    | 10.0      | **684×**| 1.4e-13 |

The mesh time is ~constant while the per-k time grows ×7 per order — so the
speedup multiplies by ~7 for each extra harmonic order.  Across grids (order 3)
the speedup is a steady ~100–120×; at 120³ order 4 it is ~1000×+ (seconds vs
~12 h).

### 3. Convergence

`|J^(s)|` from the mesh stabilizes as the grid refines (per-k gives the same
value at each grid, err ~1e-13), i.e. both describe the same converged physics.
Higher orders need finer grids (the near-node resonances), as expected for a
Weyl semimetal.

## API

```python
from qxti.analytics.mesh_response import (
    precompute_band_data, harmonic_currents, uniform_mp_grid)

kpts, w = uniform_mp_grid(bounds, shape)         # shifted Monkhorst-Pack
band = precompute_band_data(                     # eigh + velocities ONCE
        H_func, kpts, shape, bounds,
        mu=0.0, T_au=0.0, dimension=3,
        distribution=my_dist)                    # None -> energy Fermi step
J = harmonic_currents(                           # reuse `band` across the sweep
        band, w,
        E_field=[0, 0, 1e-3],                    # complex e^{-iωt} amplitude
        omega=0.02, max_order=4,
        gamma=1/T2, gamma_pop=1/T1)              # γ_pop defaults to γ
# J[s] = Σ_k w_k Tr[v ρ^(s)(k, s·ω)]  (complex 3-vector), s = 1..max_order
```

`mesh_harmonic_currents(H_func, …)` is the one-call convenience wrapper
(`precompute_band_data` + `harmonic_currents`) when you don't reuse the band data.

- `H_func(kx, ky, kz) -> (nb, nb)` hermitian, atomic units (e.g.
  `hamiltonian._matrix_at` for any QXTI model).
- `perk_harmonic_currents(...)` computes the same via `rho_order_s` — kept as the
  slow reference for tests.
- `σ^(s)` follows by the usual prefactor (`(1j**s) · spin / V_BZ / E^s`), applied
  identically to whichever path you use.

## Multi-frequency drives: the time-domain engine

The closed form above assumes a **single carrier ω** and returns ρ^(s)(sω).  For a
drive with **more than one laser/frequency** (e.g. two-colour, or a chirped/multi-
pulse field) the harmonics mix: the response lives at every ω_i±ω_j±… and the
single-ω peaks are ill-defined.  `time_domain_currents` solves the **same** A1
recursion in the **time domain** with the FULL field E(t):

```
S^(N)(t) = Σ_α E_α(t) [D_k ρ^(N-1)(t)]_α          # Wilson covariant grad, per t
ρ^(N)(ω) = FFT_t[S^(N)(t)] / (−ω − ω_mn + iΓ_mn)   # H_0 diagonal → per-element solve
ρ^(N)(t) = IFFT_ω[ρ^(N)(ω)]
```

Every mixing product appears automatically in the FFT of the current — no
enumeration of the ~F^N frequency channels.  Because H_0 is diagonal in the band
basis the propagator is an exact per-element frequency denominator (no time-
stepping error).  The observable current carries the documented per-order phase
`i^(s-1)`, which makes `J^(s)(t)` **real** (physical, Hermitian spectrum).

**Auto-dispatch (both engines).** `compute_hhg_spectrum` (cmd) and
`compute_susceptibility_spectrum` (xtp) select automatically: **1 laser → the fast
closed form**; **>1 laser → the time-domain engine**.  The time window is the union
of all pulses (`laser_system.temporal_bounds()`: first onset → last end, plus the
configured pre/post offsets), so two pulses with a temporal offset are integrated
over their full span.

### `pfddm_pulse` — realistic single-pulse response on demand

For a **single** pulse the fast closed form does **not** solve the pulse: it takes the
CW harmonic amplitude `χ⁽ˢ⁾(sω₀)` and simply **dresses it with the envelope**,
`J⁽ˢ⁾(t) = χ⁽ˢ⁾ · env(t)ˢ · e^{−isω₀t}`.  This is the **quasi-CW / adiabatic
approximation**: it places harmonic `s` exactly at `sω₀`, broadened only by `envˢ`.
It is exact for **many-cycle** pulses but misses the true harmonic **bandwidth**,
**spectral shift/chirp**, and **few-cycle** effects — and it is what depends on the
Hilbert-envelope reconstruction.

The input flag **`[cmd] pfddm_pulse`** chooses the single-pulse treatment:

| value | meaning |
|---|---|
| `envelope` *(default)* | fast closed-form `χ⁽ˢ⁾(sω₀) × envˢ` — quasi-CW, exact for many-cycle pulses |
| `full_field` | route the single pulse through the **same full-field `time_domain_currents` recursion** used for multi-colour drives → the **realistic pulsed response** (true harmonic bandwidth, chirp, few-cycle; immune to the envelope reconstruction) |

Multi-colour drives (>1 laser) **always** use the full-field path regardless of the
flag.  A single pulse run with `full_field` is reported with
`method = "pfddm-full-field"` (so it is not mistaken for a genuine multi-colour run);
it reproduces the closed-form peaks in the many-cycle limit and agrees with the
`envelope` result on the gauge-invariant linear response H1 (validated in
`tests/test_pfddm_pulse.py`).  The cost is that of the time-domain solve (`Nt` steps,
`(nk,nb,nb,Nt)` per-order frequency tensors), so `envelope` stays the default for
speed and `full_field` is opt-in for when pulse realism matters.

**Validation** (`tools/study_time_domain_engine.py` →
`outputs/time_domain_engine_study.png`):

- **A. Correctness** — in the single-carrier limit the time-domain harmonics match
  the closed form to **~1e-15** (every order, every time resolution) and `J(t)` is
  real to ~1e-15.  (100% agreement — the two are the same physics.)
- **B. Multi-frequency** — two lasers produce the mixing peaks ω1±ω2, 2ω1, 2ω2,
  2ω1±ω2, … that the single-carrier path cannot.
- **C. Timing** — the time integration is inherently costlier than the single-ω
  closed form (Nt time steps), but is optimized to **~6× faster than the naive
  version** (order-independent propagator `1/denom` built once → multiply not
  divide; Wilson links precomputed once; BLAS batched matmul for the covariant
  transport; a BLAS `tensordot` current trace; only the previous order's ρ(t) kept
  in memory).  Sub-second on the test grids (0.1–0.7 s at Nk≈1.7k, Nt≈256–512).
- **D. Window** — two temporally-offset pulses are both captured and integrated over
  the full window.

## Status — wired into production

The integration is **done**.  Orders ≥ 2 in both
`theory_response.compute_hhg_spectrum` and `compute_susceptibility_spectrum` go
through `precompute_band_data` + `harmonic_currents` (band data diagonalized
**once**, reused across the whole ω × drive-direction sweep); order 1 stays the
streaming Kubo pass.  This turns the high-order HHG/susceptibility sweeps from
hours into seconds with no change in the physics.  Smoke-checked end-to-end
(orders 1–4 finite, sane magnitudes) and the full suite passes (142).

### The SHG-tool order-2 tensor (`order2_tensor_at_omega` / `_order2_gridbased`)

Five SHG-phase/helicity tools import `order2_tensor_at_omega` (full `σ^(2)_{ijk}`
tensor), so it is **kept** rather than deleted.  It now uses the **same** one-shot
Wilson covariant gradient as the mesh — the double-counted `−i[A,ρ]` commutator
was removed (it cancels the intraband/population channel; the fix is confirmed by
two of three diagonal-input `σ^(2)` components matching the mesh to **1e-14**).
The remaining ~6% difference on the third component is the **diagonal population
channel of ρ^(2)** that the mesh keeps (matching `rho_order_s`) and the tensor
path still drops — i.e. the mesh is the more complete of the two.  Both now read
the same configured `distribution`, so they no longer disagree on occupation.

See also: [[covariant-gradient-gauge-fix]], [[theory-vs-simulation-engine]],
`docs/ARCHITECTURE.md`, `qxti/analytics/rho_analytic.py` (the per-k reference).
