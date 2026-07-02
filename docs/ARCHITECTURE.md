# QXTI Architecture

## Overview

QXTI is designed as a modular object-oriented framework for perturbative optical-response simulations in generic tight-binding Hamiltonians.

All calculations are intended to run internally in atomic units.

## Core Design Rules

1. The Hamiltonian layer must not depend on laser objects.
2. The laser layer must not depend on Hamiltonian objects.
3. `CMD` is the first layer allowed to combine Hamiltonian and laser information.
4. The graphics layer is responsible only for plotting.
5. `PDG` is responsible only for organizing and exporting data.
6. `QXTISimulation` coordinates the workflow but should not absorb physics-specific logic.
7. Atomic units are the default internal system everywhere.
8. Runtime configuration should originate from `inputParams.cfg`.

## Package Responsibilities

### `qxti/core`

- `config.py`: parse and validate simulation inputs.
- `simulation.py`: orchestrate the end-to-end run (HHG / time-domain).
- `susceptibility_scan.py`: `SusceptibilityScanRunner` (conductivity/susceptibility tensors).
- `ldos_scan.py`: `LDOSRunner` (density of states).
- `results.py`: define containers for simulation outputs.

### `qxti/physics`

- Hamiltonian abstractions and concrete models.
- Laser pulse definitions and composite laser systems.
- Operators and observables used during response calculations.

### `qxti/grids`

- Reciprocal-space grids.
- Time grids.
- Frequency grids.

### `qxti/solvers`

- Common solver interfaces.
- Perturbative and domain-specific implementations.

### `qxti/response`

- Coupling and response engines such as `CMD` and `XTP`.

### `qxti/data`

- Data organization, loading, and export utilities.

### `qxti/graphics`

- Plotting utilities for bands, DOS, response observables, and harmonics.

### `qxti/utils`

- Shared constants, validators, math helpers, and I/O helpers.

## Calculation families

A single input file (`inputs/inputParams.<model>.cfg`) describes one model and
can drive three independent calculations, selected by a flag to `main.py`. Every
family writes to a standardized folder derived from the input name, and
`graphics.py` plots whichever families have data on disk.

| Flag    | Section(s)                        | Engine / runner                              | Output dir                | Produces |
|---------|-----------------------------------|----------------------------------------------|---------------------------|----------|
| `-cmd`  | `[cmd]`, `[laser]`, `[timegrid]`  | `QXTISimulation` (CMD time-domain or theory) | `outputs/<model>/cmd`     | HHG current spectrum, harmonics |
| `-xtp`  | `[xtp]` (includes the sweep solver) | `SusceptibilityScanRunner`                 | `outputs/<model>/xtp`     | σ(ω), χ⁽ⁿ⁾(ω) tensors |
| `-ldos` | `[ldos]`                          | `LDOSRunner` → `qxti.analytics.dos`          | `outputs/<model>/ldos`    | DOS/PDOS plus bulk, surface (one/both faces), or finite-plate LDOS observables |

`-cmd` was previously spelled `-hhg`, which is kept as a deprecated alias. The
susceptibility-sweep solver parameters (relaxation times, distribution, gauge,
RKF45 settings) now live **inside `[xtp]`**; the standalone `[susceptibility_solver]`
section is still read for backward compatibility but is no longer required.

```bash
python main.py inputs/inputParams.graphene.cfg -cmd      # harmonics
python main.py inputs/inputParams.graphene.cfg -xtp      # tensors
python main.py inputs/inputParams.graphene.cfg -ldos     # density of states
python qxti/graphics/graphics.py inputs/inputParams.graphene.cfg   # auto-detects & plots
```

`graphics.py` with no flag plots **every** family that has data on disk
(hamiltonian, harmonics, susceptibility, ldos, response); families without data
are skipped. The same `-cmd`/`-xtp`/`-ldos` flags (or `--family`) restrict it to one.

### Density of states (`-ldos`)

`qxti/analytics/dos.py` now supports **four LDOS/DOS engines** selected with
`[ldos] method = ...`. All inputs remain in atomic units and all plots are drawn
in eV and 1/Angstrom where appropriate.

### 1. Bulk periodic DOS (`method = eigenvalues`)

This is the original bulk engine: it diagonalizes `H(k)` on the `[kgrid]` mesh
and broadens the eigenvalues:

```
g(E) = Σ_k w_k Σ_n  L_η(E − E_n(k)),     ∫ g(E) dE = basis_size
```

with `w_k` the same Brillouin-zone quadrature weights the XTP/theory engines use
(normalized to 1) and `L_η` a normalized Lorentzian or Gaussian of width `η`. The
Lorentzian choice equals the Green-function spectral trace
`−Im Tr[(E+iη−H)⁻¹]/π`, so it is the broadened bulk DOS computed through a cheap
Hermitian diagonalization. Orbital projection gives the PDOS
`g_α(E) = Σ_k w_k Σ_n |U_{αn}|² L_η(E−E_n)` with `Σ_α g_α = g`, and a straight
k-path yields the momentum-resolved spectral function `A(k,E)`. It has two
display modes (both `-Im Tr G/π`, drawn with the original surface-Green example's
`Blues` + linear-alpha colormap, log scale and fixed log color range):

- **E vs k** (`spectral_enabled`): `A(k,E)` along a straight k-path — the
  broadened band structure.
- **kx vs ky** (`spectral_plane_enabled`): `A(kx,ky;E₀)` at a fixed energy `E₀`
  (default the Fermi level) — a constant-energy / Fermi-surface map.

**Broadening choice:** Lorentzian (default) is exact for the Green-function DOS
and fine for gapless metals/semimetals; for **gapped** insulators its `1/d²`
tails leak into the gap, so prefer `broadening = gaussian` there (it decays fast,
keeps the gap clean, and the sum rule stays exact).

### 2. Semi-infinite surface / edge (`method = surface`)

Setting `method = surface` makes the crystal **semi-infinite along one direction**
(`surface_normal`) and solves the surface Green function by **López-Sancho
decimation**:

```
H(k) --Fourier along normal--> H00, H01 (principal-layer blocks)
G_surf(k_∥, E) = (E + iη - H00 - Σ_surf)^{-1},   Σ_surf from Sancho iteration
A_surf(k_∥, E) = -Im Tr G_surf / π
```

This breaks translational symmetry in the normal direction and reveals **surface
states** (3-D models) or **edge states** (2-D models) inside the projected bulk
gap, plus an optional constant-energy map `A_surf(kx,ky;E₀)` (Fermi arcs for Weyl
semimetals) and the k‖-integrated surface LDOS `g_surf(E)`.

`surface_side` chooses the termination. The López-Sancho decimation yields **both**
opposite faces at once (`bottom`/`top`), and `surface_side = both` returns their
sum. Their surface states are complementary — e.g. a Weyl slab's Fermi arcs route
one way on one face and the other way on the opposite face, so `both` shows **all**
arcs (including the ones connecting through the BZ centre); for a Haldane ribbon it
shows both chiral edge channels. Because the two faces are *infinitely* separated,
there is **no finite-size hybridisation gap** — this is the exact, fast replacement
for a finite ribbon/slab calculation (a thin finite slab would gap the
deeply-penetrating arcs). Pick a `surface_normal` for which opposite-chirality Weyl
nodes project to **distinct** surface-BZ points, and cut at the node energy.

### 3. Finite plaque (`method = finite`)

`method = finite` removes periodicity in **both** lattice directions. That means
there is **no good crystal momentum at all**, so an `E-k` spectral map is no
longer the correct observable. Instead QXTI builds a fully finite real-space
Hamiltonian and returns:

- finite DOS `g_finite(E)`
- optional projected DOS
- a finite-state spectrum colored by edge localization
- a site-resolved `LDOS(r,E)`

This mode is useful when you want a true finite flake/plate and spatial
information. At the moment the plaque builder is implemented for the 2-D Haldane
model.

### Summary of LDOS modes

- `eigenvalues`: periodic bulk DOS / PDOS / `A(k,E)`
- `surface`: semi-infinite Green function; `surface_side = bottom`/`top`/`both`
  (both faces at once — the exact, fast replacement for a finite ribbon/slab)
- `finite`: finite in two directions, no good `k`, use `LDOS(r,E)` instead

## Simulation Flow

```text
inputParams.cfg
        ↓
Config
        ↓
QXTISimulation
        ↓
Hamiltonian + LaserSystem + Grids
        ↓
OperatorFactory
        ↓
CMD
        ↓
rho^(0), rho^(1), rho^(2), rho^(3)
        ↓
XTP + ObservableCalculator
        ↓
P(t), J(t), chi, HHG
        ↓
SimulationResult
        ↓
PDG
        ↓
Graphics
```

## Implementation Note

The current repository state intentionally provides the full skeleton first. Scientific logic can now be added incrementally without changing the high-level structure.
