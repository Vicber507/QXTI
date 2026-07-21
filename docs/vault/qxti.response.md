---
tags: [package, response, engine]
updated: 2026-07-21
---

# 📦 qxti.response — CMD, XTP, distribuciones

> El motor en el **tiempo** (`CMD`) y el motor de **observables BZ** (`XTP`), más los
> modelos de ocupación/relajación. [[Home]] · [[Architecture Map]] · [[Concept - Response Engines]]

**Depende de:** [[qxti.grids]], [[qxti.physics]], [[qxti.solvers]], [[qxti.utils]]
**Usado por:** [[qxti.core]], [[qxti.analytics]] (mesh_response usa CMD; theory/dos usan XTP), [[qxti.data]]

## Archivos

| Archivo | Rol | Símbolos |
| --- | --- | --- |
| [cmd.py](../../qxti/response/cmd.py) | Propagación perturbativa ρ⁽ˢ⁾(k,t) en el tiempo | `CMD` |
| [xtp.py](../../qxti/response/xtp.py) | Integra ρ⁽ˢ⁾ en BZ → P(t), J(t), χ⁽ˢ⁾(ω) | `XTP` |
| [distributions.py](../../qxti/response/distributions.py) | Ocupaciones + relajación T1/T2 | `fermi_dirac`, `valence_occupation`, `T1T2Relaxation` |
| [susceptibility_tensor_calculator.py](../../qxti/response/susceptibility_tensor_calculator.py) | Tensor χ desde varios probes (LSQ) | `SusceptibilityTensorCalculator` |

## CMD — el motor de tiempo

Recursión ρ⁽⁰⁾…ρ⁽ᴺ⁾. Orden 0 = `diag(f_n)` (ver [[Concept - Occupation Gauge Basis]]).
Cada orden s≥1 resuelve `dρ⁽ˢ⁾/dt = −(iω_mn+γ_mn)ρ⁽ˢ⁾ + E(t)·[D_k ρ⁽ˢ⁻¹⁾]` con integrador
exponencial trapezoidal (FFT o directo). El **término fuente** usa el **gradiente covariante de
un solo tiro** (Wilson links): `D_k ρ = ∂_k ρ − i[A,ρ]`. Salidas: `solve_time_domain(dir)`
(streaming a disco), `solve_time_domain_in_memory`, `solve_frequency_domain`.

⛔ **Invariantes** (ver [[Playbook - Invariants Not to Break]])
- Gradiente covariante **de un solo tiro**: nunca sumar `−i[A,ρ]` aparte (rompe simetría en órdenes ≥2).
- Base de banda (ω_mn diagonal); `gauge="length"` (velocity/dipole lanzan `NotImplementedError`).
- T1/T2 se aplican al **término fuente** (diag→γ_pop=1/T1, off-diag→γ_coh=1/T2), no a ρ directamente.
- `_default_worker_count()` usa **performance cores** (evita e-cores). Ver [[Concept - Memory and Parallelism]].

➕ **Extender:** observables al vuelo → `order_observe_callbacks` (los usa el
[[qxti.data|StreamingCurrentAccumulator]] para J intra/inter sin materializar ρ).

## XTP — observables macroscópicos

Desde `rho_orders` (dict o disco), integra en BZ: `polarization`, `current`,
`current_decomposition` (intra/inter, requiere `band_gauge_frame`), `*_frequency_domain`,
`linear_susceptibility/conductivity`, `effective_susceptibility_spectrum`,
`susceptibility_tensor_spectrum`. Total vía `einsum("mn,tnm->t", v, ρ)`; intra vía diagonal.

⛔ Los pesos de cuadratura BZ (`_axis_integration_weights`) son la **fuente de verdad** que
copian [[qxti.analytics|mesh_response/dos]]. `bz_mask` es multiplicativo (no cambia la regla de suma).
Ver [[Concept - Inter-Intra Decomposition]] y [[Concept - BZ Grid and Degeneracy Guard]].

## distributions.py

`valence_occupation` llena las `Nb//2` bandas más bajas **por-k** (argsort por k, no global) —
crítico en conos de Weyl inclinados. `T1T2Relaxation.from_rates(γ_pop, γ_coh)`.
Ver [[Concept - Occupation Gauge Basis]].

---

Relacionado: [[Concept - Perturbative Recursion]] · [[qxti.analytics]] · [[Concept - Response Engines]]
