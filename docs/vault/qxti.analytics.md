---
tags: [package, analytics, engine]
updated: 2026-07-21
---

# 📦 qxti.analytics — motores cerrados + DOS

> La forma **cerrada/vectorizada** de la respuesta (rápida, de producción) y la **referencia
> per-k** (validación). Aquí vive la "vía theory". [[Home]] · [[Concept - Response Engines]]

**Depende de:** [[qxti.response]] (CMD, XTP, distributions), [[qxti.core]] (config, simulation),
[[qxti.utils]] (memory, progress)
**Usado por:** [[qxti.core]] (simulation `-cmd` theory, susceptibility_scan `-xtp` theory, ldos_scan `-ldos`)

## Archivos

| Archivo | Rol | Símbolos clave |
| --- | --- | --- |
| [mesh_response.py](../../qxti/analytics/mesh_response.py) | Recursión **mesh-vectorizada** (`pfddm`) | `harmonic_currents_meshed`, `time_domain_currents`, `precompute_band_data`, `perk_harmonic_currents` |
| [theory_response.py](../../qxti/analytics/theory_response.py) | Cablea mesh a producción (HHG + σ/χ) = `pfddm` | `compute_hhg_spectrum`, `compute_susceptibility_spectrum`, `compute_linear_response_spectrum` |
| [tddm.py](../../qxti/analytics/tddm.py) | **`tddm`** full NO-perturbativo (gauge velocidad) | `compute_hhg_spectrum_tddm`, `compute_susceptibility_spectrum_tddm`, `_tddm_current_time`, `_vector_potential_from_field` |
| [rho_analytic.py](../../qxti/analytics/rho_analytic.py) | **Referencia per-k** (Hipólito 2018) | `rho_order_s`, `sigma1_kubo`, `sigma_analytic`, `compare_rho_vs_qxti` |
| [dos.py](../../qxti/analytics/dos.py) | 4 motores de DOS/LDOS | `compute_dos_spectrum` |
| [hipolito2018.py](../../qxti/analytics/hipolito2018.py) | σ⁽¹⁾ analítica rápida (validación) | `analytical_sigma1_fast`, `load_model` |

> El eje de motores (`pfddm`/`ptddm`/`tddm`) y la comparación están en [[Concept - Response Engines]].
> `tddm` (gauge velocidad, streamable sin halo, exp-Euler en base instantánea) es el motor para
> **fuera del régimen perturbativo**; en `-cmd` da el espectro directo, en `-xtp` extrae χ⁽ˢ⁾ por
> **escalado en amplitud**.

Guía detallada: [[MESH_RESPONSE]] (en `docs/`).

## mesh_response.py — el motor de producción

`harmonic_currents_meshed(H_func, kpts, shape, bounds, weights, E_field, omega, max_order, *,
gamma, gamma_pop, mu, T_au, dimension, distribution, n_workers, reserve_gb, h_batch,
return_intraband, progress_cb)`:
- Recursión cerrada en s·ω, **streaming por bloques de k** con `halo = max_order−1` (interior
  bit-exacto), ThreadPool, **RAM guard** ([[Concept - Memory and Parallelism]]).
- `return_intraband=True` → devuelve `(J_total, J_intra)` (intra = traza diagonal `Σ_n v_nn ρ_nn`).
- Gradiente covariante vía Wilson links (`cov_grad`, `np.roll` de U). Ver [[Concept - Perturbative Recursion]].

`time_domain_currents(band, weights, E_t, dt, max_order, *, return_intraband)` → rama **multi-láser**
(FFT en el tiempo; todos los productos de mezcla). Reduce a la forma cerrada en cada n·ω para CW.

## theory_response.py — producción "theory"

- `compute_hhg_spectrum(config)`: mono-láser → `harmonic_currents_meshed(..., return_intraband=True)`
  y arma el **split intra/inter real** en el dataset. Multi-láser → `_hhg_multilaser_result`
  (usa `time_domain_currents`). Reusa los *builders* de [[qxti.core|simulation]].
- `compute_susceptibility_spectrum`: orden 1 Kubo streaming; órdenes ≥2 `_mesh_susceptibility`
  (ThreadPool freq×dir, **solo entradas diagonales**).
- ⛔ Import **top-level** de `simulation` (ok); el ciclo lo rompe simulation con su import lazy.

## rho_analytic.py — la referencia (no producción)

`rho_order_s(H_func, kx, ky, kz, E_field, omega, gamma, mu, T_au, max_order=3, dk_grad=1e-3,
dk_vel=1e-4, distribution=None)` → ρ⁽ˢ⁾ per-k en base de banda local. Coste **~7^(s−1)
diagonalizaciones/k** → s≤3 práctico, s≥5 exploratorio. **Mismo** gradiente Wilson que CMD/mesh.
⚠️ Pasa `distribution=valence_occupation` para igualar la config (si no, usa Fermi por energía).

## La cadena de validación (léela)

```
rho_analytic (per-k, lento, "verdad")  ══ machine precision ══  mesh_response (producción)
        │                                                             │
        └── también valida ──▶ CMD (tiempo)  ══ FFT en n·ω ══════════┘
theory_response = mesh_response cableado a la config (misma normalización que XTP)
```

Detalle y matices en [[Concept - Response Engines]].

## dos.py

`method`: `eigenvalues` (bulk `−Im Tr G/π`), `surface` (López-Sancho, arcos de Fermi,
`side=bottom/top/both`), `finite` (placa real-space, hoy solo Haldane). Pesos BZ = los de XTP.

➕ **Extender:** nueva vía de cálculo → [[Playbook - Add an Observable or Order]];
nuevo método LDOS → [[Playbook - Add an LDOS Method or Plot]].

---

Relacionado: [[Concept - Perturbative Recursion]] · [[Concept - Inter-Intra Decomposition]] · [[qxti.response]]
