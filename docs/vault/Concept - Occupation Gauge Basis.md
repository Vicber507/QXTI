---
tags: [concept, physics]
updated: 2026-07-21
---

# 🧠 Concept — Ocupación, gauge y base

> Tres decisiones que fijan el ρ⁽⁰⁾ y la representación. [[Home]] · [[qxti.response]]

## Ocupación (ρ⁽⁰⁾ = diag(f_n))

`[cmd]/[xtp] distribution`: `fermi_dirac` | `valence_occupation` | `maxwell_boltzmann` |
`bose_einstein`. Definidas en [[qxti.response|distributions.py]].

⛔ **`valence_occupation` llena las `Nb//2` bandas más bajas POR-K** (argsort por cada k, no
global). Un argsort global mal-asigna el llenado cuando las bandas de valencia y conducción se
solapan en energía a través de la BZ (conos de Weyl **inclinados**). Iguala la `UniformValence`
de antelope. Para modelos con gap/no-inclinados, global y per-k coinciden — pero per-k es el correcto.

⚠️ Al usar la **referencia per-k** `rho_order_s`, pasa `distribution=valence_occupation`
explícitamente; su default es Fermi por energía y **no** coincide con la config.

## Relajación T1/T2

`T1T2Relaxation` (población T1, coherencia T2). Se aplican al **término fuente** de la recursión:
γ_pop = 1/T1 en la diagonal, γ_coh = 1/T2 fuera. `T=∞` ⇒ sin relajación; `0` ⇒ error.
`coherence_time`/`population_time` en el `.cfg` son T2/T1. Ver [[Concept - Perturbative Recursion]].

> Nota histórica: se probó añadir *smearing* puro encima de T1/T2 y se **quitó** (empeora la
> convergencia). No lo reintroduzcas sin una razón fuerte.

## Base y gauge

- **Base:** `band` (autobase de H(k)) es la de producción. La recursión asume ω_mn diagonal.
  `orbital` es una post-transformación. El **streaming** de J requiere `band`.
- **Gauge:** `length` (dipolo). ⛔ `velocity`/`dipole` gauge lanzan `NotImplementedError` en
  [[qxti.response|CMD]] — no están implementados.
- La **fase** de los autovectores la suaviza [[qxti.physics|BandGaugeFrame]] a lo largo de la malla
  (gauge no-Abeliano en degeneraciones **no** resuelto — ver [[Playbook - Invariants Not to Break]]).

---

Relacionado: [[Concept - Perturbative Recursion]] · [[Concept - Inter-Intra Decomposition]] · [[qxti.response]]
