---
tags: [concept, physics, grid]
updated: 2026-07-21
---

# 🧠 Concept — Malla BZ, cuadratura y guard de degeneración

> Cómo se integra sobre la zona de Brillouin y por qué la malla se "empuja" fuera de las
> degeneraciones. [[Home]] · [[qxti.grids]] · [[qxti.response]]

## Pesos de cuadratura (una sola fuente de verdad)

La integral BZ `Σ_k w_k (…)` usa pesos `w_k` normalizados a 1. La regla la fija
[[qxti.response|XTP._axis_integration_weights]]:
- **Simpson 1/3** para N impar uniforme,
- **trapezoidal** en otro caso,
- **punto medio** si `shifted=True` (Monkhorst-Pack).

⛔ **Invariante:** [[qxti.analytics|mesh_response]] y [[qxti.analytics|dos]] **copian** estos pesos.
Si cambias la cuadratura, cámbiala en XTP y verifica que DOS siga cumpliendo `∫g dE = basis_size`.

## Guard de degeneración (evitar armónicos espurios)

Si la malla-k cae **exactamente** sobre un nodo de Dirac/Weyl (gap=0, conexión de Berry singular),
aparecen **armónicos espurios**. Solución automática:
- `[kgrid] auto_degeneracy_guard = true` (default) en [[qxti.core|simulation.build_kgrid]] escanea el
  gap mínimo y, si toca una degeneración, **empuja N** al grid simétrico limpio más cercano
  (offset Monkhorst-Pack 0.5).
- `[kgrid] shifted = true` desplaza a puntos medios (nunca borde/centro).

### Guard consciente de la conexión de Berry

El guard de gap solo reacciona al **cero exacto** (gap < 1e-6·bandwidth). Pero la respuesta
intrabanda/anómala por debajo del gap la dirige la **conexión de Berry** `A_mn = v_mn/(ε_m−ε_n) ~ 1/gap`,
que ya es enorme en **todo un vecindario** del nodo — así una malla puede *superar* el piso de
degeneración exacta y aun así caer en un punto **casi-nodo** donde `|A|` se dispara (los near-misses
genéricos de frank8). `[kgrid] berry_singularity_guard = true` (default) escanea también
`max_k |v_mn|/gap` (velocidad por diferencias finitas + `eigh`), y si el peor pico supera
`berry_guard_ratio`× el valor típico (percentil 90, criterio scale-free) **empuja N a la malla que
minimiza ese peor pico** (`_grid_berry_diagnostics`). Preserva la simetría k→−k.

⚠️ **Alcance honesto:** esto hace la respuesta en malla gruesa **más robusta/reproducible** (evita los
peores spikes), pero **no** converge por sí solo una respuesta cuyo peso está genuinamente concentrado
en los nodos (frank8 HHG). Para eso: **refino adaptativo cerca del nodo** (octree) o el solver en
**gauge de velocidad `tddm`** (que nunca forma `1/gap`). Ver memoria `frank8-belowgap-nodes`.

⛔ **Invariante:** preservar la **simetría k→−k** (offset exacto 0.5). Un offset irracional esquiva
todo punto racional **pero** rompe esa simetría (y con ella la cancelación de respuestas prohibidas).
Ver [[qxti.grids]].

> Memoria del proyecto: los WSM gapless (frank8) tienen "hojas de resonancia" de ancho ~γ/v cerca
> del nodo; necesitan mallas enormes. A veces la no-convergencia es **física**, no un bug de malla.

## Máscara BZ (opcional)

`[xtp] bz_mask_*` aplica una máscara radial **multiplicativa** cerca del borde de zona. No cambia
la regla de suma; solo suprime contribuciones de k de borde. Útil para aislar la física de nodo.

---

Relacionado: [[qxti.grids]] · [[Concept - Perturbative Recursion]] · [[Models and Inputs]]
