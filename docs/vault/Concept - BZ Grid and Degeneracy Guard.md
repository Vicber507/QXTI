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
