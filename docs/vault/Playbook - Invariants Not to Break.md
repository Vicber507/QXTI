---
tags: [playbook, invariants, critical]
updated: 2026-07-21
---

# 🛠️ Playbook — Invariantes que NO se rompen

> La lista de *lazos* críticos. Léela antes de cualquier cambio. Cada ⛔ enlaza a su explicación.
> [[Home]] · [[Architecture Map]]

## Física / correctitud

1. ⛔ **Gradiente covariante de un solo tiro** (Wilson). Nunca sumar `−i[A,ρ]` por separado —
   doble conteo en órdenes ≥2. → [[Concept - Perturbative Recursion]]
2. ⛔ **`intra + inter = total` exacto**; signos consistentes (`j=−v`). Un split que no suma al
   total es un bug. → [[Concept - Inter-Intra Decomposition]]
3. ⛔ **`valence_occupation` por-k** (no argsort global). → [[Concept - Occupation Gauge Basis]]
4. ⛔ **Unidades atómicas** en el núcleo; eV/Å solo en gráficas. → [[Concept - Atomic Units]]
5. ⛔ **Pesos BZ = los de XTP** (Simpson/trapecio/punto-medio); DOS cumple `∫g dE = basis_size`. →
   [[Concept - BZ Grid and Degeneracy Guard]]
6. ⛔ **Simetría k→−k** de la malla `shifted` (offset 0.5). → [[Concept - BZ Grid and Degeneracy Guard]]
7. ⛔ **Base `band` + gauge `length`**; `velocity`/`dipole` gauge no implementados. →
   [[Concept - Occupation Gauge Basis]]
8. ⛔ **`H_batch` bit-exacto vs `H`** escalar. → [[Playbook - Add a Model]]

## Arquitectura / orden

9. ⛔ **Hamiltoniano ⊥ láser** (no se importan entre sí; se combinan en CMD). → [[Architecture Map]]
10. ⛔ **`graphics` no calcula física**; solo lee `.npz`. → [[qxti.graphics]]
11. ⛔ **Import lazy `simulation → theory_response`** (dentro de la función). No lo subas al tope →
    ciclo de imports. → [[Concept - Response Engines]]
12. ⛔ **No renombrar claves del dataset** sin actualizar [[qxti.graphics]] (las lee por nombre). →
    [[qxti.data]]
13. ⛔ **`with_standard_output_dirs()` idempotente**; respeta rutas explícitas del usuario. → [[qxti.core]]

## Rendimiento / robustez

14. ⛔ **Dejar ≥1 GB de RAM libre** (`reserve_gb`, multiplataforma). → [[Concept - Memory and Parallelism]]
15. ⛔ **Resultado independiente de `n_workers`** (paralelismo bit-exacto). →
    [[Concept - Memory and Parallelism]]
16. ⛔ **Malla-k con ≥2 puntos por eje activo** o CMD sale de la ruta covariante. →
    [[Concept - Perturbative Recursion]]

## Deudas conocidas (mejorar con cuidado, no romper)

- **Gauge no-Abeliano en degeneraciones:** [[qxti.physics|BandGaugeFrame]] alinea fase secuencial
  por eje, no globalmente suave en 2D/3D. `operators.berry_connection` salta pares casi degenerados.
  Un fix sería transporte no-Abeliano en subespacios degenerados.
- **`float16_complex` scratch:** ε~1e-3, casa con la tolerancia del solver pero es arriesgado en
  órdenes altos. → [[qxti.utils]]
- **Comparar mesh vs per-k:** hazlo a **malla fina** (pasos de gradiente igualados); a malla gruesa
  difieren legítimamente. → [[Concept - Response Engines]]
- **Stubs vacíos** (`pdg`, `results`, `constants`, `*_solver`, …): son puntos de extensión, no
  código a "reparar". → [[Architecture Map]]

## Antes de dar por bueno un cambio

```bash
pytest tests/          # suite completa
# y para física de respuesta, revalida a malla fina:
pytest tests/test_mesh_response.py tests/test_analytics_rho.py tests/test_response_integration.py
```

---

Relacionado: [[Architecture Map]] · [[Concept - Response Engines]] · todos los `Concept -*`
