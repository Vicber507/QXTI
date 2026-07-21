---
tags: [playbook, howto, physics]
updated: 2026-07-21
---

# 🛠️ Playbook — Añadir un observable u orden de respuesta

> Meta: un observable nuevo (o subir el orden perturbativo) sin romper la cadena de validación.
> [[Home]] · [[qxti.response]] · [[qxti.analytics]] · [[Concept - Response Engines]]

## Subir el orden perturbativo

- Es un **parámetro**, no código nuevo: `[cmd] max_order = N` (o `[xtp] susceptibility_orders`).
- La recursión ρ⁽ˢ⁾ ya es genérica en s ([[Concept - Perturbative Recursion]]).
- ⚠️ Coste: la **referencia per-k** `rho_order_s` escala ~7^(s−1)/k → s≤3 práctico, s≥5 exploratorio.
  Los motores **mesh/CMD** escalan bien; usa esos para producción.
- El `halo` de streaming es `max_order − 1`: subir el orden **aumenta la RAM por bloque** →
  [[Concept - Memory and Parallelism]].

## Añadir un observable nuevo (p.ej. otra proyección de J)

1. **Cálculo:** si sale de ρ⁽ˢ⁾, agrégalo en [[qxti.response|XTP]] (integración BZ) **y/o** en
   [[qxti.data|StreamingCurrentAccumulator]] si lo quieres al vuelo. Mantén la definición idéntica en
   ambos (como intra/inter).
2. **Dataset:** exponlo con una clave nueva en el dict de [[qxti.data|HarmonicData]] y guárdalo con
   `save_dataset_npz`. Nombra la clave de forma consistente (`current_<algo>`).
3. **Gráfica:** lee la clave en [[qxti.graphics|plot_harmonics.py]] y dibújala (sin calcular física).
4. **Motor cerrado (opcional):** si quieres la versión "theory", refléjalo en
   [[qxti.analytics|mesh_response]]/`theory_response`.

## Checklist de invariantes

- [ ] Cualquier split cumple **parte + parte = total** exacto → [[Concept - Inter-Intra Decomposition]].
- [ ] Signo de corriente `j = −v` consistente ([[qxti.physics|operators]]).
- [ ] Gradiente covariante **de un solo tiro** (no sumes Berry aparte) →
      [[Concept - Perturbative Recursion]].
- [ ] Resultado **independiente de `n_workers`** (bit-exacto).
- [ ] Si tocas mesh, **revalida** vs `rho_analytic` per-k a malla fina (no gruesa) →
      [[Concept - Response Engines]].

## Prueba

`pytest tests/test_mesh_response.py tests/test_response_integration.py tests/test_analytics_rho.py`.
Si el observable es nuevo, añade un test que compare mesh vs per-k (o CMD vs mesh) a malla fina.

---

Relacionado: [[Concept - Response Engines]] · [[Concept - Perturbative Recursion]] · [[Playbook - Invariants Not to Break]]
