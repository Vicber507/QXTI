---
tags: [concept, engine, validation]
updated: 2026-07-21
---

# 🧠 Concept — Los motores de respuesta y la cadena de validación

> Hay **cuatro** formas de calcular la respuesta perturbativa. Saber cuál es producción y cuál es
> referencia evita "arreglar" un motor comparándolo mal con otro. [[Home]] · [[qxti.analytics]] · [[qxti.response]]

## Los cuatro motores

| Motor | Dónde | Dominio | Rol | Coste |
| --- | --- | --- | --- | --- |
| **CMD** | [[qxti.response|cmd.py]] | tiempo | propaga ρ⁽ˢ⁾(k,t); independiente | O(orden·Nk·Nt log Nt) |
| **mesh_response** | [[qxti.analytics|mesh_response.py]] | frecuencia (n·ω) | **producción** cerrada, vectorizada | O(orden·Nk) |
| **theory_response** | [[qxti.analytics|theory_response.py]] | frecuencia | cablea mesh a la config (normaliza como XTP) | = mesh |
| **rho_analytic** | [[qxti.analytics|rho_analytic.py]] | frecuencia | **referencia per-k** (la "verdad") | O(Nk·7^(s−1)) |

`time_domain_currents` (dentro de mesh_response) es la variante **multi-láser** de la vía cerrada.

## La cadena de validación (no la rompas)

```
rho_analytic (per-k, lento)  ══ precisión de máquina ══  mesh_response (producción)
      ║                                                        ║
      ╚═══ valida ═══▶  CMD (tiempo) ══ FFT en n·ω, uniforme ══╝
```

- `test_mesh_matches_perk_to_machine_precision` fija el acuerdo **mesh ↔ per-k** (con pasos de
  gradiente igualados).
- ⚠️ **Trampa de comparación:** a malla **gruesa**, mesh (gradiente Wilson con paso 2π/N) y per-k
  (`dk_grad=1e-3` local) usan **pasos distintos** → difieren en órdenes ≥2. Eso **no** es bug;
  convergen al refinar. Para comparar de verdad: iguala `dk_grad` al paso de malla o usa malla fina.

## Cuál usa cada flag

- `-cmd` `response_method`: `theory` (mesh, rápido) | `simulation` (CMD) | `both`.
- `-xtp` `susceptibility_method`: `theory` (Kubo o1 + mesh o≥2) | `simulation` (CMD por freq) | `both`.
- Multi-láser ⇒ siempre rama tiempo (`time_domain_currents`).

Detalle de despacho en [[qxti.core|simulation / susceptibility_scan]] y [[Data Flow]].

## El ciclo de import (cuidado)

`theory_response` importa `simulation` en el **tope**; `simulation` importa `compute_hhg_spectrum`
de forma **lazy** (dentro de la función). ⛔ No lo muevas al tope → import circular. Es la razón de
que theory pueda reusar los *builders* de simulation sin duplicar código.

---

Relacionado: [[Concept - Perturbative Recursion]] · [[Concept - Inter-Intra Decomposition]] · [[Architecture Map]]
