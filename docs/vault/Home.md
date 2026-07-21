---
tags: [moc, home]
updated: 2026-07-21
---

# 🏠 QXTI Vault — Home

> **Empieza aquí.** Este vault es el "mapa mental" del código QXTI: cómo está
> estructurado, cómo se conectan las piezas (los *lazos*), qué reglas **no** se
> pueden romper, y cómo añadir cosas nuevas sin desordenar el sistema.
>
> **Cómo usarlo (para el usuario y para Claude):** al empezar un chat nuevo, lee
> este `Home` y el [[Architecture Map]]. Antes de tocar un módulo, abre su nota
> `qxti.<paquete>` y su [[Playbook - Invariants Not to Break|lista de invariantes]].
> Abre la carpeta `docs/` como *vault* en Obsidian para que los wikilinks y el
> grafo funcionen.

QXTI = simulación **perturbativa** de respuesta óptica (HHG, σ/χ, DOS) sobre
Hamiltonianos *tight-binding* genéricos. Todo internamente en **unidades atómicas**
([[Concept - Atomic Units]]).

---

## 🗺️ Mapas (leer primero)

- [[Architecture Map]] — grafo de dependencias, capas, y las **reglas de oro**.
- [[Data Flow]] — qué pasa de principio a fin en `-cmd`, `-xtp`, `-ldos`.

## 📦 Paquetes (`qxti/`)

| Nota | Rol en una línea |
| --- | --- |
| [[qxti.core]] | Orquestación: config + los 3 *runners* (`-cmd`/`-xtp`/`-ldos`). |
| [[qxti.physics]] | Hamiltoniano, operadores, gauge de banda, láser. |
| [[qxti.grids]] | Mallas: k, tiempo, frecuencia. |
| [[qxti.solvers]] | Integradores ODE (RKF45 / Adams-Bashforth). |
| [[qxti.response]] | Motores: `CMD` (tiempo) y `XTP` (observables BZ) + distribuciones. |
| [[qxti.analytics]] | Motores cerrados: `mesh_response`, `theory_response`, `rho_analytic`, `dos`. |
| [[qxti.data]] | Contenedores de datos, `StreamingCurrentAccumulator`, I/O `.npz`. |
| [[qxti.graphics]] | Ploteo (auto-detecta familias en disco). |
| [[qxti.utils]] | RAM guard, I/O de arrays, progreso/ETA. |
| [[Models and Inputs]] | `models/*.py` (Hamiltonianos) e `inputs/*.cfg`. |

## 🧠 Conceptos (el "por qué" y los *lazos* físicos)

- [[Concept - Response Engines]] — las 4 vías de cálculo y la **cadena de validación**.
- [[Concept - Perturbative Recursion]] — la recursión ρ⁽⁰…ᴺ⁾ + gradiente covariante (Wilson).
- [[Concept - Inter-Intra Decomposition]] — intra/inter, x/y/z, RCP/LCP (y por qué el inter "alto" es físico).
- [[Concept - Occupation Gauge Basis]] — `valence_occupation` por-k, gauge length, base de banda.
- [[Concept - Atomic Units]] — Hartree/Bohr adentro, eV/Å solo en gráficas.
- [[Concept - Memory and Parallelism]] — RAM guard (≥1 GB libre) + workers.
- [[Concept - BZ Grid and Degeneracy Guard]] — pesos de cuadratura + guard de degeneración.

## 🛠️ Playbooks (cómo añadir sin romper)

- [[Playbook - Add a Model]]
- [[Playbook - Add a Config Option]]
- [[Playbook - Add an Observable or Order]]
- [[Playbook - Add an LDOS Method or Plot]]
- [[Playbook - Invariants Not to Break]] ← **lee esto antes de cualquier cambio.**

## 📄 Docs largos previos (referencia de diseño, en `docs/`)

- [[ARCHITECTURE]] — visión de diseño original (⚠️ menciona `PDG` como organizador; hoy
  eso es un *stub* vacío, la organización real vive en [[qxti.core]] + [[qxti.graphics]]).
- [[HAMILTONIAN]] · [[LASER]] · [[MESH_RESPONSE]] — guías detalladas por tema.

---

### Convenciones de este vault

- **Invariante** = un *lazo* que si lo rompes, rompes física o rendimiento. Marcado con ⛔.
- **Extender aquí** = el punto único donde se añade una variante nueva. Marcado con ➕.
- Los enlaces a código son relativos: `../../qxti/...`.
- Cuando cambies el código, **actualiza la nota del paquete afectado** (mantén el mapa vivo).
