---
tags: [concept, performance, memory]
updated: 2026-07-21
---

# 🧠 Concept — RAM guard y paralelismo

> Cómo QXTI corre mallas grandes (p.ej. 120³ de 8 bandas) sin apagar la máquina, y cómo
> paraleliza. [[Home]] · [[qxti.utils]] · [[qxti.analytics]] · [[qxti.response]]

## RAM guard — dejar ≥1 GB libre (mac/win/linux)

[[qxti.utils|memory.py]] mide la RAM disponible de forma multiplataforma (psutil → `/proc/meminfo`
Linux, `vm_stat` macOS, `GlobalMemoryStatusEx` Windows). Los motores lo usan así:
- `memory_budget_bytes(reserve_gb)` — cuánto puedo pedir dejando `reserve_gb` libre.
- `pick_block_count(n_units, bytes_per_unit, reserve_gb, halo_units)` — cuántos planos-k por bloque.
- `ensure_headroom(need_bytes, reserve_gb)` — chequea; si falta, `gc.collect()` y recheck; si aún
  falta, `MemoryGuardError`.

⛔ **Invariante:** `reserve_gb` (config: `[cmd]/[ldos] reserve_gb`, default 1.0) **no se viola**.
Si la RAM no es medible, se **permite** (no bloquea). Es la solución permanente al "se apaga sola".

## Streaming por bloques de k (el truco de RAM)

[[qxti.analytics|mesh_response.harmonic_currents_meshed]] procesa la malla en **bloques de planos-k**
con `halo = max_order − 1` (los planos extra que necesita el gradiente). El **interior** de cada
bloque es **bit-exacto** vs la malla completa. Así 120³×8 bandas pasó de ~39 GB (thrashing) a un
"piso" de ~2 GB. `_plan_stream` decide el tamaño de bloque con el budget de RAM.

En [[qxti.response|CMD]], la ruta *streaming* acumula J(t) con
[[qxti.data|StreamingCurrentAccumulator]] y **no** escribe el ρ del último orden a disco; el scratch
de ρ puede ir en `float16_complex` (4 bytes/elem, ver [[qxti.utils|io_utils]]).

## Paralelismo (por dónde)

| Cálculo | Unidad paralela | Mecanismo |
| --- | --- | --- |
| mesh / theory (`-cmd`, `-xtp` o≥2) | bloques/planos-k, freq×dir | **ThreadPool** (NumPy suelta el GIL) |
| CMD tiempo | chunks de k | ThreadPool |
| `-xtp` simulation | **por frecuencia** | **ProcessPool** (cada worker: CMD+XTP en scratch) |

`n_workers = 0` ⇒ auto (`_default_worker_count`, **performance cores**; evita e-cores por el GIL).
Speedup real ~2.5× con 6-8 cores (limitado por GIL/BLAS). Bit-exacto vs serie.

⛔ **Invariantes**
- El resultado **no** depende de `n_workers` (paralelismo bit-exacto). Si cambia con los workers,
  hay un bug de acumulación/halo.
- No sobre-suscribir: el worker de proceso (`-xtp`) fuerza `n_workers=1` internamente.

---

Relacionado: [[qxti.utils]] · [[Concept - Perturbative Recursion]] · [[Playbook - Invariants Not to Break]]
